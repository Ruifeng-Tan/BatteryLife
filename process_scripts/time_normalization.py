# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Time Normalization for BatteryLife Dataset
===========================================

This script normalizes time data across all battery datasets to ensure
consistent cumulative time format (in seconds) across all cycles.

Features:
- Normalizes time to cumulative seconds format
- Handles various time format inconsistencies
- Parallel processing for efficiency
- Generates detailed processing report

Key Issues Addressed:
1. Internal time resets within cycles (e.g., XJTU, CALB)
   - Multiple resets to 0 within a single cycle due to step-based timing
   - Typically occurs at charge->rest->discharge->rest transitions

2. Time jumps between cycles
   - Abrupt jumps or gaps in time continuity
   - May indicate removed/missing cycles

3. Inconsistent time formats across datasets
   - Some datasets start from 0 each cycle (HUST, MATR)
   - Others use cumulative time (Stanford, RWTH)

4. Special formats requiring conversion
   - Nanosecond timestamps (ISU_ILCC): values > 1e15
   - Large initial values (RWTH, HNEI): starting at millions of seconds

Usage:
    # Specify input directory
    python time_normalization.py --input /path/to/processed_data

    # Specify number of parallel workers
    python time_normalization.py --input /path/to/processed_data --workers 16

    # Quiet mode
    python time_normalization.py --input /path/to/processed_data --quiet

"""

import os
import pickle
import json
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
from tqdm import tqdm
import argparse
import copy
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings('ignore')


def process_single_battery_file(args):
    """
    Process a single battery file (for parallel processing).

    Args:
        args: Tuple of (pkl_file_path, output_path, dataset_name, inplace_mode)

    Returns:
        Tuple of (battery_id, issues_dict)
    """
    pkl_file, _, _, _ = args  # Only pkl_file is used; others kept for compatibility

    try:
        # Load battery data
        with open(pkl_file, 'rb') as f:
            battery_data = pickle.load(f)

        battery_id = battery_data.get('cell_id', pkl_file.stem)

        # Check if cycle_data exists
        if 'cycle_data' not in battery_data or not battery_data['cycle_data']:
            return battery_id, None

        # Deep copy to ensure data integrity
        normalized_data = copy.deepcopy(battery_data)

        # Process time normalization
        issues = normalize_battery_time(normalized_data)

        # Save normalized data (always overwrite original)
        with open(pkl_file, 'wb') as f:
            pickle.dump(normalized_data, f)

        return battery_id, issues

    except Exception as e:
        return pkl_file.stem, {'has_issues': True, 'error': str(e)}


def normalize_battery_time(normalized_data: Dict) -> Dict:
    """
    Normalize time data for a battery to cumulative seconds format.
    Modifies the data in-place (already deep copied).

    Args:
        normalized_data: Battery data dictionary (will be modified)

    Returns:
        issues_dict
    """
    # Initialize issues tracking
    issues = {
        'has_issues': False,
        'internal_resets': [],
        'time_jumps': [],
        'negative_times': [],
        'nanosecond_conversion': False,
        'original_time_format': None,
        'data_removed': []
    }

    # Get cycle data
    cycle_data = normalized_data['cycle_data']

    # Check for special cases (nanosecond timestamps)
    first_cycle = cycle_data[0] if cycle_data else None
    if first_cycle and 'time_in_s' in first_cycle:
        first_time = first_cycle['time_in_s'][0] if first_cycle['time_in_s'] else 0

        # Check for nanosecond timestamp (ISU_ILCC)
        if first_time > 1e15:
            issues['nanosecond_conversion'] = True
            issues['has_issues'] = True
            # Convert all times from nanoseconds to seconds
            for cycle in cycle_data:
                if 'time_in_s' in cycle and cycle['time_in_s']:
                    cycle['time_in_s'] = [t / 1e9 for t in cycle['time_in_s']]

    # Process time normalization
    cumulative_time = 0
    prev_cycle_end = 0

    for cycle_idx, cycle in enumerate(cycle_data):
        cycle_number = cycle.get('cycle_number', cycle_idx + 1)

        # Get time data
        if 'time_in_s' not in cycle or not cycle['time_in_s']:
            continue

        original_times = cycle['time_in_s']

        # Fix internal resets within the cycle
        fixed_times, reset_info = fix_internal_resets(original_times)

        if reset_info['reset_count'] > 0:
            issues['internal_resets'].append({
                'cycle': cycle_number,
                'reset_count': reset_info['reset_count'],
                'reset_positions': reset_info['reset_positions'][:5]  # First 5 positions
            })
            issues['has_issues'] = True

        # Check for negative times
        if any(t < 0 for t in fixed_times):
            issues['negative_times'].append(cycle_number)
            issues['has_issues'] = True
            # Fix negative times
            fixed_times = [max(0, t) for t in fixed_times]

        # Make times relative to cycle start (if not already)
        min_time = min(fixed_times) if fixed_times else 0
        if min_time > 0:
            relative_times = [t - min_time for t in fixed_times]
        else:
            relative_times = fixed_times

        # Check for time jump from previous cycle
        if cycle_idx > 0 and prev_cycle_end > 0:
            expected_start = prev_cycle_end
            actual_start = cumulative_time
            gap = actual_start - expected_start

            # Check if there might be removed cycles
            if abs(gap) > 7200:  # More than 2 hours gap
                issues['time_jumps'].append({
                    'after_cycle': cycle_number - 1,
                    'to_cycle': cycle_number,
                    'gap_seconds': gap,
                    'possible_data_removal': True
                })
                issues['has_issues'] = True

                # Check if cycles might have been removed
                if cycle_idx > 0:
                    prev_cycle_num = cycle_data[cycle_idx - 1].get('cycle_number', cycle_idx)
                    if cycle_number - prev_cycle_num > 1:
                        issues['data_removed'].append({
                            'between_cycles': [prev_cycle_num, cycle_number],
                            'missing_cycles': cycle_number - prev_cycle_num - 1
                        })

        # Add cumulative offset
        normalized_times = [t + cumulative_time for t in relative_times]

        # Update cycle with normalized times
        cycle['time_in_s'] = normalized_times

        # Update cumulative time for next cycle
        if normalized_times:
            cumulative_time = normalized_times[-1]
            prev_cycle_end = cumulative_time

    # Determine original time format
    if cycle_data and len(cycle_data) > 1:
        first_cycle_start = cycle_data[0].get('time_in_s', [0])[0]
        second_cycle_start = cycle_data[1].get('time_in_s', [0])[0] if len(cycle_data) > 1 else 0

        if issues['nanosecond_conversion']:
            issues['original_time_format'] = 'nanosecond_timestamp'
        elif first_cycle_start < 100 and second_cycle_start < 100:
            issues['original_time_format'] = 'zero_based_per_cycle'
        else:
            issues['original_time_format'] = 'cumulative'

    return issues


def fix_internal_resets(times: List[float]) -> Tuple[List[float], Dict]:
    """
    Fix internal time resets within a cycle.

    This implements the logic: "Identify zeros after the first zero,
    split into segments, and concatenate"

    Args:
        times: List of time values

    Returns:
        Tuple of (fixed_times, reset_info)
    """
    if not times or len(times) <= 1:
        return times, {'reset_count': 0, 'reset_positions': []}

    # Find all reset points
    segments = []
    current_segment = []
    reset_positions = []

    for i in range(len(times)):
        if i == 0:
            # First element starts the first segment
            current_segment = [times[i]]
        else:
            # Check for reset conditions
            is_reset = False

            # Method 1: Explicit zero (after first element)
            if times[i] == 0 and i > 0:
                is_reset = True

            # Method 2: Significant decrease (more than 50% drop)
            elif times[i] < times[i-1] * 0.5 and times[i-1] > 10:
                is_reset = True

            # Method 3: Large backward jump (more than 100 seconds)
            elif times[i] < times[i-1] - 100:
                is_reset = True

            if is_reset:
                # Save current segment and start new one
                if current_segment:
                    segments.append(current_segment)
                current_segment = [times[i]]
                reset_positions.append(i)
            else:
                # Continue current segment
                current_segment.append(times[i])

    # Add final segment
    if current_segment:
        segments.append(current_segment)

    # If no resets found, return original
    if len(segments) == 1:
        return times, {'reset_count': 0, 'reset_positions': []}

    # Concatenate segments with continuous time
    continuous_times = []
    accumulated_time = 0

    for segment in segments:
        # Make segment relative to its start
        segment_start = segment[0] if segment else 0
        relative_segment = [t - segment_start for t in segment]

        # Add to continuous times with accumulated offset
        for t in relative_segment:
            continuous_times.append(t + accumulated_time)

        # Update accumulated time
        if relative_segment:
            accumulated_time = continuous_times[-1]

    return continuous_times, {
        'reset_count': len(reset_positions),
        'reset_positions': reset_positions
    }


class ParallelTimeNormalizationProcessor:
    """
    Time normalization processor with parallel processing.
    Always modifies files in-place.
    """

    def __init__(self, input_dir: str, verbose: bool = True, n_workers: int = None):
        """
        Initialize the processor.

        Args:
            input_dir: Path to processed_data directory
            verbose: Print progress
            n_workers: Number of parallel workers (default: CPU count - 1)
        """
        self.input_dir = Path(input_dir)
        self.verbose = verbose
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.output_dir = self.input_dir  # Always modify in-place

        # Initialize global issues report
        self.issues_report = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'input_directory': str(self.input_dir),
                'output_directory': str(self.output_dir),
                'inplace_mode': True,
                'total_datasets': 0,
                'total_batteries': 0,
                'batteries_with_issues': 0,
                'parallel_workers': self.n_workers
            },
            'dataset_issues': {},
            'battery_issues': {}
        }

    def process_all_datasets(self):
        """
        Process all datasets with parallel processing.
        """
        # Get all dataset directories
        datasets = [d for d in os.listdir(self.input_dir)
                   if os.path.isdir(self.input_dir / d)
                   and d != 'Life labels']

        self.issues_report['metadata']['total_datasets'] = len(datasets)

        if self.verbose:
            print(f"\nFound {len(datasets)} datasets to process")
            print(f"Using {self.n_workers} parallel workers")
            print(f"Mode: In-place modification")
            print("-" * 60)

        # Process each dataset
        for dataset_name in tqdm(sorted(datasets), desc="Processing datasets", disable=not self.verbose):
            self.process_dataset_parallel(dataset_name)

        # Save issues report
        self.save_issues_report()

        # Print summary
        if self.verbose:
            self.print_summary()

    def process_dataset_parallel(self, dataset_name: str):
        """
        Process a single dataset using parallel processing.

        Args:
            dataset_name: Name of the dataset
        """
        dataset_input_path = self.input_dir / dataset_name
        dataset_output_path = dataset_input_path  # Always in-place

        # Get all pickle files
        pkl_files = list(dataset_input_path.glob('*.pkl'))

        if not pkl_files:
            return

        # Initialize dataset issues tracking
        if dataset_name not in self.issues_report['dataset_issues']:
            self.issues_report['dataset_issues'][dataset_name] = {
                'total_batteries': 0,
                'batteries_with_internal_resets': [],
                'batteries_with_time_jumps': [],
                'batteries_with_negative_time': [],
                'special_format': None
            }

        dataset_issues = self.issues_report['dataset_issues'][dataset_name]
        dataset_issues['total_batteries'] = len(pkl_files)

        # Prepare arguments for parallel processing
        process_args = [(pkl_file, dataset_output_path, dataset_name, True)  # Always inplace=True
                       for pkl_file in pkl_files]

        # Process in parallel with progress bar
        desc = f"  {dataset_name} ({len(pkl_files)} files)"

        if len(pkl_files) < 5:  # Small dataset, use serial
            results = []
            for args in tqdm(process_args, desc=desc, leave=False, disable=not self.verbose):
                results.append(process_single_battery_file(args))
        else:  # Large dataset, use parallel
            with Pool(self.n_workers) as pool:
                results = list(tqdm(
                    pool.imap(process_single_battery_file, process_args),
                    total=len(process_args),
                    desc=desc,
                    leave=False,
                    disable=not self.verbose
                ))

        # Aggregate results
        for battery_id, issues in results:
            if issues and issues.get('has_issues'):
                self.issues_report['metadata']['batteries_with_issues'] += 1
                self.issues_report['battery_issues'][battery_id] = issues

                # Update dataset-level tracking
                if issues.get('internal_resets'):
                    dataset_issues['batteries_with_internal_resets'].append(battery_id)
                if issues.get('time_jumps'):
                    dataset_issues['batteries_with_time_jumps'].append(battery_id)
                if issues.get('negative_times'):
                    dataset_issues['batteries_with_negative_time'].append(battery_id)

            self.issues_report['metadata']['total_batteries'] += 1

    def save_issues_report(self):
        """
        Save the issues report to a JSON file.
        """
        # Save report in input directory
        report_file = self.input_dir / 'time_normalization_report.json'

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(self.issues_report, f, indent=2, ensure_ascii=False)

        if self.verbose:
            print(f"\nIssues report saved to: {report_file}")

    def print_summary(self):
        """
        Print a summary of the normalization process.
        """
        print("\n" + "=" * 60)
        print("TIME NORMALIZATION SUMMARY")
        print("=" * 60)

        metadata = self.issues_report['metadata']
        print(f"Mode: In-place modification")
        print(f"Total datasets processed: {metadata['total_datasets']}")
        print(f"Total batteries processed: {metadata['total_batteries']}")
        print(f"Batteries with issues fixed: {metadata['batteries_with_issues']}")
        print(f"Parallel workers used: {metadata['parallel_workers']}")

        print("\nIssues by dataset:")
        for dataset, issues in self.issues_report['dataset_issues'].items():
            if any([
                issues['batteries_with_internal_resets'],
                issues['batteries_with_time_jumps'],
                issues['batteries_with_negative_time']
            ]):
                print(f"\n{dataset}:")
                if issues['batteries_with_internal_resets']:
                    print(f"  - Internal resets: {len(issues['batteries_with_internal_resets'])} batteries")
                if issues['batteries_with_time_jumps']:
                    print(f"  - Time jumps: {len(issues['batteries_with_time_jumps'])} batteries")
                if issues['batteries_with_negative_time']:
                    print(f"  - Negative time: {len(issues['batteries_with_negative_time'])} batteries")

        # Report data removal suspicions
        data_removal_count = sum(1 for b in self.issues_report['battery_issues'].values()
                                if b.get('data_removed'))
        if data_removal_count > 0:
            print(f"\nPossible data removal detected in {data_removal_count} batteries")
            print("Check the JSON report for details")

        print("\n" + "=" * 60)


def main():
    """
    Main function to run parallel time normalization.
    Directly modifies original battery files to normalize time to cumulative seconds.
    """
    parser = argparse.ArgumentParser(
        description='Time normalization for BatteryLife dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Specify input directory
  python time_normalization.py --input /path/to/processed_data


  # Use 32 parallel workers
  python time_normalization.py --input /path/to/processed_data --workers 32

  # Quiet mode
  python time_normalization.py --input /path/to/processed_data --quiet
        """
    )

    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help="Input directory path containing battery data"
    )

    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=None,
        help='Number of parallel workers (default: CPU count - 1)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress progress output'
    )

    args = parser.parse_args()

    # Set default workers if not specified
    if args.workers is None:
        num_workers = max(1, cpu_count() - 1)
    else:
        num_workers = args.workers

    if not args.quiet:
        print("\n" + "="*60)
        print("TIME NORMALIZATION")
        print("="*60)
        print(f"Input directory: {args.input}")
        print(f"Using {num_workers} parallel workers")
        print("Mode: Direct modification of original files")
        print("-" * 60)

    # Create processor with direct modification
    processor = ParallelTimeNormalizationProcessor(
        input_dir=args.input,
        verbose=not args.quiet,
        n_workers=num_workers
    )

    # Process all datasets
    processor.process_all_datasets()

    if not args.quiet:
        print("\nTime normalization completed successfully!")
        print(f"Original files have been modified in: {processor.input_dir}")


if __name__ == '__main__':
    main()