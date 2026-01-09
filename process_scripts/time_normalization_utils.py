# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

"""
Time Normalization Utilities for Battery Data Preprocessing
============================================================

This module provides utilities to normalize time series data in battery cycling datasets
to ensure consistent cumulative time format across all cycles.

Key Features:
- Converts various time formats to cumulative seconds
- Fixes internal time resets within cycles
- Compresses large intra-cycle time gaps to 1 second
- Handles nanosecond timestamps and other special formats
- Maintains time continuity across cycles

Usage:
    from time_normalization_utils import normalize_cycle_times

    # Normalize a list of CycleData objects
    normalized_cycles = normalize_cycle_times(cycles, battery_id)
"""

from typing import List, Dict, Tuple, Optional
import copy


def normalize_cycle_times(cycles: List, battery_id: Optional[str] = None) -> List:
    """
    Normalize time data for all cycles to cumulative seconds format.

    This function processes a list of CycleData objects and ensures that:
    1. Time is in seconds (converts from nanoseconds if needed)
    2. Time is cumulative across all cycles
    3. Internal resets within cycles are fixed
    4. Time values are monotonically increasing

    Args:
        cycles: List of CycleData objects to normalize
        battery_id: Optional battery ID for dataset-specific handling

    Returns:
        List of CycleData objects with normalized time values

    Example:
        >>> cycles = [CycleData(...), CycleData(...)]
        >>> normalized = normalize_cycle_times(cycles, "ISU_ILCC_18650_NMC_25C_1")
    """
    if not cycles:
        return cycles

    # Deep copy to avoid modifying original data
    normalized_cycles = copy.deepcopy(cycles)

    # Check for special time formats (e.g., nanoseconds)
    normalized_cycles = handle_special_time_formats(normalized_cycles, battery_id)

    # Process time normalization
    cumulative_time = 0.0

    for cycle in normalized_cycles:
        if not hasattr(cycle, 'time_in_s') or not cycle.time_in_s:
            continue

        # Fix internal resets within the cycle (with dataset-specific handling)
        fixed_times, _ = fix_internal_resets(cycle.time_in_s, battery_id)

        # Make times relative to cycle start
        min_time = min(fixed_times) if fixed_times else 0
        relative_times = [t - min_time for t in fixed_times]

        # Add cumulative offset
        normalized_times = [t + cumulative_time for t in relative_times]

        # Update cycle with normalized times
        cycle.time_in_s = normalized_times

        # Update cumulative time for next cycle
        if normalized_times:
            cumulative_time = normalized_times[-1]

    return normalized_cycles


def handle_special_time_formats(cycles: List, battery_id: Optional[str] = None) -> List:
    """
    Handle special time formats for specific datasets.

    Special cases:
    - ISU_ILCC: Nanosecond timestamps (values > 1e15)
    - RWTH: Millisecond timestamps (need to divide by 1000)
    - HNEI: Large initial time offsets

    Args:
        cycles: List of CycleData objects
        battery_id: Optional battery ID for dataset identification

    Returns:
        List of CycleData objects with converted time formats
    """
    if not cycles or not cycles[0].time_in_s:
        return cycles

    first_time = cycles[0].time_in_s[0] if cycles[0].time_in_s else 0

    # Check for nanosecond timestamps (ISU_ILCC dataset)
    if first_time > 1e15 or (battery_id and 'ISU_ILCC' in battery_id):
        for cycle in cycles:
            if hasattr(cycle, 'time_in_s') and cycle.time_in_s:
                cycle.time_in_s = [t / 1e9 for t in cycle.time_in_s]

    # Handle RWTH dataset millisecond timestamps
    elif battery_id and 'RWTH' in battery_id:
        for cycle in cycles:
            if hasattr(cycle, 'time_in_s') and cycle.time_in_s:
                cycle.time_in_s = [t / 1000 for t in cycle.time_in_s]

    return cycles


def fix_internal_resets(times: List[float], battery_id: Optional[str] = None) -> Tuple[List[float], Dict]:
    """
    Fix internal time resets within a single cycle and compress large time gaps.

    This function identifies and corrects time resets that occur within a cycle,
    which are common in step-based cycling protocols (charge->rest->discharge->rest).
    It also compresses large intra-cycle time gaps to 1 second.

    Detection methods:
    1. Explicit reset to zero (after first element)
    2. Significant decrease (>50% drop when previous value >10s)
    3. Large backward jump (>100s backward)
    4. Large forward gap (>5 min = 300s) - compressed to 1 second

    Args:
        times: List of time values from a single cycle
        battery_id: Optional battery ID (for logging purposes)

    Returns:
        Tuple of (fixed_times, reset_info) where:
        - fixed_times: List with continuous time values
        - reset_info: Dict with reset count, positions, gaps removed, and gap details

    Example:
        >>> times = [0, 100, 200, 0, 100, 200]  # Reset at index 3
        >>> fixed, info = fix_internal_resets(times)
        >>> fixed
        [0, 100, 200, 200, 300, 400]
    """
    # Threshold for large gaps: 5 minutes = 300 seconds
    LARGE_GAP_THRESHOLD = 300

    if not times or len(times) <= 1:
        return times, {'reset_count': 0, 'reset_positions': [], 'large_gaps_removed': 0, 'large_gap_positions': []}

    # Process time values to fix resets and compress large gaps
    continuous_times = []
    accumulated_time = 0.0
    reset_positions = []
    large_gap_positions = []
    large_gaps_removed = 0

    # First time point
    continuous_times.append(accumulated_time)

    for i in range(1, len(times)):
        time_diff = times[i] - times[i-1]

        # Check for reset conditions
        is_reset = False
        is_large_gap = False

        # Method 1: Explicit zero (after first element)
        if times[i] == 0 and i > 0:
            is_reset = True
            reset_positions.append(i)
        # Method 2: Significant decrease (more than 50% drop)
        elif times[i] < times[i-1] * 0.5 and times[i-1] > 10:
            is_reset = True
            reset_positions.append(i)
        # Method 3: Large backward jump (more than 100 seconds)
        elif times[i] < times[i-1] - 100:
            is_reset = True
            reset_positions.append(i)
        # Method 4: Handle large intra-cycle gaps (>5 min) for ALL datasets
        elif time_diff > LARGE_GAP_THRESHOLD:
            is_large_gap = True
            large_gaps_removed += 1
            large_gap_positions.append({
                'index': i,
                'original_gap_seconds': time_diff,
                'original_gap_minutes': time_diff / 60
            })

        if is_reset:
            # Handle reset: continue from previous accumulated time
            accumulated_time = continuous_times[-1]
            continuous_times.append(accumulated_time)
        elif is_large_gap:
            # Compress large gap to 1 second
            accumulated_time += 1.0
            continuous_times.append(accumulated_time)
        else:
            # Normal time progression
            accumulated_time += time_diff
            continuous_times.append(accumulated_time)

    return continuous_times, {
        'reset_count': len(reset_positions),
        'reset_positions': reset_positions,
        'large_gaps_removed': large_gaps_removed,
        'large_gap_positions': large_gap_positions
    }


def validate_time_continuity(cycles: List) -> Dict:
    """
    Validate time continuity across all cycles.

    Checks for:
    - Negative time differences (time going backward)
    - Large time jumps between consecutive points
    - Non-monotonic time series

    Args:
        cycles: List of CycleData objects to validate

    Returns:
        Dict containing validation results and any issues found

    Example:
        >>> validation = validate_time_continuity(normalized_cycles)
        >>> if validation['has_issues']:
        ...     print(f"Found {len(validation['issues'])} issues")
    """
    validation_result = {
        'has_issues': False,
        'issues': [],
        'total_cycles': len(cycles),
        'cycles_with_issues': []
    }

    for cycle_idx, cycle in enumerate(cycles):
        if not hasattr(cycle, 'time_in_s') or not cycle.time_in_s:
            continue

        cycle_number = getattr(cycle, 'cycle_number', cycle_idx + 1)
        times = cycle.time_in_s

        # Check for negative differences
        for i in range(1, len(times)):
            diff = times[i] - times[i-1]
            if diff < 0:
                validation_result['has_issues'] = True
                validation_result['issues'].append({
                    'cycle': cycle_number,
                    'type': 'negative_time_diff',
                    'position': i,
                    'diff': diff
                })
                if cycle_number not in validation_result['cycles_with_issues']:
                    validation_result['cycles_with_issues'].append(cycle_number)

            # Check for large jumps (> 2 hours)
            if abs(diff) > 7200:
                validation_result['has_issues'] = True
                validation_result['issues'].append({
                    'cycle': cycle_number,
                    'type': 'large_time_jump',
                    'position': i,
                    'jump_size': diff
                })
                if cycle_number not in validation_result['cycles_with_issues']:
                    validation_result['cycles_with_issues'].append(cycle_number)

    return validation_result


def get_cumulative_time_array(cycles: List) -> List[float]:
    """
    Extract all time points from cycles as a single cumulative array.

    Useful for visualization and analysis of the complete time series.

    Args:
        cycles: List of CycleData objects

    Returns:
        List of all time points in cumulative order

    Example:
        >>> all_times = get_cumulative_time_array(cycles)
        >>> plt.scatter(range(len(all_times)), all_times)
    """
    all_times = []

    for cycle in cycles:
        if hasattr(cycle, 'time_in_s') and cycle.time_in_s:
            all_times.extend(cycle.time_in_s)

    return all_times