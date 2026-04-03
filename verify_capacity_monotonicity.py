"""Minimal shareable capacity monotonicity checker."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.utils import (
    check_monotonicity,
    extract_segments,
    extract_segments_dataloader,
    find_pkl_files,
    get_cell_id,
    get_dataset_name,
)


EXTRACTOR_FUNCTIONS = {
    "masking": extract_segments,
    "dataloader": extract_segments_dataloader,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Check whether extracted charge/discharge capacities are non-decreasing "
            "within each cycle."
        )
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=os.path.expanduser("~/workspace/others/BatteryLife/dataset/"),
        help="Root directory containing BatteryLife dataset folders.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset directory names to include.",
    )
    parser.add_argument(
        "--extractor",
        choices=sorted(EXTRACTOR_FUNCTIONS.keys()),
        default="masking",
        help="Capacity extractor used before monotonicity checking.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Fixed tolerance. A violation is recorded when diff < -tolerance.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "results"),
        help="Directory for timestamped run outputs.",
    )
    return parser.parse_args(argv)


def _load_pickle(pkl_path):
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as exc:  # pragma: no cover - 防御性分支
        warnings.warn(f"Failed to read {pkl_path}: {exc}")
        return None


def _check_indices_continuous(indices, violation_index):
    """Check if the indices around the violation point are consecutive."""
    if violation_index + 1 >= len(indices):
        return False
    return int(indices[violation_index + 1]) - int(indices[violation_index]) == 1


def _process_single_file(task):
    pkl_path, extractor_name, tolerance = task
    dataset = get_dataset_name(pkl_path)
    data = _load_pickle(pkl_path)
    if data is None:
        return None

    cell_id = get_cell_id(data, pkl_path)
    total_cycles = len(data.get("cycle_data", []))
    extractor = EXTRACTOR_FUNCTIONS[extractor_name]

    try:
        segments, nominal_capacity = extractor(pkl_path, data=data)
    except Exception as exc:
        warnings.warn(f"Failed to extract segments from {pkl_path}: {exc}")
        return (dataset, cell_id, total_cycles, 0.0, [], [])

    if nominal_capacity <= 0:
        return (dataset, cell_id, total_cycles, 0.0, [], [])

    violation_rows = []
    monotonicity_check_records = []
    worst_diff = 0.0
    for segment in segments:
        for phase_name, capacity_values, indices in (
            ("charge", segment.charge_capacity, segment.charge_indices),
            ("discharge", segment.discharge_capacity, segment.discharge_indices),
        ):
            violations = check_monotonicity(
                capacity_values,
                tolerance=tolerance,
            )
            if violations:
                # Check index continuity for this cycle/phase
                continuous = _check_indices_continuous(indices, violations[0][0])
                monotonicity_check_records.append(
                    {
                        "dataset": dataset,
                        "file_name": os.path.basename(pkl_path),
                        "cell_id": cell_id,
                        "cycle_number": segment.cycle_number,
                        "phase": phase_name,
                        "continuous": continuous,
                        "violation_count": len(violations),
                        "violations": [
                            {"index": int(vi), "diff": float(df)}
                            for vi, df in violations
                        ],
                    }
                )

            for violation_index, diff_value in violations:
                violation_rows.append(
                    (
                        dataset,
                        cell_id,
                        segment.cycle_number,
                        phase_name,
                        violation_index,
                        diff_value,
                        tolerance,
                    )
                )
                worst_diff = min(worst_diff, diff_value)

    return (dataset, cell_id, total_cycles, worst_diff, violation_rows, monotonicity_check_records)


def _run_tasks(tasks, workers):
    progress_kwargs = {
        "total": len(tasks),
        "desc": "Processing pkl files",
    }

    if workers <= 1:
        return [
            result
            for result in tqdm(map(_process_single_file, tasks), **progress_kwargs)
            if result is not None
        ]

    with Pool(processes=workers) as pool:
        return [
            result
            for result in tqdm(
                pool.imap_unordered(_process_single_file, tasks),
                **progress_kwargs,
            )
            if result is not None
        ]


def _build_run_dir(output_dir, extractor_name, timestamp):
    run_dir = Path(output_dir) / f"capacity_mono_{extractor_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_config(run_dir, args, timestamp):
    config = {
        "extractor": args.extractor,
        "tolerance": args.tolerance,
        "datasets": args.datasets,
        "data_path": args.data_path,
        "timestamp": timestamp,
    }
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def _write_summary(run_dir, dataset_stats):
    with open(run_dir / "summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "num_cells",
                "total_cycles",
                "violation_cells",
                "violation_cycles",
                "violation_rate_pct",
                "cell_violation_rate_pct",
                "worst_diff",
            ]
        )
        for dataset in sorted(dataset_stats):
            stats = dataset_stats[dataset]
            violation_cycles = len(stats["violation_cycles"])
            violation_cells = len(stats["violation_cells"])
            total_cycles = stats["total_cycles"]
            num_cells = stats["num_cells"]
            writer.writerow(
                [
                    dataset,
                    num_cells,
                    total_cycles,
                    violation_cells,
                    violation_cycles,
                    f"{(100.0 * violation_cycles / total_cycles) if total_cycles else 0.0:.4f}",
                    f"{(100.0 * violation_cells / num_cells) if num_cells else 0.0:.4f}",
                    f"{stats['worst_diff']:.10f}",
                ]
            )


def _write_cell_stats(run_dir, cell_stats):
    with open(run_dir / "cell_stats.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "cell_id",
                "total_cycles",
                "violation_cycles",
                "violation_rate_pct",
                "worst_diff",
            ]
        )
        for dataset, cell_id in sorted(cell_stats):
            stats = cell_stats[(dataset, cell_id)]
            violation_cycles = len(stats["violation_cycles"])
            total_cycles = stats["total_cycles"]
            writer.writerow(
                [
                    dataset,
                    cell_id,
                    total_cycles,
                    violation_cycles,
                    f"{(100.0 * violation_cycles / total_cycles) if total_cycles else 0.0:.4f}",
                    f"{stats['worst_diff']:.10f}",
                ]
            )


def _write_violations_detail(run_dir, detail_rows):
    with open(run_dir / "violations_detail.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "cell_id",
                "cycle_number",
                "phase",
                "violation_index",
                "diff",
                "tolerance",
            ]
        )
        for row in sorted(detail_rows, key=lambda item: item[:5]):
            writer.writerow(
                [
                    row[0],
                    row[1],
                    row[2],
                    row[3],
                    row[4],
                    f"{row[5]:.10f}",
                    f"{row[6]:.10f}",
                ]
            )


def _write_violation_index(run_dir, args, timestamp, results):
    grouped = defaultdict(dict)
    for dataset, cell_id, total_cycles, _worst_diff, violation_rows, _mono_records in results:
        if not violation_rows:
            continue

        per_cycle = defaultdict(lambda: defaultdict(list))
        for _dataset, _cell_id, cycle_number, phase_name, _index, diff_value, _tol in violation_rows:
            per_cycle[cycle_number][phase_name].append(diff_value)

        grouped[dataset][cell_id] = {
            "total_cycles": total_cycles,
            "violations": [
                {
                    "cycle_number": cycle_number,
                    "phases": {
                        phase_name: {
                            "count": len(diff_values),
                            "worst_diff": min(diff_values),
                        }
                        for phase_name, diff_values in sorted(phase_map.items())
                    },
                }
                for cycle_number, phase_map in sorted(per_cycle.items())
            ],
        }

    payload = {
        "meta": {
            "extractor": args.extractor,
            "tolerance": args.tolerance,
            "timestamp": timestamp,
        },
        "datasets": {dataset: dict(cells) for dataset, cells in sorted(grouped.items())},
    }
    with open(run_dir / "violation_index.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main(argv=None):
    args = parse_args(argv)
    pkl_files = find_pkl_files(args.data_path, datasets=args.datasets)
    if not pkl_files:
        print("No pkl files found. Check --data_path and --datasets.")
        return 1

    tasks = [(pkl_path, args.extractor, args.tolerance) for pkl_path in pkl_files]
    results = _run_tasks(tasks, args.workers)

    dataset_stats = defaultdict(
        lambda: {
            "num_cells": 0,
            "total_cycles": 0,
            "violation_cells": set(),
            "violation_cycles": set(),
            "worst_diff": 0.0,
        }
    )
    cell_stats = {}
    detail_rows = []
    all_monotonicity_records = []

    for dataset, cell_id, total_cycles, worst_diff, violation_rows, monotonicity_check_records in results:
        dataset_entry = dataset_stats[dataset]
        dataset_entry["num_cells"] += 1
        dataset_entry["total_cycles"] += total_cycles
        dataset_entry["worst_diff"] = min(dataset_entry["worst_diff"], worst_diff)

        cell_entry = cell_stats.setdefault(
            (dataset, cell_id),
            {
                "total_cycles": total_cycles,
                "violation_cycles": set(),
                "worst_diff": 0.0,
            },
        )
        cell_entry["total_cycles"] = total_cycles
        cell_entry["worst_diff"] = min(cell_entry["worst_diff"], worst_diff)

        for row in violation_rows:
            detail_rows.append(row)
            cycle_number = row[2]
            dataset_entry["violation_cells"].add(cell_id)
            dataset_entry["violation_cycles"].add((cell_id, cycle_number))
            cell_entry["violation_cycles"].add(cycle_number)

        all_monotonicity_records.extend(monotonicity_check_records)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _build_run_dir(args.output_dir, args.extractor, timestamp)
    _write_config(run_dir, args, timestamp)
    _write_summary(run_dir, dataset_stats)
    _write_cell_stats(run_dir, cell_stats)
    _write_violations_detail(run_dir, detail_rows)
    _write_violation_index(run_dir, args, timestamp, results)

    # Save monotonicity check results to pkl in project root
    project_root = Path(__file__).resolve().parent
    pkl_output_path = project_root / "monotonicity_check_results.pkl"
    with open(pkl_output_path, "wb") as f:
        pickle.dump(all_monotonicity_records, f)

    print(f"Wrote results to {run_dir}")
    print(f"Wrote monotonicity check results to {pkl_output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
