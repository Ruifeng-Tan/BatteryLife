"""View monotonicity check results and plot violation examples."""

import argparse
import os
import pickle
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUTPUT_DIR = Path("/data/trf/python_project/BatteryLife/dev/Issue 22")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = Path("/data/trf/python_works/BatteryLife/dataset")

CURRENT_C_THRESHOLD = 0.01
CAPACITY_START_TOL = 1e-2

DISCHARGE_FIRST_PREFIXES = frozenset(
    ["RWTH", "CALB_0", "CALB_25", "CALB_45"]
)

ZN_COIN_CHARGE_FIRST_FILES = frozenset(
    [
        "ZN-coin_402-1_20231209225636_01_1.pkl",
        "ZN-coin_402-2_20231209225727_01_2.pkl",
        "ZN-coin_402-3_20231209225844_01_3.pkl",
        "ZN-coin_403-1_20231209225922_01_4.pkl",
        "ZN-coin_428-1_20231212185048_01_2.pkl",
        "ZN-coin_428-2_20231212185058_01_4.pkl",
        "ZN-coin_429-1_20231212185129_01_5.pkl",
        "ZN-coin_429-2_20231212185157_01_8.pkl",
        "ZN-coin_430-1_20231212185250_02_6.pkl",
        "ZN-coin_430-2_20231212185305_02_7.pkl",
        "ZN-coin_430-3_20231212185323_03_2.pkl",
    ]
)

NEED_KEYS = [
    "current_in_A",
    "voltage_in_V",
    "charge_capacity_in_Ah",
    "discharge_capacity_in_Ah",
    "time_in_s",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize monotonicity check results and plot violation examples."
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(Path(__file__).resolve().parent / "monotonicity_check_results.pkl"),
        help="Path to monotonicity_check_results.pkl",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=str(DATA_PATH),
        help="Root directory containing the dataset pkl files.",
    )
    parser.add_argument(
        "--extractor",
        choices=["masking", "dataloader"],
        default="masking",
        help="Capacity extractor used for plotting (must match the one used for checking).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for selecting examples.",
    )
    return parser.parse_args()


def _load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _get_nominal_capacity(file_name, data):
    if file_name.startswith("RWTH"):
        return 1.85
    if file_name.startswith("SNL_18650_NCA_25C_20-80"):
        return 3.2
    raw_value = data.get("nominal_capacity_in_Ah", 0.0)
    if raw_value is None:
        return 0.0
    try:
        val = float(raw_value)
    except (TypeError, ValueError):
        return 0.0
    return val if np.isfinite(val) else 0.0


def _compute_charge_discharge_masks(current_arr, nominal_capacity):
    """Masking-style: threshold on C-rate."""
    values = np.asarray(current_arr, dtype=float)
    if nominal_capacity <= 0 or values.size == 0:
        empty = np.zeros(values.shape, dtype=bool)
        return empty, empty
    current_in_c = values / nominal_capacity
    charge_mask = current_in_c >= CURRENT_C_THRESHOLD
    discharge_mask = current_in_c <= -CURRENT_C_THRESHOLD
    return charge_mask, discharge_mask


def _find_trim_bounds(capacity_norm, tol=CAPACITY_START_TOL):
    values = np.asarray(capacity_norm, dtype=float)
    if values.size < 2:
        return None
    increasing = np.diff(values) > 0
    start_candidates = np.flatnonzero((values[:-1] < tol) & increasing)
    if start_candidates.size == 0:
        return None
    start = int(start_candidates[0])
    end_candidates = np.flatnonzero(increasing[start:])
    if end_candidates.size == 0:
        return None
    end = int(start + end_candidates[-1] + 2)
    return start, end


def _extract_masked_capacity(capacity_arr, mask, nominal_capacity):
    mask = np.asarray(mask, dtype=bool)
    masked_capacity = np.asarray(capacity_arr, dtype=float)[mask]
    if nominal_capacity <= 0 or masked_capacity.size < 2:
        return np.array([], dtype=float), np.array([], dtype=int)
    capacity_norm = masked_capacity / nominal_capacity
    bounds = _find_trim_bounds(capacity_norm)
    if bounds is None:
        return np.array([], dtype=float), np.array([], dtype=int)
    start, end = bounds
    original_indices = np.flatnonzero(mask)
    trimmed_indices = original_indices[start:end]
    return capacity_norm[start:end], trimmed_indices


def _get_prefix(file_name):
    prefix = file_name.split("_")[0]
    if prefix == "CALB":
        prefix = "_".join(file_name.split("_")[:2])
    return prefix


def _is_discharge_first(file_name, prefix):
    return (
        prefix in DISCHARGE_FIRST_PREFIXES
        or (file_name not in ZN_COIN_CHARGE_FIRST_FILES and prefix == "ZN-coin")
    )


def _numpy_bfill_2d(arr_2d):
    filled = arr_2d.copy()
    for row_index in range(filled.shape[0] - 2, -1, -1):
        nan_mask = np.isnan(filled[row_index])
        if np.any(nan_mask):
            filled[row_index, nan_mask] = filled[row_index + 1, nan_mask]
    return filled


def _extract_dataloader_segments_with_file(cycle_dict, nominal_capacity, file_name):
    """Dataloader-style extraction with file name for discharge_first logic."""
    arrays = {}
    n_points = None
    for key in NEED_KEYS:
        values = np.asarray(cycle_dict.get(key, []), dtype=float)
        if n_points is None:
            n_points = values.size
        if values.size < 2:
            return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)
        arrays[key] = values

    data_2d = np.column_stack([arrays[key] for key in NEED_KEYS])
    charge_col = NEED_KEYS.index("charge_capacity_in_Ah")
    discharge_col = NEED_KEYS.index("discharge_capacity_in_Ah")

    data_2d[data_2d[:, charge_col] < 0] = np.nan
    data_2d[data_2d[:, discharge_col] < 0] = np.nan
    data_2d = _numpy_bfill_2d(data_2d)

    current_arr = data_2d[:, NEED_KEYS.index("current_in_A")]
    charge_capacity_arr = data_2d[:, charge_col]
    discharge_capacity_arr = data_2d[:, discharge_col]
    current_in_c = current_arr / nominal_capacity

    charge_indices = np.flatnonzero(current_in_c >= CURRENT_C_THRESHOLD)
    discharge_indices = np.flatnonzero(current_in_c <= -CURRENT_C_THRESHOLD)
    if charge_indices.size == 0 or discharge_indices.size == 0:
        return np.array([]), np.array([], dtype=int), np.array([]), np.array([], dtype=int)

    charge_end_index = int(charge_indices[-1])
    discharge_end_index = int(discharge_indices[-1])

    prefix = _get_prefix(file_name)
    discharge_first = _is_discharge_first(file_name, prefix)

    if discharge_first:
        discharge_capacity = discharge_capacity_arr[:discharge_end_index]
        discharge_cap_indices = np.arange(discharge_end_index)
        charge_region_capacity = charge_capacity_arr[discharge_end_index:]
        charge_region_c_rate = current_in_c[discharge_end_index:]
        charge_cap_indices = np.flatnonzero(
            np.abs(charge_region_c_rate) > CURRENT_C_THRESHOLD
        ) + discharge_end_index
        charge_capacity = charge_region_capacity[
            np.abs(charge_region_c_rate) > CURRENT_C_THRESHOLD
        ]
    else:
        charge_capacity = charge_capacity_arr[:charge_end_index]
        charge_cap_indices = np.arange(charge_end_index)
        discharge_region_capacity = discharge_capacity_arr[charge_end_index:]
        discharge_region_c_rate = current_in_c[charge_end_index:]
        discharge_cap_indices = np.flatnonzero(
            np.abs(discharge_region_c_rate) > CURRENT_C_THRESHOLD
        ) + charge_end_index
        discharge_capacity = discharge_region_capacity[
            np.abs(discharge_region_c_rate) > CURRENT_C_THRESHOLD
        ]

    return (
        np.asarray(charge_capacity, dtype=float) / nominal_capacity,
        np.asarray(charge_cap_indices, dtype=int),
        np.asarray(discharge_capacity, dtype=float) / nominal_capacity,
        np.asarray(discharge_cap_indices, dtype=int),
    )


def _plot_cycle_segment(pkl_path, cycle_number, phase, output_path, continuous=None, extractor="masking"):
    """Load raw data and plot the charge/discharge segment for a given cycle."""
    data = _load_pkl(pkl_path)
    file_name = os.path.basename(pkl_path)
    nominal_capacity = _get_nominal_capacity(file_name, data)

    cycle_dict = data["cycle_data"][cycle_number - 1]  # 1-indexed
    current_arr = np.asarray(cycle_dict["current_in_A"], dtype=float)
    capacity_arr = np.asarray(
        cycle_dict[f"{phase}_capacity_in_Ah"], dtype=float
    )
    n_points = len(current_arr)
    indices = np.arange(n_points)
    current_in_c = current_arr / nominal_capacity

    if extractor == "dataloader":
        charge_cap, charge_idx, discharge_cap, discharge_idx = (
            _extract_dataloader_segments_with_file(cycle_dict, nominal_capacity, file_name)
        )
        if phase == "charge":
            active_indices = charge_idx
        else:
            active_indices = discharge_idx
    else:
        charge_mask, discharge_mask = _compute_charge_discharge_masks(
            current_arr, nominal_capacity
        )
        if phase == "charge":
            _, active_indices = _extract_masked_capacity(
                cycle_dict["charge_capacity_in_Ah"], charge_mask, nominal_capacity
            )
        else:
            _, active_indices = _extract_masked_capacity(
                cycle_dict["discharge_capacity_in_Ah"], discharge_mask, nominal_capacity
            )

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Left y-axis: current C-rate
    color_current = "tab:blue"
    ax1.set_xlabel("Index")
    ax1.set_ylabel("Current (C-rate)", color=color_current)
    ax1.plot(indices, current_in_c, color=color_current, linewidth=0.8, label="Current")
    ax1.axhline(y=0.01, color="tab:orange", linestyle="--", linewidth=0.6, alpha=0.7, label="+0.01C")
    ax1.axhline(y=-0.01, color="tab:green", linestyle="--", linewidth=0.6, alpha=0.7, label="-0.01C")
    ax1.tick_params(axis="y", labelcolor=color_current)
    ax1.legend(loc="upper left")

    # Right y-axis: capacity
    ax2 = ax1.twinx()
    color_capacity = "tab:red"
    ax2.set_ylabel(f"{phase.capitalize()} Capacity (normalized)", color=color_capacity)
    ax2.plot(indices, capacity_arr / nominal_capacity, color=color_capacity, linewidth=0.8, label="Capacity")
    ax2.tick_params(axis="y", labelcolor=color_capacity)

    # Overlay markers for extracted data points (green for both extractors)
    if len(active_indices) > 0:
        active_capacity_norm = capacity_arr[active_indices] / nominal_capacity
        ax2.scatter(
            active_indices, active_capacity_norm,
            color="green", s=10, marker="o", zorder=5,
            label="Extracted points", edgecolors="none",
        )

    # Merge legends from both axes into one
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    fig.legend(
        handles1 + handles2, labels1 + labels2,
        loc="lower center", bbox_to_anchor=(0.5, -0.02),
        ncol=3, framealpha=0.9,
    )

    title = f"{file_name} | Cycle {cycle_number} | {phase.capitalize()}"
    if continuous is not None:
        tag = "CONTINUOUS" if continuous else "DISCONTINUOUS"
        title += f" | {tag}"
    plt.title(title, fontsize=10)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot: {output_path}")


def main():
    args = parse_args()
    with open(args.input, "rb") as f:
        records = pickle.load(f)

    if not records:
        print("No violation records found.")
        return

    # Categorize records
    continuous_records = [r for r in records if r.get("continuous")]
    discontinuous_records = [r for r in records if not r.get("continuous")]

    # Per-dataset stats
    dataset_stats = defaultdict(
        lambda: {
            "violation_cycles": set(),
            "unique_batteries": set(),
            "first_100_cycles": 0,
            "continuous_count": 0,
            "discontinuous_count": 0,
        }
    )

    for rec in records:
        dataset = rec.get("dataset", "unknown")
        stats = dataset_stats[dataset]
        stats["violation_cycles"].add((rec["file_name"], rec["cycle_number"]))
        stats["unique_batteries"].add(rec["file_name"])
        if rec["cycle_number"] <= 100:
            stats["first_100_cycles"] += 1
        if rec.get("continuous"):
            stats["continuous_count"] += 1
        else:
            stats["discontinuous_count"] += 1

    # Print summary table
    print(f"{'='*90}")
    print(f"Monotonicity Check Results Summary")
    print(f"{'='*90}")
    print(f"Total violation records (cycle-phase pairs): {len(records)}")
    print(f"Datasets with violations: {len(dataset_stats)}")
    print()

    hdr = (
        f"{'Dataset':<20} {'Violation Cycles':>16} {'Unique Batt.':>14} "
        f"{'<=100 Cycles':>14} {'Violations':>12} "
        f"{'Continuous':>12} {'Discontinuous':>14} {'Disc. Rate':>12}"
    )
    print(hdr)
    print(f"{'-'*len(hdr)}")

    total_violation_cycles = 0
    total_violations = 0
    total_unique_batteries = set()
    total_first_100 = 0
    total_continuous = 0
    total_discontinuous = 0

    for dataset in sorted(dataset_stats):
        stats = dataset_stats[dataset]
        vc = len(stats["violation_cycles"])
        ub = len(stats["unique_batteries"])
        f100 = stats["first_100_cycles"]
        vr = len([
            r for r in records if r.get("dataset") == dataset
        ])
        cont = stats["continuous_count"]
        disc = stats["discontinuous_count"]
        total_violation_cycles += vc
        total_unique_batteries |= stats["unique_batteries"]
        total_first_100 += f100
        total_violations += vr
        total_continuous += cont
        total_discontinuous += disc

        disc_rate = f"{100.0 * disc / vr:.2f}%" if vr > 0 else "N/A"
        print(
            f"{dataset:<20} {vc:>16} {ub:>14} {f100:>14} {vr:>12} "
            f"{cont:>12} {disc:>14} {disc_rate:>12}"
        )

    print(f"{'-'*len(hdr)}")
    overall_disc_rate = (
        f"{100.0 * total_discontinuous / total_violations:.2f}%"
        if total_violations else "N/A"
    )
    print(
        f"{'TOTAL':<20} {total_violation_cycles:>16} {len(total_unique_batteries):>14} "
        f"{total_first_100:>14} {total_violations:>12} "
        f"{total_continuous:>12} {total_discontinuous:>14} {overall_disc_rate:>12}"
    )
    print()
    print(f"Summary:")
    print(
        f"  - {total_discontinuous}/{total_violations} "
        f"({100.0 * total_discontinuous / total_violations:.2f}%) "
        f"violation records have DISCONTINUOUS indices"
    )
    print(
        f"  - {total_continuous}/{total_violations} "
        f"({100.0 * total_continuous / total_violations:.2f}%) "
        f"violation records have continuous indices"
    )
    print()

    # ---- Plotting ----
    random.seed(args.seed)

    # Build lookup: (dataset, file_name) -> pkl_path
    print("Scanning data_path for pkl files...")
    pkl_lookup = {}
    for sub in sorted(os.listdir(args.data_path)):
        sub_path = Path(args.data_path) / sub
        if sub_path.is_dir():
            for pkl_file in sub_path.glob("*.pkl"):
                pkl_lookup[pkl_file.name] = str(pkl_file)

    def _resolve_pkl_path(file_name):
        """Try to find the pkl file by name."""
        if file_name in pkl_lookup:
            return pkl_lookup[file_name]
        # Try direct search in data_path
        for root, _dirs, files in os.walk(args.data_path):
            if file_name in files:
                return os.path.join(root, file_name)
        return None

    def _pick_and_plot(pool, label, count=2):
        if not pool:
            print(f"No {label} records to plot.")
            return
        selected = random.sample(pool, min(count, len(pool)))
        print(f"\nPlotting {len(selected)} {label} example(s):")
        for rec in selected:
            file_name = rec["file_name"]
            cycle_number = rec["cycle_number"]
            phase = rec["phase"]
            dataset = rec.get("dataset", "unknown")

            pkl_path = _resolve_pkl_path(file_name)
            if pkl_path is None:
                print(f"  SKIP: Cannot find pkl file for {file_name}")
                continue

            # Sanitize file name for output
            safe_name = file_name.replace(".pkl", "").replace("/", "_").replace("\\", "_")
            tag = "cont" if rec.get("continuous") else "discont"
            output_name = f"{safe_name}_cycle{cycle_number}_{phase}_{tag}.png"
            output_path = OUTPUT_DIR / output_name

            try:
                _plot_cycle_segment(
                    pkl_path, cycle_number, phase, output_path,
                    continuous=rec.get("continuous"),
                    extractor=args.extractor,
                )
            except Exception as exc:
                print(f"  ERROR plotting {file_name} cycle {cycle_number}: {exc}")

    # Pick 2 continuous and 2 discontinuous
    _pick_and_plot(continuous_records, "continuous", count=2)
    _pick_and_plot(discontinuous_records, "discontinuous", count=2)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
