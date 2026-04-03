import os
import pickle
import warnings

import numpy as np

from .common import CURRENT_C_THRESHOLD, get_nominal_capacity
from .types import CycleSegments

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


def _numpy_bfill_2d(arr_2d):
    filled = arr_2d.copy()
    for row_index in range(filled.shape[0] - 2, -1, -1):
        nan_mask = np.isnan(filled[row_index])
        if np.any(nan_mask):
            filled[row_index, nan_mask] = filled[row_index + 1, nan_mask]
    return filled


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


def extract_segments(pkl_path, *, data=None):
    """按 data_loader.py 中的 charge/discharge segmentation 逻辑复现提取。"""
    if data is None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

    file_name = os.path.basename(pkl_path)
    nominal_capacity = get_nominal_capacity(file_name, data)
    if nominal_capacity <= 0:
        warnings.warn(f"Invalid nominal capacity for {file_name}: {nominal_capacity}")
        return [], 0.0

    prefix = _get_prefix(file_name)
    discharge_first = _is_discharge_first(file_name, prefix)
    segments = []

    for cycle_index, cycle_dict in enumerate(data.get("cycle_data", []), start=1):
        arrays = {}
        n_points = None
        skip_cycle = False
        for key in NEED_KEYS:
            values = np.asarray(cycle_dict.get(key, []), dtype=float)
            if n_points is None:
                n_points = values.size
            if values.size < 2:
                skip_cycle = True
                break
            arrays[key] = values

        if skip_cycle or n_points is None or n_points < 2:
            continue

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
            continue

        charge_end_index = int(charge_indices[-1])
        discharge_end_index = int(discharge_indices[-1])

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

        segments.append(
            CycleSegments(
                cycle_number=cycle_index,
                charge_capacity=np.asarray(charge_capacity, dtype=float) / nominal_capacity,
                discharge_capacity=np.asarray(discharge_capacity, dtype=float) / nominal_capacity,
                charge_indices=np.asarray(charge_cap_indices, dtype=int),
                discharge_indices=np.asarray(discharge_cap_indices, dtype=int),
            )
        )

    return segments, nominal_capacity
