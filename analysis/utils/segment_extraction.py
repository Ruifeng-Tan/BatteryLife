import os
import pickle
import warnings

import numpy as np

from .common import CAPACITY_START_TOL, CURRENT_C_THRESHOLD, get_nominal_capacity
from .types import CycleSegments


def _compute_charge_discharge_masks(current_arr, nominal_capacity):
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
    # Also return the original indices of the trimmed capacity points
    original_indices = np.flatnonzero(mask)
    trimmed_indices = original_indices[start:end]
    return capacity_norm[start:end], trimmed_indices


def extract_segments(pkl_path, *, data=None):
    """按 masking 规则提取每个 cycle 的 charge/discharge capacity。"""
    if data is None:
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

    file_name = os.path.basename(pkl_path)
    nominal_capacity = get_nominal_capacity(file_name, data)
    if nominal_capacity <= 0:
        warnings.warn(f"Invalid nominal capacity for {file_name}: {nominal_capacity}")
        return [], 0.0

    segments = []
    for cycle_index, cycle_dict in enumerate(data.get("cycle_data", []), start=1):
        current_arr = np.asarray(cycle_dict.get("current_in_A", []), dtype=float)
        if current_arr.size < 2:
            continue

        charge_mask, discharge_mask = _compute_charge_discharge_masks(
            current_arr,
            nominal_capacity,
        )
        charge_capacity, charge_indices = _extract_masked_capacity(
            cycle_dict.get("charge_capacity_in_Ah", []),
            charge_mask,
            nominal_capacity,
        )
        discharge_capacity, discharge_indices = _extract_masked_capacity(
            cycle_dict.get("discharge_capacity_in_Ah", []),
            discharge_mask,
            nominal_capacity,
        )

        if charge_capacity.size == 0 and discharge_capacity.size == 0:
            continue

        segments.append(
            CycleSegments(
                cycle_number=cycle_index,
                charge_capacity=charge_capacity,
                discharge_capacity=discharge_capacity,
                charge_indices=charge_indices,
                discharge_indices=discharge_indices,
            )
        )

    return segments, nominal_capacity
