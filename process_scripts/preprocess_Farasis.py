# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc
import re

from tqdm import tqdm
from typing import List
from pathlib import Path

from batteryml.data import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor


FARASIS_C_RATE = 0.333333


@PREPROCESSORS.register()
class FarasisPreprocessor(BasePreprocessor):
    def process(self, parent_dir, **kwargs) -> List[BatteryData]:
        root = Path(parent_dir)
        raw_root = root / 'raw_data' if (root / 'raw_data').exists() else root
        parquet_files = sorted(raw_root.rglob('*.parquet'))

        process_batteries_num = 0
        skip_batteries_num = 0
        file_iter = parquet_files if self.silent else tqdm(parquet_files, desc='Processing Farasis parquet files')

        for parquet_file in file_iter:
            rel = parquet_file.relative_to(raw_root).with_suffix('')
            cell_name = f"Farasis_{'_'.join(rel.parts)}"

            whether_to_skip = self.check_processed_file(cell_name)
            if whether_to_skip is True:
                skip_batteries_num += 1
                continue

            ok, message = _process_single_file(parquet_file, raw_root, self.output_dir)
            if ok:
                process_batteries_num += 1
                if not self.silent:
                    tqdm.write(f'File: {cell_name} dumped to pkl file')
            else:
                skip_batteries_num += 1
                if not self.silent:
                    tqdm.write(f'Skip {parquet_file}: {message}')

        return process_batteries_num, skip_batteries_num  # type: ignore[return-value]


def organize_cell(timeseries_df, name):
    required_cols = {'current (A)', 'voltage (V)'}
    if not required_cols.issubset(set(timeseries_df.columns)):
        return None

    df = timeseries_df.copy()

    current_numeric = pd.to_numeric(df['current (A)'], errors='coerce')
    voltage_numeric = pd.to_numeric(df['voltage (V)'], errors='coerce')
    # Scheme-1 pass-through: keep original parquet values if already numeric.
    current_passthrough = df['current (A)'] if pd.api.types.is_numeric_dtype(df['current (A)']) else current_numeric
    voltage_passthrough = df['voltage (V)'] if pd.api.types.is_numeric_dtype(df['voltage (V)']) else voltage_numeric

    if 'rpt' in df.columns:
        df['rpt'] = pd.to_numeric(df['rpt'], errors='coerce').astype(np.float32)
    if 'del_time' in df.columns:
        df['del_time'] = pd.to_numeric(df['del_time'], errors='coerce').fillna(0.0).astype(np.float32)
    if 'temp' in df.columns:
        df['temp'] = pd.to_numeric(df['temp'], errors='coerce').astype(np.float32)

    time_all_full = _build_time_in_seconds(df)
    valid = (
        np.isfinite(current_numeric.to_numpy(dtype=float))
        & np.isfinite(voltage_numeric.to_numpy(dtype=float))
        & np.isfinite(time_all_full)
    )
    if int(np.count_nonzero(valid)) < 2:
        return None

    df = df.loc[valid].reset_index(drop=True)
    current_all_numeric = current_numeric.loc[valid].to_numpy(dtype=float)
    voltage_all_numeric = voltage_numeric.loc[valid].to_numpy(dtype=float)
    current_all_passthrough = current_passthrough.loc[valid].to_numpy()
    voltage_all_passthrough = voltage_passthrough.loc[valid].to_numpy()
    time_all = time_all_full[valid]

    cycle_ids, cycle_attributes = _infer_cycle_ids_and_attributes(df)
    cycle_data = []
    all_voltages = []
    cumulative_time_offset = 0.0

    # Build base arrays once and slice by explicit integer indices to keep strict alignment.
    temp_all = None
    if 'temp' in df.columns:
        temp_all = pd.to_numeric(df['temp'], errors='coerce').to_numpy(dtype=float)

    for cycle_number in sorted(np.unique(cycle_ids)):
        if int(cycle_number) <= 0:
            continue

        cycle_idx = np.flatnonzero(cycle_ids == cycle_number)
        if len(cycle_idx) < 2:
            continue

        time_cycle = time_all[cycle_idx]

        keep_indices = np.arange(len(time_cycle), dtype=int)

        sampled_idx = cycle_idx[keep_indices]

        # Strictly index-aligned series from the same sampled row indices.
        current_in_A = current_all_passthrough[sampled_idx]
        voltage_in_V = voltage_all_passthrough[sampled_idx]
        current_in_A_numeric = current_all_numeric[sampled_idx]
        time_in_s = time_all[sampled_idx]

        # Keep time continuous across cycles instead of resetting to zero each cycle.
        time_in_s = time_in_s - float(time_in_s[0]) + cumulative_time_offset
        charge_capacity_in_Ah, discharge_capacity_in_Ah = _integrate_capacity(current_in_A_numeric, time_in_s)

        if len(time_in_s) < 2:
            continue

        cumulative_time_offset = float(time_in_s[-1])

        temperature_in_C = [float('nan')] * len(time_in_s)
        if temp_all is not None:
            temp_array = temp_all[cycle_idx]
            if np.isfinite(temp_array).any():
                filled_temp = np.nan_to_num(temp_array, nan=float(np.nanmedian(temp_array[np.isfinite(temp_array)])))
                temperature_in_C = filled_temp[keep_indices].tolist()

        cycle_data.append(CycleData(
            cycle_number=int(cycle_number),
            voltage_in_V=voltage_in_V.tolist(),
            current_in_A=current_in_A.tolist(),
            temperature_in_C=temperature_in_C,
            discharge_capacity_in_Ah=discharge_capacity_in_Ah.tolist(),
            charge_capacity_in_Ah=charge_capacity_in_Ah.tolist(),
            time_in_s=time_in_s.tolist(),
            attribute=cycle_attributes.get(int(cycle_number), 'Cycling')
        ))

        all_voltages.append(voltage_all_numeric[sampled_idx])

    if len(cycle_data) == 0:
        return None

    nominal_capacity = _infer_nominal_capacity(name)

    voltage_concat = np.concatenate(all_voltages)
    min_voltage = float(np.nanmin(voltage_concat))
    max_voltage = float(np.nanmax(voltage_concat))
    anode_material, cathode_material = _infer_materials(name)

    charge_protocol = [CyclingProtocol(rate_in_C=FARASIS_C_RATE, start_soc=0.0, end_soc=1.0)]
    discharge_protocol = [CyclingProtocol(rate_in_C=FARASIS_C_RATE, start_soc=1.0, end_soc=0.0)]

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='pouch',
        anode_material=anode_material,
        cathode_material=cathode_material,
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=nominal_capacity,
        min_voltage_limit_in_V=min_voltage,
        max_voltage_limit_in_V=max_voltage,
        SOC_interval=[0, 1]
    )


def _infer_cycle_ids_and_attributes(df):
    n = len(df)
    cycle_ids = np.zeros(n, dtype=int)
    cycle_attrs = {}
    if n == 0:
        return cycle_ids, cycle_attrs

    if 'rpt' not in df.columns:
        cycle_ids[:] = 1
        cycle_attrs[1] = 'Cycling'
        return cycle_ids, cycle_attrs

    # New rule:
    # - contiguous non-empty rpt segments -> attribute 'RPT'
    # - contiguous empty rpt segments -> attribute 'Cycling'
    # Empty segments are naturally split by adjacent numbered rpt segments.
    rpt_raw = pd.to_numeric(df['rpt'], errors='coerce').to_numpy(dtype=float)
    is_blank = ~np.isfinite(rpt_raw)

    cycle_no = 1
    start = 0
    while start < n:
        if is_blank[start]:
            end = start + 1
            while end < n and is_blank[end]:
                end += 1
            cycle_ids[start:end] = cycle_no
            cycle_attrs[cycle_no] = 'Cycling'
            cycle_no += 1
            start = end
            continue

        rpt_value = rpt_raw[start]
        end = start + 1
        while end < n and (not is_blank[end]) and float(rpt_raw[end]) == float(rpt_value):
            end += 1
        cycle_ids[start:end] = cycle_no
        cycle_attrs[cycle_no] = 'RPT'
        cycle_no += 1
        start = end

    return cycle_ids, cycle_attrs


def _build_time_in_seconds(cycle_df):
    if 'del_time' in cycle_df.columns:
        dt = pd.to_numeric(cycle_df['del_time'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        dt = np.maximum(dt, 0.0)
        if len(dt) > 0:
            dt[0] = 0.0
        return np.cumsum(dt)

    if 'time_stamp' in cycle_df.columns:
        ts = pd.to_datetime(cycle_df['time_stamp'], errors='coerce')
        if ts.notna().sum() >= 2:
            seconds = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
            return np.maximum.accumulate(np.nan_to_num(seconds, nan=0.0))

    return np.arange(len(cycle_df), dtype=float)


def _integrate_capacity(current_in_A, time_in_s):
    current = np.asarray(current_in_A, dtype=float)
    time_s = np.asarray(time_in_s, dtype=float)

    n = min(len(current), len(time_s))
    if n == 0:
        return np.array([]), np.array([])

    current = current[:n]
    time_s = time_s[:n]

    dt = np.diff(time_s, prepend=time_s[0])
    dt = np.maximum(dt, 0.0)

    # Reset capacity at each contiguous charge/discharge segment.
    # This is trapz-style accumulation, but kept as cumulative per-point values.
    # np.trapz / np.trapezoid returns one scalar for the whole interval; here we
    # need a running curve, so we accumulate trapezoid increments segment-wise.
    charge_capacity = np.zeros(n, dtype=float)
    discharge_capacity = np.zeros(n, dtype=float)

    eps = 0.0
    state = np.zeros(n, dtype=np.int8)
    state[current > eps] = 1
    state[current < -eps] = -1

    # Segment boundaries where charge/discharge/rest state changes.
    change_idx = np.flatnonzero(state[1:] != state[:-1]) + 1
    bounds = np.concatenate(([0], change_idx, [n]))

    for j in range(len(bounds) - 1):
        s = int(bounds[j])
        e = int(bounds[j + 1])
        if e - s <= 0:
            continue

        st = int(state[s])
        if st == 0:
            continue

        if st > 0:
            y = np.maximum(current[s:e], 0.0)
            out = charge_capacity
        else:
            y = np.maximum(-current[s:e], 0.0)
            out = discharge_capacity

        dt_seg = dt[s:e]
        prev_y = np.concatenate(([0.0], y[:-1]))
        # per-point trapezoid increment in As, then convert to Ah
        inc = 0.5 * (prev_y + y) * dt_seg / 3600.0
        out[s:e] = np.cumsum(inc)

    return charge_capacity, discharge_capacity


def _extract_cell_group_and_id(name):
    group_match = re.search(r'Farasis_(PA|PB|PC|PD)', name)
    group = group_match.group(1) if group_match else None

    # Use the three-digit cell ID in parquet file names like 001_single_parquet...
    id_match = re.search(r'_(\d{3})_single_parquet', name)
    cell_id = int(id_match.group(1)) if id_match else None
    return group, cell_id


def _infer_nominal_capacity(name):
    group, cell_id = _extract_cell_group_and_id(name)

    if group == 'PA' or (cell_id is not None and 1 <= cell_id <= 19):
        return 73.0
    if group in {'PB', 'PC'} or (cell_id is not None and 20 <= cell_id <= 112):
        return 76.0
    if group == 'PD' or (cell_id is not None and 113 <= cell_id <= 123):
        return 84.0

    # Keep previous default behavior for any unforeseen filename pattern.
    return 0.0


def _infer_materials(name):
    # PD folder cells 113-123 use Si-C(10 wt% Si)/NMC9 per dataset notes.
    group, cell_idx = _extract_cell_group_and_id(name)
    if group == 'PD' and cell_idx is not None and 113 <= cell_idx <= 123:
        return 'Si-C (10 wt% Si)', 'NMC9'
    return 'Graphite', 'NMC811'


def _process_single_file(parquet_file_path, raw_root_path, output_dir_path):
    parquet_file = Path(parquet_file_path)
    raw_root = Path(raw_root_path)
    output_dir = Path(output_dir_path)

    rel = parquet_file.relative_to(raw_root).with_suffix('')
    cell_name = f"Farasis_{'_'.join(rel.parts)}"

    df = None
    battery = None
    try:
        parquet_obj = pq.ParquetFile(parquet_file)
        available_cols = set(parquet_obj.schema.names)
        selected_cols = [col for col in REQUIRED_PARQUET_COLUMNS if col in available_cols]
        if ('current (A)' not in selected_cols) or ('voltage (V)' not in selected_cols):
            return False, 'missing required columns'

        # Keep full source resolution for cycle split and pkl output.
        df = parquet_obj.read(columns=selected_cols, use_threads=False).to_pandas()
        battery = organize_cell(df, cell_name)

        if battery is None:
            return False, 'invalid or incomplete cycle data'

        battery.dump(output_dir / f'{battery.cell_id}.pkl')
        return True, battery.cell_id
    except Exception as exc:
        return False, str(exc)
    finally:
        if df is not None:
            del df
        if battery is not None:
            del battery
        gc.collect()

REQUIRED_PARQUET_COLUMNS = [
    'current (A)',
    'voltage (V)',
    'rpt',
    'time_stamp',
    'del_time',
    'temp'
]