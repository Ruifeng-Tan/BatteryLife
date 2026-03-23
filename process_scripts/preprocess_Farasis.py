# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import gc
import re
import multiprocessing as mp

from tqdm import tqdm
from typing import List
from pathlib import Path

from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor


FARASIS_C_RATE = 0.333333
PULSE_CURRENT_THRESHOLD_A = 80.0
FORMATION_CURRENT_THRESHOLD_A = 5.0
MIN_DISCHARGE_STEP_DURATION_S = 600.0
MAX_PULSE_STEP_DURATION_S = 20.0


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

            ok, message = _run_single_file_isolated(parquet_file, raw_root, self.output_dir)
            if ok:
                process_batteries_num += 1
                if not self.silent:
                    tqdm.write(f'File: {cell_name} dumped to pkl file')
            else:
                skip_batteries_num += 1
                if not self.silent:
                    tqdm.write(f'Skip {parquet_file}: {message}')

        return process_batteries_num, skip_batteries_num  # type: ignore[return-value]


def organize_cell(timeseries_df, name, already_sampled=False):
    required_cols = {'current (A)', 'voltage (V)'}
    if not required_cols.issubset(set(timeseries_df.columns)):
        return None

    df = timeseries_df.copy()
    df['current (A)'] = pd.to_numeric(df['current (A)'], errors='coerce').fillna(0.0).astype(np.float32)
    df['voltage (V)'] = pd.to_numeric(df['voltage (V)'], errors='coerce').astype(np.float32)

    if 'rpt' in df.columns:
        df['rpt'] = pd.to_numeric(df['rpt'], errors='coerce').astype(np.float32)
    if 'step_no' in df.columns:
        df['step_no'] = pd.to_numeric(df['step_no'], errors='coerce').fillna(0).astype(np.int32)
    if 'del_time' in df.columns:
        df['del_time'] = pd.to_numeric(df['del_time'], errors='coerce').fillna(0.0).astype(np.float32)
    if 'temp' in df.columns:
        df['temp'] = pd.to_numeric(df['temp'], errors='coerce').astype(np.float32)

    df = df.dropna(subset=['voltage (V)'])
    if len(df) < 2:
        return None

    cycle_ids, cycle_attributes = _infer_cycle_ids_and_attributes(df)
    cycle_data = []
    all_voltages = []
    cumulative_time_offset = 0.0

    # Build base arrays once and slice by explicit integer indices to keep strict alignment.
    current_all = df['current (A)'].to_numpy(dtype=float)
    voltage_all = df['voltage (V)'].to_numpy(dtype=float)
    temp_all = None
    if 'temp' in df.columns:
        temp_all = pd.to_numeric(df['temp'], errors='coerce').to_numpy(dtype=float)
    time_all = _build_time_in_seconds(df)

    for cycle_number in sorted(np.unique(cycle_ids)):
        if int(cycle_number) <= 0:
            continue

        cycle_idx = np.flatnonzero(cycle_ids == cycle_number)
        if len(cycle_idx) < 2:
            continue

        current_cycle = current_all[cycle_idx]
        voltage_cycle = voltage_all[cycle_idx]
        time_cycle = time_all[cycle_idx]

        keep_indices = np.arange(len(time_cycle), dtype=int)

        sampled_idx = cycle_idx[keep_indices]

        # Strictly index-aligned series from the same sampled row indices.
        current_in_A = current_all[sampled_idx]
        voltage_in_V = voltage_all[sampled_idx]
        time_in_s = time_all[sampled_idx]

        # Keep time continuous across cycles instead of resetting to zero each cycle.
        time_in_s = time_in_s - float(time_in_s[0]) + cumulative_time_offset
        charge_capacity_in_Ah, discharge_capacity_in_Ah = _integrate_capacity(current_in_A, time_in_s)

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

        all_voltages.append(voltage_in_V)

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
    current = pd.to_numeric(df['current (A)'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
    n = len(current)
    cycle_ids = np.zeros(n, dtype=int)
    cycle_attrs = {}
    if n == 0:
        return cycle_ids, cycle_attrs

    if 'rpt' not in df.columns:
        cycle_ids[:] = 1
        cycle_attrs[1] = 'Cycling'
        return cycle_ids, cycle_attrs

    rpt = pd.to_numeric(df['rpt'], errors='coerce').ffill().bfill().fillna(0).to_numpy(dtype=int)

    cycle_no = 1
    start = 0
    for i in range(1, n):
        if rpt[i] != rpt[i - 1]:
            cycle_ids[start:i] = cycle_no
            seg_curr = current[start:i]
            max_abs_i = float(np.nanmax(np.abs(seg_curr))) if len(seg_curr) > 0 else 0.0
            cycle_attrs[cycle_no] = 'Formation' if (cycle_no == 1 and max_abs_i <= FORMATION_CURRENT_THRESHOLD_A) else 'Cycling'
            cycle_no += 1
            start = i

    cycle_ids[start:n] = cycle_no
    seg_curr = current[start:n]
    max_abs_i = float(np.nanmax(np.abs(seg_curr))) if len(seg_curr) > 0 else 0.0
    cycle_attrs[cycle_no] = 'Formation' if (cycle_no == 1 and max_abs_i <= FORMATION_CURRENT_THRESHOLD_A) else 'Cycling'

    return cycle_ids, cycle_attrs


def _build_time_in_seconds(cycle_df):
    if 'time_stamp' in cycle_df.columns:
        ts = pd.to_datetime(cycle_df['time_stamp'], errors='coerce')
        if ts.notna().sum() >= 2:
            seconds = (ts - ts.iloc[0]).dt.total_seconds().to_numpy(dtype=float)
            return np.maximum.accumulate(np.nan_to_num(seconds, nan=0.0))

    if 'del_time' in cycle_df.columns:
        dt = pd.to_numeric(cycle_df['del_time'], errors='coerce').fillna(0.0).to_numpy(dtype=float)
        dt = np.maximum(dt, 0.0)
        return np.cumsum(dt)

    return np.arange(len(cycle_df), dtype=float)


def _integrate_capacity(current_in_A, time_in_s):
    dt = np.diff(time_in_s, prepend=time_in_s[0])
    dt = np.maximum(dt, 0.0)

    # Reset capacity at each contiguous charge/discharge segment.
    # This avoids unbounded accumulation over a long rpt block.
    charge_capacity = np.zeros(len(current_in_A), dtype=float)
    discharge_capacity = np.zeros(len(current_in_A), dtype=float)

    charge_running = 0.0
    discharge_running = 0.0
    prev_state = 0  # 1: charge, -1: discharge, 0: rest

    for i in range(len(current_in_A)):
        curr = float(current_in_A[i])
        delta_h = float(dt[i]) / 3600.0

        if curr > 0.0:
            if prev_state != 1:
                charge_running = 0.0
            charge_running += curr * delta_h
            charge_capacity[i] = charge_running
            discharge_capacity[i] = 0.0
            prev_state = 1
        elif curr < 0.0:
            if prev_state != -1:
                discharge_running = 0.0
            discharge_running += (-curr) * delta_h
            discharge_capacity[i] = discharge_running
            charge_capacity[i] = 0.0
            prev_state = -1
        else:
            charge_capacity[i] = 0.0
            discharge_capacity[i] = 0.0
            prev_state = 0

    return charge_capacity, discharge_capacity


def _build_downsample_indices(time_in_s, current_in_A, interval_s):
    if len(time_in_s) <= 2:
        return np.arange(len(time_in_s), dtype=int)

    keep_indices = [0]
    last_kept_time = float(time_in_s[0])

    for idx in range(1, len(time_in_s) - 1):
        if float(time_in_s[idx]) - last_kept_time >= interval_s:
            keep_indices.append(idx)
            last_kept_time = float(time_in_s[idx])

    if keep_indices[-1] != len(time_in_s) - 1:
        keep_indices.append(len(time_in_s) - 1)

    # Always keep phase transitions to preserve R-D-R-C structure after sampling.
    eps = 1e-4
    state = np.zeros(len(current_in_A), dtype=np.int8)
    state[current_in_A > eps] = 1
    state[current_in_A < -eps] = -1
    transitions = np.where(state[1:] != state[:-1])[0] + 1

    merged = np.unique(
        np.concatenate((np.asarray(keep_indices, dtype=int), np.asarray(transitions, dtype=int)))
    )
    return np.asarray(merged, dtype=int)


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

        # Always split cycles on raw 1s source data, then downsample to 30s at write time.
        df = parquet_obj.read(columns=selected_cols, use_threads=False).to_pandas()
        battery = organize_cell(df, cell_name, already_sampled=False)

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


def _worker_process_single_file(parquet_file_path, raw_root_path, output_dir_path, result_queue):
    result_queue.put(_process_single_file(parquet_file_path, raw_root_path, output_dir_path))


def _run_single_file_isolated(parquet_file, raw_root, output_dir):
    ctx = mp.get_context('spawn')
    result_queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_worker_process_single_file,
        args=(str(parquet_file), str(raw_root), str(output_dir), result_queue)
    )
    proc.start()
    proc.join()

    if proc.exitcode != 0:
        return False, f'worker exited with code {proc.exitcode}'
    if result_queue.empty():
        return False, 'worker returned no result'

    return result_queue.get()


REQUIRED_PARQUET_COLUMNS = [
    'current (A)',
    'voltage (V)',
    'rpt',
    'step_no',
    'time_stamp',
    'del_time',
    'temp'
]