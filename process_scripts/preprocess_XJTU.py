# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import zipfile
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import Tuple
from pathlib import Path
from scipy.io import loadmat
from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor
from .time_normalization_utils import normalize_cycle_times


XJTU_BATCH_CONFIG = {
    'Batch-1': {
        'charge_rates': [2.0],
        'discharge_rates': [1.0],
        'soc_interval': [0, 1],
        'min_voltage_limit': 2.5,
    },
    'Batch-2': {
        'charge_rates': [3.0],
        'discharge_rates': [1.0],
        'soc_interval': [0, 1],
        'min_voltage_limit': 2.5,
    },
    'Batch-3': {
        'charge_rates': [2.0],
        'discharge_rates': [0.5, 1.0, 2.0, 3.0, 5.0],
        'soc_interval': [0, 1],
        'min_voltage_limit': 2.5,
    },
    'Batch-4': {
        'charge_rates': [2.0],
        'discharge_rates': [0.5, 1.0, 2.0, 3.0, 5.0],
        # Main aging cycles stop at 3.0V (partial discharge), while periodic
        # capacity-test cycles discharge to 2.5V.
        'soc_interval': [0, 1],
        'min_voltage_limit': 2.5,
    },
    'Batch-5': {
        'charge_rates': [0.5, 1.0, 3.0],
        'discharge_rates': [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        'soc_interval': [0, 1],
        # Random-walk cycles stop at 3.0V, but periodic 1C capacity tests use 2.5V cutoff.
        'min_voltage_limit': 2.5,
    },
    'Batch-6': {
        'charge_rates': [2.0, 1.0],
        'discharge_rates': [0.667, 0.5],
        'soc_interval': [0, 1],
        'min_voltage_limit': 2.5,
    },
}

XJTU_REMOVE_FIRST_CYCLE_BATCHES = {'Batch-1', 'Batch-2', 'Batch-3', 'Batch-4', 'Batch-6'}


@PREPROCESSORS.register()
class XJTUPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> Tuple[int, int]:
        cells = []
        paths = []
        cells_files_path = ['Batch-1', 'Batch-2', 'Batch-3', 'Batch-4', 'Batch-5', 'Batch-6']
        raw_file = Path(parentdir) / 'Battery Dataset.zip'
        # Unzip the raw file
        if not os.path.exists(raw_file.parent / 'Battery Dataset'):
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                pbar = zip_ref.namelist()
                if not self.silent:
                    pbar = tqdm(pbar)
                for file in pbar:
                    if not self.silent:
                        pbar.set_description(f'Unzip XJTU file {file}')
                    zip_ref.extract(file, raw_file.parent)
        else:
            if not self.silent:
                tqdm.write('Skipping XJTU dataset, already exists')

        for files_path in cells_files_path:
            mat_path = raw_file.parent / 'Battery Dataset' / files_path
            if not mat_path.exists():
                if not self.silent:
                    tqdm.write(f'Skipping XJTU {files_path}, folder not found under Battery Dataset')
                continue
            mat_files = os.listdir(mat_path)
            mats = [i for i in mat_files if i.endswith('.mat')]
            for mat in mats:
                cells.append(mat)
                paths.append(mat_path)

        process_batteries_num = 0
        skip_batteries_num = 0
        for path, cell in zip(paths, tqdm(cells, desc='Processing XJTU file')):
            cell = cell.split('.mat')[0]
            cell_name = 'XJTU_' + cell
            # Step1: judge whether to skip the processed file
            whether_to_skip = self.check_processed_file(cell_name)
            if whether_to_skip == True:
                skip_batteries_num += 1
                continue

            mat = loadmat(str(path / cell))
            data = mat['data']
            cell_df = pd.DataFrame()
            for cycle in range(1, data.shape[1]+1):
                cycle_data_df = get_one_cycle(data, cycle)
                cycle_data_df['cycle_number'] = cycle
                cell_df = pd.concat([cell_df, cycle_data_df], ignore_index=True)

            # split capacity columns
            batch_name = path.name
            cell_df = split_capacity_column(
                cell_df,
                cycle_number_column_name='cycle_number',
                current_column_name='current_A',
                capacity_column_name='capacity_Ah',
                nominal_capacity=2.0,
            )

            # Step3: organize the cell data
            battery = organize_cell(cell_df, cell_name, batch_name)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num


def organize_cell(timeseries_df, name, batch_name):
    # Preserve raw sample order within each cycle to avoid mixing points from
    # different steps that can share the same relative_time_min (e.g., reset to 0).
    timeseries_df = timeseries_df.copy()
    timeseries_df['_record_order'] = np.arange(len(timeseries_df), dtype=int)
    timeseries_df['cycle_number'] = pd.to_numeric(timeseries_df['cycle_number'], errors='coerce')
    timeseries_df['relative_time_min'] = pd.to_numeric(timeseries_df['relative_time_min'], errors='coerce')
    timeseries_df = timeseries_df.dropna(subset=['cycle_number', 'relative_time_min'])
    timeseries_df = timeseries_df.sort_values(['cycle_number', '_record_order'], kind='mergesort')
    cycle_data = []
    remove_first_cycle = batch_name in XJTU_REMOVE_FIRST_CYCLE_BATCHES
    for cycle_index, df in timeseries_df.groupby('cycle_number'):
        # skip the first RPT test cycle
        if remove_first_cycle and cycle_index < 2:
            continue
        out_cycle_number = int(cycle_index - 1) if remove_first_cycle else int(cycle_index)
        attribute = str(df['attribute'].iloc[0]) if 'attribute' in df else 'Cycling'
        cycle_data.append(CycleData(
            cycle_number=out_cycle_number,
            voltage_in_V=df['voltage_V'].tolist(),
            current_in_A=df['current_A'].tolist(),
            temperature_in_C=None,
            discharge_capacity_in_Ah=df['discharge_cap'].tolist(),
            charge_capacity_in_Ah=df['charge_cap'].tolist(),
            time_in_s=list(df['relative_time_min'].values * 60),
            step_description=get_cycle_step_description(df),
            attribute=attribute,
        ))
    batch_config = XJTU_BATCH_CONFIG.get(batch_name, XJTU_BATCH_CONFIG['Batch-1'])

    charge_protocol = [
        CyclingProtocol(rate_in_C=rate, start_soc=0.0, end_soc=1.0)
        for rate in batch_config['charge_rates']
    ]
    discharge_protocol = [
        CyclingProtocol(
            rate_in_C=rate,
            start_soc=1.0,
            end_soc=0.0,
        )
        for rate in batch_config['discharge_rates']
    ]
    soc_interval = batch_config['soc_interval']

    # Normalize time data across all cycles
    cycle_data = normalize_cycle_times(cycle_data, name)
    cycle_data = enforce_strictly_increasing_time(cycle_data)

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='cylindrical_18650',
        anode_material='graphite',
        cathode_material='LiNi0.5Co0.2Mn0.3O2',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=2.0,
        min_voltage_limit_in_V=batch_config['min_voltage_limit'],
        max_voltage_limit_in_V=4.2,
        SOC_interval=soc_interval
    )


def enforce_strictly_increasing_time(cycles, min_step=1e-6):
    """
    Ensure time_in_s is strictly increasing for each cycle and globally across
    cycles without changing sample count or sample order.

    This is XJTU-specific post-processing to avoid identical timestamps mapping
    to different current/voltage points after reset correction.
    """
    global_last = None
    for cycle in cycles:
        if not hasattr(cycle, 'time_in_s') or not cycle.time_in_s:
            continue

        fixed = []
        for i, t in enumerate(cycle.time_in_s):
            t = float(t)
            if i == 0:
                if global_last is not None and t <= global_last:
                    t = global_last + min_step
            else:
                if t <= fixed[-1]:
                    t = fixed[-1] + min_step
            fixed.append(t)

        cycle.time_in_s = fixed
        global_last = fixed[-1]

    return cycles

def get_value(data, cycle,variable):
    variable_name = ['system_time', 'relative_time_min', 'voltage_V', 'current_A', 'capacity_Ah', 'power_Wh',
                     'temperature_C', 'description']
    if isinstance(variable,str):
        variable = variable_name.index(variable)
    assert cycle <= data.shape[1]
    assert variable <= 7
    value = data[0][cycle-1][variable]
    if variable == 7:
        value = value[0]
    else:
        value = value.reshape(-1)
    return value

def get_one_cycle(data, cycle):
    assert cycle <= data.shape[1]
    cycle_data = pd.DataFrame()
    cycle_data['system_time'] = get_value(data, cycle=cycle,variable='system_time')
    cycle_data['relative_time_min'] = get_value(data, cycle=cycle,variable='relative_time_min')
    cycle_data['voltage_V'] = get_value(data, cycle=cycle,variable='voltage_V')
    cycle_data['current_A'] = get_value(data, cycle=cycle,variable='current_A')
    cycle_data['capacity_Ah'] = get_value(data, cycle=cycle,variable='capacity_Ah')
    cycle_data['power_Wh'] = get_value(data, cycle=cycle,variable='power_Wh')
    cycle_data['temperature_C'] = get_value(data, cycle=cycle,variable='temperature_C')
    cycle_data['description'] = get_value(data, cycle=cycle,variable='description')
    return cycle_data

def split_capacity_column(
    df,
    cycle_number_column_name,
    current_column_name,
    capacity_column_name,
    nominal_capacity,
):
    cycle_number = list(set(df[cycle_number_column_name].values))
    for cycle in cycle_number:
        current_records = df.loc[df[cycle_number_column_name] == cycle, current_column_name].values
        current_c_rate = current_records / nominal_capacity
        capacity_records = df.loc[df[cycle_number_column_name] == cycle, capacity_column_name].values

        # get start and end index for charge period
        cutoff_indices = np.nonzero(current_c_rate >= 0.01)
        charge_start_index = cutoff_indices[0][0]
        charge_end_index = cutoff_indices[0][-1]

        # get start and end index for discharge period
        cutoff_indices = np.nonzero(current_c_rate <= -0.01)
        discharge_start_index = cutoff_indices[0][0]
        discharge_end_index = cutoff_indices[0][-1]

        # get index for rest period
        rest_indices = np.nonzero(np.abs(current_c_rate) < 0.01)

        # set the charge and discharge columns
        # format:
        #   if in charging, the discharge columns will be set into 0.
        #   if in discharging, the charge columns will be set into 0.
        #   if in resting, both charge and discharge columns will be set into 0.
        discharge_capacity_records = capacity_records.copy()
        discharge_capacity_records[charge_start_index: charge_end_index + 1] = 0
        discharge_capacity_records[rest_indices] = 0

        charge_capacity_records = capacity_records.copy()
        charge_capacity_records[discharge_start_index: discharge_end_index + 1] = 0
        charge_capacity_records[rest_indices] = 0

        df.loc[df[cycle_number_column_name] == cycle, 'discharge_cap'] = discharge_capacity_records
        df.loc[df[cycle_number_column_name] == cycle, 'charge_cap'] = charge_capacity_records

    set_cycle_attributes_from_description(df, cycle_number_column_name)
    return df


def get_cycle_step_description(cycle_df):
    if 'description' not in cycle_df:
        return ''
    descriptions = pd.Series(cycle_df['description']).dropna().astype(str)
    descriptions = descriptions[descriptions.str.len() > 0]
    return ' | '.join(pd.unique(descriptions))


def set_cycle_attributes_from_description(df, cycle_number_column_name):
    for cycle, cycle_df in df.groupby(cycle_number_column_name):
        description = get_cycle_step_description(cycle_df)
        attribute = 'RPT' if is_capacity_test_description(description) else 'Cycling'
        df.loc[cycle_df.index, 'attribute'] = attribute
    return df


def is_capacity_test_description(description):
    description = str(description).lower()
    return (
        'test capacity' in description
        or 'capacity test' in description
        or ('1.0c charge' in description and '0.5c discharge' in description)
    )
