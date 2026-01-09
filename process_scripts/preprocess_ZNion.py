# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import numpy as np
import pandas as pd
from datetime import datetime

from tqdm import tqdm
from typing import List
from pathlib import Path

from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor
from .time_normalization_utils import normalize_cycle_times


@PREPROCESSORS.register()
class ZNionPreprocessor(BasePreprocessor):
    def process(self, parent_dir, **kwargs) -> List[BatteryData]:
        filenames = ['Batch-1', 'Batch-2', 'Batch-3']
        for filename in filenames:
            path = Path(parent_dir) / filename
            files_path = os.listdir(path)
            cells = [i for i in files_path if i.endswith('.xlsx')]

            process_batteries_num = 0
            skip_batteries_num = 0
            for cell in tqdm(cells, desc=f'Processing {filename} ZN-coin cells'):
                df = pd.DataFrame()
                capacity = 0
                cell_name = cell.split('.')[0]
                cell_name = cell_name.split('我的设备_')[0] + cell_name.split('我的设备_')[1]
                if cell_name in FILES_TO_DROP:
                    continue
                if filename == 'Batch-3':
                    cell_name = 'ZN-coin_' + cell_name + f'_{filename}'
                else:
                    cell_name = 'ZN-coin_' + cell_name

                # Step1: judge whether to skip the processed file if exists
                whether_to_skip = self.check_processed_file(cell_name)
                if whether_to_skip == True:
                    skip_batteries_num += 1
                    continue

                if filename == 'Batch-1':
                    # Step2: get timeseries and cycle_data from files of each cell
                    df = pd.read_excel(path / cell, sheet_name='记录')
                    capacity_records_df = pd.read_excel(path / cell, sheet_name='循环')
                    capacity = capacity_records_df.loc[capacity_records_df['循环序号'] == 10, '放电容量/mAh'].values[0] * 0.001

                    # check if there have some problematic cycles
                    try:
                        # some batteries have problematic cycles which specific charge capacity are < 1
                        cycles = capacity_records_df.loc[capacity_records_df['充电比容量/mAh/g'] < 1, '循环序号'].values[0] - 1
                    except:
                        cycles = capacity_records_df['循环序号'].max()

                    df = clean_cycles(df, cycles, filename)

                elif filename == 'Batch-2':
                    # Step2: get timeseries and cycle_data from files of each cell
                    capacity_records_df = pd.read_excel(path / cell, sheet_name='循环')
                    capacity = capacity_records_df.loc[capacity_records_df['循环序号'] == 10, '放电容量/mAh'].values[0] * 0.001
                    df1 = pd.read_excel(path / cell, sheet_name='记录')

                    # check if there have some problematic cycles
                    try:
                        # some batteries have problematic cycles which specific charge capacity are < 1
                        cycles = capacity_records_df.loc[capacity_records_df['充电比容量/mAh/g'] < 1, '循环序号'].values[0] - 1
                    except:
                        cycles = capacity_records_df['循环序号'].max()
                    try:
                        df2 = pd.read_excel(path / cell, sheet_name='记录(2)')
                        df = pd.concat([df1, df2], ignore_index=True)
                    except:
                        df = df1

                    df = clean_cycles(df, cycles, filename)

                elif filename == 'Batch-3':
                    capacity_records_df = pd.read_excel(path / cell, sheet_name='Cycle')
                    capacity = capacity_records_df.loc[capacity_records_df['Cycle'] == 10, 'CapD/mAh'].values[0] * 0.001

                    # Step2: get timeseries and cycle_data from files of each cell
                    sheets_name = pd.ExcelFile(path / cell).sheet_names
                    total_df = pd.DataFrame()
                    for name in sheets_name:
                        if name.startswith('Record'):
                            df = pd.read_excel(path / cell, sheet_name=name)
                            total_df = pd.concat([total_df, df], ignore_index=True)

                    df = total_df

                    # check if there have some problematic cycles
                    try:
                        # some batteries have problematic cycles which specific charge capacity are < 1
                        cycles = \
                        capacity_records_df.loc[capacity_records_df['CoulombEfficiency/%'] < 1, 'Cycle'].values[0] - 1
                    except:
                        cycles = capacity_records_df['Cycle'].max()

                    df = clean_cycles(df, cycles, filename)

                # reset the cycle number and unify the capacity data
                df = reset_cell(df, filename, capacity)



                battery = organize_cell(df, cell_name, capacity, filename)
                self.dump_single_file(battery)
                process_batteries_num += 1

                if not self.silent:
                    tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, C, filename):
    if filename == 'Batch-1':
        timeseries_df = timeseries_df.sort_values('测试时间')
    elif filename == 'Batch-2':
        timeseries_df = timeseries_df.sort_values('系统时间')

    cycle_data = []
    if filename == 'Batch-3':
        for cycle_index, df in timeseries_df.groupby('Cycle'):
            time_in_s = convert_to_s(list(df['TestTime'].values), cycle_index)
            current = df['Current/mA'] / 1000
            charge_capacity = df['charge_cap'] / 1000
            discharge_capacity = df['discharge_cap'] / 1000
            cycle_data.append(CycleData(
                cycle_number=int(cycle_index),
                voltage_in_V=df['Voltage/V'].tolist(),
                current_in_A=current.tolist(),
                temperature_in_C=None,
                discharge_capacity_in_Ah=discharge_capacity.tolist(),
                charge_capacity_in_Ah=charge_capacity.tolist(),
                time_in_s=time_in_s
            ))
    else:
        for cycle_index, df in timeseries_df.groupby('循环序号'):
            current = df['电流/mA'] / 1000
            charge_capacity = df['charge_cap'] / 1000
            discharge_capacity = df['discharge_cap'] / 1000
            if filename == 'Batch-1':
                time_in_s = convert_to_s(list(df['测试时间'].values), cycle_index)
                cycle_data.append(CycleData(
                    cycle_number=int(cycle_index),
                    voltage_in_V=df['电压/V'].tolist(),
                    current_in_A=current.tolist(),
                    temperature_in_C=None,
                    discharge_capacity_in_Ah=discharge_capacity.tolist(),
                    charge_capacity_in_Ah=charge_capacity.tolist(),
                    time_in_s=time_in_s
                ))
            elif filename == 'Batch-2':
                time_list = df['系统时间'].values.tolist()
                time_in_s = []
                for time in time_list:
                    time = datetime.strptime(time, '%Y-%m-%d %H:%M:%S.%f')
                    time_in_seconds = int(time.timestamp())
                    time_in_s.append(time_in_seconds)
                cycle_data.append(CycleData(
                    cycle_number=int(cycle_index),
                    voltage_in_V=df['电压/V'].tolist(),
                    current_in_A=current.tolist(),
                    temperature_in_C=None,
                    discharge_capacity_in_Ah=discharge_capacity.tolist(),
                    charge_capacity_in_Ah=charge_capacity.tolist(),
                    time_in_s=time_in_s
                ))
    # Charge Protocol is constant current
    charge_protocol = [CyclingProtocol(
        rate_in_C=8.0, start_soc=0.0, end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=8.0, start_soc=1.0, end_soc=0.0
    )]

    soc_interval = [0, 1]

    # Normalize time data across all cycles
    cycle_data = normalize_cycle_times(cycle_data, name)

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='coin',
        anode_material='MnO2',
        cathode_material='Zinc',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=C,
        min_voltage_limit_in_V=0.8,
        max_voltage_limit_in_V=1.8,
        SOC_interval=soc_interval
    )

def reset_cell(df, batch_name, capacity):
    if batch_name == 'Batch-3':
        cycles = df['Cycle']
        cycle_number = set([index for index in cycles.tolist() if index != 0])
        for current_index, new_index in zip(cycle_number, range(1, len(cycle_number) + 1)):
            # reset the cycle number
            df.loc[df['Cycle'] == current_index, 'Cycle'] = new_index

            current_records = df.loc[df['Cycle'] == new_index, 'Current/mA'].values
            current_c_rate = current_records / capacity
            capacity_records = df.loc[df['Cycle'] == new_index, 'Capacity/mAh'].values

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

            df.loc[df['Cycle'] == new_index, 'discharge_cap'] = discharge_capacity_records
            df.loc[df['Cycle'] == new_index, 'charge_cap'] = charge_capacity_records
    else:
        cycles = df['循环序号']
        cycle_number = set([index for index in cycles.tolist() if index != 0])
        for current_index, new_index in zip(cycle_number, range(1, len(cycle_number) + 1)):
            # reset the cycle number
            df.loc[df['循环序号'] == current_index, '循环序号'] = new_index

            current_records = df.loc[df['循环序号'] == new_index, '电流/mA'].values
            current_c_rate = current_records / capacity
            capacity_records = df.loc[df['循环序号'] == new_index, '容量/mAh'].values

            # get start and end index for charge period
            cutoff_indices = np.nonzero(current_c_rate > 0.01)
            charge_start_index = cutoff_indices[0][0]
            charge_end_index = cutoff_indices[0][-1]

            # get start and end index for discharge period
            cutoff_indices = np.nonzero(current_c_rate < -0.01)
            discharge_start_index = cutoff_indices[0][0]
            discharge_end_index = cutoff_indices[0][-1]

            # get start and end index for discharge period
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

            df.loc[df['循环序号'] == new_index, 'discharge_cap'] = discharge_capacity_records
            df.loc[df['循环序号'] == new_index, 'charge_cap'] = charge_capacity_records

    return df

def clean_cycles(df, cycles, filename):
    if filename == 'Batch-3':
        # drop probelmatic cycles
        df = df[df['Cycle'] <= cycles]

        # drop formation cycles
        df = df[df['Cycle'] > 9]
    else:
        # drop probelmatic cycles
        df = df[df['循环序号'] <= cycles]

        # drop formation cycles
        df = df[df['循环序号'] > 9]

    return df

def drop_abnormal_cycles(df, cell_name):
    if cell_name == 'ZN-coin_451-1_20240116203425_03_4_Batch-3':
        cycles = df['Cycle']
        cycle_number = set([index for index in cycles.tolist() if index != 0])
        for cycle in cycle_number:
            cycle_df = df[df['Cycle'] == cycle]
            voltage = cycle_df['Voltage/V'].values
            if cycle == 29:
                print(cycle_df)


def convert_to_s(time_list, cycle_index):
    time_in_s = []
    for time in time_list:
        h = float(str(time).split(':')[0])
        m = float(str(time).split(':')[1])
        s = float(str(time).split(':')[2])
        seconds = h * 3600 + m * 60 + s
        time_in_s.append(seconds)
    return time_in_s

# problematic cells
FILES_TO_DROP = [
    '202软包_20231218215923_05_8',
    '429-3_20231212185225_02_5',
    '403-2_20231209230005_01_5',
    '407-2_20231209231809_02_1',
    '411-3_20231209232908_09_6',
    '413-2_20231209233232_06_3',
    '414-1_20231209233323_06_5',
    '416-1_20231209233742_10_3',
    '417-1_20231209233942_10_6',
    '417-2_20231209234017_10_7',
    '421-2_20231205230026_01_5',
    '424-2_20231205230111_02_6',
    '425-3_20231205230128_03_2',
    '427-1_20231205230150_03_6',
    '427-3_20231205230157_03_8',
    '431-1_20231212185404_03_6',
    '431-2_20231212185411_03_7',
    '431-3_20231212185417_03_8',
    '443-1_20240104212454_09_4',
    '446-2_20240104212544_07_3',
]
