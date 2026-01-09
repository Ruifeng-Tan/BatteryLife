# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import numpy as np
import pandas as pd
import openpyxl

from tqdm import tqdm
from typing import List
from pathlib import Path

from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor
from .time_normalization_utils import normalize_cycle_times


@PREPROCESSORS.register()
class CALBPreprocessor(BasePreprocessor):
    def process(self, parent_dir, **kwargs) -> List[BatteryData]:
        path = Path(parent_dir)
        files_path_list = ['0度', '25度', '35度', '45度']  # drop the -10 batch for its capacity retention bigger than 0.925
        process_batteries_num = 0
        skip_batteries_num = 0
        for files_path in files_path_list:
            file_path = os.listdir(path / files_path)
            files = [i for i in file_path if i.endswith('.xlsx')]
            for file in tqdm(files):
                cell_name = 'CALB_' + files_path.split('度')[0] + '_' + file.split('.')[0]
                if cell_name.startswith('CALB_45_B254'):
                    continue

                # step1: judge whether to skip the processed file if exists
                whether_to_skip = self.check_processed_file(cell_name)
                if whether_to_skip == True:
                    skip_batteries_num += 1
                    continue

                data = pd.read_excel(path / files_path/ file, sheet_name='record')
                df = pd.DataFrame(data)
                if (files_path == '25度') or (files_path == '35度') or ('B254' in file) or ('B256' in file):
                    df = df[df['循环号'] > 1]

                # split capacity columns
                df = split_capacity_column(df, cycle_number_column_name='循环号', current_column_name='电流(A)', capacity_column_name='容量(Ah)', nominal_capacity=58)

                # organize data
                battery = organize_cell(df, cell_name, 58, files_path)
                self.dump_single_file(battery)
                process_batteries_num += 1

                if not self.silent:
                    tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, C, temperature):
    temperature_in_C_value = 0
    charge_rate_in_C = 0
    discharge_rate_in_C = 0
    lower_cutoff_voltage = 0
    upper_cutoff_voltage = 0
    if temperature.startswith('0'):
        temperature_in_C_value = 0
        charge_rate_in_C = 1.0
        discharge_rate_in_C = 1.0
        lower_cutoff_voltage = 2.2
        upper_cutoff_voltage = 4.35
    elif temperature.startswith('-10'):
        temperature_in_C_value = -10
        charge_rate_in_C = 'stepcharge'
        discharge_rate_in_C = 'stepcharge'
        lower_cutoff_voltage = 2.75
        upper_cutoff_voltage = 4.35
    elif temperature.startswith('25'):
        temperature_in_C_value = 25
        charge_rate_in_C = 'stepcharge'
        discharge_rate_in_C = 'stepcharge'
        lower_cutoff_voltage = 2.75
        upper_cutoff_voltage = 4.35
    elif temperature.startswith('35'):
        temperature_in_C_value = 35
        charge_rate_in_C = 'stepcharge'
        discharge_rate_in_C = 'stepcharge'
        lower_cutoff_voltage = 2.75
        upper_cutoff_voltage = 4.35
    elif temperature.startswith('45'):
        temperature_in_C_value = 45
        charge_rate_in_C = 5.0
        discharge_rate_in_C = 15.0
        lower_cutoff_voltage = 2.5
        upper_cutoff_voltage = 4.25

    if '-10' in name:
        cycle_data = []
        for cycle_index, df in timeseries_df.groupby('外循环'):
            cycle_data.append(CycleData(
                cycle_number=int(cycle_index),
                voltage_in_V=df['电压(V)'].tolist(),
                current_in_A=df['电流(A)'].tolist(),
                temperature_in_C=list([temperature_in_C_value] * len(df)),
                discharge_capacity_in_Ah=df['安时(AH)'].tolist(),
                charge_capacity_in_Ah=df['安时(AH)'].tolist(),
                time_in_s=df['步时间(s)'].tolist()
            ))
    else:
        cycle_data = []
        for cycle_index, df in timeseries_df.groupby('循环号'):
            times = []
            for time in list(df['绝对时间'].values):
                time = time.split(' ')[1]
                h = float(time.split(':')[0])
                m = float(time.split(':')[1])
                s = float(time.split(':')[2])
                seconds = (h * 3600 + m * 60 + s)
                times.append(seconds)

            if '_0' in name:
                capacities = df['容量(Ah)'].tolist()
                discharge_capacities = df['放电容量(Ah)'].tolist()
                charge_capacities = list(np.array(capacities) - np.array(discharge_capacities))
            else:
                charge_capacities = []
                for cc in list(df['充电容量(Ah)'].values):
                    if len(charge_capacities) == 0:
                        charge_capacities.append(cc)
                    else:
                        accumulate_cc = charge_capacities[-1] + cc
                        charge_capacities.append(accumulate_cc)

            cycle_data.append(CycleData(
                cycle_number=int(cycle_index),
                voltage_in_V=df['电压(V)'].tolist(),
                current_in_A=df['电流(A)'].tolist(),
                temperature_in_C=list([temperature_in_C_value] * len(df)),
                discharge_capacity_in_Ah=df['放电容量(Ah)'].tolist(),
                charge_capacity_in_Ah=charge_capacities,
                time_in_s=times
            ))
    # Charge Protocol is constant current
    charge_protocol = [CyclingProtocol(
        rate_in_C=charge_rate_in_C, start_soc=0.0, end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=discharge_rate_in_C, start_soc=1.0, end_soc=0.0
    )]

    soc_interval = [0, 1]

    # Normalize time data across all cycles
    cycle_data = normalize_cycle_times(cycle_data, name)

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='Prismatic',
        anode_material='graphite',
        cathode_material='NMC',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=C,
        min_voltage_limit_in_V=lower_cutoff_voltage,
        max_voltage_limit_in_V=upper_cutoff_voltage,
        SOC_interval=soc_interval
    )

def split_capacity_column(df, cycle_number_column_name, current_column_name, capacity_column_name, nominal_capacity):
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

    return df