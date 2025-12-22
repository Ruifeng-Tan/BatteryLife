# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import json
import zipfile
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from pathlib import Path
from datetime import datetime

from batteryml import BatteryData, CycleData, CyclingProtocol
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor
from .time_normalization_utils import normalize_cycle_times


@PREPROCESSORS.register()
class ISU_ILCCPreprocessor(BasePreprocessor):
    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        raw_file = Path(parentdir) / '22582234.zip'
        # Unzip the raw file
        # Skip extraction if the file already exists
        if not os.path.exists(raw_file.parent / '22582234'):
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                pbar = zip_ref.namelist()
                if not self.silent:
                    pbar = tqdm(pbar)
                for file in pbar:
                    if not self.silent:
                        pbar.set_description(f'Unzip ISU_ILCC file {file}')
                    zip_ref.extract(file, raw_file.parent / '22582234')
        else:
            if not self.silent:
                tqdm.write('Skipping ISU_ILCC dataset, already exists')

        # Iterate through each file in the zip
        cycle_zip_path = raw_file.parent / '22582234/Cycling_json.zip'
        rpt_zip_path = raw_file.parent / '22582234/RPT_json.zip'
        # zip_files = [i for i in files_path if i.endswith('Cycling_json.zip')]
        if not os.path.exists(cycle_zip_path.parent / 'Cycling_json'):
            with zipfile.ZipFile(cycle_zip_path, 'r') as zip_ref:
                pbar = zip_ref.namelist()
                if not self.silent:
                    pbar = tqdm(pbar)
                for file in pbar:
                    if not self.silent:
                        pbar.set_description(f'Unzip cycling files {file}')
                    zip_ref.extract(file, cycle_zip_path.parent)
        else:
            if not self.silent:
                tqdm.write('Skipping cycling files, already exists')

        if not os.path.exists(rpt_zip_path.parent / 'RPT_json'):
            with zipfile.ZipFile(rpt_zip_path, 'r') as zip_ref:
                pbar = zip_ref.namelist()
                if not self.silent:
                    pbar = tqdm(pbar)
                for file in pbar:
                    if not self.silent:
                        pbar.set_description(f'Unzip RPT files {file}')
                    zip_ref.extract(file, rpt_zip_path.parent)
        else:
            if not self.silent:
                tqdm.write('Skipping RPT files, already exists')

        zip_path = raw_file.parent / '22582234'
        valid_cells = pd.read_csv(zip_path / 'Valid_cells.csv').values.flatten().tolist()
        batch2 = ['G57C1', 'G57C2', 'G57C3', 'G57C4', 'G58C1', 'G26C3', 'G49C1', 'G49C2', 'G49C3', 'G49C4', 'G50C1',
                  'G50C3', 'G50C4']

        process_batteries_num = 0
        skip_batteries_num = 0
        for cell in tqdm(valid_cells, desc='Processing ISU_ILCC cells'):
            print(f'processing cell {cell}')
            if cell == 'G42C4' or cell == 'G9C4' or cell == 'G25C4' or 'G26' in cell or 'G11' in cell:
                continue
            # Step1: judge whether to skip the processed file
            whether_to_skip = self.check_processed_file('ISU-ILCC_' + cell)
            if whether_to_skip == True:
                skip_batteries_num += 1
                continue

            if cell not in batch2:
                subfolder = 'Release 1.0'
            else:
                subfolder = 'Release 2.0'

            # Step2: get the raw data into DataFrame format
            cycling_dict = convert_cycling_to_dict(zip_path, cell, subfolder)
            df = pd.DataFrame()

            for index in tqdm(range(len(cycling_dict['QV_charge']['I']))):
                cycle_df = pd.DataFrame()
                cycle_number_df = pd.DataFrame([index+1] * (len(cycling_dict['QV_charge']['I'][index]) + len(cycling_dict['QV_discharge']['I'][index])))
                cycle_df['cycle_number'] = cycle_number_df
                cycle_df['I'] = pd.concat([pd.DataFrame(cycling_dict['QV_charge']['I'][index]), pd.DataFrame(cycling_dict['QV_discharge']['I'][index])],
                                          ignore_index=True)
                cycle_df['V'] = pd.concat([pd.DataFrame(cycling_dict['QV_charge']['V'][index]), pd.DataFrame(cycling_dict['QV_discharge']['V'][index])],
                                          ignore_index=True)
                cycle_df['t'] = pd.concat([pd.DataFrame(cycling_dict['QV_charge']['t'][index]), pd.DataFrame(cycling_dict['QV_discharge']['t'][index])],
                                          ignore_index=True)
                charge_zero_df = [0] * len(cycling_dict['QV_charge']['Q'][index])
                discharge_zero_df = [0] * len(cycling_dict['QV_discharge']['Q'][index])
                cycle_df['Q_charge'] = pd.concat([pd.DataFrame(cycling_dict['QV_charge']['Q'][index]), pd.DataFrame(discharge_zero_df)], ignore_index=True)
                cycle_df['Q_discharge'] = pd.concat([pd.DataFrame(charge_zero_df), pd.DataFrame(cycling_dict['QV_discharge']['Q'][index])], ignore_index=True)
                df = pd.concat([df, cycle_df], ignore_index=True)\

            # Step3: drop the cycles after every RPT test
            df = clean_cell(df, zip_path, cell, subfolder)

            # Step4: organize the cell data
            cell = 'ISU-ILCC_' + cell
            battery = organize_cell(df, cell)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name):
    timeseries_df = timeseries_df.sort_values('t')
    cycle_data = []
    for cycle_index, df in timeseries_df.groupby('cycle_number'):
        time_list = df['t'].values.tolist()
        time_in_s = []
        for time in time_list:
            if is_valid_datetime(str(time)):
                time = datetime.strptime(str(time), '%Y-%m-%d %H:%M:%S')
                time_in_seconds = time.timestamp()
                time_in_s.append(time_in_seconds)
            else:
                time_in_s.append(time)
        df['t'] = time_in_s

        cycle_data.append(CycleData(
            cycle_number=int(cycle_index),
            voltage_in_V=df['V'].tolist(),
            current_in_A=df['I'].tolist(),
            temperature_in_C=None,
            discharge_capacity_in_Ah=df['Q_discharge'].tolist(),
            charge_capacity_in_Ah=df['Q_charge'].tolist(),
            time_in_s=time_in_s
        ))
    # Charge Protocol is constant current
    charge_start_soc, discharge_end_soc = calculate_soc_start_and_end(timeseries_df, name)

    rates = CYCLING_RATES[name[:-2]]
    charge_protocol = [CyclingProtocol(
        rate_in_C=float(rates[0]), start_soc=charge_start_soc[name], end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=float(rates[1]), start_soc=1.0, end_soc=discharge_end_soc[name]
    )]

    soc_interval = [charge_start_soc[name], 1]

    # Normalize time data across all cycles
    cycle_data = normalize_cycle_times(cycle_data, name)

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor=' 502030-size Li-polymer',
        anode_material='graphite',
        cathode_material='NMC',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=0.25,
        min_voltage_limit_in_V=3.0,
        max_voltage_limit_in_V=4.2,
        SOC_interval=soc_interval
    )

def is_valid_datetime(datetime_str):
    try:
        datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S')
        return True
    except ValueError:
        return False

def clean_cell(df, zip_path, cell, subfolder):
    i = 0
    should_exclude = []
    data_dict_cycle = convert_cycling_to_dict(zip_path, cell, subfolder)
    data_dict_rpt = convert_RPT_to_dict(zip_path, cell, subfolder)
    cycle_start = data_dict_cycle['QV_discharge']['t']

    # drop the cycles after every RPT test
    for cycle_number in range(len(cycle_start)):
        current_cycle_start = data_dict_cycle['QV_discharge']['t'][cycle_number][0]
        if not data_dict_rpt['start_stop_time']['start'][i]:
            continue
        elif data_dict_rpt['start_stop_time']['start'][i] < current_cycle_start:
            should_exclude.append(cycle_number+1)
            i = i + 1
    df = df[~df['cycle_number'].isin(set(should_exclude))]

    cycle_number = set([index for index in df['cycle_number'].tolist() if index != 0])
    for current_index, new_index in zip(cycle_number, range(1, len(cycle_number) + 1)):
        df.loc[df['cycle_number'] == current_index, 'cycle_number'] = new_index
    return df

def convert_cycling_to_dict(zip_path, cell, subfolder):
    with open(zip_path / 'Cycling_json/{}/{}.json'.format(subfolder,cell),'r') as file:
        data_dict = json.loads(json.load(file))

    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    for iii, start_time in enumerate(data_dict['time_series_charge']['start']):
        if start_time != '[]':
            data_dict['time_series_charge']['start'][iii] = np.datetime64(start_time)
            data_dict['time_series_charge']['stop'][iii] = np.datetime64(data_dict['time_series_charge']['stop'][iii])
            data_dict['time_series_discharge']['start'][iii] = np.datetime64(
                data_dict['time_series_discharge']['start'][iii])
            data_dict['time_series_discharge']['stop'][iii] = np.datetime64(
                data_dict['time_series_discharge']['stop'][iii])
        else:
            data_dict['time_series_charge']['start'][iii] = []
            data_dict['time_series_charge']['stop'][iii] = []
            data_dict['time_series_discharge']['start'][iii] = []
            data_dict['time_series_discharge']['stop'][iii] = []

    for iii in range(len(data_dict['time_series_charge']['start'])):
        data_dict['QV_charge']['t'][iii] = list(map(np.datetime64, data_dict['QV_charge']['t'][iii]))
        data_dict['QV_discharge']['t'][iii] = list(map(np.datetime64, data_dict['QV_discharge']['t'][iii]))

    return data_dict

def convert_RPT_to_dict(zip_path, cell, subfolder):
    with open(zip_path / 'RPT_json/{}/{}.json'.format(subfolder,cell),'r') as file:
        data_dict = json.loads(json.load(file))

    # Convert time series data from string to np.datetime64
    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    for iii in range(len(data_dict['start_stop_time']['start'])):
        data_dict['QV_charge_C_2']['t'][iii] = list(map(np.datetime64, data_dict['QV_charge_C_2']['t'][iii]))
        data_dict['QV_discharge_C_2']['t'][iii] = list(map(np.datetime64, data_dict['QV_discharge_C_2']['t'][iii]))
        data_dict['QV_charge_C_5']['t'][iii] = list(map(np.datetime64, data_dict['QV_charge_C_5']['t'][iii]))
        data_dict['QV_discharge_C_5']['t'][iii] = list(map(np.datetime64, data_dict['QV_discharge_C_5']['t'][iii]))

    return data_dict

def calculate_soc_start_and_end(df, name, nominal_capacity=0.25):
    charge_start_soc, discharge_end_soc = {}, {}

    charge_capacity = df.loc[df['cycle_number'] == 1, 'Q_charge'].max()
    soc_charge_interval = charge_capacity / nominal_capacity
    if soc_charge_interval > 1:
        soc_charge_interval = 1
    charge_start_soc[name] = 1 - soc_charge_interval

    discharge_capacity = df.loc[df['cycle_number'] == 1, 'Q_discharge'].max()
    soc_discharge_interval = discharge_capacity / nominal_capacity
    if soc_discharge_interval > 1:
        soc_discharge_interval = 0
    discharge_end_soc[name] = 1 - soc_discharge_interval

    if charge_start_soc[name] == 1 and discharge_end_soc[name] == 1:
        charge_start_soc[name] = 0
        discharge_end_soc[name] = 1
    return charge_start_soc, discharge_end_soc

CYCLING_RATES = {
    'ISU-ILCC_G1': [0.5, 0.5],
    'ISU-ILCC_G2': [0.5, 0.5],
    'ISU-ILCC_G3': [0.5, 0.5],
    'ISU-ILCC_G4': [1, 0.5],
    'ISU-ILCC_G5': [1, 0.5],
    'ISU-ILCC_G6': [2, 0.5],
    'ISU-ILCC_G7': [2, 0.5],
    'ISU-ILCC_G8': [2, 0.5],
    'ISU-ILCC_G9': [2, 0.5],
    'ISU-ILCC_G10': [2.5, 0.5],
    'ISU-ILCC_G12': [3, 0.5],
    'ISU-ILCC_G13': [3, 0.5],
    'ISU-ILCC_G14': [3, 0.5],
    'ISU-ILCC_G15': [3, 0.5],
    'ISU-ILCC_G16': [0.5, 0.5],
    'ISU-ILCC_G17': [1, 0.5],
    'ISU-ILCC_G18': [2.5, 0.5],
    'ISU-ILCC_G19': [2.5, 0.5],
    'ISU-ILCC_G20': [0.8, 0.5],
    'ISU-ILCC_G21': [1.2, 0.5],
    'ISU-ILCC_G22': [1.4, 0.5],
    'ISU-ILCC_G23': [1.6, 0.5],
    'ISU-ILCC_G24': [1.8, 0.5],
    'ISU-ILCC_G25': [1.8, 0.6],
    'ISU-ILCC_G26': [1.4, 2.2],
    'ISU-ILCC_G27': [0.6, 2.4],
    'ISU-ILCC_G28': [2.4, 1.6],
    'ISU-ILCC_G29': [1.6, 1.8],
    'ISU-ILCC_G30': [0.8, 0.8],
    'ISU-ILCC_G31': [1.2, 1],
    'ISU-ILCC_G32': [1, 1.4],
    'ISU-ILCC_G33': [2, 1.2],
    'ISU-ILCC_G34': [2.2, 2],
    'ISU-ILCC_G35': [1.825, 0.5],
    'ISU-ILCC_G36': [2.075, 0.5],
    'ISU-ILCC_G37': [0.725, 0.5],
    'ISU-ILCC_G38': [1.875, 0.5],
    'ISU-ILCC_G39': [1.475, 0.5],
    'ISU-ILCC_G40': [1.825, 1.025],
    'ISU-ILCC_G41': [2.075, 1.775],
    'ISU-ILCC_G42': [0.725, 2.375],
    'ISU-ILCC_G43': [1.875, 2.325],
    'ISU-ILCC_G44': [0.775, 1.275],
    'ISU-ILCC_G45': [1.125, 1.725],
    'ISU-ILCC_G46': [1.225, 2.025],
    'ISU-ILCC_G47': [2.325, 1.925],
    'ISU-ILCC_G48': [2.375, 2.225],
    'ISU-ILCC_G49': [0.975, 0.675],
    'ISU-ILCC_G50': [2.425, 1.625],
    'ISU-ILCC_G51': [2.275, 1.875],
    'ISU-ILCC_G52': [1.425, 0.875],
    'ISU-ILCC_G53': [2.025, 0.825],
    'ISU-ILCC_G54': [0.925, 1.125],
    'ISU-ILCC_G55': [1.025, 2.475],
    'ISU-ILCC_G56': [2.175, 0.975],
    'ISU-ILCC_G57': [1.775, 1.175],
    'ISU-ILCC_G58': [2.475, 0.575],
    'ISU-ILCC_G59': [1.325, 1.825],
    'ISU-ILCC_G60': [0.675, 1.325],
    'ISU-ILCC_G61': [2.125, 1.975],
    'ISU-ILCC_G62': [1.575, 2.425],
    'ISU-ILCC_G63': [1.975, 1.675],
    'ISU-ILCC_G64': [1.175, 1.425],
}