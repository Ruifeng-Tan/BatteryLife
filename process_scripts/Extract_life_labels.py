import pickle
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

dataset_name = ('ISU_ILCC')
dataset_root_path = '../datasets/processed/'
dataset_path = f'{dataset_root_path}/{dataset_name}'
files = os.listdir(dataset_path)
need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']

name_lables = {}
abadon_count = 0
for file_name in tqdm(files):
    if dataset_name != 'CALB':
        data = pickle.load(open(f'{dataset_path}/{file_name}', 'rb'))
        cycle_data = data['cycle_data']
        last_cycle = cycle_data[-1]
        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = data['nominal_capacity_in_Ah']
        SOC_interval = data['SOC_interval']  # get the charge and discharge soc interval
        SOC_interval = SOC_interval[1] - SOC_interval[0]
        if SOC_interval == 0:
            SOC_interval = 1 # fully charge and discharge
        last_cycle_soh = max(last_cycle['discharge_capacity_in_Ah']) / nominal_capacity / SOC_interval


        if last_cycle_soh >= 0.825:
            # [0.825, inf)
            # exclude this cell from the dataset
            abadon_count += 1
            continue
        elif last_cycle_soh > 0.8:
            # (0.8, 0.825)
            # Linear Regression based on the last 20 cycles to obtain the cycle life label
            regress_cycle_num = 20
            total_SOHs = []
            total_cycle_numbers = np.array([i + 1 for i in range(len(cycle_data) - regress_cycle_num, len(cycle_data))])
            for correct_cycle_index, sub_cycle_data in enumerate(cycle_data[-regress_cycle_num:]):
                Qd = max(sub_cycle_data['discharge_capacity_in_Ah'])
                cycle_number = sub_cycle_data['cycle_number']
                soh = Qd / nominal_capacity / SOC_interval
                total_SOHs.append(soh)

            total_SOHs = np.array(total_SOHs).reshape(-1, 1)
            linear_regressor = LinearRegression()
            linear_regressor.fit(total_SOHs, total_cycle_numbers)
            eol = linear_regressor.predict(np.array([0.80]).reshape(-1, 1))[0]
            eol = int(eol)
        else:
            # (-inf, 0.8]
            eol, find_eol = None, False
            for correct_cycle_index, sub_cycle_data in enumerate(cycle_data):
                Qd = max(sub_cycle_data['discharge_capacity_in_Ah'])
                soh = Qd / nominal_capacity / SOC_interval
                if soh <= 0.8 and not find_eol:
                    eol = correct_cycle_index + 1
                    find_eol = True
                    break
            # if not find_eol:
            #     # The end of life is not found in the battery
            #     eol = len(cycle_data) + 1

        name_lables[file_name] = eol
        print(file_name, eol)

    elif dataset_name == 'CALB':
        data = pickle.load(open(f'{dataset_path}/{file_name}', 'rb'))
        df = pd.read_csv(f'./CALB_capacity/{file_name}.csv')
        df = df.fillna(method='backfill')
        cycle_data = data['cycle_data']
        if file_name.startswith('RWTH'):
            nominal_capacity = 1.85
        elif file_name.startswith('SNL_18650_NCA_25C_20-80'):
            nominal_capacity = 3.2
        else:
            nominal_capacity = df['discharge_capacity'].values[0]# use the capacity of first cycle as nominal capacity for CALB dataset

        SOC_interval = data['SOC_interval']  # get the charge and discharge soc interval
        SOC_interval = SOC_interval[1] - SOC_interval[0]

        total_SOHs = df['discharge_capacity'].values / nominal_capacity / SOC_interval
        total_cycle_numbers = df['cycle_number'].values
        nan_mask = np.isnan(total_SOHs)
        total_SOHs = total_SOHs[~nan_mask]
        total_cycle_numbers = total_cycle_numbers[~nan_mask]

        find_eol = False
        if min(total_SOHs) < 0.925:
            use_extrapolation = True
        if min(total_SOHs) <= 0.9:
            find_eol = True
            for cycle_number, soh in enumerate(total_SOHs):
                print(cycle_number, soh)
                if soh <= 0.9:
                    eol = cycle_number + 1
                    name_lables[file_name] = eol
                    break

        if not find_eol:
            if use_extrapolation:
                regress_cycle_num = 20
                if file_name != 'CALB_25_T25-2.pkl':
                    tmp_SOHs = total_SOHs[-regress_cycle_num:].reshape(-1, 1)
                    tmp_cycle_numbers = total_cycle_numbers[-regress_cycle_num:]
                else:
                    # the last several cycles have sudden capacity rise
                    for tmp_i, soh in enumerate(total_SOHs):
                        if soh <= 0.925:
                            tmp_SOHs = total_SOHs[tmp_i - 19:tmp_i + 1].reshape(-1, 1)
                            tmp_cycle_numbers = total_cycle_numbers[tmp_i - 19:tmp_i + 1]
                            break

                linear_regressor = LinearRegression()
                linear_regressor.fit(tmp_SOHs, tmp_cycle_numbers)
                eol = linear_regressor.predict(np.array([0.9]).reshape(-1, 1))[0]
                eol = int(eol)
            else:
                abadon_count += 1
                continue
            name_lables[file_name] = eol



print(f'Totally {len(name_lables)} batteries have life labels | {abadon_count} batteries are excluded.')
print(f'Labels are saved in {dataset_root_path}/{dataset_name}_labels.json')
if dataset_name == 'UL_PUR':
    with open(f'{dataset_root_path}/UL-PUR_labels.json', 'w') as f:
        json.dump(name_lables, f)
elif dataset_name == 'ZNcoin':
    with open(f'{dataset_root_path}/ZN-coin_labels.json', 'w') as f:
        json.dump(name_lables, f)
elif dataset_name == 'NAion':
    with open(f'{dataset_root_path}/NA-ion_labels.json', 'w') as f:
        json.dump(name_lables, f)
else:
    with open(f'{dataset_root_path}/{dataset_name}_labels.json', 'w') as f:
        json.dump(name_lables, f)

