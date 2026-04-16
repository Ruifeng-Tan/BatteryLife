import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm

def load_calb_capacity_from_excel(sheets, CALB_summary_file):
    capacity_data_dict = {}

    for sheet in sheets:
        cells_data = pd.read_excel(CALB_summary_file, sheet_name=sheet)
        columns_name = cells_data.columns.tolist()

        if sheet == '0℃循环':
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('A1')]
            cells_name = ['CALB_0_' + i.replace('A', 'B') for i in columns_name if i.startswith('A1')]

        elif sheet == '25℃ 循环':
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('T25')]
            cells_name = ['CALB_25_' + i for i in columns_name if i.startswith('T25')]

        elif sheet == '35℃ 循环':
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('B')]
            cells_name = ['CALB_35_' + i for i in columns_name if i.startswith('B')]

        elif sheet == '45℃循环':
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('B')]
            cells_name = ['CALB_45_' + i for i in columns_name if i.startswith('B')]

        else:
            continue

        cycles_column_idx = [i + 1 for i in start_column_idx]
        times_column_idx = [i + 2 for i in start_column_idx]
        discharge_column_idx = [i + 4 for i in start_column_idx]

        for cell_name, cycle_idx, discharge_idx, time_idx in zip(
            tqdm(cells_name, desc=f'Loading {sheet} Capacity Data'),
            cycles_column_idx,
            discharge_column_idx,
            times_column_idx
        ):

            if cell_name.startswith('CALB_45_B254'):
                continue

            cycles = cells_data.iloc[:, cycle_idx].tolist()
            discharge = cells_data.iloc[:, discharge_idx].tolist()
            times = cells_data.iloc[:, time_idx].tolist()

            cycle_df = pd.DataFrame({
                'cycle_number': cycles,
                'time_in_s': times,
                'discharge_capacity_in_Ah': discharge
            })

            capacity_data_dict[cell_name] = cycle_df

    return capacity_data_dict


def cal_life_labels(dataset_name, dataset_root_path, output_path):
    dataset_path = f'{dataset_root_path}/{dataset_name}'
    files = os.listdir(dataset_path)

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
            # print(file_name, last_cycle_soh)

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

                # # only keep life label >100 cells
                # if eol < 100:
                #     continue
                # if not find_eol:
                #     # The end of life is not found in the battery
                #     eol = len(cycle_data) + 1

            name_lables[file_name] = eol
            # print(file_name, eol)

        elif dataset_name == 'CALB':
            data = pickle.load(open(f'{dataset_path}/{file_name}', 'rb'))
            df = capacity_data_dict[file_name.split('.pkl')[0]].copy()
            df = df.bfill()
            cycle_data = data['cycle_data']

            nominal_capacity = df['discharge_capacity_in_Ah'].values[0]# use the capacity of first cycle as nominal capacity for CALB dataset

            SOC_interval = data['SOC_interval']  # get the charge and discharge soc interval
            SOC_interval = SOC_interval[1] - SOC_interval[0]

            total_SOHs = df['discharge_capacity_in_Ah'].values / nominal_capacity / SOC_interval
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
                    # skip the abnormal drop cycle
                    if file_name.startswith('CALB_35_B229'):
                        if cycle_number == 696:
                            continue

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


    if not os.path.exists(f'{output_path}'):
        os.makedirs(f'{output_path}')

    print(f'Totally {len(name_lables)} batteries have life labels | {abadon_count} batteries are excluded.')
    print(f'Labels are saved in {output_path}/{dataset_name}_labels.json')
    if dataset_name == 'UL_PUR':
        with open(f'{output_path}/UL-PUR_labels.json', 'w') as f:
            json.dump(name_lables, f)
    elif dataset_name == 'ZNcoin':
        with open(f'{output_path}/ZN-coin_labels.json', 'w') as f:
            json.dump(name_lables, f)
    elif dataset_name == 'NAion':
        with open(f'{output_path}/NA-ion_labels.json', 'w') as f:
            json.dump(name_lables, f)
    else:
        with open(f'{output_path}/{dataset_name}_labels.json', 'w') as f:
            json.dump(name_lables, f)



if __name__ == '__main__':
    dataset_name = ('CALCE')
    output_path = './Life_labels'
    dataset_root_path = '../datasets/processed/'
    need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']


    if dataset_name == 'CALB':
        print(f'Calculate dataset {dataset_name} life labels')
        CALB_capacity_folder = 'CALB_capacity'
        CALB_sheets = ['0℃循环', '25℃ 循环', '35℃ 循环', '45℃循环']
        CALB_summary_file = 'D:/python_project/BatteryML-main/datasets/raw/CALB/汇总表-L148N58-循环.xlsx'
        capacity_data_dict = load_calb_capacity_from_excel(CALB_sheets, CALB_summary_file)
    else:
        print(f'Calculate dataset {dataset_name} life labels')
        CALB_capacity_folder = None
        capacity_data_dict = {}

    cal_life_labels(dataset_name, dataset_root_path, output_path)