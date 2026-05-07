from data_provider.data_split_recorder import split_recorder
import os
import json


def get_agingCondition_battery_num(file_names, aging_conditions):
    label_files_path = os.listdir(label_path)
    label_json_files = [i for i in label_files_path if i.endswith('.json')]
    cell_names = []
    label_file_names = [i.replace('--', '-#') if 'Tongji' in i else i for i in file_names]
    for file in label_json_files:
        if file.startswith('Stanford_labels'):
            continue
        with open(os.path.join(label_path, file), 'r') as f:
            label_data = json.load(f)
        # print(file, len(label_data))
        for key, value in label_data.items():
            if value <= 100:
                # print(key, value)
                continue
            cell_names.append(key)  
    cell_names = [i for i in cell_names if i in label_file_names]
    cell_names = [i.replace('-#', '--') if 'Tongji' in i else i for i in cell_names]
    unique_aging_conditions = []
    for cell_name in cell_names:
        agingConditionID = aging_conditions[cell_name]
        if agingConditionID not in unique_aging_conditions:
            unique_aging_conditions.append(agingConditionID)

    return len(cell_names), len(unique_aging_conditions)


def get_aging_condition_ids(file_names, aging_conditions):
    aging_condition_ids = []
    for file_name in file_names:
        if file_name not in aging_conditions:
            continue
        aging_condition_id = aging_conditions[file_name]
        if aging_condition_id not in aging_condition_ids:
            aging_condition_ids.append(aging_condition_id)
    return set(aging_condition_ids)


def get_seen_unseen_agingCondition_num(train_files, val_files, test_files, aging_conditions):
    train_val_aging_condition_ids = get_aging_condition_ids(
        train_files + val_files,
        aging_conditions,
    )
    test_aging_condition_ids = get_aging_condition_ids(test_files, aging_conditions)
    seen_aging_condition_ids = test_aging_condition_ids & train_val_aging_condition_ids
    unseen_aging_condition_ids = test_aging_condition_ids - train_val_aging_condition_ids
    return len(seen_aging_condition_ids), len(unseen_aging_condition_ids)


def print_dataset_overview(dataset_name, train_files, val_files, test_files, aging_conditions):
    all_files = train_files + val_files + test_files
    battery_num, aging_condition_num = get_agingCondition_battery_num(
        all_files,
        aging_conditions,
    )
    seen_num, unseen_num = get_seen_unseen_agingCondition_num(
        train_files,
        val_files,
        test_files,
        aging_conditions,
    )
    print(dataset_name)
    print((battery_num, aging_condition_num))
    print(
        'Seen conditions num in test set:',
        seen_num,
        'Unseen conditions num in test set:',
        unseen_num,
    )

label_path = '/data/trf/python_works/BatteryLife/dataset/Life labels'
agingconditionID_path = './name2agingConditionID.json'

with open(agingconditionID_path) as file:
    aging_conditions = json.load(file)

dataset_infos = [
    ('Li-ion', split_recorder.MIX_large_train_files, split_recorder.MIX_large_val_files, split_recorder.MIX_large_test_files),
    ('HNEI', split_recorder.HNEI_train_files, split_recorder.HNEI_val_files, split_recorder.HNEI_test_files),
    ('MATR', split_recorder.MATR_train_files, split_recorder.MATR_val_files, split_recorder.MATR_test_files),
    ('MICH', split_recorder.MICH_train_files, split_recorder.MICH_val_files, split_recorder.MICH_test_files),
    ('XJTU', split_recorder.XJTU_train_files, split_recorder.XJTU_val_files, split_recorder.XJTU_test_files),
    ('Stanford', split_recorder.Stanford_train_files, split_recorder.Stanford_val_files, split_recorder.Stanford_test_files),
    ('RWTH', split_recorder.RWTH_train_files, split_recorder.RWTH_val_files, split_recorder.RWTH_test_files),
    ('MICH_EXP', split_recorder.MICH_EXP_train_files, split_recorder.MICH_EXP_val_files, split_recorder.MICH_EXP_test_files),
    ('Tongji', split_recorder.Tongji_train_files, split_recorder.Tongji_val_files, split_recorder.Tongji_test_files),
    ('HUST', split_recorder.HUST_train_files, split_recorder.HUST_val_files, split_recorder.HUST_test_files),
    ('SNL', split_recorder.SNL_train_files, split_recorder.SNL_val_files, split_recorder.SNL_test_files),
    ('ISU_ILCC', split_recorder.ISU_ILCC_train_files, split_recorder.ISU_ILCC_val_files, split_recorder.ISU_ILCC_test_files),
    ('CALCE', split_recorder.CALCE_train_files, split_recorder.CALCE_val_files, split_recorder.CALCE_test_files),
    ('UL_PUR', split_recorder.UL_PUR_train_files, split_recorder.UL_PUR_val_files, split_recorder.UL_PUR_test_files),
    ('CALB', split_recorder.CALB_train_files, split_recorder.CALB_val_files, split_recorder.CALB_test_files),
    ('ZN-coin', split_recorder.ZNcoin_train_files, split_recorder.ZNcoin_val_files, split_recorder.ZNcoin_test_files),
    ('NA-ion2021', split_recorder.NAion_2021_train_files, split_recorder.NAion_2021_val_files, split_recorder.NAion_2021_test_files),
]

for dataset_name, train_files, val_files, test_files in dataset_infos:
    print_dataset_overview(dataset_name, train_files, val_files, test_files, aging_conditions)

print('--------------')
batlinet_train_files = (
    split_recorder.CALCE_train_files + split_recorder.MATR_train_files +
    split_recorder.HUST_train_files + split_recorder.HNEI_train_files +
    split_recorder.RWTH_train_files + split_recorder.SNL_train_files +
    split_recorder.UL_PUR_train_files
)
batlinet_val_files = (
    split_recorder.CALCE_val_files + split_recorder.MATR_val_files +
    split_recorder.HUST_val_files + split_recorder.HNEI_val_files +
    split_recorder.RWTH_val_files + split_recorder.SNL_val_files +
    split_recorder.UL_PUR_val_files
)
batlinet_test_files = (
    split_recorder.CALCE_test_files + split_recorder.MATR_test_files +
    split_recorder.HUST_test_files + split_recorder.HNEI_test_files +
    split_recorder.RWTH_test_files + split_recorder.SNL_test_files +
    split_recorder.UL_PUR_test_files
)
print_dataset_overview(
    'BatLiNet',
    batlinet_train_files,
    batlinet_val_files,
    batlinet_test_files,
    aging_conditions,
)






