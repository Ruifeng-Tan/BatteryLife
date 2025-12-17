from tqdm import tqdm
import pandas as pd
import argparse


def main(raw_data_file_path, output_path):
    sheets = ['0℃循环', '25℃ 循环', '35℃ 循环', '45℃循环']

    for sheet in sheets:
        cells_data = pd.read_excel(raw_data_file_path, sheet_name=sheet)
        columns_name = cells_data.columns.tolist()
        if sheet == '0℃循环':
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('A1')]
            cells_name = ['CALB_0_' + i.replace('A', 'B') for i in columns_name if i.startswith('A1')]
        elif sheet == '25℃ 循环':
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('T25')]
            cells_name = ['CALB_25_' + i for i in columns_name if i.startswith('T25')]
        else:
            start_column_idx = [columns_name.index(i) for i in columns_name if i.startswith('B')]
            if sheet == '35℃ 循环':
                cells_name = ['CALB_35_' + i for i in columns_name if i.startswith('B')]
            elif sheet == '45℃循环':
                cells_name = ['CALB_45_' + i for i in columns_name if i.startswith('B')]

        cycles_column_idx = [i + 1 for i in start_column_idx]
        times_column_idx = [i + 2 for i in start_column_idx]
        discharge_column_idx = [i + 4 for i in start_column_idx]

        for cell_name, cycle_idx, discharge_idx, time_idx in zip(tqdm(cells_name, desc='Extracting capacity data'),
                                                                 cycles_column_idx, discharge_column_idx, times_column_idx):
            if cell_name.startswith('CALB_45_B254'):
                continue

            cycles = cells_data.iloc[:, cycle_idx].tolist()
            discharge = cells_data.iloc[:, discharge_idx].tolist()
            times = cells_data.iloc[:, time_idx].values.tolist()

            cycle_df = pd.DataFrame()
            cycle_df['cycle_number'] = cycles
            cycle_df['time_in_s'] = times
            cycle_df['discharge_capacity_in_Ah'] = discharge

            # refine the cycle_df
            cycle_df = cycle_df.dropna()
            cycle_number_list = cycle_df['cycle_number'].values.tolist()
            new_cycle_list = [i for i in range(1, len(cycle_number_list) + 1)]
            cycle_df['cycle_number'] = new_cycle_list

            cycle_df.to_csv(f'./{output_path}/{cell_name}.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generate discharge capacities from cleaned battery data'
    )
    parser.add_argument(
        '--raw_data_file_path',
        type=str,
        default='/data/trf/python_works/BatteryLife/dataset/',
        help='Root directory of cleaned_data'
    )
    parser.add_argument(
        '--output_path',
        type=str,
        default='/data/trf/python_works/BatteryLife/dataset/processed/',
        help='Root directory for output (processed_SOH)'
    )
    args = parser.parse_args()

    raw_data_file_path = args.raw_data_file_path
    output_path = args.output_path

    main(raw_data_file_path, output_path)