import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
font = {'family' : 'Arial'}

matplotlib.rcParams['mathtext.fontset'] = 'custom'

matplotlib.rcParams['mathtext.rm'] = 'Arial'

matplotlib.rcParams['mathtext.it'] = 'Arial'

matplotlib.rc('font', **font)
matplotlib.rcParams['pdf.fonttype'] = 42 # make the text editable for Adobe Illustrator
matplotlib.rcParams['ps.fonttype'] = 42

ZN_COIN_CHARGE_FIRST_FILE_NAMES = {
    'ZN-coin_402-1_20231209225636_01_1.pkl',
    'ZN-coin_402-2_20231209225727_01_2.pkl',
    'ZN-coin_402-3_20231209225844_01_3.pkl',
    'ZN-coin_403-1_20231209225922_01_4.pkl',
    'ZN-coin_428-1_20231212185048_01_2.pkl',
    'ZN-coin_428-2_20231212185058_01_4.pkl',
    'ZN-coin_429-1_20231212185129_01_5.pkl',
    'ZN-coin_429-2_20231212185157_01_8.pkl',
    'ZN-coin_430-1_20231212185250_02_6.pkl',
    'ZN-coin_430-2_20231212185305_02_7.pkl',
    'ZN-coin_430-3_20231212185323_03_2.pkl',
}


def is_discharge_first(file_name, prefix):
    return prefix in ['RWTH', 'CALB_0', 'CALB_25', 'CALB_45'] or (prefix == 'ZN-coin' and file_name not in ZN_COIN_CHARGE_FIRST_FILE_NAMES)


def set_ax_linewidth(ax, bw=1.5):
    ax.spines['bottom'].set_linewidth(bw)
    ax.spines['left'].set_linewidth(bw)
    ax.spines['top'].set_linewidth(bw)
    ax.spines['right'].set_linewidth(bw)


def set_ax_font_size(ax, fontsize=10):
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.tick_params(axis='x', labelsize=fontsize)


def resample_charge_discharge_curves(voltages, currents, capacity_in_battery):
    '''
    resample the charge and discharge curves based on the natural records
    :param voltages:charge or dicharge voltages
    :param currents: charge or discharge current
    :param capacity_in_battery: remaining capacities in the battery
    :return:interploted records
    '''
    charge_discharge_len = 300
    charge_discharge_len = charge_discharge_len // 2
    raw_bases = np.arange(1, len(voltages) + 1)
    interp_bases = np.linspace(1, len(voltages) + 1, num=charge_discharge_len, endpoint=True)
    interp_voltages = np.interp(interp_bases, raw_bases, voltages)
    interp_currents = np.interp(interp_bases, raw_bases, currents)
    interp_capacity_in_battery = np.interp(interp_bases, raw_bases, capacity_in_battery)
    return interp_voltages, interp_currents, interp_capacity_in_battery


def main(data_path='./dataset/MICH/MICH_BLForm2_pouch_NMC_45C_0-100_1-1C_b.pkl', is_discharge=False):
    data = pickle.load(open(data_path, 'rb'))
    cycle_data = data['cycle_data']
    nominal_capacity = data['nominal_capacity_in_Ah']
    need_keys = ['current_in_A', 'voltage_in_V', 'charge_capacity_in_Ah', 'discharge_capacity_in_Ah', 'time_in_s']
    file_name = data_path.split('/')[-1]
    prefix = file_name.split('_')[0]
    if prefix == 'CALB':
        prefix = file_name.split('_')[:2]
        prefix = '_'.join(prefix)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))
    for correct_cycle_index, sub_cycle_data in enumerate(cycle_data[:100]):
        cycle_df = pd.DataFrame()
        for key in need_keys:
            cycle_df[key] = sub_cycle_data[key]
        cycle_df['cycle_number'] = correct_cycle_index + 1
        cycle_df.loc[cycle_df['charge_capacity_in_Ah'] < 0] = np.nan
        cycle_df.bfill(inplace=True)  # deal with NaN
        voltage_records = cycle_df['voltage_in_V'].values
        current_records = cycle_df['current_in_A'].values
        current_records_in_C = current_records / nominal_capacity
        charge_capacity_records = cycle_df['charge_capacity_in_Ah'].values
        discharge_capacity_records = cycle_df['discharge_capacity_in_Ah'].values

        cutoff_voltage_indices = np.nonzero(
            current_records_in_C >= 0.01)  # This includes constant-voltage charge data, 49th cycle of MATR_b1c18 has some abnormal voltage records
        charge_end_index = cutoff_voltage_indices[0][
            -1]  # after charge_end_index, there are rest after charge, discharge, and rest after discharge data

        cutoff_voltage_indices = np.nonzero(current_records_in_C <= -0.01)
        discharge_end_index = cutoff_voltage_indices[0][-1]

        if is_discharge_first(file_name, prefix):
            discharge_voltages = voltage_records[:discharge_end_index]
            discharge_capacities = discharge_capacity_records[:discharge_end_index]
            discharge_currents = current_records[:discharge_end_index]

            charge_voltages = voltage_records[discharge_end_index:]
            charge_capacities = charge_capacity_records[discharge_end_index:]
            charge_currents = current_records[discharge_end_index:]
            charge_current_in_C = charge_currents / nominal_capacity

            charge_voltages = charge_voltages[np.abs(charge_current_in_C) > 0.01]
            charge_capacities = charge_capacities[np.abs(charge_current_in_C) > 0.01]
            charge_currents = charge_currents[np.abs(charge_current_in_C) > 0.01]
        else:
            discharge_voltages = voltage_records[charge_end_index:]
            discharge_capacities = discharge_capacity_records[charge_end_index:]
            discharge_currents = current_records[charge_end_index:]
            discharge_current_in_C = discharge_currents / nominal_capacity

            discharge_voltages = discharge_voltages[np.abs(discharge_current_in_C) > 0.01]
            discharge_capacities = discharge_capacities[np.abs(discharge_current_in_C) > 0.01]
            discharge_currents = discharge_currents[np.abs(discharge_current_in_C) > 0.01]

            charge_voltages = voltage_records[:charge_end_index]
            charge_capacities = charge_capacity_records[:charge_end_index]
            charge_currents = current_records[:charge_end_index]

        if is_discharge:
            ax1.plot(discharge_capacities, discharge_voltages, marker='o')
            discharge_voltages, discharge_currents, discharge_capacities = resample_charge_discharge_curves(
                discharge_voltages,
                discharge_currents,
                discharge_capacities,
            )
            ax2.plot(discharge_capacities, discharge_voltages, marker='x')
        else:
            ax1.plot(charge_capacities, charge_voltages, marker='o')
            charge_voltages, charge_currents, charge_capacities = resample_charge_discharge_curves(
                charge_voltages,
                charge_currents,
                charge_capacities,
            )
            ax2.plot(charge_capacities, charge_voltages, marker='x')

    ax2.set_xlabel('Normalized capacity')
    ax2.set_ylabel('Voltage (V)')
    ax1.set_xlabel('Capacity (Ah)')
    ax1.set_ylabel('Voltage (V)')
    fig.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0.3)  # Adjust the spacing between subplots
    plt.savefig('./figures/111.png')


if __name__ == '__main__':
    main()