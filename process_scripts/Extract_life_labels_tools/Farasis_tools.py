import os
import pickle

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm


FARASIS_LIFE_FILE = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..',
        '..',
        'datasets',
        'raw',
        'Farasis',
        'number_relationship.xlsx',
    )
)


def _cycle_life_label_to_int(label):
    if not np.isfinite(label):
        return np.nan
    return int(np.ceil(float(label)))


def get_farasis_cell_index_from_file_name(file_name):
    parts = os.path.splitext(file_name)[0].split('_')
    for part in reversed(parts):
        if part.isdigit():
            return int(part)
    raise ValueError(f'Cannot parse Farasis cell index from {file_name}')


def load_farasis_life_from_excel(farasis_life_file=FARASIS_LIFE_FILE):
    farasis_life_data = pd.read_excel(farasis_life_file, sheet_name=0)
    required_columns = {'cell index', 'efc life'}
    missing_columns = required_columns - set(farasis_life_data.columns)
    if missing_columns:
        raise ValueError(
            f'Farasis life file is missing columns: {sorted(missing_columns)}'
        )

    life_data = {}
    for _, row in farasis_life_data.iterrows():
        cell_index = row['cell index']
        efc_life = row['efc life']
        if pd.isna(cell_index) or pd.isna(efc_life):
            continue
        life_data[int(cell_index)] = _cycle_life_label_to_int(efc_life)
    return life_data


def extract_farasis_life_labels_from_excel(
    files,
    farasis_life_file=FARASIS_LIFE_FILE,
):
    life_data = load_farasis_life_from_excel(farasis_life_file)
    indexed_lables = {}
    abadon_count = 0
    for file_name in tqdm(files, desc='Extracting Farasis labels from life Excel'):
        if not file_name.endswith('.pkl'):
            continue
        cell_index = get_farasis_cell_index_from_file_name(file_name)
        if cell_index not in life_data:
            abadon_count += 1
            continue
        indexed_lables[cell_index] = (file_name, life_data[cell_index])
    name_lables = {
        file_name: label
        for _, (file_name, label) in sorted(indexed_lables.items())
    }
    return name_lables, abadon_count


def _build_farasis_raw_index(raw_root):
    raw_index = {}
    if not os.path.isdir(raw_root):
        return raw_index
    raw_root = os.path.abspath(raw_root)
    for dirpath, _, filenames in os.walk(raw_root):
        for filename in filenames:
            if not filename.endswith('.parquet'):
                continue
            raw_path = os.path.join(dirpath, filename)
            rel = os.path.relpath(os.path.splitext(raw_path)[0], raw_root)
            raw_index[f"Farasis_{'_'.join(rel.split(os.sep))}"] = raw_path
    return raw_index


def _read_raw_rpt_runs(raw_path, nominal_capacity):
    empty = pd.DataFrame(columns=['rpt_order', 'rpt', 'efc_throughput_Ah', 'raw_qd_Ah'])
    if raw_path is None or not os.path.exists(raw_path):
        return empty

    parquet_file = pq.ParquetFile(raw_path)
    cols = set(parquet_file.schema.names)
    if 'rpt' not in cols:
        return empty

    throughput_columns = [
        col for col in [
            'total_dischg_thrgh_ah_rounded',
            'total_chg_thrgh_ah_rounded',
        ] if col in cols
    ]
    if throughput_columns:
        columns = ['rpt'] + throughput_columns
    elif 'current (A)' in cols:
        columns = ['rpt', 'current (A)']
    else:
        return empty

    df = parquet_file.read(columns=columns, use_threads=False).to_pandas()
    rpt = pd.to_numeric(df['rpt'], errors='coerce').to_numpy(dtype=float)

    valid = np.isfinite(rpt)
    if not valid.any():
        return empty

    prev_valid = np.r_[False, valid[:-1]]
    prev_rpt = np.r_[np.nan, rpt[:-1]]
    run_start = valid & (~prev_valid | (rpt != prev_rpt))

    start_idx = np.flatnonzero(run_start)
    throughput_before_row_Ah = None
    if 'total_dischg_thrgh_ah_rounded' in df.columns:
        throughput_before_row_Ah = np.abs(pd.to_numeric(
            df['total_dischg_thrgh_ah_rounded'], errors='coerce'
        ).to_numpy(dtype=float))
    elif 'total_chg_thrgh_ah_rounded' in df.columns:
        throughput_before_row_Ah = pd.to_numeric(
            df['total_chg_thrgh_ah_rounded'], errors='coerce'
        ).to_numpy(dtype=float)

    if throughput_before_row_Ah is None:
        current = pd.to_numeric(df['current (A)'], errors='coerce').to_numpy(dtype=float)
        discharge_current = np.nan_to_num(np.maximum(-current, 0.0), nan=0.0)
        abs_throughput_increment_Ah = discharge_current / 3600.0
        throughput_before_row_Ah = (
            np.cumsum(abs_throughput_increment_Ah) - abs_throughput_increment_Ah
        )

    runs = pd.DataFrame({
        'rpt_order': np.arange(len(start_idx), dtype=int),
        'rpt': rpt[start_idx],
        'efc_throughput_Ah': throughput_before_row_Ah[start_idx],
        'raw_qd_Ah': np.nan,
    })
    return runs[['rpt_order', 'rpt', 'efc_throughput_Ah', 'raw_qd_Ah']].reset_index(drop=True)


def _collect_farasis_rpt_records(cycle_data, get_cycle_discharge_capacity_for_soh):
    rpt_records = []
    for rpt_order, cycle in enumerate(cycle for cycle in cycle_data if str(cycle['attribute']) == 'RPT'):
        qd = get_cycle_discharge_capacity_for_soh(cycle)
        if np.isfinite(qd) and qd > 0:
            rpt_records.append({'rpt_order': int(rpt_order), 'pkl_qd_Ah': float(qd)})
    if len(rpt_records) == 0:
        return pd.DataFrame(columns=['rpt_order', 'pkl_qd_Ah', 'soh'])

    rpt_df = pd.DataFrame(rpt_records)
    initial_capacity = float(rpt_df['pkl_qd_Ah'].iloc[0])
    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        return pd.DataFrame(columns=['rpt_order', 'pkl_qd_Ah', 'soh'])
    rpt_df['soh'] = rpt_df['pkl_qd_Ah'] / initial_capacity
    return rpt_df


def _collect_farasis_demo_reference_curve(cycle_data, raw_rpt_runs, get_cycle_discharge_capacity_for_soh):
    rpt_df = _collect_farasis_rpt_records(cycle_data, get_cycle_discharge_capacity_for_soh)
    if raw_rpt_runs is None or raw_rpt_runs.empty or rpt_df.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    raw = raw_rpt_runs.replace([np.inf, -np.inf], np.nan).dropna(
        subset=['rpt_order', 'efc_throughput_Ah']
    ).sort_values('rpt_order').reset_index(drop=True)
    pkl = rpt_df.sort_values('rpt_order').reset_index(drop=True)
    if raw.empty or pkl.empty:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    n = min(len(raw), len(pkl))
    raw = raw.iloc[:n].reset_index(drop=True)
    pkl = pkl.iloc[:n].reset_index(drop=True)
    initial_capacity = float(pkl['pkl_qd_Ah'].iloc[0])
    if not np.isfinite(initial_capacity) or initial_capacity <= 0:
        return np.asarray([], dtype=float), np.asarray([], dtype=float)

    efc = (
        raw['efc_throughput_Ah'].to_numpy(dtype=float)
        - float(raw['efc_throughput_Ah'].iloc[0])
    ) / initial_capacity
    soh = pkl['pkl_qd_Ah'].to_numpy(dtype=float) / initial_capacity
    return efc, soh


def _linear_interpolate_x_at_y(x_values, y_values, target_y):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) == 0:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if y[0] <= target_y:
        return float(x[0])

    for i in range(1, len(x)):
        y0, y1 = y[i - 1], y[i]
        if y0 > target_y and y1 <= target_y:
            x0, x1 = x[i - 1], x[i]
            if abs(y1 - y0) < 1e-12:
                return float(x1)
            ratio = (target_y - y0) / (y1 - y0)
            return float(x0 + ratio * (x1 - x0))
    return np.nan


def _last_two_linear_extrapolate_x_at_y(x_values, y_values, target_y):
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if len(x) < 2:
        return np.nan

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if y[-1] <= target_y:
        return np.nan
    x0, x1 = x[-2], x[-1]
    y0, y1 = y[-2], y[-1]
    if abs(y1 - y0) < 1e-12:
        return np.nan
    x_target = x0 + (target_y - y0) / (y1 - y0) * (x1 - x0)
    if np.isfinite(x_target) and x_target >= x[-1]:
        return float(x_target)
    return np.nan


def _author_linear_life_x_at_y(x_values, y_values, target_y):
    life = _linear_interpolate_x_at_y(x_values, y_values, target_y)
    if np.isfinite(life):
        return life
    life = _last_two_linear_extrapolate_x_at_y(x_values, y_values, target_y)
    if np.isfinite(life):
        return life
    return np.nan


def extract_farasis_life_labels(
    files,
    dataset_path,
    dataset_root_path,
    target_soh,
    get_cycle_discharge_capacity_for_soh,
    cycle_life_label_to_int,
):
    labels = {}
    excluded = 0
    raw_root = os.path.normpath(os.path.join(dataset_root_path, '..', 'raw', 'Farasis'))
    raw_index = _build_farasis_raw_index(raw_root)
    pkl_files = sorted(file_name for file_name in files if file_name.endswith('.pkl'))

    for file_name in tqdm(pkl_files, desc='Extracting Farasis 90% SOH EFC labels'):
        data = pickle.load(open(os.path.join(dataset_path, file_name), 'rb'))
        cycle_data = data['cycle_data']
        nominal_capacity = float(data['nominal_capacity_in_Ah'])

        raw_path = raw_index.get(os.path.splitext(file_name)[0])
        raw_rpt_runs = _read_raw_rpt_runs(raw_path, nominal_capacity)
        efc_values, soh_values = _collect_farasis_demo_reference_curve(
            cycle_data,
            raw_rpt_runs,
            get_cycle_discharge_capacity_for_soh,
        )
        eol_efc = _author_linear_life_x_at_y(
            efc_values, soh_values, target_soh
        )
        eol_efc_label = cycle_life_label_to_int(eol_efc)
        if np.isfinite(eol_efc):
            labels[file_name] = eol_efc_label
            print(file_name, eol_efc_label)
        else:
            excluded += 1

    return labels, excluded
