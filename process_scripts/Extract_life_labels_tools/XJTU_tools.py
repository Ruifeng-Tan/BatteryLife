import os
import pickle

import numpy as np
from sklearn.linear_model import LinearRegression
from tqdm import tqdm


def get_soc_span(data):
    soc_interval = data['SOC_interval']
    soc_span = abs(float(soc_interval[1]) - float(soc_interval[0]))
    if np.isfinite(soc_span) and soc_span > 1e-12:
        return soc_span
    return 1.0


def compute_cycle_soh(cycle, nominal_capacity, soc_span):
    current = np.asarray(cycle['current_in_A'], dtype=float)
    discharge = np.asarray(cycle['discharge_capacity_in_Ah'], dtype=float)
    n = min(len(current), len(discharge))
    if n == 0 or not np.isfinite(nominal_capacity) or nominal_capacity <= 0:
        return np.nan

    current = current[:n]
    discharge = discharge[:n]
    mask = np.isfinite(current) & np.isfinite(discharge)
    if not np.any(mask):
        return np.nan

    discharge_mask = mask & (current < 0)
    values = discharge[discharge_mask] if np.any(discharge_mask) else discharge[mask]
    qd = float(np.nanmax(values))
    return qd / float(nominal_capacity) / float(soc_span)


def collect_cycle_soh_curve(cycle_data, nominal_capacity, soc_span):
    cycle_numbers = []
    soh_values = []
    for i, cycle in enumerate(cycle_data):
        cycle_number = cycle['cycle_number']
        try:
            cycle_number = int(cycle_number)
        except Exception:
            cycle_number = i + 1
        soh = compute_cycle_soh(cycle, nominal_capacity, soc_span)
        cycle_numbers.append(cycle_number)
        soh_values.append(soh)
    return np.asarray(cycle_numbers, dtype=float), np.asarray(soh_values, dtype=float)


def cycle_life_label_to_int(label):
    if not np.isfinite(label):
        return np.nan
    return int(np.ceil(float(label)))


def _linear_interpolate_last_descending_x_at_y(x_values, y_values, target_y):
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

    crossings = []
    for i in range(1, len(x)):
        y0, y1 = y[i - 1], y[i]
        if y0 > target_y and y1 <= target_y:
            x0, x1 = x[i - 1], x[i]
            if abs(y1 - y0) < 1e-12:
                crossings.append(float(x1))
            else:
                ratio = (target_y - y0) / (y1 - y0)
                crossings.append(float(x0 + ratio * (x1 - x0)))

    if len(crossings) > 0:
        return max(crossings)
    if y[0] <= target_y:
        return float(x[0])
    return np.nan


def _linear_extrapolate_x_at_y_from_tail(x_values, y_values, target_y, regress_cycle_num=20):
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
    n = min(regress_cycle_num, len(x))
    x_tail = x[-n:]
    y_tail = y[-n:].reshape(-1, 1)
    try:
        linear_regressor = LinearRegression()
        linear_regressor.fit(y_tail, x_tail)
        return int(linear_regressor.predict(np.array([target_y]).reshape(-1, 1))[0])
    except Exception:
        return np.nan


def extract_xjtu_life_labels(
    files,
    dataset_path,
):
    labels_80 = {}

    for file_name in tqdm(sorted(files), desc='Extracting XJTU 80% SOH cycle labels'):
        if not file_name.endswith('.pkl'):
            continue

        data = pickle.load(open(os.path.join(dataset_path, file_name), 'rb'))
        cycle_data = data['cycle_data']
        nominal_capacity = float(data['nominal_capacity_in_Ah'])
        soc_span = get_soc_span(data)
        cycle_numbers, soh_values = collect_cycle_soh_curve(cycle_data, nominal_capacity, soc_span)

        life_80 = _linear_interpolate_last_descending_x_at_y(cycle_numbers, soh_values, 0.80)
        if not np.isfinite(life_80):
            finite_soh = soh_values[np.isfinite(soh_values)]
            last_soh = finite_soh[-1] if len(finite_soh) else np.nan
            if np.isfinite(last_soh) and 0.8 < last_soh < 0.825:
                life_80 = _linear_extrapolate_x_at_y_from_tail(cycle_numbers, soh_values, 0.80)

        labels_80[file_name] = cycle_life_label_to_int(life_80)

    return labels_80
