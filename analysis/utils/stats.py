import numpy as np


def check_monotonicity(arr, tolerance=0.0):
    """检查一维数组是否满足 non-decreasing；返回违反点索引与差分值。"""
    values = np.asarray(arr, dtype=float)
    if values.size < 2:
        return []

    diffs = np.diff(values)
    violation_indices = np.flatnonzero(diffs < -tolerance)
    return [(int(idx), float(diffs[idx])) for idx in violation_indices]
