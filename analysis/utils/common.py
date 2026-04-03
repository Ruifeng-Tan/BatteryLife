import numpy as np

# 电流阈值：|I|/nominal_capacity >= 0.01C 视为活跃充放电点
CURRENT_C_THRESHOLD = 0.01

# 容量段起点允许接近零的阈值
CAPACITY_START_TOL = 1e-2


def get_nominal_capacity(file_name, data):
    """获取标称容量，并保留现有分析使用的特殊覆盖逻辑。"""
    if file_name.startswith("RWTH"):
        return 1.85
    if file_name.startswith("SNL_18650_NCA_25C_20-80"):
        return 3.2

    raw_value = data.get("nominal_capacity_in_Ah", 0.0)
    if raw_value is None:
        return 0.0

    try:
        nominal_capacity = float(raw_value)
    except (TypeError, ValueError):
        return 0.0

    if not np.isfinite(nominal_capacity):
        return 0.0
    return nominal_capacity
