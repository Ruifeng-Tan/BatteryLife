from dataclasses import dataclass, field

import numpy as np


@dataclass
class CycleSegments:
    """单个 cycle 的充放电容量段。"""

    cycle_number: int
    charge_capacity: np.ndarray
    discharge_capacity: np.ndarray
    charge_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    discharge_indices: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
