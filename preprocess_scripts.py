import subprocess
import os
import re
import shutil
import zipfile
import numpy as np
import pandas as pd

from tqdm import tqdm
from numba import njit
from typing import List
from pathlib import Path
from scipy.signal import medfilt

from batteryml import BatteryData, CycleData
from batteryml.builders import PREPROCESSORS
from batteryml.preprocess.base import BasePreprocessor

preprocess_commands = [
    'batteryml preprocess CALCE ./datasets/raw/CALCE ./datasets/processed/CALCE/',
    'batteryml preprocess HNEI ./datasets/raw/HNEI ./datasets/processed/HNEI/',
    'batteryml preprocess HUST ./datasets/raw/HUST ./datasets/processed/HUST/',
    'batteryml preprocess MATR ./datasets/raw/MATR ./datasets/processed/MATR/',
    'batteryml preprocess MICH ./datasets/raw/MICH ./datasets/processed/MICH/',
    'batteryml preprocess MICH_EXP ./datasets/raw/MICH_EXP ./datasets/processed/MICH_EXP/',
    'batteryml preprocess RWTH ./datasets/raw/RWTH ./datasets/processed/RWTH/',
    'batteryml preprocess SNL ./datasets/raw/SNL ./datasets/processed/SNL/',
    'batteryml preprocess Stanford ./datasets/raw/Stanford ./datasets/processed/Stanford/',
    'batteryml preprocess UL_PUR ./datasets/raw/UL_PUR ./datasets/processed/UL_PUR/',
    'batteryml preprocess XJTU ./datasets/raw/XJTU ./datasets/processed/XJTU/',
    'batteryml preprocess ISU_ILCC ./datasets/raw/ISU_ILCC ./datasets/processed/ISU_ILCC/',
    'batteryml preprocess Tongji ./datasets/raw/Tongji ./datasets/processed/Tongji/',
    'batteryml preprocess CALB ./datasets/raw/CALB ./datasets/processed/CALB/',
    'batteryml preprocess NA ./datasets/raw/NAion ./datasets/processed/NA-ion/',
    'batteryml preprocess ZNion ./datasets/raw/ZNion ./datasets/processed/ZN-ion/',
    # use the same command for the following datasets, e.g., SDU et al.
]

for command in preprocess_commands:
    subprocess.run(command, shell=True)
