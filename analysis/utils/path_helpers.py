import glob
import os
from pathlib import Path


def get_dataset_name(pkl_path):
    """从 pkl 路径提取数据集目录名。"""
    return os.path.basename(os.path.dirname(pkl_path))


def find_pkl_files(data_path, datasets=None):
    """递归查找数据集目录下的 pkl 文件，忽略 label/seen 路径。"""
    all_pkls = glob.glob(os.path.join(data_path, "**", "*.pkl"), recursive=True)
    all_pkls = [
        pkl_path
        for pkl_path in all_pkls
        if "label" not in pkl_path.lower() and "seen" not in pkl_path.lower()
    ]

    if datasets:
        allowed = set(datasets)
        all_pkls = [
            pkl_path
            for pkl_path in all_pkls
            if get_dataset_name(pkl_path) in allowed
        ]

    return sorted(all_pkls)


def get_cell_id(data, pkl_path):
    """优先从数据字典读取 cell_id，缺失时回退到文件名。"""
    cell_id = data.get("cell_id")
    if cell_id is None:
        return Path(pkl_path).stem
    cell_id = str(cell_id).strip()
    return cell_id or Path(pkl_path).stem
