import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'process_scripts'))

from preprocess_Farasis import _process_single_file

# 改这两个路径
parquet_file = Path('dataset/raw/Farasis/PA-b1/001_single_parquet_with_soc_and_corrections.parquet')
raw_root = Path('dataset/raw/Farasis')

output_dir = Path('dataset/processed/Farasis')
output_dir.mkdir(parents=True, exist_ok=True)

ok, msg = _process_single_file(str(parquet_file), str(raw_root), str(output_dir))
print('ok:', ok)
print('msg:', msg)
if ok:
    pkl = output_dir / f'{msg}.pkl'
    print('pkl:', pkl)
    print('size_bytes:', pkl.stat().st_size)