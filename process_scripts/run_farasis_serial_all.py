from pathlib import Path
import sys
from datetime import datetime

project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "process_scripts"))

from preprocess_Farasis import _process_single_file


def main():
    raw_root = project_root / "dataset" / "raw" / "Farasis"
    output_dir = project_root / "dataset" / "processed" / "Farasis"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(raw_root.rglob("*.parquet"))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = project_root / f"run_farasis_serial_report_{ts}.txt"

    ok_count = 0
    fail_count = 0
    lines = []
    lines.append(f"start={datetime.now().isoformat()}")
    lines.append(f"raw_root={raw_root}")
    lines.append(f"output_dir={output_dir}")
    lines.append(f"total_files={len(parquet_files)}")
    lines.append("")

    for i, parquet_file in enumerate(parquet_files, start=1):
        rel_no_suffix = parquet_file.relative_to(raw_root).with_suffix("")
        cell_name = f"Farasis_{'_'.join(rel_no_suffix.parts)}"
        target_pkl = output_dir / f"{cell_name}.pkl"

        if target_pkl.exists():
            lines.append(f"[{i}/{len(parquet_files)}] SKIP {parquet_file.relative_to(raw_root)} -> existing {target_pkl.name}")
            print(f"[{i}/{len(parquet_files)}] SKIP {parquet_file.relative_to(raw_root)}")
            continue

        ok, msg = _process_single_file(str(parquet_file), str(raw_root), str(output_dir))
        rel = parquet_file.relative_to(raw_root)
        if ok:
            ok_count += 1
            lines.append(f"[{i}/{len(parquet_files)}] OK   {rel} -> {msg}.pkl")
            print(f"[{i}/{len(parquet_files)}] OK   {rel}")
        else:
            fail_count += 1
            lines.append(f"[{i}/{len(parquet_files)}] FAIL {rel} -> {msg}")
            print(f"[{i}/{len(parquet_files)}] FAIL {rel} -> {msg}")

    lines.append("")
    lines.append(f"end={datetime.now().isoformat()}")
    lines.append(f"ok_count={ok_count}")
    lines.append(f"fail_count={fail_count}")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"report={report_path}")
    print(f"ok_count={ok_count}")
    print(f"fail_count={fail_count}")


if __name__ == "__main__":
    main()
