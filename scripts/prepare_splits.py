"""Create patient-level, leakage-safe train/val/test splits.

Writes CSVs to `--out-dir` containing one row per patient_id and assigned split.

Example:
  python scripts/prepare_splits.py \
    --input-manifest data/processed/manifest.csv \
    --group-col patient_id \
    --out-dir data/splits --seed 42 --test-size 0.20 --val-size 0.12
"""
from __future__ import annotations
import argparse
import os
import sys
import json
from typing import Optional

import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-manifest", required=True, help="CSV with at least a patient_id column")
    p.add_argument("--group-col", default="patient_id", help="Column to group by (patient id)")
    p.add_argument("--out-dir", default="data/splits", help="Where to write split CSVs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.12)
    p.add_argument("--time-based", action="store_true", help="If set, perform time-based holdout by last date")
    return p.parse_args()


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def write_splits(df_patients: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    for split in ["train", "val", "test"]:
        path = os.path.join(out_dir, f"{split}.csv")
        df_patients[df_patients["split"] == split][["patient_id", "n_records"]].to_csv(path, index=False)
    meta = {
        "n_patients": int(df_patients.shape[0]),
        "counts": df_patients.groupby("split").size().to_dict(),
    }
    with open(os.path.join(out_dir, "split_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)


def main():
    args = parse_args()
    if not os.path.exists(args.input_manifest):
        print(f"ERROR: manifest not found: {args.input_manifest}")
        sys.exit(2)

    df = pd.read_csv(args.input_manifest)
    if args.group_col not in df.columns:
        print(f"ERROR: group column '{args.group_col}' not found in manifest columns: {df.columns.tolist()}")
        sys.exit(2)

    # Aggregate to patient-level
    pat = df.groupby(args.group_col).agg(n_records=(args.group_col, "size"))
    pat = pat.rename_axis("patient_id").reset_index()

    # Simple randomized split by patient
    pat = pat.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    n = len(pat)
    n_test = int(n * args.test_size)
    n_val = int(n * args.val_size)

    pat["split"] = "train"
    if n_test > 0:
        pat.loc[: n_test - 1, "split"] = "test"
    if n_val > 0:
        start = n_test
        pat.loc[start : start + n_val - 1, "split"] = "val"

    # Basic checks
    duplicates = pat[pat.duplicated(subset=["patient_id"], keep=False)]
    if not duplicates.empty:
        print("ERROR: duplicate patient ids found after grouping; aborting")
        sys.exit(2)

    write_splits(pat, args.out_dir)
    print(f"Wrote patient-level splits to {args.out_dir}")


if __name__ == "__main__":
    main()
