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
from sklearn.model_selection import train_test_split


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-manifest", required=True, help="CSV with at least a patient_id column")
    p.add_argument("--group-col", default="patient_id", help="Column to group by (patient id)")
    p.add_argument("--label-col", default=None, help="Optional label column for stratified patient-level splits")
    p.add_argument(
        "--patient-label-mode",
        default="any_positive",
        choices=["any_positive", "majority", "mean_threshold"],
        help="How to aggregate per-record labels to a patient-level label when --label-col is provided",
    )
    p.add_argument(
        "--mean-threshold",
        type=float,
        default=0.5,
        help="Threshold for positive class when --patient-label-mode=mean_threshold",
    )
    p.add_argument("--out-dir", default="data/splits", help="Where to write split CSVs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.12)
    p.add_argument("--time-based", action="store_true", help="If set, perform time-based holdout by last date")
    return p.parse_args()


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def to_binary_labels(series: pd.Series) -> pd.Series:
    vals = pd.to_numeric(series, errors="coerce")
    if vals.isna().any():
        raise ValueError("Found non-numeric labels in label column")
    uniq = set(vals.unique().tolist())
    if not uniq.issubset({0, 1}):
        raise ValueError(f"Label column must be binary 0/1. Got labels: {sorted(uniq)}")
    return vals.astype(int)


def aggregate_patient_labels(df: pd.DataFrame, group_col: str, label_col: str, mode: str, threshold: float) -> pd.DataFrame:
    label = to_binary_labels(df[label_col])
    tmp = df.copy()
    tmp[label_col] = label

    agg = tmp.groupby(group_col).agg(
        n_records=(group_col, "size"),
        label_sum=(label_col, "sum"),
        label_mean=(label_col, "mean"),
    ).reset_index()

    if mode == "any_positive":
        agg["patient_label"] = (agg["label_sum"] > 0).astype(int)
    elif mode == "majority":
        agg["patient_label"] = (agg["label_mean"] >= 0.5).astype(int)
    else:  # mean_threshold
        agg["patient_label"] = (agg["label_mean"] >= threshold).astype(int)

    return agg.rename(columns={group_col: "patient_id"})


def write_splits(df_patients: pd.DataFrame, out_dir: str):
    ensure_dir(out_dir)
    for split in ["train", "val", "test"]:
        path = os.path.join(out_dir, f"{split}.csv")
        df_patients[df_patients["split"] == split][["patient_id", "n_records"]].to_csv(path, index=False)
    meta = {
        "n_patients": int(df_patients.shape[0]),
        "counts": df_patients.groupby("split").size().to_dict(),
    }
    if "patient_label" in df_patients.columns:
        meta["label_distribution"] = {
            "overall": df_patients["patient_label"].value_counts(dropna=False).to_dict(),
            "by_split": {
                split: part["patient_label"].value_counts(dropna=False).to_dict()
                for split, part in df_patients.groupby("split")
            },
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

    if args.label_col is not None and args.label_col not in df.columns:
        print(f"ERROR: label column '{args.label_col}' not found in manifest columns: {df.columns.tolist()}")
        sys.exit(2)

    # Aggregate to patient-level
    if args.label_col is not None:
        try:
            pat = aggregate_patient_labels(
                df,
                group_col=args.group_col,
                label_col=args.label_col,
                mode=args.patient_label_mode,
                threshold=args.mean_threshold,
            )
        except ValueError as exc:
            print(f"ERROR: {exc}")
            sys.exit(2)
    else:
        pat = df.groupby(args.group_col).agg(n_records=(args.group_col, "size"))
        pat = pat.rename_axis("patient_id").reset_index()

    n = len(pat)
    if n < 3:
        print("ERROR: need at least 3 patients to build train/val/test splits")
        sys.exit(2)

    if args.test_size <= 0 or args.val_size <= 0 or (args.test_size + args.val_size) >= 1.0:
        print("ERROR: choose positive --test-size and --val-size with test+val < 1.0")
        sys.exit(2)

    strat = pat["patient_label"] if "patient_label" in pat.columns else None

    # split 1: train vs temp
    try:
        train_pat, temp_pat = train_test_split(
            pat,
            test_size=(args.test_size + args.val_size),
            random_state=args.seed,
            shuffle=True,
            stratify=strat,
        )
    except ValueError as exc:
        print(f"WARNING: stratified split failed ({exc}); falling back to non-stratified split")
        train_pat, temp_pat = train_test_split(
            pat,
            test_size=(args.test_size + args.val_size),
            random_state=args.seed,
            shuffle=True,
            stratify=None,
        )

    # split 2: val vs test from temp
    val_ratio_in_temp = args.val_size / (args.test_size + args.val_size)
    strat_temp = temp_pat["patient_label"] if "patient_label" in temp_pat.columns else None
    try:
        val_pat, test_pat = train_test_split(
            temp_pat,
            test_size=(1.0 - val_ratio_in_temp),
            random_state=args.seed,
            shuffle=True,
            stratify=strat_temp,
        )
    except ValueError as exc:
        print(f"WARNING: stratified val/test split failed ({exc}); falling back to non-stratified split")
        val_pat, test_pat = train_test_split(
            temp_pat,
            test_size=(1.0 - val_ratio_in_temp),
            random_state=args.seed,
            shuffle=True,
            stratify=None,
        )

    train_pat = train_pat.copy()
    val_pat = val_pat.copy()
    test_pat = test_pat.copy()
    train_pat["split"] = "train"
    val_pat["split"] = "val"
    test_pat["split"] = "test"
    pat = pd.concat([train_pat, val_pat, test_pat], ignore_index=True)

    # Basic checks
    duplicates = pat[pat.duplicated(subset=["patient_id"], keep=False)]
    if not duplicates.empty:
        print("ERROR: duplicate patient ids found after grouping; aborting")
        sys.exit(2)

    write_splits(pat, args.out_dir)
    print(f"Wrote patient-level splits to {args.out_dir}")


if __name__ == "__main__":
    main()
