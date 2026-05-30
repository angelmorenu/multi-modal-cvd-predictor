"""Run repeated group K-fold split generation and produce per-fold manifest files.

This script creates train/val patient lists for repeated group K-fold CV and writes a small
driver shell script (`run_jobs.sh`) with placeholders to run training per fold.

Usage (dry-run):
  python scripts/run_nested_cv.py --split-csv data/splits/train.csv --out results/repro --folds 5 --repeats 3

To actually run training (if your `src/train.py` supports a `--patients-file` arg):
  python scripts/run_nested_cv.py ... --run --train-cmd "python src/train.py --patients-file {patients_json} --out-dir {out_dir}"
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--split-csv", required=True, help="CSV of patient-level splits (train/val/test) or patients manifest")
    p.add_argument("--out", default="results/repro", help="Output folder to write fold manifests and run script")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run", action="store_true", help="If set, attempt to run training commands")
    default_cmd = (
        "echo 'DRYRUN: train with patients={patients_json} out={out_dir}'"
    )
    p.add_argument("--train-cmd", default=default_cmd, help="Command template to run training; use {patients_json} and {out_dir} placeholders")
    p.add_argument("--use-train-script", action="store_true", help="If set, override train-cmd and call src/train.py with conservative defaults")
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce", help="Loss function to use")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for focal loss")
    p.add_argument("--augment_ecg", action="store_true", help="Enable ECG data augmentation")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight decay")
    return p.parse_args()

# Helper function to ensure output directory exists
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

# Main function to load patient splits, generate repeated group K-fold splits, write fold manifests, and create a run script
def main():
    args = parse_args()
    ensure_dir(args.out)
    if not os.path.exists(args.split_csv):
        print(f"ERROR: split csv not found: {args.split_csv}")
        sys.exit(2)

    df = pd.read_csv(args.split_csv)
    if "patient_id" not in df.columns:
        print("ERROR: expected 'patient_id' column in split CSV")
        sys.exit(2)

    # Use label if present for stratification; otherwise create a pseudo-label that balances folds
    label_col = "label" if "label" in df.columns else None
    patients = df["patient_id"].tolist()

    # Build a simple index-based repeated k-fold for patients
    rkf = RepeatedStratifiedKFold(n_splits=args.folds, n_repeats=args.repeats, random_state=args.seed)

    y = df[label_col].fillna(0).astype(int).values if label_col else [0] * len(df)

    jobs: List[str] = []
    fold_index = 0
    for train_idx, val_idx in rkf.split(patients, y):
        fold_index += 1
        train_pats = [patients[i] for i in train_idx]
        val_pats = [patients[i] for i in val_idx]
        fold_dir = Path(args.out) / f"fold_{fold_index:03d}"
        ensure_dir(fold_dir)
        (fold_dir / "train_patients.json").write_text(json.dumps(train_pats))
        (fold_dir / "val_patients.json").write_text(json.dumps(val_pats))
        jobs.append(str(fold_dir))

    # Create a simple run script with placeholders
    run_sh = Path(args.out) / "run_jobs.sh"
    with run_sh.open("w") as fh:
        fh.write("#!/usr/bin/env bash\nset -euo pipefail\n\n")
        for d in jobs:
            patients_json = os.path.join(d, "train_patients.json")
            out_dir = os.path.join(d, "artifacts")
            # If user asked to use the real train script, override the template with conservative defaults
            if args.use_train_script:
                cmd_template = (
                    "python src/train.py --processed_dir data/processed --artifacts_dir {out_dir} --epochs 1 --batch_size 8 --seed {seed}"
                )
                cmd = cmd_template.format(patients_json=patients_json, out_dir=out_dir, seed=args.seed)
            elif args.train_cmd:
                cmd = args.train_cmd.format(patients_json=patients_json, out_dir=out_dir)
            else:
                cmd = f"echo 'Run training with patients file={patients_json} and out_dir={out_dir}'"
            fh.write(cmd + "\n")
    os.chmod(run_sh, 0o755)
    print(f"Wrote {len(jobs)} fold manifests and run script to {args.out}")

    if args.run:
        # run the jobs sequentially using the same command logic as when writing the run script
        import subprocess

        for d in jobs:
            patients_json = os.path.join(d, "train_patients.json")
            out_dir = os.path.join(d, "artifacts")
            if args.use_train_script:
                cmd_template = (
                    "python -m src.train "
                    "--processed_dir data/processed "
                    "--artifacts_dir {out_dir} "
                    "--epochs 15 --batch_size 8 --seed {seed} "
                    "--patients-file {patients_json} "
                    "--loss {loss} "
                    "--focal_gamma {focal_gamma} "
                    "--weight_decay {weight_decay} "
                    "--dropout {dropout}"
                )
                if args.augment_ecg:
                    cmd_template += " --augment_ecg"
                cmd = cmd_template.format(
                    patients_json=patients_json,
                    out_dir=out_dir,
                    seed=args.seed,
                    loss=args.loss,
                    focal_gamma=args.focal_gamma,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                )
            elif args.train_cmd:
                cmd = args.train_cmd.format(patients_json=patients_json, out_dir=out_dir)
            else:
                cmd = f"echo 'Run training with patients file={patients_json} and out_dir={out_dir}'"

            print(f"Running: {cmd}")
            subprocess.check_call(cmd, shell=True)


if __name__ == "__main__":
    main()
