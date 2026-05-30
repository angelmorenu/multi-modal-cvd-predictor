"""Scaffold an architecture sweep for ECG encoders.

This script produces job directories for each candidate architecture and writes a
`run.sh` that contains the command to train that architecture. The script does not
run heavy training by default; it prepares reproducible job specs.

Example:
  python experiments/arch_sweep.py --out results/arch_sweep --archs resnet1d,inceptiontime --trials 20
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/arch_sweep", help="Output folder for job specs")
    p.add_argument("--archs", default="resnet1d,inceptiontime,xresnet,transformer", help="Comma-separated arch ids")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def main():
    args = parse_args()
    ensure_dir(args.out)
    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    for arch in archs:
        jobdir = Path(args.out) / arch
        ensure_dir(jobdir)
        spec = {
            "arch": arch,
            "trials": args.trials,
            "seed": args.seed,
            "notes": "Adjust hyperparameters per-arch. Replace TRAIN_CMD with actual train command."
        }
        (jobdir / "spec.json").write_text(json.dumps(spec, indent=2))
        # write a run script placeholder
        run_sh = jobdir / "run.sh"
        train_cmd = (
            "echo 'Train {arch} (trials={trials})' && "
            "# Replace with: python src/train.py --arch {arch} --trials {trials} --out-dir {out}'"
        ).format(arch=arch, trials=args.trials, out=jobdir)
        run_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + train_cmd + "\n")
        os.chmod(run_sh, 0o755)
    print(f"Wrote arch sweep job specs for: {', '.join(archs)} to {args.out}")


if __name__ == "__main__":
    main()
