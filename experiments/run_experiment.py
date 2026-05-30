"""Minimal experiment runner.

This script is intentionally lightweight. By default run with `--dry-run`
to validate shapes and config without importing heavy dependencies.
"""
import argparse
import os
import numpy as np
import sys


def load_config(path: str):
    try:
        import yaml
    except Exception:
        print(f"[WARN] PyYAML not available; skipping config load for {path}")
        return None
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def dry_run_checks(processed_dir: str, ecg_key: str, label_key: str):
    ecg_path = os.path.join(processed_dir, ecg_key)
    label_path = os.path.join(processed_dir, label_key)
    if not os.path.exists(ecg_path):
        print(f"[ERROR] ECG file not found: {ecg_path}")
        return 2
    if not os.path.exists(label_path):
        print(f"[ERROR] label file not found: {label_path}")
        return 3

    X = np.load(ecg_path)
    y = np.load(label_path)
    print(f"Loaded ECG: {ecg_path} -> shape {X.shape}")
    print(f"Loaded labels: {label_path} -> shape {y.shape}")

    # basic sanity checks
    if X.shape[0] != y.shape[0]:
        print(f"[WARN] sample count mismatch: X={X.shape[0]} y={y.shape[0]}")
    # show a sample slice
    sample = X[0]
    print(f"Sample dtype: {sample.dtype}, shape: {sample.shape}")
    return 0


def smoke_model_forward(in_channels: int, num_classes: int, seq_len: int):
    """Build model using the experiments.models.resnet1d builder and run a single forward pass."""
    try:
        from experiments.models.resnet1d import build_resnet1d
    except Exception as e:
        print(f"[ERROR] Could not import model builder: {e}")
        return 4
    try:
        import torch
    except Exception as e:
        print(f"[ERROR] torch not available: {e}")
        return 5

    model = build_resnet1d(in_channels=in_channels, num_classes=num_classes)
    model.eval()
    x = torch.randn(2, in_channels, seq_len)
    with torch.no_grad():
        out = model(x)
    print(f"Model forward OK -> output shape {out.shape}")
    return 0


def main():
    p = argparse.ArgumentParser(description="Experiment runner (dry-run first)")
    p.add_argument("--config", type=str, help="path to YAML config", default=None)
    p.add_argument("--processed-dir", type=str, default="data/processed")
    p.add_argument("--ecg-key", type=str, default="ecg_train.npy")
    p.add_argument("--label-key", type=str, default="tabular_train_y.npy")
    p.add_argument("--dry-run", action="store_true", default=True, help="Only run shape checks and model smoke test")
    p.add_argument("--in-channels", type=int, default=1)
    p.add_argument("--num-classes", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=2000)
    args = p.parse_args()

    cfg = None
    if args.config:
        cfg = load_config(args.config)
        if cfg is not None:
            print(f"Loaded config from {args.config}")

    if args.dry_run:
        rc = dry_run_checks(args.processed_dir, args.ecg_key, args.label_key)
        if rc != 0:
            sys.exit(rc)
        rc2 = smoke_model_forward(args.in_channels, args.num_classes, args.seq_len)
        sys.exit(rc2)

    print("Non-dry runs are not yet implemented. Use --dry-run for now.")


if __name__ == '__main__':
    main()
