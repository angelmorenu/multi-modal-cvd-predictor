#!/usr/bin/env python3
"""Evaluate a trained CVD model on a prepared external ECG dataset.

This script expects the standardized outputs from
`scripts/prepare_external_ecg.py`:

- `external_ecg_<dataset>_signals.npy`
- `external_ecg_<dataset>_labels.npy`

Optionally, a tabular feature matrix can be supplied to evaluate the fusion
checkpoint with real tabular inputs. If no tabular features are given, the
script evaluates the ECG pathway with zero tabular inputs, which is useful for
external ECG-only validation.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, brier_score_loss, confusion_matrix, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model import MultiModalCVD, load_checkpoint


def load_external_arrays(signals_path: Path, labels_path: Path):
    signals = np.load(signals_path).astype("float32")
    labels = np.load(labels_path).astype("int64")
    if signals.ndim == 2:
        signals = signals[:, None, :]
    elif signals.ndim != 3:
        raise SystemExit(f"Expected signals with shape (N, C, T) or (N, T), got {signals.shape}")
    return signals, labels


def load_optional_tabular(tabular_path: str | None, n_samples: int, tab_dim: int | None):
    if tabular_path:
        tab = np.load(tabular_path).astype("float32")
        if tab.ndim == 1:
            tab = tab[None, :]
        if tab.shape[0] != n_samples:
            raise SystemExit(f"Tabular array has {tab.shape[0]} rows but ECG has {n_samples} rows")
        return tab, tab.shape[1]

    if tab_dim is None:
        raise SystemExit("tab_dim could not be inferred; provide --tab-dim or --tabular-features")

    return np.zeros((n_samples, tab_dim), dtype="float32"), tab_dim


def infer_tab_dim(proc_dir: Path | None) -> int | None:
    if proc_dir is None:
        return None
    candidate = proc_dir / "tabular_test_X.npy"
    if candidate.exists():
        return int(np.load(candidate).shape[1])
    return None


def compute_metrics(y_true, probs):
    preds = (probs >= 0.5).astype("int64")
    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "pr_auc": float(average_precision_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y_true, probs)),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "n_samples": int(len(y_true)),
        "n_positive": int(np.sum(y_true == 1)),
        "n_negative": int(np.sum(y_true == 0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate a model on an external ECG dataset")
    parser.add_argument("--signals", required=True, help="Path to external_ecg_<dataset>_signals.npy")
    parser.add_argument("--labels", required=True, help="Path to external_ecg_<dataset>_labels.npy")
    parser.add_argument("--artifacts-dir", default="artifacts", help="Directory containing model checkpoint(s)")
    parser.add_argument("--checkpoint", default="model.pt", help="Checkpoint filename within artifacts-dir")
    parser.add_argument("--proc-dir", default="data/processed", help="Processed dir used to infer tabular dimensionality")
    parser.add_argument("--tabular-features", default=None, help="Optional numpy array of tabular features aligned to ECG samples")
    parser.add_argument("--tab-dim", type=int, default=None, help="Override tabular feature dimensionality")
    parser.add_argument("--ecg-len", type=int, default=2000, help="Pad/crop ECGs to this length")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument("--out-dir", default="results/external", help="Where to save metrics and predictions")
    args = parser.parse_args()

    signals_path = Path(args.signals)
    labels_path = Path(args.labels)
    art_dir = Path(args.artifacts_dir)
    proc_dir = Path(args.proc_dir) if args.proc_dir else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals, labels = load_external_arrays(signals_path, labels_path)
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        raise SystemExit("No valid labels found in external label array (expected 0/1 labels)")
    if not np.all(valid_mask):
        signals = signals[valid_mask]
        labels = labels[valid_mask]

    # pad/crop to the requested ECG length
    if signals.shape[-1] < args.ecg_len:
        pad = np.zeros((signals.shape[0], signals.shape[1], args.ecg_len - signals.shape[-1]), dtype=np.float32)
        signals = np.concatenate([signals, pad], axis=-1)
    else:
        signals = signals[..., : args.ecg_len]

    inferred_tab_dim = args.tab_dim if args.tab_dim is not None else infer_tab_dim(proc_dir)
    tab_x, tab_dim = load_optional_tabular(args.tabular_features, signals.shape[0], inferred_tab_dim)

    device = torch.device(args.device if args.device in {"cpu", "cuda", "mps"} else "cpu")
    model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=signals.shape[1], ecg_embed_dim=128, n_classes=2)

    ckpt_path = art_dir / args.checkpoint
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    load_checkpoint(model, str(ckpt_path), map_location=device)
    model.to(device)
    model.eval()

    probs = np.zeros(signals.shape[0], dtype=np.float32)
    batch_size = 64
    with torch.inference_mode():
        for i in range(0, signals.shape[0], batch_size):
            j = min(i + batch_size, signals.shape[0])
            tab_b = torch.from_numpy(tab_x[i:j]).float().to(device)
            ecg_b = torch.from_numpy(signals[i:j]).float().to(device)
            logits = model(tab_b, ecg_b)
            probs[i:j] = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]

    metrics = compute_metrics(labels, probs)
    meta = {
        "timestamp": time.time(),
        "signals": str(signals_path),
        "labels": str(labels_path),
        "checkpoint": str(ckpt_path),
        "device": str(device),
        "tab_dim": int(tab_dim),
        "ecg_len": int(args.ecg_len),
        "metrics": metrics,
    }

    np.save(out_dir / "y_true.npy", labels)
    np.save(out_dir / "y_prob.npy", probs)
    np.save(out_dir / "y_pred.npy", (probs >= 0.5).astype("int64"))
    with open(out_dir / "external_eval_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
