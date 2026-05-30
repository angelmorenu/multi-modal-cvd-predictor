#!/usr/bin/env python3
# src/eval.py (improved)
import argparse
import json
import logging
import os
from pathlib import Path
import time

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    brier_score_loss, confusion_matrix
)

# keep model import
from src.model import MultiModalCVD, load_checkpoint

DEFAULT_ECG_LEN = 2000

def load_tab(proc_dir: Path, split: str):
    X = np.load(proc_dir / f"tabular_{split}_X.npy")
    y = np.load(proc_dir / f"tabular_{split}_y.npy").astype("int64")
    if X.ndim == 1:
        X = X[None, :]
    return X.astype("float32"), y

def load_ecg(proc_dir: Path, split: str, ecg_len: int):
    p = proc_dir / f"ecg_{split}.npy"
    if not p.exists():
        return None
    E = np.load(p).astype("float32")
    if E.ndim == 1:
        E = E[None, :]
    # pad/crop
    if E.shape[1] < ecg_len:
        pad = np.zeros((E.shape[0], ecg_len - E.shape[1]), dtype="float32")
        E = np.concatenate([E, pad], axis=1)
    else:
        E = E[:, :ecg_len]
    return E

def batch_inference(model, tab_np, ecg_np, device, batch_size=64):
    model.to(device)
    model.eval()
    n = tab_np.shape[0]
    probs = np.zeros(n, dtype="float32")
    with torch.inference_mode():
        for i in range(0, n, batch_size):
            j = min(i + batch_size, n)
            tab_b = torch.from_numpy(tab_np[i:j]).float().to(device)
            ecg_b = torch.from_numpy(ecg_np[i:j])[:, None, :].float().to(device)
            logits = model(tab_b, ecg_b)
            batch_probs = torch.softmax(logits, dim=-1).cpu().numpy()[:, 1]
            probs[i:j] = batch_probs
    return probs

def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype("int64")
    acc = accuracy_score(y_true, preds)
    try:
        roc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    except Exception:
        roc = float("nan")
    pr = average_precision_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    cm = confusion_matrix(y_true, preds)
    return dict(accuracy=acc, roc_auc=roc, pr_auc=pr, brier=brier, confusion_matrix=cm, preds=preds)


def run_failure_checks(y_true, probs, preds, threshold):
    checks = {
        "n_samples": len(y_true),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(preds)),
        "prob_min": float(np.min(probs)),
        "prob_max": float(np.max(probs)),
        "prob_std": float(np.std(probs)),
        "threshold": float(threshold),
        "warnings": [],
        "is_degenerate": False,
    }

    all_positive = bool(np.all(preds == 1))
    all_negative = bool(np.all(preds == 0))
    all_probs_above = bool(np.all(probs >= threshold))
    all_probs_below = bool(np.all(probs < threshold))
    almost_constant_probs = np.std(probs) < 1e-3

    if all_positive:
        checks["warnings"].append("All predictions are positive (class=1)")
    if all_negative:
        checks["warnings"].append("All predictions are negative (class=0)")
    if all_probs_above:
        checks["warnings"].append("All predicted probabilities are above threshold")
    if all_probs_below:
        checks["warnings"].append("All predicted probabilities are below threshold")
    if almost_constant_probs:
        checks["warnings"].append("Predicted probabilities are almost constant")

    checks["is_degenerate"] = any([all_positive, all_negative, all_probs_above, all_probs_below, almost_constant_probs])
    return checks

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--proc", default="data/processed", help="processed data dir")
    p.add_argument("--art", default="artifacts", help="artifacts dir")
    p.add_argument("--ecg-len", type=int, default=DEFAULT_ECG_LEN)
    p.add_argument("--device", default="cpu", help="cuda,mps or cpu")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--fail-on-degenerate", action="store_true", help="Exit with code 3 if degenerate predictions are detected")
    p.add_argument("--min-roc-auc", type=float, default=None, help="Optional minimum ROC AUC sanity threshold; warns/fails if lower")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    proc_dir = Path(args.proc)
    art_dir = Path(args.art)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    logging.info("Loading tabular data...")
    Xt, yt = load_tab(proc_dir, "test")

    logging.info("Loading ECG (optional)...")
    Ecg = load_ecg(proc_dir, "test", args.ecg_len)
    if Ecg is None:
        logging.info("No ECG file found; using zeros as fallback.")
        Ecg = np.zeros((Xt.shape[0], args.ecg_len), dtype="float32")

    # align/truncate if necessary (defensive)
    if Xt.shape[0] != yt.shape[0]:
        logging.warning("X and y length mismatch; truncating to min length.")
        n = min(Xt.shape[0], yt.shape[0])
        Xt, yt, Ecg = Xt[:n], yt[:n], Ecg[:n]

    tab_dim = Xt.shape[1]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" or args.device == "mps" else "cpu")

    # instantiate model (adjust parameters to your model)
    model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)

    ckpt_path = art_dir / "model.pt"
    if not ckpt_path.exists():
        logging.error("Checkpoint not found: %s", ckpt_path)
        raise SystemExit(2)

    logging.info("Loading checkpoint %s to device %s", ckpt_path, device)
    load_checkpoint(model, str(ckpt_path), map_location=str(device))  # keep your loader if it handles map_location
    # alternatively: model.load_state_dict(torch.load(ckpt_path, map_location=device))

    probs = batch_inference(model, Xt, Ecg, device, batch_size=args.batch_size)

    metrics = compute_metrics(yt, probs, threshold=args.threshold)
    logging.info("Metrics: acc=%.4f roc=%.4f pr=%.4f brier=%.4f", metrics["accuracy"], metrics["roc_auc"], metrics["pr_auc"], metrics["brier"])
    logging.info("Confusion matrix:\n%s", metrics["confusion_matrix"])

    checks = run_failure_checks(yt, probs, metrics["preds"], threshold=args.threshold)
    for w in checks["warnings"]:
        logging.warning("Degeneracy check: %s", w)

    low_roc_auc = False
    if (
        args.min_roc_auc is not None
        and not np.isnan(metrics["roc_auc"])
        and metrics["roc_auc"] < args.min_roc_auc
    ):
        low_roc_auc = True
        logging.warning("ROC AUC %.4f is below --min-roc-auc %.4f", metrics["roc_auc"], args.min_roc_auc)

    # save outputs and provenance
    np.save(out_dir / "y_true.npy", yt)
    np.save(out_dir / "y_prob.npy", probs)
    np.save(out_dir / "y_pred.npy", metrics["preds"])

    meta = {
        "timestamp": time.time(),
        "proc_dir": str(proc_dir),
        "art_dir": str(art_dir),
        "ckpt": str(ckpt_path),
        "device": str(device),
        "n_samples": int(yt.shape[0]),
        "metrics": {k: (v.tolist() if hasattr(v, "tolist") else str(v)) for k, v in metrics.items() if k != "confusion_matrix"},
        "failure_checks": checks,
        "low_roc_auc": low_roc_auc,
    }
    with open(out_dir / "eval_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logging.info("Saved outputs to %s", out_dir)

    if args.fail_on_degenerate and (checks["is_degenerate"] or low_roc_auc):
        logging.error("Evaluation failed sanity checks (degenerate predictions and/or low ROC AUC).")
        raise SystemExit(3)

if __name__ == "__main__":
    main()