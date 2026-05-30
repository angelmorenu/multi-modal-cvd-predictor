#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train.py — Multi-Modal CVD project

Modes supported:
  (A) ECG-only (Week 4 sanity check)
  (B) Fusion (Tabular + ECG) (Week 5)

Data expectations (flexible, any subset may be missing):
- Tabular (np.save’d by preprocess.py):
    data/processed/tabular_train_X.npy
    data/processed/tabular_train_y.npy
    data/processed/tabular_val_X.npy
    data/processed/tabular_val_y.npy
    (optional) data/processed/tabular_test_X.npy, tabular_test_y.npy

- ECG waveform arrays (preferred for CNN; one-lead, padded/cropped in code):
    data/processed/ecg_train.npy   # shape (N_train, T)
    data/processed/ecg_val.npy     # shape (N_val,   T)
    (optional) data/processed/ecg_test.npy

If ECG arrays are not available, you can still:
  - Train tabular-only baseline (fusion with zero ECG), or
  - (Alternative) Use ECG feature csv by projecting to an embedding (not included by default).

Author: Angel Morenu
Course: EEE 6778 – Applied Machine Learning II (Fall 2025)
"""

from __future__ import annotations
import os
import argparse
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

# Local imports
# Support both running as a module (python -m src.train) and running the
# script directly (python src/train.py). When the file is executed directly,
# relative imports (from .model) fail with "no known parent package". Try
# the relative import first, then fall back to the package-style absolute
# import.
try:
    from .model import MultiModalCVD, ECG1DCNN, save_checkpoint  # noqa
except Exception:
    from src.model import MultiModalCVD, ECG1DCNN, save_checkpoint  # noqa

# -----------------------
# Utilities
# -----------------------
def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_tensor(x, dtype=torch.float32):
    return torch.as_tensor(x, dtype=dtype)

def exists(path: str) -> bool:
    return path is not None and os.path.exists(path)

# -----------------------
# Data Loading Helpers
# -----------------------
def load_tabular_split(processed_dir: str, split: str):
    Xp = os.path.join(processed_dir, f"tabular_{split}_X.npy")
    yp = os.path.join(processed_dir, f"tabular_{split}_y.npy")
    if exists(Xp) and exists(yp):
        X = np.load(Xp)
        # ensure 2D and correct dtype
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X = X.astype(np.float32)

        y = np.load(yp).astype(np.int64)
        if y.ndim == 0:
            y = np.array([int(y)], dtype=np.int64)
        return X, y
    return None, None

def load_ecg_split(processed_dir: str, split: str):
    ep = os.path.join(processed_dir, f"ecg_{split}.npy")
    if exists(ep):
        E = np.load(ep)
        # Expect (N, T). If a single 1D array was saved, expand dims.
        if E.ndim == 1:
            E = E[None, :]
        # Ensure float32 for PyTorch
        E = E.astype(np.float32)
        return E
    return None

def make_dataloader(tab_X, tab_y, ecg, batch_size: int, shuffle: bool, ecg_len: int = 2000):
    # Align by first dimension where possible
    if tab_X is not None and ecg is not None:
        n = min(len(tab_X), len(ecg))
        tab_X = tab_X[:n]
        tab_y = tab_y[:n] if tab_y is not None else np.zeros(n, dtype=np.int64)
        ecg   = ecg[:n]
    elif tab_X is not None:
        n = len(tab_X)
        tab_y = tab_y[:n] if tab_y is not None else np.zeros(n, dtype=np.int64)
        # create zero ECG (tab-only training)
        ecg = np.zeros((n, ecg_len), dtype=np.float32)
    elif ecg is not None:
        n = len(ecg)
        tab_X = np.zeros((n, 32), dtype=np.float32)  # placeholder tabular
        tab_y = tab_y[:n] if tab_y is not None else np.zeros(n, dtype=np.int64)
    else:
        raise RuntimeError("No data found for this split.")

    # Tensors
    tab_t = to_tensor(tab_X, torch.float32)
    y_t   = to_tensor(tab_y, torch.long)
    # reshape ECG to (N, C=1, T)
    ecg_t = to_tensor(ecg, torch.float32).unsqueeze(1)

    ds = TensorDataset(tab_t, ecg_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

# -----------------------
# Loss Functions
# -----------------------
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.
    
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    where p_t is the model's estimated probability for the ground truth class.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C) logits from model
            targets: (B,) integer class labels
        Returns:
            loss scalar
        """
        p = torch.softmax(logits, dim=-1)
        p_t = p.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        focal_weight = (1 - p_t) ** self.gamma
        ce = nn.functional.cross_entropy(logits, targets, reduction='none')
        loss = focal_weight * ce
        if self.alpha is not None:
            # Apply class weighting
            alpha_t = self.alpha.gather(0, targets)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# -----------------------
# Data Augmentation
# -----------------------
def augment_ecg(ecg_batch, prob=0.5):
    """Apply light ECG augmentation during training.
    
    Args:
        ecg_batch: (B, 1, T) tensor
        prob: probability of applying augmentation to each sample
    
    Returns:
        augmented_batch: (B, 1, T) tensor
    """
    if np.random.rand() > prob:
        return ecg_batch
    
    B, C, T = ecg_batch.shape
    aug_batch = ecg_batch.clone()
    
    for b in range(B):
        aug_type = np.random.choice(['noise', 'scale', 'jitter'])
        
        if aug_type == 'noise':
            # Add small Gaussian noise
            noise = torch.randn_like(aug_batch[b]) * 0.05
            aug_batch[b] = torch.clamp(aug_batch[b] + noise, -1.0, 1.0)
        elif aug_type == 'scale':
            # Random amplitude scaling
            scale = np.random.uniform(0.8, 1.2)
            aug_batch[b] = aug_batch[b] * scale
        elif aug_type == 'jitter':
            # Time-shift by small amount
            shift = np.random.randint(-10, 11)
            if shift != 0:
                aug_batch[b] = torch.roll(aug_batch[b], shifts=shift, dims=-1)
    
    return aug_batch

# -----------------------
# Training / Eval
# -----------------------
def run_one_epoch(model, loader, criterion, optimizer=None, device: torch.device | str = "cpu", augment_ecg_flag=False):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_correct, total = 0.0, 0, 0

    for tab_x, ecg_x, y in loader:
        tab_x = tab_x.to(device)
        ecg_x = ecg_x.to(device)
        y     = y.to(device)

        # Apply ECG augmentation during training
        if is_train and augment_ecg_flag:
            ecg_x = augment_ecg(ecg_x, prob=0.5)

        logits = model(tab_x, ecg_x)  # (B, 2)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            preds = probs.argmax(dim=-1)
            total_correct += (preds == y).sum().item()
            total += y.size(0)
            total_loss += loss.item() * y.size(0)

    avg_loss = total_loss / max(1, total)
    acc = total_correct / max(1, total)
    return avg_loss, acc

# -----------------------
# Main entry point and argument parsing
# -----------------------
def main():
    p = argparse.ArgumentParser(description="Train ECG CNN (Week 4) and Fusion (Week 5).")
    p.add_argument("--processed_dir", default="data/processed", help="Dir with npy splits.")
    p.add_argument("--artifacts_dir", default="artifacts", help="Where to save checkpoints.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ecg_only", action="store_true", help="Week 4: train ECG CNN + a small head (tabular zeros).")
    p.add_argument("--ecg_len", type=int, default=2000, help="Target ECG length (pad/crop).")
    p.add_argument("--patients-file", default=None, help="Optional JSON file with list of patient_ids to use for this run (patient-level subsampling).")
    p.add_argument("--loss", choices=["ce", "weighted_ce", "focal"], default="weighted_ce", help="Loss function: ce (cross-entropy), weighted_ce (class-weighted), or focal (focal loss).")
    p.add_argument("--focal_gamma", type=float, default=2.0, help="Gamma parameter for focal loss (higher = focus on hard negatives).")
    p.add_argument("--dropout", type=float, default=0.3, help="Dropout rate in model layers.")
    p.add_argument("--weight_decay", type=float, default=1e-4, help="L2 regularization weight.")
    p.add_argument("--augment_ecg", action="store_true", help="Enable ECG data augmentation (noise, jitter, scaling).")
    args = p.parse_args()

    set_seed(args.seed)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # --- Load data
    tab_tr_X, tab_tr_y = load_tabular_split(args.processed_dir, "train")
    tab_va_X, tab_va_y = load_tabular_split(args.processed_dir, "val")

    ecg_tr = load_ecg_split(args.processed_dir, "train")  # (N, T)
    ecg_va = load_ecg_split(args.processed_dir, "val")

    # If a patients-file is provided, subset arrays using the patient manifest.
    if args.patients_file is not None:
        import json as _json
        mf = os.path.join(args.processed_dir, "manifest.csv")
        if not os.path.exists(mf):
            print(f"WARNING: manifest not found at {mf}; --patients-file will be ignored.")
        else:
            # load manifest and build mapping patient_id -> row index (assumes manifest rows align with array ordering)
            import pandas as _pd
            mdf = _pd.read_csv(mf)
            pid_to_idx = {pid: idx for idx, pid in enumerate(mdf["patient_id"].tolist())}
            with open(args.patients_file) as fh:
                sel_pids = _json.load(fh)
            sel_idxs = [pid_to_idx[p] for p in sel_pids if p in pid_to_idx]
            sel_idxs = sorted(set(sel_idxs))
            if tab_tr_X is not None:
                tab_tr_X = tab_tr_X[sel_idxs]
                if tab_tr_y is not None:
                    tab_tr_y = tab_tr_y[sel_idxs]
            if ecg_tr is not None:
                ecg_tr = ecg_tr[sel_idxs]
            # For val we keep as-is (validation is created by run_nested_cv via train/val split of patients)

    # Optional: pad/crop ECG to ecg_len
    def fit_len(ecg):
        if ecg is None: return None
        T = ecg.shape[1]
        if T == args.ecg_len: return ecg
        if T > args.ecg_len:  return ecg[:, :args.ecg_len]
        # pad end
        pad = np.zeros((ecg.shape[0], args.ecg_len - T), dtype=ecg.dtype)
        return np.concatenate([ecg, pad], axis=1)

    ecg_tr = fit_len(ecg_tr)
    ecg_va = fit_len(ecg_va)

    # --- Infer tab_dim
    tab_dim = 32
    if tab_tr_X is not None:
        tab_dim = tab_tr_X.shape[1]

    # --- Build model
    model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)
    device = torch.device(args.device)
    model.to(device)

    # --- Build loss function
    if args.loss == "weighted_ce":
        # Compute class weights from training set
        if tab_tr_y is not None:
            unique, counts = np.unique(tab_tr_y, return_counts=True)
            class_weights = 1.0 / (counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * 2  # normalize to ~2
            class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
            criterion = nn.CrossEntropyLoss(weight=class_weights_t)
        else:
            criterion = nn.CrossEntropyLoss()
    elif args.loss == "focal":
        # Compute class weights for focal loss
        if tab_tr_y is not None:
            unique, counts = np.unique(tab_tr_y, return_counts=True)
            class_weights = 1.0 / (counts + 1e-6)
            class_weights = class_weights / class_weights.sum() * 2
            class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
        else:
            class_weights_t = None
        criterion = FocalLoss(gamma=args.focal_gamma, alpha=class_weights_t, reduction='mean')
    else:  # ce
        criterion = nn.CrossEntropyLoss()

    # Loss/opt with weight decay for L2 regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # --- Dataloaders
    train_loader = make_dataloader(tab_tr_X, tab_tr_y, ecg_tr, batch_size=args.batch_size, shuffle=True)
    val_loader   = make_dataloader(tab_va_X, tab_va_y, ecg_va, batch_size=args.batch_size, shuffle=False)

    # --- Training loop
    best_val = math.inf
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_one_epoch(model, train_loader, criterion, optimizer, device, augment_ecg_flag=args.augment_ecg)
        va_loss, va_acc = run_one_epoch(model, val_loader, criterion, optimizer=None, device=device, augment_ecg_flag=False)

        print(f"[Epoch {epoch:02d}] "
              f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
              f"val_loss={va_loss:.4f} acc={va_acc:.3f}")

        # Save best (by val_loss)
        if va_loss < best_val:
            best_val = va_loss
            ckpt = os.path.join(args.artifacts_dir, "model_ecg.pt" if args.ecg_only else "model.pt")
            save_checkpoint(model, ckpt)
            print(f"  Saved checkpoint: {ckpt}")

    print("Done.")

if __name__ == "__main__":
    main()