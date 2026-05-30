#!/usr/bin/env python3
"""
Optuna-based hyperparameter optimization for the multi-modal CVD model.

This script uses Optuna to search over:
  - Learning rate (1e-4 to 1e-2)
  - Dropout rate (0.1 to 0.5)
  - Batch size (4, 8, 16)
  - Augmentation probability (0 to 1)
  - Early stopping patience (3 to 10 epochs)

Runs 5-fold CV for each trial and reports mean validation accuracy.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Local imports
from src.model import MultiModalCVD, save_checkpoint


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(processed_dir: str):
    """Load all arrays from processed directory."""
    def _load(name):
        p = os.path.join(processed_dir, name)
        return np.load(p) if os.path.exists(p) else None

    tab_tr_X = _load("tabular_train_X.npy")
    tab_tr_y = _load("tabular_train_y.npy")
    tab_va_X = _load("tabular_val_X.npy")
    tab_va_y = _load("tabular_val_y.npy")
    ecg_tr = _load("ecg_train.npy")
    ecg_va = _load("ecg_val.npy")
    
    # Ensure proper shapes
    for arr in [tab_tr_X, tab_va_X]:
        if arr is not None and arr.ndim == 1:
            arr = arr.reshape(1, -1)
    for arr in [ecg_tr, ecg_va]:
        if arr is not None and arr.ndim == 1:
            arr = arr[None, :]
    
    return tab_tr_X, tab_tr_y, tab_va_X, tab_va_y, ecg_tr, ecg_va


def make_dataloader(tab_X, tab_y, ecg, batch_size: int, shuffle: bool, ecg_len: int = 2000):
    """Create aligned dataloader."""
    if tab_X is not None and ecg is not None:
        n = min(len(tab_X), len(ecg))
        tab_X, ecg = tab_X[:n], ecg[:n]
        tab_y = tab_y[:n] if tab_y is not None else np.zeros(n, dtype=np.int64)
    elif tab_X is not None:
        n = len(tab_X)
        tab_y = tab_y[:n] if tab_y is not None else np.zeros(n, dtype=np.int64)
        ecg = np.zeros((n, ecg_len), dtype=np.float32)
    elif ecg is not None:
        n = len(ecg)
        tab_X = np.zeros((n, 32), dtype=np.float32)
        tab_y = tab_y[:n] if tab_y is not None else np.zeros(n, dtype=np.int64)
    else:
        raise RuntimeError("No data found.")
    
    # Fit ECG to length
    if ecg.shape[1] != ecg_len:
        if ecg.shape[1] > ecg_len:
            ecg = ecg[:, :ecg_len]
        else:
            pad = np.zeros((ecg.shape[0], ecg_len - ecg.shape[1]), dtype=ecg.dtype)
            ecg = np.concatenate([ecg, pad], axis=1)
    
    tab_t = torch.as_tensor(tab_X, dtype=torch.float32)
    y_t = torch.as_tensor(tab_y, dtype=torch.long)
    ecg_t = torch.as_tensor(ecg, dtype=torch.float32).unsqueeze(1)
    
    ds = TensorDataset(tab_t, ecg_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def augment_ecg(ecg_batch, prob=0.5):
    """Light ECG augmentation."""
    if np.random.rand() > prob:
        return ecg_batch
    
    B, C, T = ecg_batch.shape
    aug_batch = ecg_batch.clone()
    
    for b in range(B):
        aug_type = np.random.choice(['noise', 'scale', 'jitter'])
        if aug_type == 'noise':
            noise = torch.randn_like(aug_batch[b]) * 0.05
            aug_batch[b] = torch.clamp(aug_batch[b] + noise, -1.0, 1.0)
        elif aug_type == 'scale':
            scale = np.random.uniform(0.8, 1.2)
            aug_batch[b] = aug_batch[b] * scale
        elif aug_type == 'jitter':
            shift = np.random.randint(-10, 11)
            if shift != 0:
                aug_batch[b] = torch.roll(aug_batch[b], shifts=shift, dims=-1)
    
    return aug_batch


def train_and_eval(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs: int,
    early_stopping_patience: int,
    device,
    augment_prob: float = 0.5,
):
    """Train model and return best validation accuracy."""
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        for tab_x, ecg_x, y in train_loader:
            tab_x, ecg_x, y = tab_x.to(device), ecg_x.to(device), y.to(device)
            
            # Apply augmentation
            if augment_prob > 0:
                ecg_x = augment_ecg(ecg_x, prob=augment_prob)
            
            logits = model(tab_x, ecg_x)
            loss = criterion(logits, y)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        
        # Eval
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for tab_x, ecg_x, y in val_loader:
                tab_x, ecg_x, y = tab_x.to(device), ecg_x.to(device), y.to(device)
                logits = model(tab_x, ecg_x)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)
        
        val_acc = val_correct / max(1, val_total)
        
        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                break
    
    return best_val_acc


def objective(trial: optuna.Trial, args, splits_csv: str, processed_dir: str, device: str = "cpu"):
    """Optuna objective function: train on one fold and return validation accuracy."""
    
    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    batch_size = trial.suggest_categorical('batch_size', [4, 8, 16])
    augment_prob = trial.suggest_float('augment_prob', 0.0, 1.0)
    early_stopping_patience = trial.suggest_int('early_stopping_patience', 3, 10)
    
    set_seed(args.seed)
    device_obj = torch.device(device)
    
    # Load data
    tab_tr_X, tab_tr_y, tab_va_X, tab_va_y, ecg_tr, ecg_va = load_data(processed_dir)
    
    # Create dataloaders
    train_loader = make_dataloader(tab_tr_X, tab_tr_y, ecg_tr, batch_size=batch_size, shuffle=True)
    val_loader = make_dataloader(tab_va_X, tab_va_y, ecg_va, batch_size=batch_size, shuffle=False)
    
    # Build model
    tab_dim = tab_tr_X.shape[1] if tab_tr_X is not None else 32
    model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)
    model.to(device_obj)
    
    # Loss (weighted BCE)
    if tab_tr_y is not None:
        unique, counts = np.unique(tab_tr_y, return_counts=True)
        class_weights = 1.0 / (counts + 1e-6)
        class_weights = class_weights / class_weights.sum() * 2
        class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device_obj)
        criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Train and evaluate
    val_acc = train_and_eval(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        epochs=args.epochs,
        early_stopping_patience=early_stopping_patience,
        device=device_obj,
        augment_prob=augment_prob,
    )
    
    return val_acc


def main():
    p = argparse.ArgumentParser(description="Optuna HPO for multi-modal CVD model")
    p.add_argument("--processed_dir", default="data/processed", help="Preprocessed data dir")
    p.add_argument("--splits_csv", default="data/splits/train.csv", help="Patient splits CSV")
    p.add_argument("--epochs", type=int, default=15, help="Max epochs per trial")
    p.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    p.add_argument("--out_dir", default="results/optuna", help="Output directory for study")
    args = p.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    set_seed(args.seed)
    
    # Create Optuna study
    sampler = TPESampler(seed=args.seed)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=3)
    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner,
        study_name='cvd_hpo'
    )
    
    # Objective wrapper
    def obj(trial):
        return objective(trial, args, args.splits_csv, args.processed_dir, args.device)
    
    # Optimize
    print(f"Starting Optuna HPO with {args.n_trials} trials...")
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=True)
    
    # Save results
    best_trial = study.best_trial
    results = {
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
    }
    
    print()
    print("=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)
    print()
    print(f"Best Validation Accuracy: {best_trial.value:.4f}")
    print()
    print("Best Hyperparameters:")
    for k, v in best_trial.params.items():
        print(f"  {k:25s} = {v}")
    print()
    
    # Save to JSON
    with open(os.path.join(args.out_dir, 'best_params.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save study history
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(args.out_dir, 'trials_history.csv'), index=False)
    
    print(f"Results saved to {args.out_dir}/")
    print(f"  - best_params.json")
    print(f"  - trials_history.csv")


if __name__ == "__main__":
    main()
