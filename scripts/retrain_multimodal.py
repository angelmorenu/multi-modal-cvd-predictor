#!/usr/bin/env python3
"""
Retrain MultiModalCVD on properly split data with imbalance awareness.

Usage:
    python scripts/retrain_multimodal.py --processed data/processed --epochs 100 --device cpu
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.model import MultiModalCVD, save_checkpoint


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(processed_dir):
    """Load train/val/test data from processed directory."""
    def _load(name):
        p = Path(processed_dir) / name
        return torch.tensor(np.load(p), dtype=torch.float32) if p.exists() else None
    
    return {
        'tabular_train_X': _load('tabular_train_X.npy'),
        'tabular_train_y': _load('tabular_train_y.npy'),
        'tabular_val_X': _load('tabular_val_X.npy'),
        'tabular_val_y': _load('tabular_val_y.npy'),
        'tabular_test_X': _load('tabular_test_X.npy'),
        'tabular_test_y': _load('tabular_test_y.npy'),
        'ecg_train': _load('ecg_train.npy'),
        'ecg_val': _load('ecg_val.npy'),
        'ecg_test': _load('ecg_test.npy'),
    }


def train_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for tabular, ecg, y in loader:
        tabular = tabular.to(device)
        ecg = ecg.unsqueeze(1).to(device) if ecg.dim() == 2 else ecg.to(device)
        y = y.long().to(device)
        
        optimizer.zero_grad()
        logits = model(tabular, ecg)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * y.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / total, correct / total


def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_probs = []
    all_y = []
    
    with torch.no_grad():
        for tabular, ecg, y in loader:
            tabular = tabular.to(device)
            ecg = ecg.unsqueeze(1).to(device) if ecg.dim() == 2 else ecg.to(device)
            y = y.long().to(device)
            
            logits = model(tabular, ecg)
            loss = criterion(logits, y)
            
            total_loss += loss.item() * y.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_y.extend(y.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_y = np.array(all_y)
    
    roc_auc = roc_auc_score(all_y, all_probs) if len(np.unique(all_y)) > 1 else np.nan
    ap = average_precision_score(all_y, all_probs) if len(np.unique(all_y)) > 1 else np.nan
    
    return total_loss / total, correct / total, roc_auc, ap


def main():
    parser = argparse.ArgumentParser(description='Retrain multimodal model on proper splits')
    parser.add_argument('--processed', type=str, default='data/processed')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out-dir', type=str, default='artifacts')
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    
    # Load data
    print("Loading data...")
    data = load_data(args.processed)
    
    # Check all data exists
    for key in ['tabular_train_X', 'tabular_train_y', 'tabular_val_X', 'tabular_val_y',
                'tabular_test_X', 'tabular_test_y', 'ecg_train', 'ecg_val', 'ecg_test']:
        if data[key] is None:
            raise ValueError(f"Missing {key}")
    
    # Create datasets
    train_ds = TensorDataset(data['tabular_train_X'], data['ecg_train'], data['tabular_train_y'])
    val_ds = TensorDataset(data['tabular_val_X'], data['ecg_val'], data['tabular_val_y'])
    test_ds = TensorDataset(data['tabular_test_X'], data['ecg_test'], data['tabular_test_y'])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")
    print(f"Test: {len(test_ds)} samples")
    print()
    
    # Build model
    tab_dim = data['tabular_train_X'].shape[1]
    model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)
    model = model.to(device)
    
    # Compute class weights for training set
    train_y = data['tabular_train_y'].numpy().astype(int)
    class_counts = np.bincount(train_y)
    class_weights = torch.tensor(
        len(train_y) / (2 * class_counts),
        dtype=torch.float32,
        device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    print(f"Model: MultiModalCVD(tab_dim={tab_dim}, ecg_channels=1)")
    print(f"Loss: CrossEntropyLoss(weight={class_weights.tolist()})")
    print(f"Optimizer: AdamW(lr={args.lr})")
    print()
    
    # Training loop
    best_val_roc = -1
    best_model_state = None
    patience_counter = 0
    early_stopping_patience = 15
    
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_roc, val_ap = eval_epoch(model, val_loader, criterion, device)
        
        if epoch % 10 == 0 or epoch < 5:
            print(f"Epoch {epoch:3d} | Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.3f} roc={val_roc:.3f} ap={val_ap:.3f}")
        
        if not np.isnan(val_roc) and val_roc > best_val_roc:
            best_val_roc = val_roc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"  → New best ROC AUC: {val_roc:.4f}")
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (patience {patience_counter})")
            break
    
    # Evaluate best model on test set
    print("\n" + "="*70)
    print("FINAL EVALUATION")
    print("="*70)
    
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    test_loss, test_acc, test_roc, test_ap = eval_epoch(model, test_loader, criterion, device)
    
    print(f"Test Set Metrics:")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  ROC AUC: {test_roc:.4f}")
    print(f"  AP: {test_ap:.4f}")
    print()
    
    # Save model
    ckpt_path = out_dir / "model.pt"
    save_checkpoint(model, str(ckpt_path))
    print(f"Saved checkpoint to {ckpt_path}")
    
    # Save metrics
    metrics = {
        'train_samples': len(train_ds),
        'val_samples': len(val_ds),
        'test_samples': len(test_ds),
        'test_accuracy': float(test_acc),
        'test_roc_auc': float(test_roc) if not np.isnan(test_roc) else None,
        'test_ap': float(test_ap) if not np.isnan(test_ap) else None,
        'best_val_roc': float(best_val_roc) if best_val_roc > 0 else None,
    }
    metrics_path = out_dir / "retrain_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
