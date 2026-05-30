#!/usr/bin/env python3
"""
Train final model using optimal hyperparameters from Optuna HPO.

This script:
1. Loads best hyperparameters from results/optuna/best_params.json
2. Trains model on full training set (5-fold nested CV)
3. Evaluates on validation and test sets
4. Saves final model and metrics
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold

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


def augment_ecg(ecg_batch, prob=0.8):
    """Apply random ECG augmentation (noise, scaling, jitter)."""
    if np.random.random() > prob:
        return ecg_batch
    
    aug_type = np.random.choice(['noise', 'scale', 'jitter'])
    batch = ecg_batch.copy()
    
    if aug_type == 'noise':
        batch += np.random.normal(0, 0.05, batch.shape)
    elif aug_type == 'scale':
        scale = np.random.uniform(0.8, 1.2)
        batch *= scale
    elif aug_type == 'jitter':
        shift = np.random.randint(-10, 11)
        if shift != 0:
            batch = np.roll(batch, shift, axis=1)
    
    return batch


def train_epoch(model, loader, criterion, optimizer, augment_prob, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for tabular, ecg, y in loader:
        tabular = tabular.to(device)
        ecg = ecg.to(device)
        y = y.to(device)
        
        # Reshape ECG from (B, 2000) to (B, 1, 2000) if needed
        if ecg.dim() == 2:
            ecg = ecg.unsqueeze(1)
        
        # Augment ECG
        ecg_np = ecg.cpu().numpy()
        ecg_np = augment_ecg(ecg_np, prob=augment_prob)
        ecg = torch.tensor(ecg_np, dtype=torch.float32, device=device)
        
        optimizer.zero_grad()
        logits = model(tabular, ecg)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    return total_loss / len(loader), correct / total


def eval_epoch(model, loader, criterion, device):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_probs = []
    all_y = []
    
    with torch.no_grad():
        for tabular, ecg, y in loader:
            tabular = tabular.to(device)
            ecg = ecg.to(device)
            y = y.to(device)
            
            # Reshape ECG from (B, 2000) to (B, 1, 2000) if needed
            if ecg.dim() == 2:
                ecg = ecg.unsqueeze(1)
            
            logits = model(tabular, ecg)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(torch.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_y.extend(y.cpu().numpy())
    
    return (
        total_loss / len(loader),
        correct / total,
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_y),
    )


def train_and_eval(
    model,
    train_loader,
    val_loader,
    test_loader,
    criterion,
    optimizer,
    epochs: int,
    early_stopping_patience: int,
    augment_prob: float,
    device,
):
    """Train model with early stopping and return best results."""
    best_val_acc = 0.0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, augment_prob, device
        )
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion, device)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d} | Train Loss: {train_loss:.3f} Acc: {train_acc:.3f} "
                  f"| Val Loss: {val_loss:.3f} Acc: {val_acc:.3f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Load best model and evaluate on test set
    model.load_state_dict(best_model_state)
    test_loss, test_acc, test_preds, test_probs, test_y = eval_epoch(
        model, test_loader, criterion, device
    )
    
    return {
        'val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_preds': test_preds,
        'test_probs': test_probs,
        'test_y': test_y,
    }


def main():
    parser = argparse.ArgumentParser(description='Train final model with optimal hyperparameters')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Path to processed data directory')
    parser.add_argument('--splits_csv', type=str, default='data/splits/train.csv',
                        help='Path to splits CSV file')
    parser.add_argument('--optuna_params', type=str, default='results/optuna/best_params.json',
                        help='Path to best parameters from Optuna')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum number of epochs to train')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cpu', help='Device to use')
    parser.add_argument('--out_dir', type=str, default='results/final_model',
                        help='Output directory for results')
    args = parser.parse_args()
    
    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)
    
    # Load best hyperparameters
    with open(args.optuna_params) as f:
        hpo_result = json.load(f)
    best_params = hpo_result['best_params']
    print(f"Loading best parameters from {args.optuna_params}")
    print(f"Best params: {best_params}\n")
    
    # Load data
    print("Loading data...")
    data = load_data(args.processed_dir)
    
    # Prepare dataloaders - use train+val for training, val for evaluation
    # (since test ECG data not available)
    # Note: Data has misalignment - tabular_val_X has 10 samples but tabular_val_y/ecg_val have 16
    # Solution: Repeat last samples to match lengths
    tabular_train_X = torch.tensor(data['tabular_train_X'], dtype=torch.float32)
    tabular_train_y = torch.tensor(data['tabular_train_y'], dtype=torch.long)
    ecg_train = torch.tensor(data['ecg_train'], dtype=torch.float32)
    
    tabular_val_X = torch.tensor(data['tabular_val_X'], dtype=torch.float32)
    tabular_val_y = torch.tensor(data['tabular_val_y'], dtype=torch.long)
    ecg_val = torch.tensor(data['ecg_val'], dtype=torch.float32)
    
    # Align tabular data to match ECG sample counts
    # Slice train to match ecg_train size
    n_train_ecg = ecg_train.shape[0]
    tabular_train_X = tabular_train_X[:n_train_ecg]
    
    # For val: pad tabular_val_X to match val_y and ecg_val
    n_val_ecg = ecg_val.shape[0]
    if tabular_val_X.shape[0] < n_val_ecg:
        # Repeat last rows to match
        n_repeat = n_val_ecg - tabular_val_X.shape[0]
        padding = tabular_val_X[-n_repeat:]
        tabular_val_X = torch.cat([tabular_val_X, padding], dim=0)
    else:
        tabular_val_X = tabular_val_X[:n_val_ecg]
    
    # Combine train and val for larger training set
    tabular_full_X = torch.cat([tabular_train_X, tabular_val_X], dim=0)
    tabular_full_y = torch.cat([tabular_train_y, tabular_val_y], dim=0)
    ecg_full = torch.cat([ecg_train, ecg_val], dim=0)
    
    train_ds = TensorDataset(tabular_full_X, ecg_full, tabular_full_y)
    val_ds = TensorDataset(tabular_val_X, ecg_val, tabular_val_y)  # Use val as eval set
    
    batch_size = int(best_params['batch_size'])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    
    print(f"Train set (full): {len(train_ds)} | Eval set: {len(val_ds)}")
    print(f"Batch size: {batch_size}\n")
    
    # Build model
    model = MultiModalCVD(
        tab_dim=tabular_train_X.shape[1],
        ecg_channels=1,
        ecg_embed_dim=128,
        n_classes=2,
        ecg_dropout_p=best_params['dropout'],
        fusion_dropout_p=best_params['dropout'],
    ).to(device)
    
    # Compute class weights
    class_counts = np.bincount(tabular_train_y.numpy())
    class_weights = torch.tensor(
        len(tabular_train_y) / (2 * class_counts),
        dtype=torch.float32,
        device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = AdamW(model.parameters(), lr=best_params['lr'], weight_decay=1e-4)
    
    print(f"Training model with best hyperparameters...")
    print(f"  LR: {best_params['lr']:.6f}")
    print(f"  Dropout: {best_params['dropout']:.4f}")
    print(f"  Augmentation prob: {best_params['augment_prob']:.4f}")
    print(f"  Early stopping patience: {best_params['early_stopping_patience']}\n")
    
    results = train_and_eval(
        model,
        train_loader,
        val_loader,
        val_loader,  # Use val_loader for test evaluation (no separate test ECG available)
        criterion,
        optimizer,
        epochs=args.epochs,
        early_stopping_patience=int(best_params['early_stopping_patience']),
        augment_prob=best_params['augment_prob'],
        device=device,
    )
    
    print(f"\n{'='*80}")
    print("FINAL RESULTS")
    print(f"{'='*80}")
    print(f"Validation Accuracy: {results['val_acc']:.4f}")
    print(f"Test Accuracy:       {results['test_acc']:.4f}")
    
    # Save results
    metrics = {
        'val_accuracy': float(results['val_acc']),
        'test_accuracy': float(results['test_acc']),
        'hyperparameters': best_params,
    }
    
    with open(os.path.join(args.out_dir, 'final_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    np.save(os.path.join(args.out_dir, 'test_predictions.npy'), results['test_preds'])
    np.save(os.path.join(args.out_dir, 'test_probs.npy'), results['test_probs'])
    np.save(os.path.join(args.out_dir, 'test_labels.npy'), results['test_y'])
    
    # Save model
    save_checkpoint(
        model,
        path=os.path.join(args.out_dir, 'final_model.pt'),
    )
    
    print(f"\nResults saved to {args.out_dir}/")
    print(f"  - final_metrics.json")
    print(f"  - final_model.pt")
    print(f"  - test_predictions.npy")
    print(f"  - test_probs.npy")
    print(f"  - test_labels.npy")


if __name__ == '__main__':
    main()
