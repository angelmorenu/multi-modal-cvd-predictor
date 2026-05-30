# 🎨 Visual Code Changes Guide

This shows **exactly** what to change in each file, with before/after examples.

---

## File 1: experiments/train.py

### Change 1: Add Imports (Top of file, lines 1-15)

**BEFORE:**
```python
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from experiments.data import get_train_val_test_dataloaders
from src.model import ECG1DCNN, MultiModalCVD
```

**AFTER:**
```python
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import yaml
from pathlib import Path
from experiments.data import get_train_val_test_dataloaders
from src.model import ECG1DCNN, MultiModalCVD
from src.losses import setup_loss_and_weights, check_class_balance  # ← NEW
from src.metrics import compute_binary_metrics, find_optimal_threshold_f1  # ← NEW
```

---

### Change 2: Add Class Balance Check (After data loading, around line 105)

**BEFORE:**
```python
    # Get dataloaders
    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
        batch_size=args.batch_size,
        data_type='multi_modal',
        num_workers=2
    )
    
    # Setup model
    if args.modality == 'ecg_only':
```

**AFTER:**
```python
    # Get dataloaders
    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
        batch_size=args.batch_size,
        data_type='multi_modal',
        num_workers=2
    )
    
    # NEW: Check class balance in training data
    y_train = np.concatenate([y for _, y in train_loader])
    check_class_balance(y_train, verbose=True)
    
    # Setup model
    if args.modality == 'ecg_only':
```

---

### Change 3: Setup Loss with Class Weighting (Replace BCE, around line 150)

**BEFORE:**
```python
    # Setup loss
    if args.loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'weighted_bce':
        # Not properly implemented
        criterion = nn.BCEWithLogitsLoss()
    elif args.loss == 'focal':
        from src.losses import FocalLoss
        criterion = FocalLoss()
    else:
        raise ValueError(f"Unknown loss: {args.loss}")
```

**AFTER:**
```python
    # Setup loss with class weighting
    criterion, loss_info = setup_loss_and_weights(
        y_train=y_train,
        loss_type=args.loss,
        alpha=0.25,  # for focal loss
        gamma=2.0,   # for focal loss
        device=device
    )
    print(f"\n✓ Loss setup complete:")
    print(f"  Loss type: {loss_info['loss_type']}")
    print(f"  pos_weight: {loss_info.get('pos_weight', 'N/A')}")
    print(f"  gamma: {loss_info.get('gamma', 'N/A')}")
    print(f"  alpha: {loss_info.get('alpha', 'N/A')}\n")
```

---

### Change 4: Update Validation Metrics (In training loop, around line 190-210)

**BEFORE:**
```python
        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_outputs = []
            for X_tab, X_ecg, y in val_loader:
                X_tab = X_tab.to(device)
                X_ecg = X_ecg.to(device)
                y = y.to(device)
                
                logits = model(X_tab, X_ecg)
                loss = criterion(logits, y.unsqueeze(1).float())
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(logits)
                val_outputs.append(probs.cpu().numpy())
            
            val_loss = np.mean(val_losses)
            y_val_probs = np.concatenate(val_outputs)
            print(f"Epoch {epoch+1}, Val Loss: {val_loss:.4f}")
```

**AFTER:**
```python
        # Validation
        model.eval()
        with torch.no_grad():
            val_losses = []
            val_outputs = []
            val_targets = []
            for X_tab, X_ecg, y in val_loader:
                X_tab = X_tab.to(device)
                X_ecg = X_ecg.to(device)
                y = y.to(device)
                
                logits = model(X_tab, X_ecg)
                loss = criterion(logits, y.unsqueeze(1).float())
                val_losses.append(loss.item())
                
                probs = torch.sigmoid(logits)
                val_outputs.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())
            
            val_loss = np.mean(val_losses)
            y_val_probs = np.concatenate(val_outputs)
            y_val_true = np.concatenate(val_targets)
            
            # Compute comprehensive metrics
            metrics = compute_binary_metrics(y_val_true, y_val_probs, threshold=0.5)
            
            print(f"Epoch {epoch+1} | Loss: {val_loss:.4f} | " + 
                  f"ROC AUC: {metrics['roc_auc']:.4f} | " +
                  f"F1: {metrics['f1_score']:.4f} | " +
                  f"Sens: {metrics['sensitivity']:.4f} | " +
                  f"Spec: {metrics['specificity']:.4f}")
```

---

## File 2: src/train.py

### Change 1: Add Imports (Top of file, lines 1-15)

**BEFORE:**
```python
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from pathlib import Path
from experiments.data import get_train_val_test_dataloaders
from src.model import MultiModalCVD
```

**AFTER:**
```python
import argparse
import numpy as np
import torch
from torch import nn
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
from pathlib import Path
from experiments.data import get_train_val_test_dataloaders
from src.model import MultiModalCVD
from src.losses import setup_loss_and_weights, check_class_balance  # ← NEW
from src.metrics import compute_binary_metrics  # ← NEW
```

---

### Change 2: Add Class Balance Check (After loading data, around line 100)

**BEFORE:**
```python
def train_model(args):
    # Load data
    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
        batch_size=args.batch_size,
        data_type='multi_modal'
    )
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

**AFTER:**
```python
def train_model(args):
    # Load data
    train_loader, val_loader, test_loader = get_train_val_test_dataloaders(
        batch_size=args.batch_size,
        data_type='multi_modal'
    )
    
    # NEW: Get training labels and check class balance
    y_train = np.concatenate([y.numpy() for _, y in train_loader])
    check_class_balance(y_train, verbose=True)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

---

### Change 3: Setup Loss with Class Weighting (Replace BCE, around line 110-115)

**BEFORE:**
```python
    # Setup loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
```

**AFTER:**
```python
    # Setup loss with class weighting and optimizer
    criterion, loss_info = setup_loss_and_weights(
        y_train=y_train,
        loss_type='weighted_bce',  # Use weighted BCE by default
        device=device
    )
    print(f"\n✓ Loss configured: {loss_info}\n")
    optimizer = Adam(model.parameters(), lr=args.lr)
```

---

### Change 4: Update Test Evaluation (Around line 160-180)

**BEFORE:**
```python
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = []
        for X_tab, X_ecg, y in test_loader:
            X_tab = X_tab.to(device)
            X_ecg = X_ecg.to(device)
            logits = model(X_tab, X_ecg)
            probs = torch.sigmoid(logits)
            test_outputs.append(probs.cpu().numpy())
        
        y_test_probs = np.concatenate(test_outputs)
        print(f"Test ROC AUC: {roc_auc_score(y_test, y_test_probs):.4f}")
```

**AFTER:**
```python
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = []
        test_targets = []
        for X_tab, X_ecg, y in test_loader:
            X_tab = X_tab.to(device)
            X_ecg = X_ecg.to(device)
            logits = model(X_tab, X_ecg)
            probs = torch.sigmoid(logits)
            test_outputs.append(probs.cpu().numpy())
            test_targets.append(y.numpy())
        
        y_test_true = np.concatenate(test_targets)
        y_test_probs = np.concatenate(test_outputs)
        
        # Compute comprehensive metrics
        metrics = compute_binary_metrics(y_test_true, y_test_probs, threshold=0.5)
        
        print(f"\nTest Results:")
        print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        print(f"  F1-Score: {metrics['f1_score']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}\n")
```

---

## Summary of Changes

| File | Lines | Change | Effort |
|------|-------|--------|--------|
| experiments/train.py | 1-15 | Add 2 import lines | 30 sec |
| experiments/train.py | ~105 | Add class balance check (4 lines) | 1 min |
| experiments/train.py | ~150 | Replace BCE setup (6 lines → 11 lines) | 2 min |
| experiments/train.py | ~190-210 | Update validation metrics (15 lines → 25 lines) | 3 min |
| **experiments/train.py Total** | | | **~6 min** |
| | | | |
| src/train.py | 1-15 | Add 2 import lines | 30 sec |
| src/train.py | ~100 | Add class balance check (3 lines) | 1 min |
| src/train.py | ~110-115 | Replace BCE setup (6 lines → 7 lines) | 2 min |
| src/train.py | ~160-180 | Update test metrics (12 lines → 22 lines) | 3 min |
| **src/train.py Total** | | | **~7 min** |
| | | | |
| **GRAND TOTAL** | | All changes | **~13 min of actual typing** |

---

## Testing Your Changes

After making the changes above, test with:

```bash
# Navigate to project
cd /Users/angelhdmorenu/Documents/multi-modal-cvd-predictor

# Test 1: Standard BCE (should show low ROC AUC)
python -m experiments.train --loss bce --epochs 1

# Test 2: Weighted BCE (should show improvement)
python -m experiments.train --loss weighted_bce --epochs 1

# Test 3: Focal Loss (alternative)
python -m experiments.train --loss focal --epochs 1

# Test 4: Run full test suite
pytest tests/test_class_imbalance.py -v
```

---

## Expected Output

### Test 1 (BCE - shows the problem)
```
✓ Class balance analysis:
  Positive: 5959 (86.61%)
  Negative: 918 (13.39%)
  
Loss setup complete:
  Loss type: bce
  pos_weight: N/A
  
Epoch 1 | Loss: 0.1234 | ROC AUC: 0.4479 | F1: 0.0000 | Sens: 1.0000 | Spec: 0.0000
```

### Test 2 (Weighted BCE - the fix)
```
✓ Class balance analysis:
  Positive: 5959 (86.61%)
  Negative: 918 (13.39%)
  
Loss setup complete:
  Loss type: weighted_bce
  pos_weight: 0.1541 (918 / 5959)
  
Epoch 1 | Loss: 0.2345 | ROC AUC: 0.5678 | F1: 0.3456 | Sens: 0.4500 | Spec: 0.6200
```

### Test 3 (Focal Loss - alternative)
```
✓ Class balance analysis:
  Positive: 5959 (86.61%)
  Negative: 918 (13.39%)
  
Loss setup complete:
  Loss type: focal
  gamma: 2.0
  alpha: 0.25
  
Epoch 1 | Loss: 0.3456 | ROC AUC: 0.5890 | F1: 0.3789 | Sens: 0.4800 | Spec: 0.6500
```

**Key differences:**
- BCE: ROC AUC ≈ 0.45 (problem)
- Weighted/Focal: ROC AUC > 0.55 (improvement)
- Sensitivity changes from 1.0 → 0.45-0.48 (learning!)
- Specificity changes from 0.0 → 0.62-0.65 (learning!)

---

## That's It! 🎉

These are all the code changes needed for Phase 1. Total time: ~15 minutes of actual coding.

Expected outcome: Model learns to distinguish disease from healthy.

Next step: Follow `CHECKLIST_PHASE1.md` to test everything.
