# 🚀 Quick Start: Fixing Your CVD Model

This guide helps you implement the critical fixes identified in the comprehensive review.

## TL;DR - What's Wrong?

Your model predicts **every sample as positive** because:
1. ✗ Data is 86.6% positive class (severe imbalance)
2. ✗ No class weighting in loss function
3. ✗ Using accuracy metric (wrong for imbalanced data)
4. ✗ No threshold optimization

**Result:** ROC AUC = 0.4479 (worse than random) despite 86.6% accuracy

## What's Fixed (New Files)

### 1. `src/metrics.py` - Comprehensive Evaluation
- Bootstrap confidence intervals for all metrics
- Calibration analysis (ECE, MCE)
- ROC/PR curve analysis
- **Threshold optimization for clinical settings**
- Fairness analysis utilities

### 2. `src/losses.py` - Class Imbalance Solutions
- Weighted BCE loss with automatic pos_weight computation
- Focal Loss implementation (down-weights easy examples)
- SMOTE data balancing
- Class balance checking utilities

### 3. `PROJECT_REVIEW.md` - Complete Analysis
- Detailed issue breakdown
- 70+ hour roadmap to production
- Phase-by-phase implementation guide
- Success criteria for each phase

## Phase 1: Fix Class Imbalance (TODAY - 5 hours)

### Step 1.1: Update `experiments/train.py`

**Add imports:**
```python
from src.losses import setup_loss_and_weights, check_class_balance
from src.metrics import compute_binary_metrics, find_optimal_threshold_f1
```

**In `main()` function, after loading data (around line 100):**
```python
# Add class balance checking
print("\n" + "="*60)
print("CLASS BALANCE ANALYSIS")
print("="*60)
check_class_balance(y_train, verbose=True)

# Setup loss function with class weighting
criterion, loss_info = setup_loss_and_weights(
    y_train,
    loss_type=args.loss,  # 'bce', 'weighted_bce', or 'focal'
    focal_gamma=args.focal_gamma,
    device=device
)

print("\nLoss Function Configuration:")
for k, v in loss_info.items():
    print(f"  {k}: {v}")
```

**Update training loop to use proper criterion** (around line 150):
```python
# OLD: criterion = nn.BCEWithLogitsLoss()
# NEW: criterion (already set up above with class weighting)
```

**Add evaluation with proper metrics** (around line 180):
```python
# After getting val_probs and y_val:
val_metrics = compute_binary_metrics(y_val, val_probs, threshold=0.5, compute_ci=False)
print(f"\nValidation Metrics:")
print(f"  ROC AUC: {val_metrics['roc_auc']:.4f}")
print(f"  PR AUC: {val_metrics['pr_auc']:.4f}")
print(f"  F1-Score: {val_metrics['f1_score']:.4f}")
print(f"  Sensitivity: {val_metrics['sensitivity']:.4f}")
print(f"  Specificity: {val_metrics['specificity']:.4f}")

# Find optimal threshold
optimal_thresh, optimal_f1 = find_optimal_threshold_f1(y_val, val_probs)
print(f"\nOptimal F1 threshold: {optimal_thresh:.4f} (F1={optimal_f1:.4f})")
```

### Step 1.2: Update `src/train.py` similarly

Apply the same changes to `src/train.py` (lines match approximately)

### Step 1.3: Test with Balanced Data

```bash
cd /Users/angelhdmorenu/Documents/multi-modal-cvd-predictor
source .venv/bin/activate

# Test with weighted BCE loss
python -m experiments.train \
  --processed data/processed \
  --epochs 2 \
  --batch-size 16 \
  --loss weighted_bce \
  --device cpu \
  --augment

# You should see:
# ✓ Class balance analysis showing imbalance ratio
# ✓ Loss function using pos_weight
# ✓ Validation metrics showing ROC AUC > 0.5
# ✓ Confusion matrix with BOTH TP and TN > 0
```

### Step 1.4: Try Focal Loss

```bash
python -m experiments.train \
  --processed data/processed \
  --epochs 2 \
  --batch-size 16 \
  --loss focal \
  --focal-gamma 2.0 \
  --device cpu \
  --augment
```

### Step 1.5: Compare Results

Run both BCE and weighted BCE, capture output:

```bash
# Standard BCE (should show model bias)
python -m experiments.train --loss bce --epochs 2 > results/test_bce.log 2>&1

# Weighted BCE (should be better)
python -m experiments.train --loss weighted_bce --epochs 2 > results/test_weighted_bce.log 2>&1

# Compare ROC AUC values in logs
grep "ROC AUC" results/test_*.log
```

## Phase 2: Proper Evaluation (Next Week)

Once Phase 1 works, implement comprehensive evaluation:

```bash
# New evaluation script (create src/eval_comprehensive.py)
python src/eval_comprehensive.py \
  --proc data/processed \
  --art artifacts \
  --out-dir results/comprehensive_eval \
  --n-bootstraps 1000  # Generate confidence intervals
```

This will produce:
- Metrics with 95% confidence intervals
- Calibration curves
- ROC/PR curves with threshold annotations
- Fairness analysis by demographic groups

## Success Criteria - How to Know It's Working

### Before Fixes (Current)
```
❌ Confusion matrix: [[0, 918], [0, 5959]]
❌ ROC AUC: 0.4479 (worse than random)
❌ All predictions > 0.5
```

### After Fixes (Target)
```
✅ Confusion matrix: non-zero in all quadrants
✅ ROC AUC: > 0.60 (at minimum, 0.70+ is good)
✅ Mix of predictions across probability range
✅ Sensitivity and Specificity both reported
✅ F1-score reported instead of accuracy
```

## File-by-File Changes

### `experiments/train.py`
Lines to modify (approximately):
- **Line 30-40**: Add loss import
- **Line 100-110**: Add class balance checking
- **Line 115-130**: Setup loss with weighting
- **Line 150**: Use weighted criterion
- **Line 180-200**: Add proper evaluation metrics

### `src/train.py`
Same changes as above, different line numbers

### `src/eval.py`
- Import new metrics module
- Replace accuracy with ROC AUC as primary metric
- Add confidence interval computation
- Add calibration analysis

## Next Commands to Run

```bash
# 1. Verify metrics module works
python -c "from src.metrics import compute_binary_metrics; print('✓ Metrics module OK')"

# 2. Verify losses module works
python -c "from src.losses import setup_loss_and_weights; print('✓ Losses module OK')"

# 3. Run a quick test with class weighting
python -m pytest tests/test_train_smoke.py -v

# 4. Check the comprehensive review
cat PROJECT_REVIEW.md | head -100
```

## Questions to Ask Yourself

After implementing Phase 1:

1. **Does ROC AUC improve?** (Should go from 0.44 to 0.60+)
2. **Are both positive and negative classes predicted?** (Check confusion matrix)
3. **Does sensitivity > specificity?** (Expected for imbalanced data with standard threshold)
4. **Do predictions spread across probability range?** (Not all > 0.5)

If yes to all → Phase 1 is successful!

## Common Issues & Solutions

### Issue: "Loss module not found"
**Solution:** Make sure you're in the venv: `source .venv/bin/activate`

### Issue: "Metrics computation fails"
**Solution:** Ensure y_true is binary (0 or 1), not one-hot encoded

### Issue: "ROC AUC still doesn't improve"
**Solution:** 
1. Check that pos_weight is being applied (should be ~6.5)
2. Try focal loss instead (may work better)
3. Increase training epochs (1 epoch is just a smoke test)

### Issue: "SMOTE fails with 'not enough samples'"
**Solution:** SMOTE needs at least k_neighbors samples per class. Use simple oversampling instead (available in src/losses.py)

## Timeline

- **Today** (2-3 hours): Implement fixes to both train files
- **Tomorrow** (2 hours): Test and verify improvements
- **This week** (8 hours): Implement Phase 2 evaluation
- **Next week** (20 hours): Better models and external validation

---

**Need more details?** See `PROJECT_REVIEW.md` for comprehensive analysis.
