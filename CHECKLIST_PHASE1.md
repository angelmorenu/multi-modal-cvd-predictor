# ✅ Phase 1 Implementation Checklist

**Goal**: Fix class imbalance and get model to actually learn  
**Time Estimate**: 3-5 hours  
**Target ROC AUC**: > 0.55 (from 0.4479)

---

## 📋 Pre-Implementation

- [ ] Read `IMPLEMENTATION_GUIDE.md` (15 min)
- [ ] Have `src/losses.py` and `src/metrics.py` in your IDE (already created)
- [ ] Backup current results (optional but recommended)
- [ ] Have test data available

---

## 🔧 Step 1: Update experiments/train.py (45 min)

### Add Imports (Line 1-10)
- [ ] Add: `from src.losses import setup_loss_and_weights, check_class_balance`
- [ ] Add: `from src.metrics import compute_binary_metrics, find_optimal_threshold_f1`

### Add Class Balance Check (After data loading, ~line 105)
```python
# NEW: Check class balance
check_class_balance(y_train, verbose=True)
```
- [ ] Code added
- [ ] Indentation correct

### Setup Loss & Weights (Replace BCE instantiation, ~line 150)
```python
# NEW: Setup loss with class weights
criterion, loss_info = setup_loss_and_weights(
    y_train=y_train,
    loss_type=args.loss,  # 'bce', 'weighted_bce', or 'focal'
    alpha=0.25,  # for focal loss
    gamma=2.0,   # for focal loss
    device=device
)
print(f"Loss: {loss_info}")
```
- [ ] Code added
- [ ] Using `args.loss` parameter
- [ ] Prints loss configuration

### Update Training Loop Metrics (Line ~190)
Replace basic metric computation with:
```python
# NEW: Comprehensive metrics
metrics = compute_binary_metrics(
    y_true=y_val.cpu().numpy(),
    y_pred=probs.cpu().numpy(),
    threshold=0.5
)
print(f"ROC AUC: {metrics['roc_auc']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
```
- [ ] Code added
- [ ] Metrics printing in training loop

---

## 🔧 Step 2: Update src/train.py (45 min)

### Add Imports (Line 1-10)
- [ ] Add: `from src.losses import setup_loss_and_weights, check_class_balance`
- [ ] Add: `from src.metrics import compute_binary_metrics`

### Add Class Balance Check (After y_train loaded, ~line 100)
```python
check_class_balance(y_train, verbose=True)
```
- [ ] Code added

### Setup Loss (Replace BCE instantiation, ~line 110)
```python
criterion, loss_info = setup_loss_and_weights(
    y_train=y_train,
    loss_type='weighted_bce',
    device=device
)
print(f"Loss configuration: {loss_info}")
```
- [ ] Code added
- [ ] Using 'weighted_bce' as default

### Update Metrics (Line ~160)
```python
metrics = compute_binary_metrics(y_test, y_pred_probs, threshold=0.5)
print(f"Test ROC AUC: {metrics['roc_auc']:.4f}")
print(f"Test F1-Score: {metrics['f1_score']:.4f}")
```
- [ ] Code added

---

## ✅ Step 3: Test Implementation (30 min)

### Verify Imports
```bash
cd /Users/angelhdmorenu/Documents/multi-modal-cvd-predictor
python -c "from src.losses import setup_loss_and_weights; print('✓ Imports OK')"
```
- [ ] Command runs without error

### Test Basic Training
```bash
python -m experiments.train --loss weighted_bce --epochs 2 --batch-size 16
```
- [ ] Runs without crash
- [ ] Shows class balance info
- [ ] Shows loss configuration
- [ ] Training loop completes
- [ ] Shows final metrics (ROC AUC, sensitivity, specificity)

### Check Output
- [ ] "Loss configuration" printed
- [ ] "ROC AUC" printed (should be > 0.5, not 0.44)
- [ ] "Sensitivity" printed (should be > 0, not 1.0)
- [ ] "Specificity" printed (should be > 0, not 0.0)

---

## 🔍 Step 4: Compare Loss Functions (1 hour)

### Test 1: Standard BCE (to see the problem)
```bash
python -m experiments.train --loss bce --epochs 1 --batch-size 16
# Note: ROC AUC should be ~0.44, confusion matrix [[0, 918], [0, 5959]]
```
- [ ] Run completes
- [ ] Note ROC AUC value (should be bad ~0.4)
- [ ] Note confusion matrix (should have zeros)

### Test 2: Weighted BCE (the fix)
```bash
python -m experiments.train --loss weighted_bce --epochs 1 --batch-size 16
# Note: ROC AUC should be > 0.5, confusion matrix should be more balanced
```
- [ ] Run completes
- [ ] Note ROC AUC value (should be > 0.5)
- [ ] Note confusion matrix (should have all quadrants > 0)

### Test 3: Focal Loss (alternative)
```bash
python -m experiments.train --loss focal --epochs 1 --batch-size 16
# Note: Focal loss often works better than weighted BCE for severe imbalance
```
- [ ] Run completes
- [ ] Compare ROC AUC to weighted BCE
- [ ] Note if focal loss is better

### Compare Results
- [ ] Fill in comparison table:

| Loss Type | ROC AUC | Sensitivity | Specificity | TP | TN | FP | FN |
|-----------|---------|-------------|-------------|----|----|----|----|
| BCE       | ?       | ?           | ?           | ?  | ?  | ?  | ?  |
| Weighted  | ?       | ?           | ?           | ?  | ?  | ?  | ?  |
| Focal     | ?       | ?           | ?           | ?  | ?  | ?  | ?  |

---

## ✨ Step 5: Verify Success (30 min)

### Checklist: Model is Learning
- [ ] ROC AUC > 0.50 (improvement from 0.4479)
- [ ] Sensitivity > 0.0 (improvement from 0.0)
- [ ] Specificity > 0.0 (improvement from 0.0)
- [ ] Confusion matrix has values in all quadrants
- [ ] Predictions span 0.0-1.0 range (not all > 0.5)

### Checklist: Code Quality
- [ ] No import errors
- [ ] No runtime errors
- [ ] No warnings about deprecated functions
- [ ] All new functions documented with docstrings

### Checklist: Ready for Next Phase
- [ ] Changes committed to git
- [ ] `IMPLEMENTATION_GUIDE.md` successfully followed
- [ ] Can run: `pytest tests/test_class_imbalance.py -v`
- [ ] All tests pass

---

## 🎯 Success Criteria (FINAL CHECK)

**Before Phase 1:**
```
Accuracy: 86.65% (misleading - matches majority class)
ROC AUC: 0.4479 (worse than random)
Sensitivity: 100% (predicts all positive)
Specificity: 0% (never predicts negative)
Confusion matrix: [[0, 918], [0, 5959]]
```

**After Phase 1 (TARGET):**
```
Accuracy: 70-80% (realistic)
ROC AUC: > 0.55 (improvement)
Sensitivity: 30-70% (actually learning)
Specificity: 30-70% (actually learning)
Confusion matrix: All quadrants > 0
```

---

## 🐛 Troubleshooting

### Problem: "ModuleNotFoundError: No module named 'src.losses'"
**Solution**: You're running from wrong directory. Do:
```bash
cd /Users/angelhdmorenu/Documents/multi-modal-cvd-predictor
python -m experiments.train ...
```

### Problem: "TypeError: setup_loss_and_weights() missing required argument"
**Solution**: Check that you're passing all arguments:
```python
criterion, loss_info = setup_loss_and_weights(
    y_train=y_train,  # required
    loss_type=args.loss,  # required
    device=device  # required
)
```

### Problem: "Class balance info not printing"
**Solution**: Make sure check_class_balance is called after y_train is loaded:
```python
# After data is loaded
check_class_balance(y_train, verbose=True)
```

### Problem: "Model still predicting all positive"
**Solution**: Likely issue is still using BCE without weight. Check:
1. Did you replace the BCE instantiation?
2. Is it using `loss_type=args.loss` and you're passing `--loss weighted_bce`?
3. Try running with `--loss focal` instead

### Problem: "ROC AUC still 0.44"
**Solution**: Confirm changes are actually running:
1. Print loss configuration: `print(loss_info)`
2. Print class weights: `print(f"pos_weight: {loss_info.get('pos_weight')}")`
3. Check if weights are being applied

---

## 📞 Need Help?

1. Check `IMPLEMENTATION_GUIDE.md` for detailed walkthrough
2. Look at `src/losses.py` docstrings for function documentation
3. Look at `src/metrics.py` docstrings for metric definitions
4. Run `pytest tests/test_class_imbalance.py -v` to see working examples

---

## ⏭️ Next Steps (After Phase 1 Success)

Once you have ROC AUC > 0.55:

1. **Week 2**: Implement proper evaluation metrics
   - Bootstrap confidence intervals
   - Calibration analysis
   - Threshold optimization

2. **Week 3-4**: Improve model architecture
   - Better ECG models (ResNet, Inception)
   - Hyperparameter sweep with proper class weighting
   - Ensemble methods

3. **Week 5**: External validation
   - Run on PTBDB/CPSC datasets
   - Check generalization

---

**Status**: Ready to implement  
**Estimated Completion**: This week (3-5 hours focused work)  
**Expected Outcome**: Model learns to distinguish disease from healthy  

🚀 **Let's go fix this!**
