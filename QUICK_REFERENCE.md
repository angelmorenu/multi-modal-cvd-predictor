# ⚡ Quick Reference Card

**Print this or bookmark it while you're implementing Phase 1**

---

## 🎯 The Problem (30 seconds)
Your model predicts all samples as positive due to class imbalance (86.6%). 
ROC AUC is 0.4479 (worse than random).

## ✅ The Fix (30 seconds)
Add class weighting to your loss function. Update 2 files (30 min coding).
Result: ROC AUC > 0.55, model learns to distinguish classes.

---

## 📋 Your Task This Week

```
TASK                    TIME      DOCS TO READ
────────────────────────────────────────────────
Read overview           5 min     REVIEW_SUMMARY.md
Read detailed plan      30 min    PROJECT_REVIEW.md
Implement changes       3-5 hrs   CODE_CHANGES_VISUAL.md + 
                                  IMPLEMENTATION_GUIDE.md
Track progress          1-2 hrs   CHECKLIST_PHASE1.md
Test & verify          1 hr      CHECKLIST_PHASE1.md
────────────────────────────────────────────────
TOTAL                  5-7 hrs
```

---

## 🚀 Three Ways to Start

### 1. FAST (Ready to code now?)
```
1. Open REVIEW_SUMMARY.md
2. Skip to "Immediate Next Steps"
3. Open CHECKLIST_PHASE1.md
4. Start implementing Step 1
```

### 2. THOROUGH (Need context?)
```
1. Open PROJECT_REVIEW.md
2. Read "Executive Summary"
3. Open IMPLEMENTATION_GUIDE.md
4. Follow each step with examples
```

### 3. VISUAL (Prefer examples?)
```
1. Open CODE_CHANGES_VISUAL.md
2. Find your file (experiments/train.py or src/train.py)
3. See "BEFORE" and "AFTER" code
4. Copy the changes exactly
```

---

## 🔄 The 5 Code Changes

### Change 1: Add Imports
```python
# Add to top of file
from src.losses import setup_loss_and_weights, check_class_balance
from src.metrics import compute_binary_metrics, find_optimal_threshold_f1
```

### Change 2: Check Class Balance
```python
# After loading training data
y_train = np.concatenate([y for _, y in train_loader])
check_class_balance(y_train, verbose=True)
```

### Change 3: Setup Loss
```python
# Replace BCEWithLogitsLoss() with:
criterion, loss_info = setup_loss_and_weights(
    y_train=y_train,
    loss_type=args.loss,  # 'bce', 'weighted_bce', 'focal'
    device=device
)
```

### Change 4: Update Metrics
```python
# In validation loop
metrics = compute_binary_metrics(y_val, y_pred_probs, threshold=0.5)
print(f"ROC AUC: {metrics['roc_auc']:.4f}")
print(f"Sensitivity: {metrics['sensitivity']:.4f}")
print(f"Specificity: {metrics['specificity']:.4f}")
```

### Change 5: Add Test Collection
```python
# Collect targets while validating
val_targets.append(y.cpu().numpy())
y_val_true = np.concatenate(val_targets)
```

---

## ✅ Success Checklist

### Before Implementation
- [ ] Read appropriate doc (5-30 min)
- [ ] Have IDE open
- [ ] Have 3-5 hours of focus time

### During Implementation
- [ ] Add imports (30 sec)
- [ ] Add class balance check (1 min)
- [ ] Replace BCE setup (2 min)
- [ ] Update metrics (3 min)
- [ ] Save and test (5 min)

### After Implementation
- [ ] Python imports work: `python -c "from src.losses import setup_loss_and_weights"`
- [ ] Training runs: `python -m experiments.train --loss weighted_bce --epochs 1`
- [ ] Class balance prints (check output)
- [ ] Loss config prints (check output)
- [ ] Metrics print (check output)
- [ ] Confusion matrix has all quadrants > 0
- [ ] ROC AUC > 0.55

### Final Success
- [ ] ROC AUC improved (from 0.4479)
- [ ] Sensitivity > 0% (from 0%)
- [ ] Specificity > 0% (from 0%)
- [ ] All quadrants filled in confusion matrix
- [ ] No error messages
- [ ] Changes committed to git

---

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "ModuleNotFoundError: src.losses" | Run from project root: `cd /Users/angelhdmorenu/Documents/multi-modal-cvd-predictor` |
| "TypeError: missing argument" | Check function signature - likely missing `device=device` |
| "ROC AUC still 0.44" | Did you actually pass `--loss weighted_bce`? Check it's being used |
| "Class balance not printing" | Make sure `check_class_balance()` is called AFTER y_train is loaded |
| "Metrics not printing" | Make sure you're collecting both predictions AND targets |

---

## 🧪 Testing Commands

```bash
# Test 1: Standard BCE (shows problem)
python -m experiments.train --loss bce --epochs 1 --batch-size 16

# Test 2: Weighted BCE (shows fix)
python -m experiments.train --loss weighted_bce --epochs 1 --batch-size 16

# Test 3: Focal Loss (alternative)
python -m experiments.train --loss focal --epochs 1 --batch-size 16

# Test 4: Run all tests
pytest tests/test_class_imbalance.py -v
```

---

## 📊 Expected Output (Weighted BCE)

```
✓ Class balance analysis:
  Positive: 5959 (86.61%)
  Negative: 918 (13.39%)

✓ Loss setup complete:
  Loss type: weighted_bce
  pos_weight: 0.1541 (918 / 5959)

Epoch 1 | Loss: 0.2345 | ROC AUC: 0.5678 | F1: 0.3456 | 
Sens: 0.4500 | Spec: 0.6200
```

---

## 🎯 What's Different (Before vs After)

| Metric | Before | After |
|--------|--------|-------|
| ROC AUC | 0.4479 | > 0.55 |
| Sensitivity | 1.0000 | 0.30-0.70 |
| Specificity | 0.0000 | 0.30-0.70 |
| TP, TN, FP, FN | 0, 918, 0, 5959 | All > 0 |

---

## 📚 File Reference

| When You... | Open This |
|-------------|-----------|
| Don't know where to start | INDEX.md |
| Need quick overview | REVIEW_SUMMARY.md |
| Want all details | PROJECT_REVIEW.md |
| Need a quick reference | QUICK_START_FIXES.md |
| Ready to implement | IMPLEMENTATION_GUIDE.md or CODE_CHANGES_VISUAL.md |
| Tracking progress | CHECKLIST_PHASE1.md |
| Stuck / troubleshooting | CHECKLIST_PHASE1.md (scroll to Troubleshooting) |

---

## ⏱️ Time Breakdown

```
Reading:      0-30 min (pick your pace)
Implementing: 0.5 hours (copy/type changes)
Testing:      0.5 hours (verify changes)
Debugging:    1-2 hours (if issues)
────────────────────────
Total:        3-5 hours
```

---

## 🚀 Next Steps

1. **Pick your path** (Fast/Thorough/Visual above)
2. **Read the document** (5-30 min)
3. **Open the file** (experiments/train.py or src/train.py)
4. **Make 5 changes** (add imports, checks, loss setup, metrics)
5. **Test** (run training, check ROC AUC)
6. **Done!** (model is learning)

---

## 💡 Remember

- This is **fixable** (done all the time in ML)
- **You have all the code** (just needs integration)
- **Tests verify it works** (30+ tests included)
- **Detailed guidance exists** (multiple docs provided)
- **You can do this** (3-5 hours of focused work)

---

**Status**: Ready to implement  
**Confidence**: High (proven mathematical solutions)  
**Expected outcome**: Model learns to classify diseases  
**Time investment**: This week, 5-7 hours  

**GO MAKE IT HAPPEN! 🚀**
