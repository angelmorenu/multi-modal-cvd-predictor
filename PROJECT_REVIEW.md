# 🔍 Multi-Modal CVD Predictor – Comprehensive Project Review

**Review Date:** December 2025  
**Status:** Post-coursework improvement plan (medical-grade readiness track)  
**Reviewer Focus:** Class imbalance fixes, evaluation rigor, clinical validity

---

## Executive Summary

Your multi-modal CVD prediction project is **well-architected and reproducible**, with solid engineering foundations (Docker, CI/CD, modular design, clear documentation). However, **critical model validation issues** revealed in recent results indicate the system requires substantial remediation before clinical deployment:

### Critical Finding: Complete Model Collapse

Your last reported results show the model predicting **all samples as positive class**:

```
Accuracy: 0.8665 (deceptively high — equals majority class baseline)
PR AUC: 0.8570 (misleading due to class imbalance)
ROC AUC: 0.4479 (< 0.5 = worse than random; major red flag)
Confusion Matrix: [[0, 918], [0, 5959]] (zero discrimination)
Min probability: 0.5966, Max: 0.7376 (entire output collapsed to positive)
```

**Root Cause:** Severe class imbalance (~86.6% positive class) with **no effective mitigation strategy in place**.

**Impact:** Model has **zero clinical validity** and cannot be deployed or published without comprehensive remediation.

---

## Part 1: Understanding the Class Imbalance Problem

### Why This Happens

1. **Data Imbalance:** Your training data contains ~86.6% positive samples (CVD=1), 13.4% negative (CVD=0)
2. **Naive Loss Function:** Standard BCE loss treats both classes equally in aggregate loss
3. **Optimization Dynamics:** Model learns it can minimize loss by predicting positive for everything (86.6% accuracy for free)
4. **Sigmoid Collapse:** All logits get pushed far into (0.5, 1.0) range, losing discriminatory power
5. **Metrics Illusion:** Accuracy looks good, but ROC-AUC < 0.5 reveals complete failure

### Why Standard Metrics Fail

| Metric | Value | Interpretation |
|--------|-------|-----------------|
| Accuracy | 86.65% | ❌ Misleading (majority class baseline) |
| ROC AUC | 0.4479 | ❌ **Worse than random** |
| PR AUC | 0.8570 | ❌ Misleading (class-imbalanced average) |
| Sensitivity | 100% (5959/5959) | ✅ Correct by accident (predicts all positive) |
| Specificity | 0% (0/918) | ❌ Zero ability to identify negatives |

**Key Insight:** Accuracy and PR-AUC are **unreliable for imbalanced data**. ROC-AUC is better but still shows catastrophic failure. **Sensitivity/Specificity trade-off reveals the truth.**

---

## Part 2: Prioritized Remediation Plan

### Phase 1: Diagnostics & Documentation (Essential First Step)

**Goal:** Understand exactly what went wrong and document for future prevention.

#### Task 1.1: Create Diagnostic Notebook
- **File:** `notebooks/diagnostic_analysis.ipynb`
- **Content:**
  - Visualize class distribution (histogram + percentages)
  - Show probability distributions for true positives vs. true negatives
  - Scatter plot of predicted probabilities colored by true label
  - Confusion matrix with all thresholds (0.1 to 0.9)
  - ROC curve showing poor discrimination
  - Calibration plot (reliability diagram)
  - Feature distributions stratified by target class
  - Identify features with strongest class separation

#### Task 1.2: Add Diagnostic Tests
- **File:** `tests/test_class_imbalance.py`
- **Tests:**
  - Assert ROC-AUC > 0.6 (must beat baseline)
  - Assert specificity + sensitivity > 1.0 (Youden index)
  - Assert predicted probabilities span full [0, 1] range
  - Assert model doesn't collapse to majority class
  - Check confusion matrix diagonal for meaningful predictions

### Phase 2: Fix the Training Pipeline

**Goal:** Integrate multiple class imbalance mitigation strategies.

#### Task 2.1: Implement Class Weights in Loss Function
**File:** `src/train.py` (update)

```python
def compute_class_weights(labels):
    """Compute inverse frequency weighting."""
    unique, counts = np.unique(labels, return_counts=True)
    # Weight inversely proportional to frequency
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * len(unique)  # normalize
    return torch.tensor(weights, dtype=torch.float32)

# In training loop:
class_weights = compute_class_weights(train_labels)
criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1:])
```

#### Task 2.2: Add SMOTE for Training Data
**File:** Create `src/data_balancing.py`

```python
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

def apply_smote(X, y, sampling_strategy=0.5):
    """
    Over-sample minority class using SMOTE.
    
    Args:
        X: feature matrix
        y: labels
        sampling_strategy: target ratio (minority_class / majority_class)
    
    Returns:
        X_resampled, y_resampled
    """
    pipeline = Pipeline([
        ('smote', SMOTE(sampling_strategy=sampling_strategy, random_state=42)),
    ])
    return pipeline.fit_resample(X, y)
```

#### Task 2.3: Add Focal Loss Option
**Status:** Already partially in `experiments/train.py`  
**Action:** Backport to `src/train.py` with correct implementation

```python
class FocalLoss(nn.Module):
    """Focal loss for addressing class imbalance."""
    def __init__(self, gamma=2.0, alpha=None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight for positive class
    
    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = bce * ((1 - p_t) ** self.gamma)
        
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss
        
        return loss.mean()
```

### Phase 3: Improve Evaluation Methodology

**Goal:** Use evaluation metrics appropriate for imbalanced data and find optimal threshold.

#### Task 3.1: Create Comprehensive Metrics Module
**File:** Create `src/eval_metrics.py`

```python
def evaluate_comprehensive(y_true, y_pred_probs, thresholds=None):
    """
    Comprehensive evaluation with multiple thresholds and metrics.
    
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    for thresh in thresholds:
        y_pred = (y_pred_probs >= thresh).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        metrics = {
            'threshold': thresh,
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'f1': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'youden': (tp/(tp+fn) + tn/(tn+fp) - 1) if (tp+fn)>0 and (tn+fp)>0 else 0,
            'roc_auc': roc_auc_score(y_true, y_pred_probs),  # threshold-independent
            'pr_auc': average_precision_score(y_true, y_pred_probs),
        }
        results.append(metrics)
    
    return pd.DataFrame(results)
```

#### Task 3.2: Add Threshold Optimization
**File:** Create `src/threshold_optimizer.py`

```python
def find_optimal_threshold(y_true, y_pred_probs, metric='f1'):
    """Find threshold that maximizes specified metric."""
    thresholds = np.arange(0.01, 1.0, 0.01)
    best_thresh = 0.5
    best_value = -1
    
    for thresh in thresholds:
        y_pred = (y_pred_probs >= thresh).astype(int)
        
        if metric == 'f1':
            value = f1_score(y_true, y_pred)
        elif metric == 'youden':
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            value = tp/(tp+fn) + tn/(tn+fp) - 1
        # ... other metrics
        
        if value > best_value:
            best_value = value
            best_thresh = thresh
    
    return best_thresh, best_value
```

### Phase 4: Implement Stratified Cross-Validation

**Goal:** Robust evaluation unbiased by random data splits.

#### Task 4.1: Add Stratified K-Fold Cross-Validation
**File:** Create `src/cv_utils.py`

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cross_validate

def evaluate_with_stratified_cv(X, y, model, n_splits=5, scoring=None):
    """
    Evaluate model with stratified k-fold cross-validation.
    
    Ensures each fold maintains original class distribution.
    """
    if scoring is None:
        scoring = {
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall',
        }
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_results = cross_validate(model, X, y, cv=skf, scoring=scoring, return_train_score=True)
    
    return cv_results
```

---

## Part 3: Repository Cleanup and Organization

### Files to Remove (Unused/Redundant)
- `requirements-interpretability.txt` → Merge into `requirements.txt` with optional markers
- `Report/*.tex` → Keep only final PDF, archive old versions
- `data/ptbxl_records/record_*.npy` → Large files, replace with script to regenerate
- `__pycache__/` directories → Add to `.gitignore` (already done)
- Old notebook checkpoints → Use `nbstripout` filter

### Files to Create/Enhance
- ✅ `notebooks/diagnostic_analysis.ipynb` — **Priority 1**
- ✅ `tests/test_class_imbalance.py` — **Priority 1**
- ✅ `src/eval_metrics.py` — **Priority 2**
- ✅ `src/data_balancing.py` — **Priority 2**
- ✅ `src/threshold_optimizer.py` — **Priority 2**
- ✅ `src/cv_utils.py` — **Priority 2**
- `docs/METHODOLOGY.md` — Data preprocessing and modeling choices
- `docs/CLINICAL_NOTES.md` — Medical context and interpretations
- `.env.example` — Configuration template

### Structure After Cleanup

```
multi-modal-cvd-predictor/
├── src/
│   ├── model.py
│   ├── train.py (updated)
│   ├── eval.py (updated)
│   ├── preprocess.py
│   ├── eval_metrics.py (NEW)
│   ├── data_balancing.py (NEW)
│   ├── threshold_optimizer.py (NEW)
│   └── cv_utils.py (NEW)
├── notebooks/
│   ├── diagnostic_analysis.ipynb (NEW — PRIORITY 1)
│   ├── Setup.ipynb
│   └── train_eval.ipynb
├── tests/
│   ├── test_class_imbalance.py (NEW — PRIORITY 1)
│   ├── test_train_smoke.py
│   └── test_model_shapes.py
├── scripts/ (keep all, but prioritize)
├── docs/
│   ├── METHODOLOGY.md (NEW)
│   ├── CLINICAL_NOTES.md (NEW)
│   └── TROUBLESHOOTING.md (NEW)
└── ...
```

---

## Part 4: Testing Strategy

### Unit Tests
- ✅ `tests/test_class_imbalance.py` — Assert no majority class collapse
- Verify class weights computation
- Verify SMOTE output shape and label distribution
- Verify threshold optimization finds distinct boundaries

### Integration Tests
- Train-eval pipeline with all imbalance mitigation techniques
- Verify metrics stable across multiple runs
- Check cross-validation consistency

### Regression Tests
- Add baseline performance benchmarks
- Alert on significant drops in ROC-AUC or F1

### Smoke Tests
- Keep existing tests in `tests/test_train_smoke.py`
- Add smoke tests for new balancing methods

---

## Part 5: Documentation Improvements

### Create `docs/METHODOLOGY.md`
Document:
- Data sources and preprocessing pipeline
- Feature engineering rationale
- Class imbalance mitigation strategy
- Train/val/test split logic
- Hyperparameter selection process
- Evaluation methodology

### Update `MODEL_CARD.md`
Add:
- **Known limitations:** Class imbalance remediation status
- **Performance metrics:** With confidence intervals and stratified results
- **Fairness considerations:** Subgroup analysis
- **Recommended use cases:** Current proof-of-concept stage
- **Not recommended:** Clinical deployment without external validation

### Create `docs/CLINICAL_NOTES.md`
Add:
- CVD epidemiology context
- Why multi-modal fusion matters clinically
- Interpretation guidelines for predictions
- When to trust vs. question model outputs

---

## Part 6: Immediate Action Items (Next 2 Weeks)

### Week 1
- [ ] Create and run diagnostic notebook (Task 1.1)
- [ ] Create class imbalance tests (Task 1.2)
- [ ] Implement class weights in `src/train.py` (Task 2.1)
- [ ] Document findings in `PROJECT_REVIEW.md` updates

### Week 2
- [ ] Add SMOTE/data balancing module (Task 2.2)
- [ ] Implement focal loss option (Task 2.3)
- [ ] Create comprehensive metrics module (Task 3.1)
- [ ] Add threshold optimization (Task 3.2)
- [ ] Run diagnostic tests and document results

### Week 3-4
- [ ] Implement stratified CV (Task 4.1)
- [ ] Create `docs/METHODOLOGY.md`
- [ ] Update `MODEL_CARD.md`
- [ ] Re-train with all improvements
- [ ] Update results and figures

---

## Part 7: Medical-Grade Readiness Checklist

### Data Quality ✅ Partially
- [x] Data sources documented
- [x] Missing value handling defined
- [ ] Outlier detection and handling
- [ ] Data quality report with statistics
- [ ] Fairness analysis by demographics

### Model Rigor ✅ In Progress
- [x] Baseline comparison
- [x] Feature importance analysis
- [x] Cross-validation framework
- [ ] Uncertainty quantification
- [ ] External validation on new datasets

### Evaluation ✅ Under Improvement
- [x] Multiple metrics (ROC, PR, F1, etc.)
- [x] Confusion matrix analysis
- [ ] Threshold optimization
- [ ] Calibration validation
- [ ] Confidence intervals / bootstrap testing

### Interpretability ✅ Implemented
- [x] SHAP feature importance
- [x] ECG saliency maps
- [x] UI explanations
- [ ] Fairness explanations
- [ ] Uncertainty communication

### Documentation ✅ In Progress
- [x] README with reproduction steps
- [x] Model card skeleton
- [ ] Methods paper ready
- [ ] Clinical validation protocol
- [ ] Deployment checklist

### Deployment Readiness ⚠️ Not Ready
- [x] Docker containerization
- [x] Streamlit UI
- [ ] HIPAA compliance audit
- [ ] FDA 510(k) pathway evaluation
- [ ] Formal validation study protocol

---

## Part 8: Key Metrics to Track

### Model Performance
| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| ROC AUC | > 0.75 | 0.4479 | ❌ Critical |
| F1-Score | > 0.70 | ? | ⚠️ Unknown |
| Sensitivity @ 0.9 specificity | > 0.60 | ? | ⚠️ Unknown |
| Specificity @ 0.9 sensitivity | > 0.60 | ? | ⚠️ Unknown |
| Youden Index | > 0.50 | ? | ⚠️ Unknown |

### Code Quality
- Test coverage: >= 70%
- Documentation coverage: >= 80%
- Type hints: >= 50% of functions
- Linting score (pylint): >= 8.0/10

### Reproducibility
- All results regeneratable from seed
- Dependency versions pinned
- Data preprocessing fully scripted
- Model checkpoints versioned

---

## Part 9: Long-Term Vision (6-12 Months)

### Clinical Validation (Phase 2)
- Prospective evaluation on new patient cohort
- Multi-center data collection
- Subgroup fairness analysis
- External dataset validation (MIMIC-IV, eICU)

### Regulatory Path (Phase 3)
- FDA 510(k) submission for ECG-based device
- CE mark (EU) if applicable
- Clinical trial protocol development
- Risk management (ISO 14971)

### Model Improvements (Ongoing)
- Larger training datasets
- Advanced architectures (Transformers for ECG)
- Multi-task learning (CVD + comorbidities)
- Uncertainty quantification (Bayesian)

### Deployment (Phase 4)
- EHR integration (FHIR APIs)
- Telemedicine platform adoption
- Wearable device support
- Federated learning for privacy

---

## Part 10: Success Criteria

### Short-term (2 months)
- ✅ ROC AUC > 0.70 on test set with class imbalance mitigation
- ✅ F1-Score > 0.65 (balanced metric)
- ✅ All tests pass, including class imbalance validation
- ✅ Diagnostic notebook completed with clear explanation of fixes

### Medium-term (6 months)
- ✅ External validation on 2+ independent datasets
- ✅ ROC AUC > 0.75 across all splits
- ✅ Fairness analysis: performance parity across demographics
- ✅ Publication-ready manuscript drafted

### Long-term (12 months)
- ✅ FDA 510(k) submission initiated
- ✅ Multi-center prospective validation completed
- ✅ Deployed in 2-3 clinical sites
- ✅ Peer-reviewed publication accepted

---

## Conclusion

Your project has **excellent engineering foundations** and **comprehensive documentation** that put it ahead of most academic ML projects. The class imbalance issue, while critical, is **well-understood and solvable** with the strategies outlined here.

**The path forward is clear:** Fix the model validation issues (2-3 weeks), validate improvements with rigorous testing (2 weeks), then begin external validation (2-3 months). With disciplined execution of this plan, your system can transition from a solid academic project to a **clinically credible tool.**

### Next Step
1. Read this document end-to-end
2. Create the diagnostic notebook (Part 1 → Task 1.1)
3. Run tests and assess current state
4. Report back with findings
5. Execute remediation tasks in priority order

**Estimated effort:** 4-6 weeks for core improvements, 6-12 months for medical-grade readiness.

---

**Document maintained by:** Angel Morenu  
**Last updated:** December 2025  
**Version:** 1.0

