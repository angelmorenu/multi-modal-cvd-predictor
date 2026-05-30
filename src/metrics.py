#!/usr/bin/env python3
"""
Comprehensive metrics module for evaluating imbalanced binary classification.

Provides:
- Bootstrap confidence intervals for all metrics
- Calibration analysis (ECE, MCE)
- ROC/PR curves with threshold analysis
- Threshold optimization utilities
- Clinical evaluation metrics

Author: Angel Morenu
Course: EEE 6778 – Applied Machine Learning II (Fall 2025)
"""

import warnings
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from scipy.stats import bootstrap
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss, confusion_matrix,
    matthews_corrcoef
)
from sklearn.calibration import calibration_curve


# ============================================================================
# Confidence Interval Utilities
# ============================================================================

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn,
    n_bootstraps: int = 1000,
    ci: float = 95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: Binary labels (0 or 1)
        y_pred: Predicted probabilities or scores
        metric_fn: Function that computes metric(y_true, y_pred)
        n_bootstraps: Number of bootstrap samples
        ci: Confidence level (default 95)
        random_state: Random seed
        
    Returns:
        (lower_ci, point_estimate, upper_ci)
    """
    np.random.seed(random_state)
    n = len(y_true)
    metrics = []
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, n, replace=True)
        try:
            m = metric_fn(y_true[idx], y_pred[idx])
            if not np.isnan(m) and not np.isinf(m):
                metrics.append(m)
        except (ValueError, ZeroDivisionError):
            continue
    
    if not metrics:
        return np.nan, np.nan, np.nan
    
    metrics = np.array(metrics)
    point_estimate = np.median(metrics)
    alpha = (100 - ci) / 2
    lower = np.percentile(metrics, alpha)
    upper = np.percentile(metrics, 100 - alpha)
    
    return lower, point_estimate, upper


# ============================================================================
# Threshold Optimization
# ============================================================================

def find_optimal_threshold_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Tuple[float, float]:
    """
    Find threshold that maximizes F1-score.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        
    Returns:
        (optimal_threshold, max_f1_score)
    """
    thresholds = np.sort(np.unique(y_pred))
    f1_scores = []
    
    for thresh in thresholds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        if len(np.unique(y_pred_binary)) < 2:  # Skip if all same class
            continue
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)
        f1_scores.append(f1)
    
    if not f1_scores:
        return 0.5, 0.0
    
    best_idx = np.argmax(f1_scores)
    optimal_thresh = thresholds[best_idx]
    max_f1 = f1_scores[best_idx]
    
    return optimal_thresh, max_f1


def find_optimal_threshold_sensitivity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_sensitivity: float = 0.95
) -> Tuple[float, float, float]:
    """
    Find threshold that achieves target sensitivity while maximizing specificity.
    
    Useful for clinical settings where false negatives are costly.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        target_sensitivity: Target sensitivity (recall for positive class)
        
    Returns:
        (threshold, achieved_sensitivity, achieved_specificity)
    """
    thresholds = np.sort(np.unique(y_pred))[::-1]  # Descending
    
    for thresh in thresholds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if sensitivity >= target_sensitivity:
            return thresh, sensitivity, specificity
    
    # If can't achieve target sensitivity, return lowest threshold
    return thresholds[-1], sensitivity, specificity


def find_optimal_threshold_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_specificity: float = 0.95
) -> Tuple[float, float, float]:
    """
    Find threshold that achieves target specificity while maximizing sensitivity.
    
    Useful when false positives are costly.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        target_specificity: Target specificity
        
    Returns:
        (threshold, achieved_sensitivity, achieved_specificity)
    """
    thresholds = np.sort(np.unique(y_pred))  # Ascending
    
    for thresh in thresholds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        if specificity >= target_specificity:
            return thresh, sensitivity, specificity
    
    # If can't achieve target specificity, return highest threshold
    return thresholds[-1], sensitivity, specificity


# ============================================================================
# Calibration Analysis
# ============================================================================

def compute_ece(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures how well predicted probabilities match true probabilities.
    Lower is better (0 = perfectly calibrated).
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        ECE value (float between 0 and 1)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    ece = np.mean(np.abs(prob_true - prob_pred))
    return ece


def compute_mce(y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum absolute difference between predicted and true probabilities.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        n_bins: Number of bins for calibration curve
        
    Returns:
        MCE value (float between 0 and 1)
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    mce = np.max(np.abs(prob_true - prob_pred))
    return mce


# ============================================================================
# Comprehensive Evaluation
# ============================================================================

def compute_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    threshold: float = 0.5,
    compute_ci: bool = True,
    n_bootstraps: int = 1000
) -> Dict[str, Any]:
    """
    Compute comprehensive binary classification metrics.
    
    Handles class imbalance by returning both accuracy and threshold-robust metrics.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        threshold: Decision threshold
        compute_ci: Whether to compute bootstrap confidence intervals
        n_bootstraps: Number of bootstrap samples for CI
        
    Returns:
        Dictionary with metrics and optional confidence intervals
    """
    y_pred_binary = (y_pred >= threshold).astype(int)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    
    metrics = {
        'threshold': threshold,
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
        
        # Threshold-based metrics
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': f1_score(y_true, y_pred_binary, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred_binary),
        
        # Threshold-independent metrics
        'roc_auc': roc_auc_score(y_true, y_pred) if len(np.unique(y_true)) > 1 else np.nan,
        'pr_auc': average_precision_score(y_true, y_pred),
        'brier': brier_score_loss(y_true, y_pred),
        
        # Calibration
        'ece': compute_ece(y_true, y_pred),
        'mce': compute_mce(y_true, y_pred),
        
        # Confusion matrix
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
    }
    
    # Add confidence intervals if requested
    if compute_ci:
        metrics['ci'] = {
            'accuracy': bootstrap_ci(y_true, y_pred_binary, accuracy_score, n_bootstraps),
            'roc_auc': bootstrap_ci(y_true, y_pred, roc_auc_score, n_bootstraps),
            'pr_auc': bootstrap_ci(y_true, y_pred, average_precision_score, n_bootstraps),
            'f1_score': bootstrap_ci(y_true, y_pred_binary, f1_score, n_bootstraps),
            'sensitivity': bootstrap_ci(y_true, y_pred_binary, 
                                       lambda y, yp: recall_score(y, yp, zero_division=0), 
                                       n_bootstraps),
        }
    
    return metrics


def compute_threshold_curves(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, np.ndarray]:
    """
    Compute ROC and PR curves across all thresholds.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        
    Returns:
        Dictionary with 'fpr', 'tpr', 'pr_precision', 'pr_recall', 'thresholds'
    """
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
    
    return {
        'fpr': fpr,
        'tpr': tpr,
        'pr_precision': precision,
        'pr_recall': recall,
        'roc_thresholds': roc_thresholds,
        'pr_thresholds': pr_thresholds,
        'roc_auc': auc(fpr, tpr),
    }


# ============================================================================
# Fairness Analysis (Basic)
# ============================================================================

def compute_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    subgroup_labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics separately for different subgroups.
    
    Useful for detecting fairness issues across demographic groups.
    
    Args:
        y_true: Binary labels
        y_pred: Predicted probabilities
        subgroup_labels: Group assignments (e.g., 0=female, 1=male)
        threshold: Decision threshold
        
    Returns:
        Dictionary with metrics for each subgroup
    """
    unique_groups = np.unique(subgroup_labels)
    results = {}
    
    for group in unique_groups:
        mask = subgroup_labels == group
        if mask.sum() < 10:  # Skip very small groups
            continue
        
        y_true_g = y_true[mask]
        y_pred_g = y_pred[mask]
        
        if len(np.unique(y_true_g)) < 2:  # Skip if only one class present
            continue
        
        y_pred_binary = (y_pred_g >= threshold).astype(int)
        
        group_name = str(group)
        results[group_name] = {
            'n': int(mask.sum()),
            'accuracy': accuracy_score(y_true_g, y_pred_binary),
            'roc_auc': roc_auc_score(y_true_g, y_pred_g),
            'sensitivity': recall_score(y_true_g, y_pred_binary, zero_division=0),
            'specificity': 1 - recall_score(~y_true_g.astype(bool), 
                                           y_pred_binary, zero_division=0),
            'positive_rate': y_true_g.mean(),
        }
    
    return results


if __name__ == '__main__':
    # Example usage
    from sklearn.datasets import make_classification
    
    X, y = make_classification(n_samples=1000, n_features=20, weights=[0.85, 0.15], random_state=42)
    y_pred = np.random.rand(1000)  # Replace with actual predictions
    
    # Compute metrics at default threshold
    metrics = compute_binary_metrics(y, y_pred)
    print("Metrics at threshold=0.5:")
    for k, v in metrics.items():
        if k != 'ci':
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    # Find optimal threshold
    opt_thresh, opt_f1 = find_optimal_threshold_f1(y, y_pred)
    print(f"\nOptimal F1 threshold: {opt_thresh:.4f} (F1={opt_f1:.4f})")
    
    # Find threshold for clinical sensitivity
    sens_thresh, sens, spec = find_optimal_threshold_sensitivity(y, y_pred, 0.95)
    print(f"Threshold for 95% sensitivity: {sens_thresh:.4f}")
    print(f"  Sensitivity: {sens:.4f}, Specificity: {spec:.4f}")
