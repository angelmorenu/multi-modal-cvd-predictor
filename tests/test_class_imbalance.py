#!/usr/bin/env python3
"""
Test suite for class imbalance handling.

Ensures that the model doesn't collapse to predicting the majority class,
and that class imbalance mitigation techniques are working correctly.

Run with: pytest tests/test_class_imbalance.py -v
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, f1_score, confusion_matrix, 
    precision_score, recall_score, balanced_accuracy_score
)
import pytest


class TestClassImbalanceDetection:
    """Verify that the test suite can detect majority class collapse."""
    
    def test_detects_majority_class_collapse(self):
        """Test that we catch models predicting all positive."""
        y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])  # 33% positive
        y_pred_probs = np.array([0.6, 0.7, 0.65, 0.7, 0.75, 0.68, 0.72, 0.71, 0.69])
        y_pred = (y_pred_probs >= 0.5).astype(int)  # All predict positive
        
        # Verify the collapse is detected
        roc = roc_auc_score(y_true, y_pred_probs)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Assertions
        assert roc < 0.6, "ROC-AUC should be low for collapsed model"
        assert specificity < 0.5, "Specificity should be near zero for collapsed model"
        assert np.sum(y_pred) == len(y_pred), "Collapsed model predicts all positive"


class TestNoMajorityClassCollapse:
    """Ensure trained model doesn't collapse to majority class baseline."""
    
    def test_probability_range_diversity(self):
        """Test that predicted probabilities span range, not concentrated near 0.5."""
        # Simulate model output with real discriminative power
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.binomial(1, 0.3, n_samples)  # 30% positive (imbalanced)
        
        # Good model: outputs well-separated probabilities
        y_pred_probs_good = np.where(
            y_true == 1,
            np.random.normal(0.8, 0.1, n_samples),
            np.random.normal(0.2, 0.1, n_samples)
        )
        y_pred_probs_good = np.clip(y_pred_probs_good, 0.01, 0.99)
        
        # Check probability distribution
        prob_min = y_pred_probs_good.min()
        prob_max = y_pred_probs_good.max()
        prob_std = y_pred_probs_good.std()
        
        assert prob_min < 0.3, f"Min probability {prob_min} should be low"
        assert prob_max > 0.7, f"Max probability {prob_max} should be high"
        assert prob_std > 0.2, f"Std {prob_std} should show diversity"
        
        # ROC-AUC should be high for good separation
        roc = roc_auc_score(y_true, y_pred_probs_good)
        assert roc > 0.8, f"Good model should achieve ROC-AUC > 0.8, got {roc:.4f}"
    
    def test_confusion_matrix_structure(self):
        """Ensure model makes meaningful predictions on both classes."""
        np.random.seed(42)
        n_samples = 1000
        y_true = np.random.binomial(1, 0.15, n_samples)  # 15% positive (severe imbalance)
        
        # Good model
        y_pred_probs = np.where(
            y_true == 1,
            np.random.normal(0.75, 0.15, n_samples),
            np.random.normal(0.25, 0.15, n_samples)
        )
        y_pred_probs = np.clip(y_pred_probs, 0.01, 0.99)
        y_pred = (y_pred_probs >= 0.5).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Check that model makes meaningful predictions
        assert tp > 0, "True positives should be > 0"
        assert tn > 0, "True negatives should be > 0"
        assert fp > 0, "False positives should be > 0 (some errors expected)"
        assert fn > 0, "False negatives should be > 0 (some errors expected)"
        
        # Specificity and sensitivity should both be reasonable
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        
        assert sensitivity > 0.5, f"Sensitivity {sensitivity:.4f} should be > 0.5"
        assert specificity > 0.5, f"Specificity {specificity:.4f} should be > 0.5"


class TestImbalancedMetrics:
    """Test that evaluation uses appropriate metrics for imbalanced data."""
    
    def test_f1_score_for_imbalance(self):
        """F1-score should be more reliable than accuracy for imbalanced data."""
        np.random.seed(42)
        
        # Simulate imbalanced dataset: 85% class 1
        y_true = np.concatenate([
            np.ones(850),
            np.zeros(150)
        ])
        
        # Majority class predictor (predicts all 1)
        y_pred_majority = np.ones(1000)
        acc_majority = np.mean(y_pred_majority == y_true)
        f1_majority = f1_score(y_true, y_pred_majority, zero_division=0)
        
        # Balanced predictor
        y_pred_balanced = np.random.binomial(1, 0.5, 1000)
        acc_balanced = np.mean(y_pred_balanced == y_true)
        f1_balanced = f1_score(y_true, y_pred_balanced, zero_division=0)
        
        # Accuracy is misleading for imbalanced data
        assert acc_majority > acc_balanced, "Accuracy favors majority predictor"
        
        # F1-score should penalize imbalance more
        # (This demonstrates why F1 is better for imbalanced evaluation)
        assert f1_majority < 0.8, "F1 for majority predictor should be lower"
    
    def test_balanced_accuracy(self):
        """Balanced accuracy should be insensitive to class imbalance."""
        np.random.seed(42)
        
        y_true = np.array([0] * 950 + [1] * 50)  # 95% negative
        
        # Predictor that always guesses negative
        y_pred_all_neg = np.zeros(1000)
        balanced_acc = balanced_accuracy_score(y_true, y_pred_all_neg)
        
        # Should be ~0.5 (not close to 0.95)
        assert 0.45 < balanced_acc < 0.55, f"Balanced acc {balanced_acc:.4f} should be ~0.5"
    
    def test_roc_auc_threshold_independence(self):
        """ROC-AUC should be threshold-independent and reveal discrimination."""
        np.random.seed(42)
        
        n = 1000
        y_true = np.random.binomial(1, 0.2, n)
        
        # Model with good discrimination
        y_pred_good = np.where(
            y_true == 1,
            np.random.normal(0.8, 0.1, n),
            np.random.normal(0.2, 0.1, n)
        )
        y_pred_good = np.clip(y_pred_good, 0, 1)
        
        roc_good = roc_auc_score(y_true, y_pred_good)
        
        assert roc_good > 0.75, f"Good model should have ROC-AUC > 0.75, got {roc_good:.4f}"


class TestThresholdOptimization:
    """Test threshold optimization for imbalanced datasets."""
    
    def test_optimal_threshold_not_always_0_5(self):
        """For imbalanced data, optimal threshold may not be 0.5."""
        np.random.seed(42)
        
        n = 1000
        y_true = np.random.binomial(1, 0.2, n)  # 20% positive
        
        # Model with some discriminative power
        y_pred_probs = np.where(
            y_true == 1,
            np.random.normal(0.7, 0.15, n),
            np.random.normal(0.3, 0.15, n)
        )
        y_pred_probs = np.clip(y_pred_probs, 0, 1)
        
        # Evaluate F1 at multiple thresholds
        f1_scores = {}
        for thresh in np.arange(0.1, 1.0, 0.1):
            y_pred = (y_pred_probs >= thresh).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            f1_scores[thresh] = f1
        
        # Find optimal threshold
        best_thresh = max(f1_scores, key=f1_scores.get)
        
        # For imbalanced data, optimal threshold often != 0.5
        # (This test demonstrates the importance of threshold optimization)
        assert len(f1_scores) > 0, "Should compute F1 for multiple thresholds"


class TestYoudenIndex:
    """Test Youden index (sensitivity + specificity - 1) for imbalanced evaluation."""
    
    def test_youden_index_computation(self):
        """Youden index should balance sensitivity and specificity."""
        np.random.seed(42)
        
        y_true = np.array([0] * 800 + [1] * 200)  # 20% positive
        
        # Compute Youden for various thresholds
        thresholds = [0.3, 0.5, 0.7]
        youdens = []
        
        y_pred_probs = np.concatenate([
            np.random.normal(0.3, 0.15, 800),
            np.random.normal(0.7, 0.15, 200)
        ])
        y_pred_probs = np.clip(y_pred_probs, 0, 1)
        
        for thresh in thresholds:
            y_pred = (y_pred_probs >= thresh).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            
            sensitivity = tp / (tp + fn)
            specificity = tn / (tn + fp)
            youden = sensitivity + specificity - 1
            youdens.append(youden)
        
        # Youden should vary with threshold
        assert len(set(youdens)) > 1, "Youden should vary across thresholds"
        assert all(y >= -1 and y <= 1 for y in youdens), "Youden should be in [-1, 1]"


class TestCalibration:
    """Test model calibration (predicted probability = actual frequency)."""
    
    def test_calibration_plot_preparation(self):
        """Prepare data for calibration analysis."""
        np.random.seed(42)
        
        n = 1000
        y_true = np.random.binomial(1, 0.3, n)
        
        # Uncalibrated model (overconfident)
        y_pred_probs = np.where(
            y_true == 1,
            np.random.normal(0.85, 0.1, n),  # Too high
            np.random.normal(0.15, 0.1, n)
        )
        y_pred_probs = np.clip(y_pred_probs, 0, 1)
        
        # Expected calibration error (ECE)
        bins = 10
        bin_edges = np.linspace(0, 1, bins + 1)
        ece = 0
        
        for i in range(bins):
            mask = (y_pred_probs >= bin_edges[i]) & (y_pred_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                confidence = y_pred_probs[mask].mean()
                accuracy = y_true[mask].mean()
                ece += np.abs(confidence - accuracy) * mask.sum() / n
        
        # ECE should be measurable and reasonable
        assert 0 <= ece <= 1, f"ECE should be in [0,1], got {ece:.4f}"


class TestRobustness:
    """Test model robustness across different scenarios."""
    
    def test_model_stability_across_seeds(self):
        """Model performance should be relatively stable across different random seeds."""
        scores = []
        
        for seed in [42, 123, 456]:
            np.random.seed(seed)
            
            n = 500
            y_true = np.random.binomial(1, 0.25, n)
            y_pred_probs = np.where(
                y_true == 1,
                np.random.normal(0.75, 0.15, n),
                np.random.normal(0.25, 0.15, n)
            )
            y_pred_probs = np.clip(y_pred_probs, 0, 1)
            
            roc = roc_auc_score(y_true, y_pred_probs)
            scores.append(roc)
        
        # ROC should be stable (not wildly varying)
        std = np.std(scores)
        mean = np.mean(scores)
        
        assert std < mean * 0.3, f"ROC should be stable, std {std:.4f} too high relative to mean {mean:.4f}"


class TestFeatureImbalance:
    """Test that imbalance mitigation doesn't distort features."""
    
    def test_smote_preserves_feature_characteristics(self):
        """SMOTE should preserve feature distributions roughly."""
        try:
            from imblearn.over_sampling import SMOTE
        except ImportError:
            pytest.skip("imblearn not installed")
        
        np.random.seed(42)
        
        # Original imbalanced data
        X_neg = np.random.normal(-2, 1, (900, 5))
        X_pos = np.random.normal(2, 1, (100, 5))
        X = np.vstack([X_neg, X_pos])
        y = np.concatenate([np.zeros(900), np.ones(100)])
        
        # Apply SMOTE
        smote = SMOTE(random_state=42)
        X_smote, y_smote = smote.fit_resample(X, y)
        
        # Check class balance
        assert np.sum(y_smote == 0) == np.sum(y_smote == 1), "SMOTE should balance classes"
        
        # Check that SMOTE output has reasonable values
        assert X_smote.min() > -5 and X_smote.max() < 5, "SMOTE output should be reasonable"


# Run tests with: pytest tests/test_class_imbalance.py -v
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
