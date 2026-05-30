#!/usr/bin/env python3
"""
Comprehensive evaluation metrics module for imbalanced classification.

Provides threshold-aware metrics, threshold optimization, calibration analysis,
and visualization-ready outputs suitable for medical applications.

Author: Angel Morenu
Course: EEE 6778 – Applied Machine Learning II (Fall 2025)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score, specificity_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, calibration_curve
)


class ThresholdAnalyzer:
    """Analyze model performance across thresholds."""
    
    def __init__(self, y_true: np.ndarray, y_pred_probs: np.ndarray):
        """
        Initialize with true labels and predicted probabilities.
        
        Args:
            y_true: Binary labels (0 or 1)
            y_pred_probs: Predicted probabilities for positive class [0, 1]
        """
        self.y_true = np.asarray(y_true)
        self.y_pred_probs = np.asarray(y_pred_probs)
        self._validate()
    
    def _validate(self):
        """Validate inputs."""
        assert len(self.y_true) == len(self.y_pred_probs), "Length mismatch"
        assert np.all((self.y_true == 0) | (self.y_true == 1)), "y_true must be binary"
        assert np.all((self.y_pred_probs >= 0) & (self.y_pred_probs <= 1)), "Probs must be in [0, 1]"
    
    def evaluate_at_thresholds(
        self, 
        thresholds: Optional[np.ndarray] = None,
        metrics: Optional[list] = None
    ) -> pd.DataFrame:
        """
        Evaluate model at multiple thresholds.
        
        Args:
            thresholds: Array of thresholds (default: 0.01 to 0.99 by 0.01)
            metrics: List of metrics to compute (default: all)
        
        Returns:
            DataFrame with metrics for each threshold
        """
        if thresholds is None:
            thresholds = np.arange(0.01, 1.0, 0.01)
        
        if metrics is None:
            metrics = ['accuracy', 'sensitivity', 'specificity', 'precision', 
                      'f1', 'youden', 'mcc', 'roc_auc', 'pr_auc']
        
        results = []
        
        for thresh in thresholds:
            y_pred = (self.y_pred_probs >= thresh).astype(int)
            
            row = {'threshold': thresh}
            
            # Threshold-dependent metrics
            if 'accuracy' in metrics:
                row['accuracy'] = accuracy_score(self.y_true, y_pred)
            
            if 'sensitivity' in metrics or 'recall' in metrics:
                row['sensitivity'] = recall_score(self.y_true, y_pred, zero_division=0)
            
            if 'specificity' in metrics:
                tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
                row['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            if 'precision' in metrics:
                row['precision'] = precision_score(self.y_true, y_pred, zero_division=0)
            
            if 'f1' in metrics:
                row['f1'] = f1_score(self.y_true, y_pred, zero_division=0)
            
            if 'youden' in metrics:
                tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                row['youden'] = sens + spec - 1
            
            if 'mcc' in metrics:
                tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
                denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
                if denom > 0:
                    row['mcc'] = (tp * tn - fp * fn) / np.sqrt(denom)
                else:
                    row['mcc'] = 0
            
            results.append(row)
        
        df = pd.DataFrame(results)
        
        # Add threshold-independent metrics (computed once, repeated)
        if 'roc_auc' in metrics:
            try:
                roc = roc_auc_score(self.y_true, self.y_pred_probs)
                df['roc_auc'] = roc
            except:
                df['roc_auc'] = np.nan
        
        if 'pr_auc' in metrics:
            try:
                pr = average_precision_score(self.y_true, self.y_pred_probs)
                df['pr_auc'] = pr
            except:
                df['pr_auc'] = np.nan
        
        return df
    
    def find_optimal_threshold(
        self, 
        metric: str = 'f1',
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Find threshold that maximizes specified metric.
        
        Args:
            metric: 'f1', 'youden', 'mcc', 'precision', 'recall', or 'accuracy'
            thresholds: Custom threshold range
        
        Returns:
            (optimal_threshold, metric_value)
        """
        df = self.evaluate_at_thresholds(thresholds=thresholds, metrics=[metric])
        
        best_idx = df[metric].idxmax()
        best_thresh = df.loc[best_idx, 'threshold']
        best_value = df.loc[best_idx, metric]
        
        return best_thresh, best_value
    
    def get_confusion_matrix_components(self, threshold: float = 0.5) -> Dict[str, int]:
        """Get TP, TN, FP, FN for given threshold."""
        y_pred = (self.y_pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        return {'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)}


class CalibrationAnalyzer:
    """Analyze and improve model calibration."""
    
    def __init__(self, y_true: np.ndarray, y_pred_probs: np.ndarray):
        """Initialize with true labels and predicted probabilities."""
        self.y_true = np.asarray(y_true)
        self.y_pred_probs = np.asarray(y_pred_probs)
    
    def expected_calibration_error(self, n_bins: int = 10) -> float:
        """
        Compute Expected Calibration Error (ECE).
        
        ECE measures average difference between predicted confidence and actual accuracy.
        Lower is better. ECE = 0 means perfectly calibrated.
        
        Args:
            n_bins: Number of bins for ECE computation
        
        Returns:
            ECE value
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        ece = 0
        
        for i in range(n_bins):
            mask = (self.y_pred_probs >= bin_edges[i]) & (self.y_pred_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                confidence = self.y_pred_probs[mask].mean()
                accuracy = self.y_true[mask].mean()
                ece += np.abs(confidence - accuracy) * mask.sum() / len(self.y_true)
        
        return ece
    
    def max_calibration_error(self, n_bins: int = 10) -> float:
        """
        Maximum Calibration Error (MCE).
        
        MCE is the maximum absolute difference between confidence and accuracy
        across all bins. Useful for identifying worst-case calibration issues.
        """
        bin_edges = np.linspace(0, 1, n_bins + 1)
        mce = 0
        
        for i in range(n_bins):
            mask = (self.y_pred_probs >= bin_edges[i]) & (self.y_pred_probs < bin_edges[i + 1])
            if mask.sum() > 0:
                confidence = self.y_pred_probs[mask].mean()
                accuracy = self.y_true[mask].mean()
                mce = max(mce, np.abs(confidence - accuracy))
        
        return mce
    
    def calibration_data_for_plot(self, n_bins: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibration curve data for plotting.
        
        Returns:
            (mean_confidences, mean_accuracies)
        """
        prob_true, prob_pred = calibration_curve(
            self.y_true, self.y_pred_probs, n_bins=n_bins, strategy='uniform'
        )
        return prob_pred, prob_true
    
    def brier_score(self) -> float:
        """
        Brier Score: Mean squared difference between predicted probability and actual label.
        
        Lower is better. Range: [0, 1]. BS = 0 is perfect, BS = 0.25 is random for balanced data.
        """
        return brier_score_loss(self.y_true, self.y_pred_probs)


class ComprehensiveEvaluator:
    """Comprehensive evaluation combining all metrics."""
    
    def __init__(self, y_true: np.ndarray, y_pred_probs: np.ndarray):
        self.y_true = np.asarray(y_true)
        self.y_pred_probs = np.asarray(y_pred_probs)
        self.threshold_analyzer = ThresholdAnalyzer(y_true, y_pred_probs)
        self.calibration_analyzer = CalibrationAnalyzer(y_true, y_pred_probs)
    
    def evaluate_comprehensive(
        self, 
        threshold: float = 0.5,
        n_bins: int = 10
    ) -> Dict:
        """
        Comprehensive evaluation report.
        
        Args:
            threshold: Decision threshold to use for threshold-dependent metrics
            n_bins: Number of bins for calibration metrics
        
        Returns:
            Dictionary with all metrics and analysis
        """
        y_pred = (self.y_pred_probs >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(self.y_true, y_pred).ravel()
        
        # Threshold-dependent metrics
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0  # precision
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # negative predictive value
        
        youden = sensitivity + specificity - 1
        
        # Threshold-independent metrics
        try:
            roc_auc = roc_auc_score(self.y_true, self.y_pred_probs)
        except:
            roc_auc = np.nan
        
        try:
            pr_auc = average_precision_score(self.y_true, self.y_pred_probs)
        except:
            pr_auc = np.nan
        
        # Calibration metrics
        ece = self.calibration_analyzer.expected_calibration_error(n_bins)
        mce = self.calibration_analyzer.max_calibration_error(n_bins)
        brier = self.calibration_analyzer.brier_score()
        
        # Threshold optimization
        optimal_thresh_f1, optimal_f1 = self.threshold_analyzer.find_optimal_threshold('f1')
        optimal_thresh_youden, optimal_youden = self.threshold_analyzer.find_optimal_threshold('youden')
        
        return {
            'threshold': threshold,
            'n_samples': len(self.y_true),
            'n_positive': int(np.sum(self.y_true)),
            'n_negative': int(len(self.y_true) - np.sum(self.y_true)),
            'positive_rate': float(np.sum(self.y_true) / len(self.y_true)),
            
            # Threshold-dependent metrics at specified threshold
            'accuracy': float(accuracy_score(self.y_true, y_pred)),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(ppv),
            'npv': float(npv),
            'f1': float(f1_score(self.y_true, y_pred, zero_division=0)),
            'youden': float(youden),
            'mcc': float(self._mcc(tp, tn, fp, fn)),
            
            # Confusion matrix
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            
            # Threshold-independent metrics
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            
            # Calibration metrics
            'ece': float(ece),
            'mce': float(mce),
            'brier': float(brier),
            
            # Optimal thresholds
            'optimal_threshold_f1': float(optimal_thresh_f1),
            'optimal_f1': float(optimal_f1),
            'optimal_threshold_youden': float(optimal_thresh_youden),
            'optimal_youden': float(optimal_youden),
            
            # Probability statistics
            'prob_mean': float(self.y_pred_probs.mean()),
            'prob_std': float(self.y_pred_probs.std()),
            'prob_min': float(self.y_pred_probs.min()),
            'prob_max': float(self.y_pred_probs.max()),
        }
    
    @staticmethod
    def _mcc(tp, tn, fp, fn):
        """Matthews Correlation Coefficient."""
        denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        if denom == 0:
            return 0
        return (tp * tn - fp * fn) / np.sqrt(denom)
    
    def threshold_analysis_dataframe(self) -> pd.DataFrame:
        """Get full threshold analysis as DataFrame."""
        return self.threshold_analyzer.evaluate_at_thresholds()
    
    def summary_table(self) -> pd.DataFrame:
        """
        Create a summary table suitable for publication.
        
        Returns:
            DataFrame with key metrics
        """
        results = self.evaluate_comprehensive()
        
        summary = {
            'Metric': [
                'N samples',
                'Positive rate',
                'ROC AUC',
                'PR AUC',
                'Sensitivity',
                'Specificity',
                'Precision',
                'F1-Score',
                'Youden Index',
                'ECE',
                'Brier Score',
            ],
            'Value': [
                f"{results['n_samples']}",
                f"{results['positive_rate']:.1%}",
                f"{results['roc_auc']:.4f}",
                f"{results['pr_auc']:.4f}",
                f"{results['sensitivity']:.4f}",
                f"{results['specificity']:.4f}",
                f"{results['precision']:.4f}",
                f"{results['f1']:.4f}",
                f"{results['youden']:.4f}",
                f"{results['ece']:.4f}",
                f"{results['brier']:.4f}",
            ]
        }
        
        return pd.DataFrame(summary)


def evaluate_and_optimize(
    y_true: np.ndarray,
    y_pred_probs: np.ndarray,
    verbose: bool = True
) -> Dict:
    """
    Convenience function: evaluate and find optimal threshold.
    
    Args:
        y_true: Binary labels
        y_pred_probs: Predicted probabilities
        verbose: Print results
    
    Returns:
        Comprehensive evaluation results
    """
    evaluator = ComprehensiveEvaluator(y_true, y_pred_probs)
    
    # Evaluate at default threshold
    results_default = evaluator.evaluate_comprehensive(threshold=0.5)
    
    # Find optimal thresholds
    optimal_f1_thresh, optimal_f1 = evaluator.threshold_analyzer.find_optimal_threshold('f1')
    optimal_youden_thresh, optimal_youden = evaluator.threshold_analyzer.find_optimal_threshold('youden')
    
    # Re-evaluate at optimal thresholds
    results_f1 = evaluator.evaluate_comprehensive(threshold=optimal_f1_thresh)
    results_youden = evaluator.evaluate_comprehensive(threshold=optimal_youden_thresh)
    
    output = {
        'threshold_0.5': results_default,
        'threshold_optimal_f1': results_f1,
        'threshold_optimal_youden': results_youden,
    }
    
    if verbose:
        print("=" * 80)
        print("COMPREHENSIVE EVALUATION REPORT")
        print("=" * 80)
        print(f"\nDataset: {results_default['n_samples']} samples, "
              f"{results_default['positive_rate']:.1%} positive")
        
        print("\n--- Default Threshold (0.5) ---")
        print(evaluator.summary_table().to_string(index=False))
        
        print(f"\n--- Optimal Threshold (F1): {optimal_f1_thresh:.3f} (F1={optimal_f1:.4f}) ---")
        results_table_f1 = ComprehensiveEvaluator(y_true, y_pred_probs).evaluate_comprehensive(threshold=optimal_f1_thresh)
        print(f"Sensitivity: {results_table_f1['sensitivity']:.4f}, "
              f"Specificity: {results_table_f1['specificity']:.4f}")
        
        print(f"\n--- Optimal Threshold (Youden): {optimal_youden_thresh:.3f} (Youden={optimal_youden:.4f}) ---")
        results_table_youden = ComprehensiveEvaluator(y_true, y_pred_probs).evaluate_comprehensive(threshold=optimal_youden_thresh)
        print(f"Sensitivity: {results_table_youden['sensitivity']:.4f}, "
              f"Specificity: {results_table_youden['specificity']:.4f}")
        print("=" * 80)
    
    return output


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.2, n)  # 20% positive
    y_pred_probs = np.where(
        y_true == 1,
        np.random.normal(0.75, 0.15, n),
        np.random.normal(0.25, 0.15, n)
    )
    y_pred_probs = np.clip(y_pred_probs, 0, 1)
    
    results = evaluate_and_optimize(y_true, y_pred_probs, verbose=True)
