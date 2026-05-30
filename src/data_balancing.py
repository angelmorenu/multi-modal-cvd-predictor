#!/usr/bin/env python3
"""
Data balancing strategies for imbalanced classification.

Provides multiple techniques to handle class imbalance:
  - Class weighting (inverse frequency)
  - SMOTE (Synthetic Minority Oversampling Technique)
  - Random undersampling
  - Combined strategies

Author: Angel Morenu
Course: EEE 6778 – Applied Machine Learning II (Fall 2025)
"""

import numpy as np
from typing import Tuple, Optional
import warnings

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline as ImbalancePipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    warnings.warn("imblearn not installed. Install with: pip install imbalanced-learn")


def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """
    Compute class weights inversely proportional to class frequency.
    
    Useful for BCE loss: loss_weight = class_weights[target_class]
    
    Args:
        labels: Binary labels (0 or 1)
    
    Returns:
        Array of shape (2,) with weights for class 0 and class 1
    
    Example:
        weights = compute_class_weights(train_labels)  # [0.52, 5.67]
        criterion = nn.BCEWithLogitsLoss(pos_weight=weights[1])
    """
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    
    # Weight inversely proportional to frequency
    # Higher weight for minority class
    weights = 1.0 / (counts + 1e-6)
    
    # Normalize so they don't get too extreme
    # Option 1: Normalize to sum to 2 (for binary classification)
    weights = weights / weights.sum() * 2
    
    return weights


def compute_pos_weight_for_bce(labels: np.ndarray) -> float:
    """
    Compute pos_weight for BCEWithLogitsLoss.
    
    pos_weight = (number of negatives) / (number of positives)
    
    Args:
        labels: Binary labels (0 or 1)
    
    Returns:
        Single value pos_weight
    
    Example:
        pos_weight = compute_pos_weight_for_bce(train_labels)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    """
    labels = np.asarray(labels)
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    
    if n_pos == 0:
        warnings.warn("No positive samples found")
        return 1.0
    
    return float(n_neg / n_pos)


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE (Synthetic Minority Oversampling Technique) to training data.
    
    Creates synthetic samples of minority class by interpolating between
    nearby minority samples. Helps address class imbalance without duplicating.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (0 or 1)
        sampling_strategy: Target ratio minority/majority after SMOTE
                          - 0.5 means minority becomes 50% of majority
                          - 1.0 means perfectly balanced
        random_state: Random seed for reproducibility
    
    Returns:
        (X_resampled, y_resampled): Data after SMOTE
    
    Example:
        X_train_balanced, y_train_balanced = apply_smote(
            X_train, y_train, sampling_strategy=0.7
        )
    
    Note:
        - SMOTE should ONLY be applied to training data, not test data
        - Apply SMOTE AFTER train/test split to avoid data leakage
        - Requires: pip install imbalanced-learn
    """
    if not HAS_IMBLEARN:
        raise ImportError(
            "imblearn is required for SMOTE. Install with: pip install imbalanced-learn"
        )
    
    X = np.asarray(X)
    y = np.asarray(y)
    
    # SMOTE requires at least k_neighbors (default 5) samples of minority class
    minority_count = np.sum(y == np.unique(y)[np.argmin(np.bincount(y))])
    if minority_count < 6:
        warnings.warn(
            f"SMOTE requires at least 6 minority samples, found {minority_count}. "
            "Skipping SMOTE."
        )
        return X, y
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=random_state,
        k_neighbors=min(5, minority_count - 1)
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled


def apply_random_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.5,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply random undersampling to majority class.
    
    Randomly removes majority class samples to balance with minority class.
    Simpler than SMOTE but loses information.
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Binary labels (0 or 1)
        sampling_strategy: Target ratio minority/majority after undersampling
                          - 0.5 means minority becomes 50% of majority
                          - 1.0 means perfectly balanced
        random_state: Random seed
    
    Returns:
        (X_resampled, y_resampled): Data after undersampling
    
    Example:
        X_train_balanced, y_train_balanced = apply_random_undersampling(
            X_train, y_train, sampling_strategy=1.0
        )
    """
    if not HAS_IMBLEARN:
        raise ImportError(
            "imblearn is required. Install with: pip install imbalanced-learn"
        )
    
    undersampler = RandomUnderSampler(
        sampling_strategy=sampling_strategy,
        random_state=random_state
    )
    
    X_resampled, y_resampled = undersampler.fit_resample(X, y)
    
    return X_resampled, y_resampled


def apply_smote_and_undersampling(
    X: np.ndarray,
    y: np.ndarray,
    smote_strategy: float = 0.7,
    undersampling_strategy: float = 1.0,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE followed by random undersampling (combined strategy).
    
    Often works better than either alone:
    1. SMOTE increases minority class
    2. Undersampling reduces majority class
    Result: More balanced dataset with less information loss than undersampling alone
    
    Args:
        X: Feature matrix
        y: Binary labels
        smote_strategy: SMOTE sampling ratio (default 0.7 = 70% of majority after SMOTE)
        undersampling_strategy: Undersampling ratio (default 1.0 = perfect balance)
        random_state: Random seed
    
    Returns:
        (X_resampled, y_resampled): Balanced data
    
    Example:
        X_train_balanced, y_train_balanced = apply_smote_and_undersampling(
            X_train, y_train, smote_strategy=0.6, undersampling_strategy=1.0
        )
    """
    if not HAS_IMBLEARN:
        raise ImportError(
            "imblearn is required. Install with: pip install imbalanced-learn"
        )
    
    pipeline = ImbalancePipeline([
        ('smote', SMOTE(sampling_strategy=smote_strategy, random_state=random_state)),
        ('undersampling', RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=random_state))
    ])
    
    X_resampled, y_resampled = pipeline.fit_resample(X, y)
    
    return X_resampled, y_resampled


def analyze_class_distribution(y: np.ndarray) -> dict:
    """
    Analyze class distribution in dataset.
    
    Args:
        y: Binary labels
    
    Returns:
        Dictionary with distribution statistics
    
    Example:
        stats = analyze_class_distribution(train_labels)
        print(f"Imbalance ratio: {stats['imbalance_ratio']:.2f}")
    """
    y = np.asarray(y)
    unique, counts = np.unique(y, return_counts=True)
    
    if len(unique) != 2:
        raise ValueError("Only binary classification supported")
    
    n_class_0 = counts[0]
    n_class_1 = counts[1]
    n_total = len(y)
    
    # Imbalance ratio: majority / minority
    imbalance_ratio = max(n_class_0, n_class_1) / min(n_class_0, n_class_1)
    
    return {
        'n_class_0': int(n_class_0),
        'n_class_1': int(n_class_1),
        'n_total': int(n_total),
        'ratio_class_0': float(n_class_0 / n_total),
        'ratio_class_1': float(n_class_1 / n_total),
        'imbalance_ratio': float(imbalance_ratio),
        'minority_class': 0 if n_class_0 < n_class_1 else 1,
        'majority_class': 1 if n_class_0 < n_class_1 else 0,
    }


def print_class_distribution(y: np.ndarray, name: str = "Dataset"):
    """Print class distribution summary."""
    stats = analyze_class_distribution(y)
    print(f"\n{name} Class Distribution:")
    print(f"  Class 0: {stats['n_class_0']:5d} ({stats['ratio_class_0']:5.1%})")
    print(f"  Class 1: {stats['n_class_1']:5d} ({stats['ratio_class_1']:5.1%})")
    print(f"  Total:   {stats['n_total']:5d}")
    print(f"  Imbalance Ratio: {stats['imbalance_ratio']:.2f}:1")


def compare_balancing_strategies(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42
) -> dict:
    """
    Compare different balancing strategies.
    
    Args:
        X: Feature matrix
        y: Binary labels
        random_state: Random seed
    
    Returns:
        Dictionary with results for each strategy
    
    Example:
        results = compare_balancing_strategies(X_train, y_train)
        for strategy, (X_bal, y_bal) in results.items():
            print(f"{strategy}: {analyze_class_distribution(y_bal)}")
    """
    results = {}
    
    # Original
    results['original'] = (X, y)
    
    # SMOTE
    if HAS_IMBLEARN:
        try:
            X_smote, y_smote = apply_smote(X, y, sampling_strategy=1.0, random_state=random_state)
            results['smote'] = (X_smote, y_smote)
        except Exception as e:
            print(f"SMOTE failed: {e}")
    
    # Undersampling
    if HAS_IMBLEARN:
        try:
            X_under, y_under = apply_random_undersampling(X, y, sampling_strategy=1.0, random_state=random_state)
            results['undersampling'] = (X_under, y_under)
        except Exception as e:
            print(f"Undersampling failed: {e}")
    
    # Combined
    if HAS_IMBLEARN:
        try:
            X_combined, y_combined = apply_smote_and_undersampling(
                X, y, smote_strategy=0.7, undersampling_strategy=1.0, random_state=random_state
            )
            results['smote_undersample'] = (X_combined, y_combined)
        except Exception as e:
            print(f"Combined strategy failed: {e}")
    
    return results


if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    
    # Create imbalanced dataset: 90% class 0, 10% class 1
    n = 1000
    X = np.random.randn(n, 10)
    y = np.concatenate([np.zeros(900), np.ones(100)])
    
    print("=" * 60)
    print_class_distribution(y, "Original Dataset")
    
    print("\nClass weights for BCE loss:")
    weights = compute_class_weights(y)
    print(f"  Class 0 weight: {weights[0]:.4f}")
    print(f"  Class 1 weight: {weights[1]:.4f}")
    
    print("\nPos weight for BCEWithLogitsLoss:")
    pos_weight = compute_pos_weight_for_bce(y)
    print(f"  pos_weight: {pos_weight:.4f}")
    
    if HAS_IMBLEARN:
        print("\n" + "=" * 60)
        print("Comparing Balancing Strategies:")
        print("=" * 60)
        
        strategies = compare_balancing_strategies(X, y)
        for strategy_name, (X_bal, y_bal) in strategies.items():
            print_class_distribution(y_bal, f"After {strategy_name.upper()}")
