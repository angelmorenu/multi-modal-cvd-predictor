#!/usr/bin/env python3
"""
Loss functions and training utilities for handling class imbalance.

Implementations:
- Weighted Binary Cross-Entropy (class weighting)
- Focal Loss (automatically down-weights easy examples)
- SMOTE preparation for data balancing
- Loss scheduling and dynamic weighting

Author: Angel Morenu
Course: EEE 6778 – Applied Machine Learning II (Fall 2025)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ============================================================================
# Class Weight Computation
# ============================================================================

def compute_class_weight_torch(y: np.ndarray) -> torch.Tensor:
    """
    Compute class weights for imbalanced binary classification.
    
    Uses inverse frequency weighting:
        weight_positive = n_total / (2 * n_positive)
        weight_negative = n_total / (2 * n_negative)
    
    For BCEWithLogitsLoss, we typically want pos_weight = n_negative / n_positive
    
    Args:
        y: Binary labels (0 or 1)
        
    Returns:
        pos_weight: Weight for positive class in BCEWithLogitsLoss
    """
    y = np.asarray(y).astype(int)
    n_positive = max(1, int(y.sum()))
    n_negative = max(1, int(len(y) - y.sum()))
    
    # pos_weight: how much to weight positive class
    # Use: nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    pos_weight = n_negative / n_positive
    
    return torch.tensor([pos_weight], dtype=torch.float32)


def compute_class_weights_sklearn(y: np.ndarray, mode: str = 'balanced') -> Tuple[float, float]:
    """
    Compute class weights using sklearn-style normalization.
    
    Args:
        y: Binary labels
        mode: 'balanced' (default) or 'balanced_subsample'
        
    Returns:
        (weight_negative, weight_positive)
    """
    y = np.asarray(y).astype(int)
    n_samples = len(y)
    n_positive = int(y.sum())
    n_negative = n_samples - n_positive
    
    if mode == 'balanced':
        weight_negative = n_samples / (2.0 * n_negative) if n_negative > 0 else 1.0
        weight_positive = n_samples / (2.0 * n_positive) if n_positive > 0 else 1.0
    else:
        weight_negative = 1.0 / n_negative if n_negative > 0 else 1.0
        weight_positive = 1.0 / n_positive if n_positive > 0 else 1.0
    
    return weight_negative, weight_positive


# ============================================================================
# Loss Functions
# ============================================================================

class WeightedBCEWithLogitsLoss(nn.Module):
    """
    Binary Cross-Entropy with class weighting.
    
    Handles class imbalance by weighting the positive class more heavily.
    
    Args:
        pos_weight: Weight for positive class. Typically n_negative / n_positive
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(self, pos_weight: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.pos_weight = pos_weight
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B,) or (B, 1) raw model outputs
            targets: (B,) or (B, 1) binary labels in {0, 1}
        """
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            pos_weight=torch.tensor([self.pos_weight], device=logits.device),
            reduction=self.reduction
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    
    From: Lin et al. 2017 "Focal Loss for Dense Object Detection"
    
    Focal loss is designed to address class imbalance by down-weighting easy examples
    and focusing training on hard negative examples.
    
    L_focal = -alpha * (1 - p_t)^gamma * log(p_t)
    
    where p_t is the model's estimated probability of the correct class.
    
    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples
               or a list [alpha_negative, alpha_positive].
        gamma: Exponent of the modulating factor (1 - p_t)^gamma to balance easy/hard examples.
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        alpha: Optional[float] = None,
        gamma: float = 2.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B,) or (B, 1) raw model outputs
            targets: (B,) or (B, 1) binary labels in {0, 1}
        """
        # Compute BCE
        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )
        
        # Compute probability
        probs = torch.sigmoid(logits)
        
        # Compute p_t (probability of correct class)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        
        # Focal weighting: (1 - p_t)^gamma
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply focal loss
        loss = focal_weight * bce
        
        # Apply alpha weighting if specified
        if self.alpha is not None:
            alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_weight * loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class CombinedLoss(nn.Module):
    """
    Combines multiple losses for better training.
    
    Example: Focal Loss + L2 regularization
    """
    
    def __init__(self, losses: list, weights: list):
        """
        Args:
            losses: List of loss functions
            weights: List of weights for each loss
        """
        super().__init__()
        assert len(losses) == len(weights)
        self.losses = losses
        self.weights = weights
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        total_loss = 0.0
        for loss_fn, weight in zip(self.losses, self.weights):
            total_loss = total_loss + weight * loss_fn(*args, **kwargs)
        return total_loss


# ============================================================================
# Training Utilities
# ============================================================================

def setup_loss_and_weights(
    y_train: np.ndarray,
    loss_type: str = 'weighted_bce',
    focal_gamma: float = 2.0,
    device: torch.device = torch.device('cpu')
) -> Tuple[nn.Module, dict]:
    """
    Setup loss function and class weights for imbalanced data.
    
    Args:
        y_train: Training labels
        loss_type: 'bce', 'weighted_bce', or 'focal'
        focal_gamma: Gamma parameter for focal loss
        device: Device to create tensors on
        
    Returns:
        (loss_function, info_dict)
    """
    y_train = np.asarray(y_train).astype(int)
    
    # Compute class statistics
    n_positive = int(y_train.sum())
    n_negative = len(y_train) - n_positive
    n_total = len(y_train)
    
    info = {
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_total': n_total,
        'positive_ratio': n_positive / n_total,
        'imbalance_ratio': n_negative / n_positive if n_positive > 0 else float('inf'),
    }
    
    if loss_type == 'bce':
        criterion = nn.BCEWithLogitsLoss()
        info['loss_type'] = 'bce'
        info['pos_weight'] = 1.0
    
    elif loss_type == 'weighted_bce':
        pos_weight = n_negative / n_positive if n_positive > 0 else 1.0
        criterion = WeightedBCEWithLogitsLoss(pos_weight=pos_weight)
        info['loss_type'] = 'weighted_bce'
        info['pos_weight'] = pos_weight
    
    elif loss_type == 'focal':
        # For focal loss, alpha can be set to negative class ratio
        alpha = n_positive / n_total
        criterion = FocalLoss(alpha=alpha, gamma=focal_gamma)
        info['loss_type'] = 'focal'
        info['focal_gamma'] = focal_gamma
        info['alpha'] = alpha
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    return criterion.to(device), info


def check_class_balance(y: np.ndarray, verbose: bool = True) -> dict:
    """
    Check class balance and report statistics.
    
    Args:
        y: Labels
        verbose: Whether to print results
        
    Returns:
        Dictionary with class statistics
    """
    y = np.asarray(y).astype(int)
    
    n_samples = len(y)
    n_positive = int(y.sum())
    n_negative = n_samples - n_positive
    
    pos_ratio = n_positive / n_samples if n_samples > 0 else 0
    neg_ratio = n_negative / n_samples if n_samples > 0 else 0
    imbalance_ratio = n_negative / n_positive if n_positive > 0 else float('inf')
    
    stats = {
        'n_samples': n_samples,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'positive_ratio': pos_ratio,
        'negative_ratio': neg_ratio,
        'imbalance_ratio': imbalance_ratio,
    }
    
    if verbose:
        print(f"Class Balance Statistics:")
        print(f"  Total samples: {n_samples}")
        print(f"  Positive class: {n_positive} ({pos_ratio*100:.1f}%)")
        print(f"  Negative class: {n_negative} ({neg_ratio*100:.1f}%)")
        print(f"  Imbalance ratio (neg/pos): {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 3:
            print(f"  ⚠️  WARNING: Severe class imbalance detected!")
            print(f"      Consider using class weighting, SMOTE, or focal loss")
    
    return stats


# ============================================================================
# SMOTE Preparation (with optional fallback)
# ============================================================================

def prepare_smote_data(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float = 0.5,
    random_state: int = 42,
    use_smote: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare balanced data using SMOTE if available, else use random oversampling.
    
    Args:
        X: Features
        y: Labels
        sampling_strategy: Target ratio of minority to majority class
        random_state: Random seed
        use_smote: Whether to try using SMOTE (falls back to random oversampling)
        
    Returns:
        (X_balanced, y_balanced)
    """
    y = np.asarray(y).astype(int)
    original_counts = np.bincount(y)
    print(f"Original class distribution: {original_counts}")
    
    if use_smote:
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                random_state=random_state,
                k_neighbors=5
            )
            X_balanced, y_balanced = smote.fit_resample(X, y)
            print(f"✓ SMOTE applied successfully")
        except Exception as e:
            print(f"⚠️  SMOTE failed ({e}), falling back to random oversampling")
            X_balanced, y_balanced = _random_oversampling(X, y, sampling_strategy, random_state)
    else:
        X_balanced, y_balanced = _random_oversampling(X, y, sampling_strategy, random_state)
    
    balanced_counts = np.bincount(y_balanced)
    print(f"Balanced class distribution: {balanced_counts}")
    
    return X_balanced, y_balanced


def _random_oversampling(
    X: np.ndarray,
    y: np.ndarray,
    sampling_strategy: float,
    random_state: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Simple random oversampling of minority class."""
    np.random.seed(random_state)
    
    unique_classes = np.unique(y)
    if len(unique_classes) != 2:
        raise ValueError("Only binary classification is supported")
    
    counts = np.bincount(y)
    majority_class = np.argmax(counts)
    minority_class = 1 - majority_class
    
    n_majority = counts[majority_class]
    n_minority = counts[minority_class]
    
    # Target number of minority samples
    n_minority_target = int(n_majority * sampling_strategy)
    n_to_oversample = max(0, n_minority_target - n_minority)
    
    if n_to_oversample == 0:
        return X, y
    
    # Sample from minority class with replacement
    minority_idx = np.where(y == minority_class)[0]
    oversampled_idx = np.random.choice(
        minority_idx,
        size=n_to_oversample,
        replace=True
    )
    
    # Combine original data with oversampled minority
    all_idx = np.concatenate([np.arange(len(X)), oversampled_idx])
    X_balanced = X[all_idx]
    y_balanced = y[all_idx]
    
    return X_balanced, y_balanced


if __name__ == '__main__':
    # Example: Demonstrate class weighting
    y_example = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])  # 80% vs 20%
    
    print("Example 1: Class Weight Computation")
    check_class_balance(y_example)
    
    print("\nExample 2: Setup Losses")
    for loss_type in ['bce', 'weighted_bce', 'focal']:
        criterion, info = setup_loss_and_weights(y_example, loss_type=loss_type)
        print(f"\n{loss_type.upper()}:")
        for k, v in info.items():
            print(f"  {k}: {v}")
    
    print("\nExample 3: SMOTE Preparation")
    X_example = np.random.randn(10, 5)
    X_balanced, y_balanced = prepare_smote_data(X_example, y_example, sampling_strategy=0.5)
    print(f"Original X shape: {X_example.shape}, Balanced X shape: {X_balanced.shape}")
