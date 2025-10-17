from __future__ import annotations
import os
from typing import Optional, Union, Dict, Any
import torch

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definitions for the Multi-Modal CVD project.

- ECG1DCNN: 1D conv feature extractor for ECG time series
- FusionNet: Fuses tabular features with ECG embedding for risk prediction
- Utilities for checkpoint save/load and simple predict_proba

Author: Angel Morenu
Course: EEE 6778 – Applied Machine Learning II (Fall 2025)
"""


import torch.nn as nn
import torch.nn.functional as F


# -------------------------
#   ECG Feature Extractor
# -------------------------
class ECG1DCNN(nn.Module):
    """
    Simple 1D CNN for ECG signals.
    Expects input of shape (B, C, T): batch, channels/leads, timepoints.
    If you only have 1 lead, set C=1.

    Output: embedding vector of size ecg_embed_dim.
    """
    def __init__(self, in_channels: int = 1, ecg_embed_dim: int = 128, dropout_p: float = 0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 32, kernel_size=7, stride=1, padding=3)
        self.bn1   = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2   = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm1d(128)
        self.pool  = nn.AdaptiveAvgPool1d(1)
        self.drop  = nn.Dropout(dropout_p)

        # Project to embedding
        self.proj  = nn.Linear(128, ecg_embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=2)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (B, 128)
        x = self.drop(x)

        x = self.proj(x)    # (B, ecg_embed_dim)
        x = F.relu(x)
        return x


# -------------------------
#       Fusion Model
# -------------------------
class FusionNet(nn.Module):
    """
    Fuses tabular features and ECG embeddings.
    - tab_dim: number of tabular features AFTER preprocessing (e.g., ColumnTransformer)
    - ecg_embed_dim: size from ECG1DCNN
    - n_classes: if 1, outputs a single logit (binary). If >1, multi-class logits.
    """
    def __init__(
        self,
        tab_dim: int,
        ecg_embed_dim: int = 128,
        hidden1: int = 256,
        hidden2: int = 128,
        n_classes: int = 2,
        dropout_p: float = 0.2,
    ):
        super().__init__()
        self.tab_dim = tab_dim
        self.ecg_embed_dim = ecg_embed_dim
        self.n_classes = n_classes

        self.mlp = nn.Sequential(
            nn.Linear(tab_dim + ecg_embed_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(hidden2, n_classes),
        )

    def forward(self, tab_x: torch.Tensor, ecg_embed: torch.Tensor) -> torch.Tensor:
        """
        tab_x: (B, tab_dim)
        ecg_embed: (B, ecg_embed_dim)
        returns logits: (B, n_classes) if n_classes>1 else (B, 1)
        """
        fused = torch.cat([tab_x, ecg_embed], dim=-1)
        logits = self.mlp(fused)
        return logits


# -------------------------
#      Full Assembly
# -------------------------
class MultiModalCVD(nn.Module):
    """
    Convenience wrapper that holds:
      - ECG1DCNN feature extractor
      - FusionNet fusion classifier
    """
    def __init__(
        self,
        tab_dim: int,
        ecg_channels: int = 1,
        ecg_embed_dim: int = 128,
        n_classes: int = 2,
        ecg_dropout_p: float = 0.1,
        fusion_dropout_p: float = 0.2,
    ):
        super().__init__()
        self.ecg = ECG1DCNN(in_channels=ecg_channels, ecg_embed_dim=ecg_embed_dim, dropout_p=ecg_dropout_p)
        self.fusion = FusionNet(
            tab_dim=tab_dim,
            ecg_embed_dim=ecg_embed_dim,
            n_classes=n_classes,
            dropout_p=fusion_dropout_p,
        )

    def forward(self, tab_x: torch.Tensor, ecg_signal: torch.Tensor) -> torch.Tensor:
        """
        tab_x: (B, tab_dim)
        ecg_signal: (B, C, T)
        """
        ecg_embed = self.ecg(ecg_signal)
        logits = self.fusion(tab_x, ecg_embed)
        return logits

    @torch.inference_mode()
    def predict_proba(self, tab_x: torch.Tensor, ecg_signal: torch.Tensor) -> torch.Tensor:
        """
        Returns probabilities with shape:
          - (B, 2) for binary (n_classes==1 or 2)
          - (B, n_classes) for multi-class (n_classes>2)
        """
        logits = self.forward(tab_x, ecg_signal)
        out_dim = logits.shape[-1]
        if out_dim == 1:
            p1 = torch.sigmoid(logits.squeeze(-1))
            probs = torch.stack([1 - p1, p1], dim=-1)
        elif out_dim == 2:
            probs = torch.softmax(logits, dim=-1)
        else:
            probs = torch.softmax(logits, dim=-1)
        return probs


# -------------------------
#   Checkpoint Utilities
# -------------------------
def save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "state_dict": model.state_dict(),
        "class": model.__class__.__name__,
    }
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, map_location: Optional[Union[str, torch.device]] = None) -> nn.Module:
    ckpt: Dict[str, Any] = torch.load(path, map_location=map_location or "cpu")
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


# -------------------------
#      Quick Sanity Run
# -------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    # Minimal shape test
    B, tab_dim, C, T = 4, 32, 1, 2000
    model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=C, ecg_embed_dim=128, n_classes=2)

    tab_x = torch.randn(B, tab_dim)
    ecg_x = torch.randn(B, C, T)
    logits = model(tab_x, ecg_x)
    print("Logits shape:", logits.shape)  # (B, 2)

    probs = model.predict_proba(tab_x, ecg_x)
    print("Probs shape:", probs.shape, "sum rows ~1?", probs.sum(dim=-1))