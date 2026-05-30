"""Simple ECG augmentation utilities.

These are intentionally lightweight and dependency-free (numpy-only)
so they can be used in data pipelines and for quick dry-runs.
"""
import numpy as np
from typing import Tuple


def random_crop(signal: np.ndarray, target_len: int, rng: np.random.Generator) -> np.ndarray:
    """Randomly crop a 1D signal to target length. If shorter, pad with zeros."""
    if signal.ndim == 2 and signal.shape[0] == 1:
        sig = signal[0]
    else:
        sig = signal
    L = len(sig)
    if L == target_len:
        return sig.copy()
    if L > target_len:
        start = rng.integers(0, L - target_len + 1)
        return sig[start:start + target_len].copy()
    # pad
    pad = target_len - L
    left = pad // 2
    right = pad - left
    return np.pad(sig, (left, right), mode='constant')


def add_noise(signal: np.ndarray, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(scale=noise_std, size=signal.shape)
    return signal + noise


def scale(signal: np.ndarray, scale_min: float, scale_max: float, rng: np.random.Generator) -> np.ndarray:
    factor = rng.uniform(scale_min, scale_max)
    return signal * factor


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, signal: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            signal = t(signal)
        return signal
