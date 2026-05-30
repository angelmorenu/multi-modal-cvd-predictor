"""Dataset and dataloader helpers for ECG experiments.
"""
from typing import Optional
import os
import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info



class ECGDataset(Dataset):
    """Loads ECG signals and labels from processed .npy arrays.

    Expects signals shaped (N, L) or (N, C, L). Returns (signal, label)
    as np.ndarray where signal is (C, L).
    """

    def __init__(self, processed_dir: str, signals_key: str = 'ecg_train.npy', labels_key: str = 'tabular_train_y.npy', transform=None, augment_params: Optional[dict] = None, base_seed: int = 42):
        self.processed_dir = processed_dir
        self.signals_path = os.path.join(processed_dir, signals_key)
        self.labels_path = os.path.join(processed_dir, labels_key)
        if not os.path.exists(self.signals_path) or not os.path.exists(self.labels_path):
            raise FileNotFoundError(f"Signals or labels not found in {processed_dir}")

        self.X = np.load(self.signals_path)
        self.y = np.load(self.labels_path)
        if self.X.shape[0] != self.y.shape[0]:
            # defensive: align to smallest
            n = min(self.X.shape[0], self.y.shape[0])
            self.X = self.X[:n]
            self.y = self.y[:n]
        self.transform = transform
        # augment_params is a dict with keys (random_crop_len, noise_std, scale_min, scale_max)
        self.augment_params = augment_params
        self.base_seed = base_seed

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sig = self.X[idx]
        # ensure shape (C, L)
        if sig.ndim == 1:
            sig = np.expand_dims(sig, 0)
        elif sig.ndim == 2 and sig.shape[0] != 1 and sig.shape[1] == 1:
            sig = sig.T

        label = self.y[idx]
        # Apply augmentations in a per-worker, per-sample deterministic way.
        if self.augment_params is not None:
            # Get worker id
            worker_info = get_worker_info()
            worker_id = worker_info.id if worker_info is not None else 0
            seed = self.base_seed + (worker_id + 1) * 1000003 + int(idx)
            rng = np.random.default_rng(seed)
            # lazy import to avoid heavy deps at module import time
            try:
                from experiments.augmentations import random_crop, add_noise, scale
            except Exception:
                random_crop = None
                add_noise = None
                scale = None

            ap = self.augment_params
            crop_len = ap.get('random_crop_len')
            noise_std = ap.get('noise_std', 0.0)
            scale_min = ap.get('scale_min', 1.0)
            scale_max = ap.get('scale_max', 1.0)

            if random_crop is not None and crop_len is not None:
                sig = random_crop(sig, int(crop_len), rng)
            if add_noise is not None and noise_std and noise_std > 0.0:
                sig = add_noise(sig, float(noise_std), rng)
            if scale is not None and (scale_min != 1.0 or scale_max != 1.0):
                sig = scale(sig, float(scale_min), float(scale_max), rng)
            # ensure (C, L) shape after augmentations
            if sig.ndim == 1:
                sig = np.expand_dims(sig, 0)

        if self.transform is not None:
            sig = self.transform(sig)

        return sig.astype('float32'), np.array(label).astype('float32')


def collate_fn(batch):
    import torch
    xs, ys = zip(*batch)
    xs = [x if x.ndim == 2 else np.expand_dims(x, 0) for x in xs]
    xs = np.stack(xs, axis=0)  # (B, C, L)
    ys = np.stack(ys, axis=0)
    return torch.from_numpy(xs), torch.from_numpy(ys)
