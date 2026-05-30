import os
import tempfile

import numpy as np


def test_ecgdataset_basic_and_augment():
    with tempfile.TemporaryDirectory() as tmpdir:
        x = np.random.randn(4, 100).astype("float32")
        y = np.array([0, 1, 0, 1], dtype="int64")
        np.save(os.path.join(tmpdir, "ecg_train.npy"), x)
        np.save(os.path.join(tmpdir, "tabular_train_y.npy"), y)

        from experiments.data import ECGDataset

        ds = ECGDataset(tmpdir)
        assert len(ds) == 4
        sig, label = ds[0]
        assert sig.shape == (1, 100)
        assert label.shape == ()

        augment_params = {
            "random_crop_len": 80,
            "noise_std": 0.1,
            "scale_min": 0.9,
            "scale_max": 1.1,
        }
        augmented = ECGDataset(tmpdir, augment_params=augment_params, base_seed=123)
        aug_sig, _ = augmented[0]

        assert aug_sig.shape == (1, 80)
        assert not np.allclose(sig[:, :80], aug_sig)
