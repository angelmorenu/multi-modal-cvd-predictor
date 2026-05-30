import tempfile
import numpy as np
import os


def test_ecgdataset_basic_and_augment():
    # create temporary processed dir
    with tempfile.TemporaryDirectory() as tmpdir:
        # write small synthetic data: 4 samples, length 100
        X = np.random.randn(4, 100).astype('float32')
        y = np.array([0, 1, 0, 1], dtype='int64')
        np.save(os.path.join(tmpdir, 'ecg_train.npy'), X)
        np.save(os.path.join(tmpdir, 'tabular_train_y.npy'), y)

        from experiments.data import ECGDataset

        # without augment
        ds = ECGDataset(tmpdir)
        assert len(ds) == 4
        sig, label = ds[0]
        assert sig.shape[0] == 1
        assert sig.shape[1] == 100

        # with augment params: noise should change values
        ap = {'random_crop_len': 80, 'noise_std': 0.1, 'scale_min': 0.9, 'scale_max': 1.1}
        ds2 = ECGDataset(tmpdir, augment_params=ap, base_seed=123)
        s0, _ = ds[0]
        s1, _ = ds2[0]
        # shapes (C,L) -> 1,80
        assert s1.shape[0] == 1
        assert s1.shape[1] == 80
        # values should differ due to augmentation
        assert not np.allclose(s0[:, :80], s1)
