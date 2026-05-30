import os
import subprocess
import sys
import tempfile

import numpy as np


def test_experiments_train_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        x = np.random.randn(8, 2000).astype("float32")
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype="int64")
        np.save(os.path.join(tmpdir, "ecg_train.npy"), x)
        np.save(os.path.join(tmpdir, "tabular_train_y.npy"), y)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd() + os.pathsep + env.get("PYTHONPATH", "")

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.train",
                "--processed",
                tmpdir,
                "--signals-key",
                "ecg_train.npy",
                "--labels-key",
                "tabular_train_y.npy",
                "--epochs",
                "1",
                "--batch-size",
                "4",
                "--device",
                "cpu",
                "--augment",
                "--random-crop-len",
                "2000",
                "--noise-std",
                "0.01",
                "--scale-min",
                "0.95",
                "--scale-max",
                "1.05",
            ],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert os.path.exists(os.path.join("experiments", "checkpoints", "last.pt"))
