import os
import subprocess
import sys
import tempfile

import numpy as np


def test_weighted_loss_and_uncertainty_smoke():
    with tempfile.TemporaryDirectory() as tmpdir:
        x = np.random.randn(10, 2000).astype("float32")
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype="int64")
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
                "--loss",
                "weighted_bce",
                "--dropout-p",
                "0.2",
                "--mc-samples",
                "2",
                "--save-uncertainty",
            ],
            cwd=os.getcwd(),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

        assert result.returncode == 0, result.stdout + "\n" + result.stderr
        assert os.path.exists(os.path.join("experiments", "checkpoints", "last.pt"))
        assert os.path.exists(os.path.join("experiments", "checkpoints", "last_metrics.json"))
