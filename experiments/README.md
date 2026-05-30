Experiments scaffold for stronger ECG models
=========================================

This folder contains a lightweight experiment scaffold to run ECG-only experiments.

Files added:
- `example_config.yaml` : example hyperparameters and paths
- `augmentations.py` : simple ECG augmentation utilities (crop, noise, scale)
- `models/resnet1d.py` : small ResNet1D model builder
- `run_experiment.py` : minimal runner supporting a dry-run shape check

Usage (quick dry-run):

```bash
# check shapes and config without training
.venv/bin/python experiments/run_experiment.py --config experiments/example_config.yaml --dry-run
```

Next steps:
- Add training loop, dataset loaders, logging (TensorBoard/weights & biases), and optional Optuna tuning.
