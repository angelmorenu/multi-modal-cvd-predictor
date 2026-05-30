# Recommendations Report

## Scope

This report summarizes the current state of the repository and prioritizes the next steps that would improve scientific validity, reproducibility, and deployment readiness.

## Key Findings

1. **The experiment pipeline is now working end-to-end.**
   - Smoke training succeeds.
   - Per-worker augmentation randomness is handled inside the dataset.
   - Validation metrics, best-checkpoint saving, TensorBoard logging, and uncertainty outputs are available.

2. **The current fusion artifact is not yet the strongest performer on the small processed validation set.**
   - The saved smoke evaluation in `results/metric_summary.json` shows the tabular baseline outperforming the ECG and fusion artifacts on the current held-out sample.
   - This suggests the main value today is in the pipeline, not in a deployable model.

3. **The repository now supports more rigorous evaluation.**
   - Repeated stratified CV with bootstrap confidence intervals is implemented.
   - DeLong-style comparison is available for pooled predictions.
   - External ECG preparation / evaluation is scaffolded for PTBDB and CPSC-style sets.

4. **Reproducibility is now substantially better.**
   - `requirements.txt` is pinned.
   - `Dockerfile` and `Makefile` enable reproducible local runs.
   - Unit tests and CI cover the new experiment and smoke-test paths.

## Priority Recommendations

### 1) Continue with external validation and provenance tracking

**Why:** The current results should not be interpreted as clinically meaningful without external validation and well-documented data provenance.

**Actions:**
- Prepare a real PTBDB / CPSC validation set.
- Record exact source URLs, download dates, and license terms.
- Run `scripts/prepare_external_ecg.py` and `scripts/eval_external_ecg.py` on the external set.

**Related todos:** 4, 11

**Suggested commands:**

```bash
.venv/bin/python scripts/download_datasets.py --all-external
.venv/bin/python scripts/prepare_external_ecg.py --input-root data/raw/ptbdb --dataset-name ptbdb --labels-csv data/raw/ptbdb_labels.csv --ecg-len 2000 --out-dir data/external
.venv/bin/python scripts/eval_external_ecg.py --signals data/external/external_ecg_ptbdb_signals.npy --labels data/external/external_ecg_ptbdb_labels.npy --artifacts-dir artifacts --checkpoint model.pt --out-dir results/external/ptbdb
```

### 2) Improve model quality on the ECG side

**Why:** The current ECG / fusion artifacts are weaker than the tabular baseline on the smoke test set.

**Actions:**
- Add a minimal hyperparameter sweep for `ResNet1D`, `InceptionTime`, and a transformer-style ECG model.
- Compare weighted BCE vs focal loss using the new trainer flags.
- Try deeper architectures and longer training on the ECG-only pipeline.

**Related todos:** 5, 6

**Suggested commands:**

```bash
.venv/bin/python -m experiments.train --processed data/processed --epochs 3 --batch-size 8 --device cpu --augment --loss weighted_bce --dropout-p 0.2 --mc-samples 5 --save-uncertainty
```

### 3) Turn the new tests into a hard gate

**Why:** The repo now has several moving parts; CI should keep the smoke tests from regressing.

**Actions:**
- Keep the current GitHub Actions workflow minimal and fast.
- Add a single synthetic-data test for any new experiment path.
- Make the Docker build the same command path used locally.

**Related todos:** 7, 8, 9

### 4) Finish the documentation package

**Why:** The repo needs a clear statement of intended use, limitations, and data provenance.

**Actions:**
- Review and finalize `MODEL_CARD.md`.
- Fill in dataset license / access details in `data/dataset_datasheets.md`.
- Publish the recommendations report as the handoff document for the project.

**Related todos:** 11, 12

## Estimated Effort

- **External validation + provenance completion:** 1–2 days.
- **ECG model improvement experiments:** 2–5 days for a first pass.
- **Documentation finalization:** 2–4 hours.

## Current Commands Worth Keeping

```bash
make dry-run
make test
make eval
make train
docker build -t multi-modal-cvd-predictor .
docker run --rm -p 8501:8501 multi-modal-cvd-predictor
```

## Handoff Notes

- The experiment pipeline, tests, and Docker / pip setup are in place.
- The main remaining scientific work is stronger ECG modeling and external validation.
- The current tabular baseline remains a useful benchmark, but the model should not be treated as deployment-ready yet.
