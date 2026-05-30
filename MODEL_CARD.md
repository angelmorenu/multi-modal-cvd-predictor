# MODEL_CARD.md

## Model Overview

This repository contains a **multi-modal cardiovascular disease (CVD) risk prediction system** that combines:

- tabular demographic / clinical inputs,
- hospital admission features, and
- ECG waveforms.

The codebase supports both classical baselines and a PyTorch multi-modal fusion model. The current repository also includes experiment scaffolding for stronger ECG-only models, calibration, interpretability, and external validation.

## Intended Use

### Primary use

- Research and benchmarking for multi-modal CVD risk prediction.
- Prototyping of tabular, ECG-only, and fusion models.
- Internal evaluation workflows with calibration and external validation.

### Not intended for

- Clinical decision making without prospective validation.
- Diagnosis, screening, or treatment planning.
- High-stakes deployment without dataset governance, bias review, and regulatory review.

## Model Inputs

- **Tabular features:** age, blood pressure, cholesterol, glucose, lifestyle and other engineered clinical features.
- **ECG signal:** fixed-length ECG waveform arrays stored in `data/processed/` or external standardized `.npy` files.
- **Optional external ECG validation:** PTBDB / CPSC-style external test sets standardized with `scripts/prepare_external_ecg.py`.

## Outputs

- Binary CVD risk score or class probability.
- Optional calibration artifacts.
- Optional uncertainty outputs from MC-dropout / ensemble-style experiments.

## Training Data

The repository references the following data sources:

- `data/cardio.csv` — tabular cardiovascular risk dataset.
- `data/hospital_admissions.csv` — hospital-admission features.
- `data/ptbxl_records/` — ECG-related sample assets / local PTB-XL artifacts.
- `data/processed/` — preprocessed arrays used by the current training and evaluation scripts.

See `data/dataset_datasheets.md` for provenance notes.

## Evaluation Summary

The repository now includes repeated cross-validation, bootstrap confidence intervals, and external ECG preparation / evaluation scripts.

### Internal summary metrics captured in `results/`

- Tabular baseline (5-fold CV): logistic regression mean ROC AUC around **0.679** and PR AUC around **0.780**.
- Tabular baseline (5-fold CV): random forest mean ROC AUC around **0.628** and PR AUC around **0.681**.
- Repeated stratified CV (5x5) on the processed 63-sample set:
  - logistic regression mean ROC AUC around **0.618** with repeat-aware 95% CI roughly **[0.586, 0.661]**.
  - random forest mean ROC AUC around **0.591** with repeat-aware 95% CI roughly **[0.565, 0.618]**.
- The pooled-prediction DeLong comparison between logistic regression and random forest was **not statistically significant** in the current run.

### Held-out artifact summary

The saved artifact smoke evaluation (`results/metric_summary.json`) shows that the current fusion artifact is not yet competitive with the tabular baseline on the small processed evaluation set. Treat those numbers as a diagnostic signal, not a clinical benchmark.

## Ethical Considerations

- **Bias and fairness:** CVD risk is sensitive to demographic and care-access bias. Subgroup analysis is still recommended before any broader use.
- **Calibration:** risk scores should be calibrated and re-evaluated under dataset shift.
- **External validity:** the repo now supports PTBDB / CPSC-style external ECG validation, but external datasets must be documented and audited.
- **Privacy:** ECGs and hospital records can be identifiable or highly sensitive; storage and sharing should follow local governance rules.

## Caveats

- The current processed sample size used for smoke tests is small.
- Some saved scikit-learn artifacts were created with older package versions and may not load cleanly in newer environments.
- The current Streamlit UI is a demo interface and should not be treated as a certified clinical workflow.
- Performance values in this card are from repository artifacts and quick CV runs, not a prospectively validated study.

## Reproducibility

Recommended commands:

```bash
make dry-run
make test
make eval
make train
```

Docker:

```bash
docker build -t multi-modal-cvd-predictor .
docker run --rm -p 8501:8501 multi-modal-cvd-predictor
```

## Version / Date

- Repository snapshot: 2026-05-29
- Documentation generated from the current `main` branch workspace state.
