# Dataset Datasheets

This file captures dataset provenance, expected usage, and practical notes for the data used by the repository.

## 1) `data/cardio.csv`

- **Type:** tabular cardiovascular risk dataset.
- **Role in repo:** tabular baseline and part of the multi-modal feature set.
- **Likely provenance:** Kaggle dataset referenced in the project README.
- **Recommended records to keep:** source URL, download date, license / terms of use, column definitions, and any filtering performed.
- **Preprocessing used in repo:** cleaning, imputation, encoding, and split generation via the preprocessing pipeline.

## 2) `data/hospital_admissions.csv`

- **Type:** tabular hospital admissions / clinical event dataset.
- **Role in repo:** tabular baseline and fusion features.
- **Likely provenance:** Kaggle dataset referenced in the project README.
- **Recommended records to keep:** exact source, access date, raw schema, and any de-identification assumptions.
- **Preprocessing used in repo:** normalization and feature engineering inside the preprocessing pipeline.

## 3) `data/ptbxl_records/`

- **Type:** ECG waveform assets derived from PTB-XL-style records and sample ECG files.
- **Role in repo:** ECG encoder training, evaluation, and saliency examples.
- **Recommended records to keep:** dataset version, acquisition source, lead configuration, sampling rate, and label mapping.
- **Preprocessing used in repo:** fixed-length cropping / padding and conversion into `.npy` arrays under `data/processed/`.

## 4) `data/processed/`

- **Type:** derived training / validation / test splits.
- **Role in repo:** primary input to the current experiments and smoke tests.
- **Contents:** ECG arrays, tabular matrices, labels, and metadata artifacts.
- **Recommended records to keep:** preprocessing script version, split seed, and feature ordering.

## 5) External ECG validation datasets

### PTBDB

- **Role:** external ECG holdout / transfer validation.
- **Expected format:** WFDB records or standardized `.npy` arrays.
- **Download target:** `scripts/download_datasets.py --ptbdb`.
- **Prep script:** `scripts/prepare_external_ecg.py`.
- **Recommended provenance record:** exact source, version, access date, label mapping, and sampling rate.

### CPSC / challenge ECG sets

- **Role:** cross-site robustness evaluation.
- **Expected format:** WFDB records or `.csv` / `.npy` exports.
- **Download target:** `scripts/download_datasets.py --cpsc` or `--all-external`.
- **Prep script:** `scripts/prepare_external_ecg.py`.
- **Recommended provenance record:** challenge name, split, access terms, and label source.

## Handling and Storage Notes

- Keep raw downloads separate from processed artifacts.
- Avoid overwriting original source files.
- Store preprocessing parameters with the generated arrays.
- If labels are missing, the prep pipeline may record `-1` for unknown labels.

## Open provenance items to verify before publication

- exact license terms for each source,
- download date and checksum,
- label definitions and class balance,
- whether any rows were filtered or merged,
- whether ECGs were resampled, truncated, or denoised.
