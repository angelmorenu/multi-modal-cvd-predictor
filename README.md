# Multi-Modal Predictors for Cardiovascular Disease Risk and Outcomes

**Author:** Angel Morenu  
**Course:** EEE 6778 – Applied Machine Learning II (Fall 2025)  
**Instructor:** Dr. Ramirez-Salgado

---

## Project Overview

Cardiovascular disease (CVD) remains the leading global cause of death. Existing predictive models often rely on a single data modality, limiting their accuracy and applicability in real-world settings.

This project develops a multi-modal machine learning system that combines:
- Tabular demographic data
- Hospital admission records
- Physiological ECG signals

to improve CVD risk prediction and make model outputs explainable, accessible, and deployable on edge devices.

---

## Datasets Used

| Dataset | Description | Link |
|--------|-------------|------|
| Cardiovascular Diseases | Demographics and lifestyle features | [View on Kaggle](https://www.kaggle.com/datasets/mexwell/cardiovascular-diseases) |
| Hospital Admissions | Clinical visit and diagnostic records | [View on Kaggle](https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data) |
| PTB-XL ECG | 12-lead ECG signals and annotations | [View on Kaggle](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset-reformatted) |

---

## Project Architecture

This hybrid workflow uses:
- scikit-learn for preprocessing and tabular baselines
- PyTorch for ECG deep learning and feature fusion
- Streamlit for the user interface
- Conceptual edge deployment (e.g., smartwatch scenario)

![Architecture Diagram](Project/multimodal_cvd_architecture.png)

---

## Environment Setup

Install with conda (choose your platform file):
```bash
# macOS (Intel or Apple Silicon M1/M2/M3; uses CPU/MPS)
conda env create -f environment.macos.yml
conda activate cvd_predictor

# Linux/Windows with NVIDIA GPU (CUDA 11.8)
conda env create -f environment.cuda.yml
conda activate cvd_predictor
```

Notes:
- On Apple Silicon, PyTorch uses the MPS backend automatically when available; the notebook prints the selected device.
- If you prefer a single cross-platform file, `environment.yml` is CPU/MPS-friendly (no CUDA packages) and should work on macOS/Intel/Linux.
- If you run into slow dependency solving, consider using mamba.

---

## Run the Project

Run setup and EDA:
```bash
jupyter notebook setup.ipynb
```

Launch the Streamlit UI:
```bash
streamlit run ui/app.py
```

---

## Deliverables

- notebooks/ – Exploratory analysis, baseline models
- src/ – Scripts for preprocessing, training, evaluation
- ui/ – Streamlit demo interface
- results/ – Logs, metrics, plots
- docs/ – Diagrams and documentation
- README.md – You’re here!
- environment.yml – Reproducible dependencies

---

## Responsible AI Goals

- Fairness: Evaluate across age/gender/race subgroups
- Transparency: Use SHAP + attention maps
- Efficiency: Support lightweight edge deployment

---

## Contact

For questions, reach out via GitHub or university email.
