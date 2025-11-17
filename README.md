# 🩺 Multi-Modal Predictors for Cardiovascular Disease Risk and Outcomes

**Author:** Angel Morenu  
**Email:** angel.morenu@ufl.edu  
**Affiliation:** University of Florida, M.S. in Applied Data Science  
**Course:** EEE 6778 – Applied Machine Learning II (Fall 2025)  
**Instructor:** Dr. Ramirez-Salgado

---

## 🧠 Project Overview

Cardiovascular disease (CVD) remains the leading global cause of death. This project develops a multi-modal machine learning system that fuses:

- Tabular demographic data  
- Hospital admission records  
- Physiological 12‑lead ECG signals

The goal is improved CVD risk prediction with explainability and edge-deployable inference.

---

## 📦 GitHub Repository

Repository: https://github.com/angelmorenu/multi-modal-cvd-predictor

Clone and navigate:
```bash
git clone https://github.com/angelmorenu/multi-modal-cvd-predictor.git
cd multi-modal-cvd-predictor
```

---

## 📊 Datasets Used

| Dataset | Description | Link |
|---|---:|---|
| Cardiovascular Diseases | Demographics and lifestyle features | https://www.kaggle.com/datasets/mexwell/cardiovascular-diseases |
| Hospital Admissions | Clinical visit and diagnostic records | https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data |
| PTB-XL ECG | 12-lead ECG signals and annotations | https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset-reformatted |

---

## 🏗️ Project Architecture

This hybrid workflow uses:
- scikit-learn for preprocessing and tabular baselines  
- PyTorch for ECG deep learning and feature fusion  
- Streamlit for the user interface  
- Conceptual Edge AI deployment (e.g., smartwatch scenario)

![Architecture Diagram](docs/multimodal_cvd_architecture.png)

---

## ⚙️ Installation and Environment Setup

Install with conda (choose the appropriate platform file):

```bash
# macOS (Intel or Apple Silicon M1/M2/M3; uses CPU/MPS)
conda env create -f environment.macos.yml
conda activate cvd_predictor

# Linux/Windows with NVIDIA GPU (CUDA 11.8)
conda env create -f environment.cuda.yml
conda activate cvd_predictor
```

Notes:
- On Apple Silicon, PyTorch uses the MPS backend when available.
- The default environment.yml is cross-platform (CPU/MPS-friendly).
- If conda dependency resolution is slow, install mamba:
```bash
conda install -c conda-forge mamba
```

---

## 🚀 Running the Project

1. Run the setup and EDA notebook:
```bash
jupyter notebook notebooks/setup.ipynb
```

2. Launch the Streamlit UI:
```bash
# Activate the conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate cvd_predictor

# Run the Streamlit app
streamlit run ui/MultiModalCVD_app.py --server.port 8502 --server.headless true
```
This opens a local UI to input demographics, upload an ECG, and view predicted risk.

  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8502
  Network URL: http://192.168.1.72:8502
  External URL: http://104.4.123.52:8502

---

## 📁 Repository Structure

```
/Volumes/Dan/MSADS Fall 2025/EEE6778/Multi_modal_CVD_Project/
├── data/
├── notebooks/
│   ├── setup.ipynb
│   └── train_eval.ipynb
├── src/
│   ├── preprocess.py
│   ├── model.py
│   └── train.py, eval.py
├── ui/
│   ├── app.py
│   └── MultiModalCVD_app.py
├── scripts/
│   ├── plot_confusion.py
│   ├── plot_calibration.py
│   └── build_report.sh
│   └── download_datasets.py
│   └── generate_predictions.py
│   └── interpretability_ecg_saliency.py
│   └── interpretability_shap.py
│   └── metric_summary.py
│   └── perf_dashboard.py
│   └── plot_calibration.py
│   └── verify_preprocessing.py
├── figures/
│   ├── confusion_matrix.png
│   ├── perf_dashboard_fusion.png
│   ├── calibration_curve_fusion.png
│   ├── prob_hist_fusion.png
│   ├── shap_force_example.png
│   └── ecg_saliency.png
├── results/
├── artifacts/
│   ├── model.pt
│   └── model_ecg.pt
├── environment.yml
├── environment.cuda.yml
├── environment.macos.yml
├── README.md
└── Report/
    ├── "Project Deliverables 3.tex"
    └── "Project Deliverables 3.pdf"
```

## Git commit & push conventions
Follow a concise, conventional commit message format to make history easy to scan and automate:

- Format: `<type>(<scope>): short subject`
  - `type` examples: feat, fix, docs, chore, refactor, test
  - `scope` examples: report, scripts, ui, data, ci

- Example commit messages used in this repo:
  - `chore(report): update Deliverable 3 cleaned LaTeX and placeholders`
  - `feat(ui): add explanations expander and JSONL logging`

- When creating a branch and pushing for the first time:
```bash
# create branch locally
git checkout -b feat/reporting
# stage files
git add path/to/file
# commit with a conventional message
git commit -m "chore(report): update Deliverable 3 cleaned LaTeX and placeholders"
# set the upstream and push
git push --set-upstream origin feat/reporting
```

- Keep notebooks output-free before committing (use `nbstripout` or the included `.gitattributes`).

This minimal convention keeps commit history consistent and makes it straightforward to find changes by scope or type.

---

## 📦 Deliverables

- Complete project repository (code + documentation)  
- Jupyter notebooks for setup and evaluation (setup.ipynb, train_eval.ipynb)  
- Streamlit application (ui/app.py and MultiModalCVD_app.py)  
- Technical IEEE report (Deliverable 2, PDF)  
- Environment YAML files for reproducibility

---

## ✅ Reproducibility Instructions (Deliverable 3 snapshot)

The repository contains scripts that regenerate the evaluation artifacts and figures used in Deliverable 3. After activating the `cvd_predictor` conda environment, follow these steps to reproduce the evaluation and the figures included in the report.

1) Re-generate predictions and the metric summary (writes per-run numpy arrays and a JSON summary):

```bash
python scripts/generate_predictions.py --outdir results/
```

This writes files like `results/tabular_y_true.npy`, `results/tabular_y_pred.npy`, `results/tabular_y_prob.npy`, and a summary JSON at `results/metric_summary.json`.

2) Recreate the main figures used in the report and UI (example commands):

```bash
# confusion matrix for fusion predictions
python scripts/plot_confusion.py --pred results/fusion_y_pred.npy --true results/fusion_y_true.npy --out figures/confusion_matrix.png

# calibration and probability histograms for the fusion model
python scripts/plot_calibration.py --probs results/fusion_y_prob.npy --true results/fusion_y_true.npy --out figures/calibration_curve_fusion.png
python scripts/perf_dashboard.py --probs results/fusion_y_prob.npy --true results/fusion_y_true.npy --out figures/perf_dashboard_fusion.png
```

3) Re-run interpretability scripts (optional — install `shap` and `captum` to get full outputs):

```bash
python scripts/interpretability_shap.py
python scripts/interpretability_ecg_saliency.py
```

4) Launch the Streamlit UI (interactive testing and demo):

```bash
streamlit run ui/MultiModalCVD_app.py --server.port 8502
```

Results snapshot (current `results/metric_summary.json` included in the repo):

- Tabular model: accuracy 0.5714, ROC AUC 0.5273, PR AUC 0.5596, Brier 0.3506
- ECG model: accuracy 0.4375, ROC AUC 0.2969, PR AUC 0.3755, Brier 0.3311
- Fusion (fallback/stable evaluation): accuracy 0.5625, ROC AUC 0.4531, PR AUC 0.5258, Brier 0.3617

These values are used in `Report/Project Deliverables 3.tex`.

---

## Commit hygiene (recommended)

Before committing notebooks or pushing branches, strip outputs so diffs remain clean. Two options:

- Install and use nbstripout (recommended):

```bash
# install once in your environment
pip install nbstripout
# enable the git filter for this repo
nbstripout --install
# strip outputs from all tracked notebooks
git ls-files "*.ipynb" -z | xargs -0 nbstripout
```

- Or add `.gitattributes` to enforce stripping on commit (this repo includes a starter `.gitattributes`).

Keeping notebooks output-free makes PR reviews much easier.

## Interpretability (SHAP & ECG saliency)

This repository includes lightweight interpretability scripts that produce visual summaries in `figures/`:

- `scripts/interpretability_shap.py` — attempts to compute SHAP values. If `shap` is not installed or available in your environment, the script falls back to plotting RandomForest feature importances (MDI). The output is written to `figures/shap_force_example.png`.
- `scripts/interpretability_ecg_saliency.py` — produces an ECG saliency visualization. If `captum` is available it can be integrated; otherwise the script produces a derivative-based saliency plot. The output is written to `figures/ecg_saliency.png`.

To get full SHAP/Captum outputs in your local environment, install the packages and re-run the scripts:

```bash
# Activate your conda environment first (example)
conda activate cvd_predictor

# Install interpretability libraries (may take a minute)
pip install -r requirements-interpretability.txt

# Run the interpretability scripts
python scripts/interpretability_shap.py
python scripts/interpretability_ecg_saliency.py
```

After running, confirm `figures/shap_force_example.png` and `figures/ecg_saliency.png` are created and then re-generate the PDF report.

Alternatively, install just the interpretability extras with pip:

```bash
pip install -r requirements-interpretability.txt
```

---

## 🤖 Responsible AI Goals

- Fairness: Evaluate across age, gender, and race subgroups.  
- Transparency: Incorporate SHAP and saliency-based explanations.  
- Efficiency: Support lightweight edge deployment for on-device inference.  
- Reproducibility: Publish code, environment files, and metrics.

---

## 👤 Author & Contact

Angel Morenu  
University of Florida – M.S. in Applied Data Science  
angel.morenu@ufl.edu  
GitHub: https://github.com/angelmorenu/multi-modal-cvd-predictor

This repository accompanies Deliverable 2 of EEE 6778 – Applied Machine Learning II (Fall 2025).
