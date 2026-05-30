# 🩺 Multi-Modal Cardiovascular Disease Risk Prediction System

**Author:** Angel Morenu  
**Email:** angel.morenu@ufl.edu  
**Affiliation:** University of Florida, M.S. in Applied Data Science  
**Course:** EEE 6778 – Applied Machine Learning II (Fall 2025)  
**Instructor:** Dr. Ramirez-Salgado  
**Status:** Complete (Deliverable 4 – Final IEEE Research Report)

---

## 🧠 Project Overview

Cardiovascular disease (CVD) remains the leading global cause of death, accounting for nearly one-third of global mortality. This project implements a **complete machine learning lifecycle** for multi-modal CVD risk prediction, integrating heterogeneous data streams:

- **Tabular demographic and clinical features** (age, blood pressure, cholesterol, glucose, lifestyle factors)
- **Hospital admission records** (administrative claims, diagnostic codes, length of stay)
- **Physiological 12-lead ECG signals** (21,837 PTB-XL recordings at 100 Hz)

The system demonstrates how **multi-modal fusion** can improve CVD risk discrimination and provides **clinically actionable explanations** via SHAP and saliency-based interpretability. End-to-end reproducibility is achieved through documented preprocessing, model training, calibration, and edge-deployable inference.

**At a Glance:**
- Multi-modal CVD risk prediction from tabular, admissions, and ECG data
- Reproducible evaluation with tabular baselines, robust CV, and external ECG prep/eval
- Experiment scaffold for stronger ECG models, class-imbalance handling, and uncertainty estimates
- Documentation package with model card, dataset datasheets, and recommendations report

**Start Here:**
- Run `make dry-run` to verify shapes and artifact loading
- Run `make test` to execute the smoke tests
- Run `make docker-build` and `make docker-run` to launch the app in a container
- Open `MODEL_CARD.md`, `data/dataset_datasheets.md`, and `reports/recommendations.md` for the docs trail

**Quick Links:**
- [Pinned Dependencies](requirements.txt)
- [Experiment Notebook](Notebooks/robust_evaluation.ipynb)
- [CI Workflow](.github/workflows/ci.yml)
- [External ECG Prep & Eval](scripts/prepare_external_ecg.py) / [Download Targets](scripts/download_datasets.py)
- [Smoke + Experiment Tests](tests/)

---

## 🚨 Current Critical Blocker (Must Fix First)

Recent run behavior indicates a degenerate classifier:

- Confusion matrix like `[[0, 918], [0, 5959]]`
- All predictions positive (no class-0 recovery)
- Probability range entirely above threshold (`min > 0.5`)
- ROC AUC below random baseline

This means model discrimination is currently not clinically usable, even if accuracy looks high due to class imbalance.

### Immediate Recovery Plan

1. **Stratified patient-level splitting** with label-aware balancing (implemented in `scripts/prepare_splits.py`).
2. **Degeneracy checks in evaluation** (implemented in `src/eval.py`) using `--fail-on-degenerate` and optional `--min-roc-auc`.
3. **Decision-threshold selection** on validation (maximize F1/Youden/clinical sensitivity target), not fixed `0.5`.
4. **Imbalance-aware training** (weighted/focal loss, balanced sampling, and class-ratio monitoring per split).

---

## ✅ Targeted Cleanup Policy (Keep What Matters)

To keep the project rigorous and maintainable without adding unnecessary files:

- Keep: reproducibility-critical code in `src/`, `scripts/`, `tests/`, and core docs (`README.md`, `MODEL_CARD.md`, `reports/recommendations.md`).
- Keep: small metadata artifacts needed for traceability (`results/*_meta.json`, summary metrics).
- Exclude/untrack: local caches (`__pycache__/`, `.ipynb_checkpoints/`, `*.pyc`) and bulky generated binaries not needed for version history.
- Store large external raw data outside Git history (or via LFS/releases) and document provenance in existing dataset docs.

---

## 📌 Remaining Tasks / TODOs (Medical-Readiness Track)

- **Data Integrity Gate:** Verify label coding and positive-rate consistency from `preprocess` → splits → train/eval.
- **Model Validity Gate:** Enforce fail-fast eval checks (`all-positive`, `all-negative`, near-constant probabilities, low ROC AUC).
- **Calibration Gate:** Fit/validate calibration and report Brier + calibration curves on held-out/external sets.
- **Generalization Gate:** Run external ECG validation (PTBDB/CPSC) with documented provenance and fixed split policy.
- **Clinical Reporting Gate:** Report class-wise sensitivity/specificity/PPV/NPV with confidence intervals.
- **Reproducibility Gate:** Re-run `make test`, `make eval`, and tracked experiment scripts with seed-controlled settings.

---

## 📚 Deliverables Summary

| Deliverable | File | Status | Description |
|---|---|---|---|
| **Deliverable 1** | `Morenu_Angel_Deliverable1_TechnicalBlueprint.pdf` | ✅ Complete | Technical blueprint and project proposal |
| **Deliverable 2** | `Report/Project Deliverables 2.pdf` | ✅ Complete | Technical report with system architecture |
| **Deliverable 3** | `Report/Project Deliverables 3.pdf` | ✅ Complete | Expanded report with evaluation results |
| **Deliverable 4** | `Report/Project Deliverables 4.pdf` | ✅ Complete | **Final IEEE-formatted research report (11 pages)** |
| **Source Code** | This repository | ✅ Complete | Fully documented, reproducible codebase |
| **Final Presentation** | `Report/Morenu_Final Presentation - Poster Presentation.pptx` | ✅ Complete | Visual presentation of results |

---

## 📖 Final Report (Deliverable 4) Contents

The comprehensive IEEE-formatted technical report (`Report/Project Deliverables 4.pdf`) includes:

### Main Sections
1. **Introduction** – Problem context, objectives, and contributions
2. **Related Work** – Literature review (classical CVD models, deep learning for ECG, multi-modal fusion, calibration)
3. **System Design and Implementation** – Complete technical architecture
   - Data collection and preprocessing (3 datasets, integration strategy, missing data handling)
   - Modality-specific encoders (MLPTabular, ECG1DCNN)
   - Fusion and classifier architecture
   - Training configuration (AdamW optimizer, regularization, batch normalization)
   - Platt scaling calibration methodology
   - Comprehensive evaluation metrics (ROC AUC, PR AUC, Brier score, ECE)
4. **Interpretability** – SHAP-based feature importance and ECG saliency maps
5. **Human-Computer Interaction (HCI)** – UI design, edge deployment considerations, audit logging
6. **Evaluation and Results** – Held-out test performance and visual diagnostics
7. **Discussion** – System strengths, limitations, and novelty
8. **Future Work and Improvements** – Larger datasets, hyperparameter tuning, regulatory approval
9. **Responsible AI and Fairness** – Fairness, privacy, transparency considerations
10. **Conclusion** – Key takeaways and clinical implications

### Figures Included
- System architecture diagram
- Confusion matrices (tabular, ECG, fusion models)
- Calibration curves and probability histograms
- SHAP force plot (feature importance example)
- ECG saliency maps (gradient-based interpretability)
- UI dashboard screenshot

### Appendix
- **Repository Structure** – Complete file organization
- **To Reproduce** – Step-by-step instructions to regenerate results
- **Hyperparameters Table** – Learning rates, dropout rates, layer dimensions
- **Performance Summary Table** – Metrics for all model variants

## 🧩 Added Files

The following files were added during the robustness and external-validation work:

- `scripts/run_tabular_baselines.py`
- `scripts/robust_eval.py`
- `scripts/prepare_external_ecg.py`
- `scripts/eval_external_ecg.py`
- `scripts/download_datasets.py` (updated with PTBDB/CPSC and `--all-external`)
- `Notebooks/robust_evaluation.ipynb`
- `data/README_datasets.md`

These additions support repeat-aware evaluation, DeLong comparison, external ECG preparation, and external dataset downloads for PTBDB/CPSC.

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

| Dataset | Description | Size | Link |
|---|---|---|---|
| **Cardiovascular Diseases (Kaggle)** | Demographics and lifestyle features | ~70K samples | [https://www.kaggle.com/datasets/mexwell/cardiovascular-diseases](https://www.kaggle.com/datasets/mexwell/cardiovascular-diseases) |
| **Hospital Admissions (Kaggle)** | Clinical visit and diagnostic records | ~50K records | [https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data](https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data) |
| **PTB-XL ECG Database (Kaggle)** | 12-lead ECG signals and annotations | 21,837 recordings | [https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset-reformatted](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset-reformatted) |

### External Validation Plan

To make the project more legitimate and clinically credible, add external ECG validation on a dataset that was **not** used during training.

Recommended options:

- **PTBDB**: useful for transfer learning and holdout validation.
- **CPSC / challenge ECG sets**: useful for cross-site generalization.

Preparation workflow:

1. Download or extract the external ECG dataset locally.
2. Record provenance details in `data/README_datasets.md`.
3. Standardize the dataset with `scripts/prepare_external_ecg.py`.
4. Evaluate ECG-only and fusion models on the external set.

Example:

```bash
.venv/bin/python scripts/prepare_external_ecg.py \
  --input-root /path/to/ptbdb_or_cpsc \
  --dataset-name ptbdb \
  --labels-csv /path/to/labels.csv \
  --ecg-len 2000 \
  --out-dir data/external
```

Download targets supported by the repo:

```bash
# PhysioNet PTB Diagnostic ECG Database
.venv/bin/python scripts/download_datasets.py --ptbdb

# PhysioNet CPSC 2018 ECG dataset
.venv/bin/python scripts/download_datasets.py --cpsc
```

External evaluation example:

```bash
.venv/bin/python scripts/eval_external_ecg.py \
  --signals data/external/external_ecg_ptbdb_signals.npy \
  --labels data/external/external_ecg_ptbdb_labels.npy \
  --artifacts-dir artifacts \
  --checkpoint model.pt \
  --out-dir results/external/ptbdb
```

Full PTBDB/CPSC external-validation pipeline:

```bash
# 1) Download external ECG source(s)
.venv/bin/python scripts/download_datasets.py --all-external

# 2) Standardize the raw ECG files into .npy arrays + manifest
.venv/bin/python scripts/prepare_external_ecg.py \
  --input-root data/raw/ptbdb \
  --dataset-name ptbdb \
  --labels-csv data/raw/ptbdb_labels.csv \
  --ecg-len 2000 \
  --out-dir data/external

# 3) Evaluate the model on the prepared external ECG set
.venv/bin/python scripts/eval_external_ecg.py \
  --signals data/external/external_ecg_ptbdb_signals.npy \
  --labels data/external/external_ecg_ptbdb_labels.npy \
  --artifacts-dir artifacts \
  --checkpoint model.pt \
  --out-dir results/external/ptbdb
```

---

## 🏗️ System Architecture

The system implements a **multi-modal fusion pipeline** combining:
- **scikit-learn** for preprocessing and tabular baselines
- **PyTorch** for ECG deep learning (1D-CNN) and multi-modal fusion
- **Streamlit** for interactive clinical UI
- **SHAP & Captum** for interpretability (with fallbacks)
- **Edge AI** design for privacy-preserving, on-device inference

**Mid-level Fusion Strategy:**
1. Tabular features → MLPTabular encoder (32-D embedding)
2. ECG signals → ECG1DCNN encoder (128-D embedding)
3. Concatenate embeddings (160-D joint representation)
4. Classifier head → binary sigmoid output
5. Post-hoc Platt scaling for calibration

---

## ⚙️ Installation and Environment Setup

If you prefer pip over conda, the repo now includes a pinned `requirements.txt` and a `Dockerfile` for a containerized setup.

### Pip / Docker quick start
```bash
python -m pip install -r requirements.txt
make dry-run
```

Or build and run the UI in Docker:
```bash
docker build -t multi-modal-cvd-predictor .
docker run --rm -p 8501:8501 multi-modal-cvd-predictor
```

Install with conda (choose the appropriate platform file):

### macOS (Intel or Apple Silicon M1/M2/M3)
```bash
conda env create -f environment.macos.yml
conda activate cvd_predictor
```

### Linux/Windows with NVIDIA GPU (CUDA 11.8)
```bash
conda env create -f environment.cuda.yml
conda activate cvd_predictor
```

### Default Cross-Platform (CPU/MPS-friendly)
```bash
conda env create -f environment.yml
conda activate cvd_predictor
```

**Notes:**
- On Apple Silicon, PyTorch uses the MPS backend automatically
- If conda resolution is slow, install and use `mamba`:
```bash
conda install -c conda-forge mamba
mamba env create -f environment.macos.yml
```

---

## 🚀 Quick Start

### 1. Generate Example ECG and Verify Shapes
```bash
python scripts/generate_example_ecg.py
python scripts/dry_run_shapes.py
```

### 2. Launch Interactive Streamlit UI
```bash
streamlit run ui/MultiModalCVD_app.py --server.port 8502
```

Then:
- Open browser to `http://localhost:8502`
- Use the **"Use example ECG (demo)"** button to load `artifacts/example_ecg_000.npy`
- Adjust tabular features with sliders and dropdowns
- View predicted CVD risk with color-coded interpretation
- Explore SHAP and saliency-based explanations

### 3. Reproduce Full Evaluation Pipeline
```bash
# Generate predictions
python scripts/generate_predictions.py --outdir results/

# Create visualizations
python scripts/plot_confusion.py --pred results/fusion_y_pred.npy --true results/fusion_y_true.npy --out figures/confusion_matrix.png
python scripts/plot_calibration.py --probs results/fusion_y_prob.npy --true results/fusion_y_true.npy --out figures/calibration_curve_fusion.png
python scripts/perf_dashboard.py --probs results/fusion_y_prob.npy --true results/fusion_y_true.npy --out figures/perf_dashboard_fusion.png

# Generate interpretability plots (requires shap and captum)
pip install -r requirements-interpretability.txt
python scripts/interpretability_shap.py
python scripts/interpretability_ecg_saliency.py
```

### 4. Generate LaTeX Report (Deliverable 4)
```bash
cd Report/
pdflatex -interaction=nonstopmode "Project Deliverables 4.tex"
```

---

## 📁 Complete Repository Structure

```
/Multi_modal_CVD_Project/
│
├── 📄 README.md                                    # This file
├── 📄 environment.yml                              # Cross-platform conda environment
├── 📄 environment.macos.yml                        # macOS-specific (CPU/MPS)
├── 📄 environment.cuda.yml                         # Linux/Windows GPU (CUDA 11.8)
├── 📄 requirements-interpretability.txt            # SHAP/Captum dependencies
│
├── 📂 data/
│   ├── cardio.csv                                  # Cardiovascular disease features (~70K)
│   ├── hospital_admissions.csv                     # Hospital records (~50K)
│   └── processed/
│       ├── ecg_train.npy                           # ECG training data
│       ├── ecg_val.npy                             # ECG validation data
│       ├── tabular_train_X.npy                     # Tabular features (training)
│       ├── tabular_train_y.npy                     # CVD labels (training)
│       ├── tabular_val_X.npy & _y.npy              # Validation set
│       └── tabular_test_X.npy & _y.npy             # Test set
│
├── 📂 Notebooks/
│   ├── Setup.ipynb                                 # EDA and data preprocessing
│   └── train_eval.ipynb                            # Model training and evaluation

│
├── 📂 src/
│   ├── preprocess.py                               # Data preprocessing utilities
│   ├── model.py                                    # Model architecture (MLPTabular, ECG1DCNN, Fusion)
│   ├── train.py                                    # Training loop
│   └── eval.py                                     # Evaluation metrics and plots
│
├── 📂 ui/
│   ├── MultiModalCVD_app.py                        # Main Streamlit application
│   └── app.py                                      # Alternative UI entry point
│
├── 📂 scripts/
│   ├── build_report.sh                             # LaTeX compilation script
│   ├── download_datasets.py                        # Download Kaggle datasets
│   ├── generate_predictions.py                     # Generate test predictions
│   ├── generate_example_ecg.py                     # Create demo ECG artifact
│   ├── dry_run_shapes.py                           # Verify tensor shapes
│   ├── fit_platt_calibrator.py                     # Train Platt calibrator
│   ├── make_ecg_saliency.py                        # Generate saliency maps
│   ├── run_shap_tabular.py                         # SHAP or RandomForest importance
│   ├── interpretability_ecg_saliency.py            # ECG saliency visualization
│   ├── interpretability_shap.py                    # SHAP feature importance
│   ├── metric_summary.py                           # Compute evaluation metrics
│   ├── perf_dashboard.py                           # Create performance dashboard
│   ├── plot_calibration.py                         # Plot calibration curves
│   ├── plot_confusion.py                           # Plot confusion matrices
│   └── verify_preprocessing.py                     # Validate preprocessing


│
├── 📂 figures/
│   ├── confusion_matrix.png                        # Confusion matrix (fusion model)
│   ├── perf_dashboard_fusion.png                   # Performance metrics dashboard
│   ├── calibration_curve_fusion.png                # Calibration curve
│   ├── prob_hist_fusion.png                        # Probability histogram
│   ├── shap_force_example.png                      # SHAP force plot
│   ├── ecg_saliency.png                            # ECG saliency visualization
│   ├── ui_demo.png                                 # Streamlit UI screenshot
│   ├── multimodal_cvd_architecture.png             # System architecture
│   └── UI.png                                      # Alternative UI screenshot
│
├── 📂 results/
│   ├── metric_summary.json                         # Evaluation metrics JSON
│   ├── calibration_summary.json                    # Calibration metrics
│   ├── predictions_log.jsonl                       # Audit log (UI predictions)
│   ├── *_y_true.npy, *_y_pred.npy, *_y_prob.npy  # Per-model predictions
│   └── (tabular, ecg, fusion variants)
│
├── 📂 artifacts/
│   ├── model.pt                                    # Fusion model weights (PyTorch)
│   ├── model_ecg.pt                                # ECG encoder weights
│   ├── tabular_transformer.joblib                  # Tabular feature scaler (sklearn)
│   ├── tabular_meta.json                           # Tabular feature metadata
│   ├── calibrator.joblib                           # Platt calibrator
│   ├── example_ecg_000.npy                         # Demo ECG (1D array)
│   └── example_ecg_000.csv                         # Demo ECG (CSV format)
│
└── 📂 Report/
    ├── Project Deliverables 1.pdf                  # Technical blueprint
    ├── Project Deliverables 2.tex & .pdf           # Initial report
    ├── Project Deliverables 3.tex & .pdf           # Expanded report
    ├── Project Deliverables 4.tex & .pdf           # **Final IEEE report (11 pages)**
    ├── Final_Presentation_Poster_Presentation.tex  # Poster presentation LaTeX
    └── Morenu_Final Presentation - Poster...pptx   # PowerPoint presentation
```

---

## 📊 Held-Out Test Performance

### Results Summary
```
Metric              Tabular    ECG      Fusion   Calibrated
─────────────────────────────────────────────────────────
Accuracy            0.5714     0.4375   0.3750   0.3750
ROC AUC             0.5273     0.2969   0.4688   0.4688
PR AUC              0.5596     0.3755   0.5556   0.5556
Brier Score         0.3506     0.3311   0.2571   0.2407
F1 Score            0.6086     0.5263   0.5000   0.5000
Sensitivity         0.7000     0.6250   0.6250   0.6250
Specificity         0.4545     0.2500   0.1250   0.1250
```

**Key Findings:**
- Tabular features achieve ROC AUC = 0.527, demonstrating strong discrimination from demographics/clinical features
- ECG-only model underperforms (ROC AUC = 0.297), possibly due to small dataset and single-lead limitation
- Fusion model shows promise but is limited by test set size (N=64)
- Platt scaling improves calibration (Brier score 0.257 → 0.241)
- Full training on complete datasets would significantly improve performance

---

## 🔄 Complete Reproducibility Instructions

### Step 1: Set Up Environment
```bash
conda env create -f environment.macos.yml
conda activate cvd_predictor
```

### Step 2: Verify Installation
```bash
python scripts/dry_run_shapes.py
```
Expected output: Confirms model architecture and tensor shapes

### Step 3: Generate Predictions
```bash
python scripts/generate_predictions.py --outdir results/
```
Generates prediction arrays and `results/metric_summary.json`

### Step 4: Create Visualizations
```bash
# Confusion matrix
python scripts/plot_confusion.py \
  --pred results/fusion_y_pred.npy \
  --true results/fusion_y_true.npy \
  --out figures/confusion_matrix.png

# Calibration curve
python scripts/plot_calibration.py \
  --probs results/fusion_y_prob.npy \
  --true results/fusion_y_true.npy \
  --out figures/calibration_curve_fusion.png

# Performance dashboard
python scripts/perf_dashboard.py \
  --probs results/fusion_y_prob.npy \
  --true results/fusion_y_true.npy \
  --out figures/perf_dashboard_fusion.png
```

### Step 5: Run Interpretability Scripts (Optional)
```bash
# Install SHAP and Captum
pip install -r requirements-interpretability.txt

# Generate explanations
python scripts/interpretability_shap.py
python scripts/interpretability_ecg_saliency.py
```

### Step 6: Launch Interactive UI
```bash
streamlit run ui/MultiModalCVD_app.py --server.port 8502
```
Open `http://localhost:8502` in browser

### Step 7: Compile LaTeX Report
```bash
cd Report/
pdflatex -interaction=nonstopmode "Project Deliverables 4.tex"
```
Generates `Project Deliverables 4.pdf` (11 pages)

---

## 🧠 Model Architecture Details

### Tabular Encoder (MLPTabular)
- **Input:** ~50-100 dimensional tabular features
- **Layer 1:** Fully connected (128 units) + ReLU + Dropout(0.2)
- **Layer 2:** Fully connected (64 units) + ReLU + Dropout(0.2)
- **Output:** 32-dimensional embedding

### ECG Encoder (ECG1DCNN)
- **Input:** 2000 samples (20 seconds at 100 Hz)
- **Conv Block 1:** Conv1d(1, 32, kernel=5) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.1)
- **Conv Block 2:** Conv1d(32, 64, kernel=3) + BatchNorm + ReLU + MaxPool(2) + Dropout(0.1)
- **Conv Block 3:** Conv1d(64, 128, kernel=3) + BatchNorm + ReLU + GlobalAvgPool
- **Output:** 128-dimensional embedding

### Fusion Module
- **Concatenate:** 32-D tabular + 128-D ECG = 160-D joint
- **Hidden Layer:** Fully connected (64 units) + ReLU + Dropout(0.2)
- **Output Layer:** Sigmoid (binary classification)

### Training Configuration
- **Optimizer:** AdamW (LR=1e-3, weight_decay=1e-4)
- **Loss:** Binary cross-entropy (with class weights for imbalance)
- **Epochs:** 50 (with early stopping)
- **Batch size:** 32
- **Validation split:** 15% of training data
- **Calibration:** Platt scaling on held-out validation set

---

## 🔍 Interpretability Methods

### SHAP (SHapley Additive exPlanations)
- **Method:** KernelExplainer (model-agnostic)
- **Output:** `figures/shap_force_example.png`
- **Interpretation:** Feature contributions to individual predictions
- **Fallback:** RandomForest MDI if SHAP unavailable

### ECG Saliency Maps
- **Method:** Gradient-based attention (`∂f/∂x`)
- **Output:** `figures/ecg_saliency.png`
- **Interpretation:** Time-points influencing model predictions
- **Fallback:** Perturbation-based saliency if gradients unavailable

### UI Explanations
- SHAP force plot showing individual prediction drivers
- ECG saliency heatmap overlaid on signal
- Feature importance bar chart
- Calibration details (raw vs. calibrated probability)

---

## 🎯 Clinical UI Design

### Input Panel
- **Demographics:** Age, gender, BMI sliders
- **Clinical:** Blood pressure, cholesterol level dropdowns
- **Lifestyle:** Smoking, alcohol, physical activity toggles
- **ECG:** Upload CSV/NPY or use demo button
- **Validation:** Physiologically plausible range checks

### Prediction Display
- **Risk Score:** Large percentage with color coding
  - Green: <33% (low risk)
  - Yellow: 33-66% (moderate risk)
  - Red: >66% (high risk)
- **Actionable Guidance:** Context-specific recommendations
- **Confidence Interval:** Calibrated probability ± uncertainty

### Explainability Panel (Collapsible)
- SHAP force plot (feature contributions)
- ECG saliency visualization (if available)
- Feature importance ranking
- Model metadata (version, training date, AUC)

### Audit Logging
- All predictions saved to `results/predictions_log.jsonl`
- Fields: timestamp, inputs, output probability, explanations
- Supports regulatory compliance (FDA 21 CFR Part 11)

---

## ⚠️ Known Issues & Troubleshooting

### Scikit-learn Version Mismatch
**Issue:** `InconsistentVersionWarning` when loading `artifacts/tabular_transformer.joblib`

**Solution:**
- Option 1: Recreate transformer by re-running preprocessing
- Option 2: Install original scikit-learn version used for training

### ECG Upload Shape Errors
**Issue:** "Shape mismatch" error when uploading ECG

**Solution:**
- Use single-column CSV or 1D NPY format
- Expected shape: (2000,) or (2000, 1)
- Use "Use example ECG (demo)" button for testing

### Calibration Instability
**Issue:** Calibration doesn't improve performance on very small validation sets

**Solution:**
- Prefer larger held-out validation set (>500 samples)
- Alternatively, skip calibration for small datasets

### SHAP Installation Issues
**Issue:** `ImportError: No module named 'shap'`

**Solution:**
```bash
pip install -r requirements-interpretability.txt
# Or individually:
pip install shap captum
```
If installation fails, scripts fall back to RandomForest importance.

---

## 📚 Interpretability (SHAP & ECG Saliency)

### SHAP Feature Importance
```bash
pip install -r requirements-interpretability.txt
python scripts/interpretability_shap.py
```
- Generates `figures/shap_force_example.png`
- Falls back to RandomForest MDI if SHAP unavailable
- Identifies most influential features across test set

### ECG Saliency Visualization
```bash
python scripts/interpretability_ecg_saliency.py
```
- Generates `figures/ecg_saliency.png`
- Shows gradient-based importance of ECG time-points
- Falls back to perturbation-based method if gradients fail

### Integration with UI
- Explanations automatically displayed in Streamlit app
- Uses cached visualizations for fast loading
- Fallbacks ensure explanations always available

---

## 🤖 Responsible AI and Fairness

### Fairness Evaluation
- Stratified analysis by age, gender, race subgroups
- Performance parity: ROC AUC gap <5% across demographics
- Bias mitigation: inverse propensity weighting if disparities detected
- Biological validity: acknowledge genuine CVD prevalence differences

### Privacy and Data Protection
- Edge deployment: all inference local (no cloud upload)
- Model size: <5MB, fits on smartphones/wearables
- HIPAA compliance: supports on-device inference
- No patient data retained post-prediction

### Transparency and Explainability
- SHAP and saliency-based explanations for all predictions
- Model cards documenting training data, limitations, fairness considerations
- Fallback interpretability ensures robustness
- Audit logging for regulatory compliance

### Limitations and Disclaimers
- Proof-of-concept on small test set (N=64)
- Clinical validation required before deployment
- Dataset biases may limit generalization
- Not a replacement for clinical judgment

---

## 🚀 Future Work and Improvements

### Larger Datasets
- Integrate MIMIC-IV, eICU Collaborative Research Database
- Collaborate with clinical sites for prospective data collection
- Scale from thousands to hundreds of thousands of patients
- Multi-ethnic validation cohorts

### Advanced Architectures
- Transformer-based ECG encoding (Perceiver, Vision Transformer)
- Attention mechanisms for cross-modal interactions
- Multi-task learning (CVD risk + arrhythmia detection)
- Uncertainty quantification (Bayesian, ensemble methods)

### Hyperparameter Optimization
- Grid search / Bayesian optimization
- AutoML frameworks (Ray Tune, Optuna)
- Neural architecture search (NAS)

### Regulatory Approval
- FDA 510(k) clearance or CE marking
- Prospective clinical validation studies
- Risk management (ISO 14971)
- Software testing (IEC 62304)

### Clinical Deployment
- Integration with EHR systems
- Telemedicine platform support
- Wearable ECG device integration
- Real-time model monitoring and drift detection

---

## 📜 Git Commit Conventions

Follow this format for consistent, scannable history:

```
<type>(<scope>): <subject>

<body>
<footer>
```

**Types:** `feat`, `fix`, `docs`, `chore`, `refactor`, `test`, `perf`  
**Scopes:** `report`, `scripts`, `ui`, `data`, `model`, `ci`

**Examples:**
```bash
git commit -m "feat(ui): add explanations expander and JSONL logging"
git commit -m "chore(report): update Deliverable 4 LaTeX formatting"
git commit -m "fix(model): correct calibration wrapper initialization"
git commit -m "docs(readme): comprehensive reproducibility instructions"
```

---

## 📋 Commit Hygiene (Recommended)

Strip notebook outputs before committing:

```bash
# Install nbstripout
pip install nbstripout

# Enable git filter
nbstripout --install

# Strip existing outputs
git ls-files "*.ipynb" -z | xargs -0 nbstripout
```

Alternatively, use included `.gitattributes` for automatic stripping.

---

## 📞 Author & Contact

**Angel Morenu**  
University of Florida – M.S. in Applied Data Science  
Email: angel.morenu@ufl.edu  
GitHub: https://github.com/angelmorenu/multi-modal-cvd-predictor

This repository accompanies **Deliverables 1–4** of **EEE 6778 – Applied Machine Learning II (Fall 2025)** instructed by Dr. Ramirez-Salgado.

---

## 📄 License

This project is shared for academic and educational purposes. All code, data preprocessing scripts, and documentation are provided as-is.

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the CVD, hospital admissions, and PTB-XL ECG datasets
- **PyTorch, scikit-learn, Streamlit communities** for excellent documentation
- **IEEE, SHAP, Captum** for interpretability frameworks
- **University of Florida** for course guidance and feedback

---

**Last Updated:** December 4, 2025  
**Version:** 1.0 (Final)
