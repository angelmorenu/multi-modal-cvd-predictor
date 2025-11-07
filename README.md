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
multi_modal_cvd_project/
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
│   └── plot_confusion.py   
|   └── ui_demo.png  
│ 
├── figures
|   └── confusion_matrix.png 
|   └── ui_demo.png
│   └── multimodal_cvd_architecture.png
├── results/                  
├── environment.yml           
├── environment.cuda.yml      
├── environment.macos.yml     
├── README.md                 
├── Morenu_Project Deliverable 1.docx
└── Morenu_Deliverable2_IEEE_Report.pdf
```

---

## 📦 Deliverables

- Complete project repository (code + documentation)  
- Jupyter notebooks for setup and evaluation (setup.ipynb, train_eval.ipynb)  
- Streamlit application (ui/app.py and MultiModalCVD_app.py)  
- Technical IEEE report (Deliverable 2, PDF)  
- Environment YAML files for reproducibility

---

## ✅ Reproducibility Instructions

To reproduce Deliverable 2 results:

1. Run the evaluation notebook:
```bash
jupyter notebook notebooks/train_eval.ipynb
```
This trains/evaluates models and saves predictions to:
```bash
results/y_true.npy
results/y_pred.npy
results/y_prob.npy
```

2. Generate the confusion-matrix figure:
```bash
python scripts/plot_confusion.py
```
Output saved to:
```bash
figures/confusion_matrix.png
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
