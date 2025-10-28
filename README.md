# 🩺 Multi-Modal Predictors for Cardiovascular Disease Risk and Outcomes

**Author:** Angel Morenu  
**Email:** angel.morenu@ufl.edu  
**Affiliation:** University of Florida, M.S. in Applied Data Science  
**Course:** EEE 6778 – Applied Machine Learning II (Fall 2025)  
**Instructor:** Dr. Ramirez-Salgado  
 

---

## 🧠 Project Overview

Cardiovascular disease (CVD) remains the leading global cause of death. Existing predictive models often rely on a single data modality, limiting their accuracy and applicability in real-world settings.

This project develops a **multi-modal machine learning system** that combines:

- Tabular demographic data  
- Hospital admission records  
- Physiological ECG signals  

to improve CVD risk prediction and make model outputs explainable, accessible, and deployable on edge devices.

---

## 📦 GitHub Repository

**Repository:** [https://github.com/angelmorenu/multi-modal-cvd-predictor](https://github.com/angelmorenu/multi-modal-cvd-predictor)

**Clone and navigate:**
```bash
git clone https://github.com/angelmorenu/multi-modal-cvd-predictor.git
cd multi-modal-cvd-predictor
```

---

## 📊 Datasets Used

| Dataset | Description | Link |
|----------|-------------|------|
| **Cardiovascular Diseases** | Demographics and lifestyle features | [Kaggle Dataset](https://www.kaggle.com/datasets/mexwell/cardiovascular-diseases) |
| **Hospital Admissions** | Clinical visit and diagnostic records | [Kaggle Dataset](https://www.kaggle.com/datasets/ashishsahani/hospital-admissions-data) |
| **PTB-XL ECG** | 12-lead ECG signals and annotations | [Kaggle Dataset](https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset-reformatted) |

---

## 🏗️ Project Architecture

This hybrid workflow uses:

- **scikit-learn** for preprocessing and tabular baselines  
- **PyTorch** for ECG deep learning and feature fusion  
- **Streamlit** for the user interface  
- Conceptual **Edge AI deployment** (e.g., smartwatch scenario)  

![Architecture Diagram](docs/multimodal_cvd_architecture.png)

---

## ⚙️ Installation and Environment Setup

Install with **conda** (choose the appropriate platform file):

```bash
# macOS (Intel or Apple Silicon M1/M2/M3; uses CPU/MPS)
conda env create -f environment.macos.yml
conda activate cvd_predictor

# Linux/Windows with NVIDIA GPU (CUDA 11.8)
conda env create -f environment.cuda.yml
conda activate cvd_predictor
```

**Notes:**
- On Apple Silicon, PyTorch automatically uses the **MPS backend** when available.
- The default `environment.yml` is cross-platform (CPU/MPS-friendly) and should work on macOS, Intel, or Linux.
- If conda dependency resolution is slow, try installing **mamba** for faster solving:
  ```bash
  conda install -c conda-forge mamba
  ```

---

## 🚀 Running the Project

### 1. Run the Setup and EDA Notebook
```bash
jupyter notebook notebooks/setup.ipynb
```

### 2. Launch the Streamlit UI
```bash
# Activate the conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate cvd_predictor

# Run the Streamlit app on a free port
streamlit run ui/MultiModalCVD_app.py --server.port 8502 --server.headless true
```

This will open a local browser window where you can input demographic features, upload an ECG signal, and view the model’s predicted risk probability.


## 📁 Repository Structure

```
multi_modal_cvd_project/
├── data/                     # Local/raw datasets (not versioned)
├── notebooks/
│   └── setup.ipynb           # Environment check & EDA
├── src/
│   ├── preprocess.py         # Data preprocessing (tabular/ECG)
│   └── model.py              # Modeling, training, fusion network
├── ui/
│   ├── app.py                # Main Streamlit app
│   └── MultiModalCVD_app.py  # Alternate prototype
├── docs/
│   └── multimodal_cvd_architecture.png
├── results/                  # Visualizations, metrics (optional)
├── environment.yml           # Cross-platform environment (CPU/MPS)
├── environment.cuda.yml      # CUDA-enabled environment
├── environment.macos.yml     # macOS (Intel/ARM) environment
├── README.md                 # Project overview & instructions
├── Morenu_Project Deliverable 1.docx
└── Morenu_Angel_Deliverable1_TechnicalBlueprint.pdf
=======
 
```

---

## 📦 Deliverables

- Complete project repository (code + documentation)  
- Jupyter notebook for setup and validation (`notebooks/setup.ipynb`)  
- Streamlit application (`ui/app.py`)  
- Technical Blueprint Report (PDF version submitted on Canvas)  
- Environment YAML files for reproducibility  

---

## 🤖 Responsible AI Goals

- **Fairness:** Evaluate across age/gender/race subgroups.  
- **Transparency:** Incorporate SHAP and attention map explanations.  
- **Efficiency:** Support lightweight edge deployment for on-device inference.  

---

## 👤 Author & Contact

**Angel Morenu**  
University of Florida, M.S. in Applied Data Science  
📧 **angel.morenu@ufl.edu**  
📁 [GitHub Repository](https://github.com/angelmorenu/multi-modal-cvd-predictor)

---

> *This repository accompanies Deliverable 1 of EEE 6778 – Applied Machine Learning II (Fall 2025), demonstrating reproducible, ethical, and explainable multi-modal AI for cardiovascular risk prediction.*
