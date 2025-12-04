#!/usr/bin/env python3
"""Run SHAP (KernelExplainer) on the transformed tabular vector for the example input.
Falls back to permutation importance plot if shap is not installed.
Saves result to `figures/shap_force_example.png`.
"""
import os
import numpy as np
from pathlib import Path
import json
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import MultiModalCVD, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
MODEL_PATH = ART / 'model.pt'
TRANSFORMER = ART / 'tabular_transformer.joblib'
META = ART / 'tabular_meta.json'
EX_ECG = ART / 'example_ecg_000.npy'

# load transformer and meta
transformer = joblib.load(TRANSFORMER) if TRANSFORMER.exists() else None
expected_cols = None
if META.exists():
    meta = json.load(open(META,'r'))
    num_cols = meta.get('num_cols',[]) or []
    cat_cols = meta.get('cat_cols',[]) or []
    expected_cols = list(num_cols)+list(cat_cols)

# infer tab_dim
tab_dim = 32
if transformer is not None and expected_cols:
    tab_dim = transformer.transform(pd.DataFrame({c:[0] for c in expected_cols})).shape[1]

# load model
model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)
if MODEL_PATH.exists():
    load_checkpoint(model, str(MODEL_PATH), map_location='cpu')
else:
    print('No model checkpoint found; exiting')
    raise SystemExit(1)

# prepare tabular transformed features for example input
if expected_cols:
    raw = {c:0 for c in expected_cols}
else:
    raw = {'age':35,'systolic':130,'cholesterol':200,'bmi':27.5,'smoker':0,'sex_male':1}

if transformer is not None and expected_cols:
    tab_df = pd.DataFrame([raw]).reindex(columns=expected_cols, fill_value=0)
    tab_X = transformer.transform(tab_df)
else:
    tab_X = np.asarray(list(raw.values()))[None, :].astype(np.float32)

# fixed ECG tensor
ecg = np.load(EX_ECG).astype(np.float32)
import torch
ecg_tensor = torch.from_numpy(ecg[None,None,:]).float()

# wrapper function taking transformed tabular vectors and returning class-1 probs

def predict_from_transformed(x):
    # x: (n, tab_dim)
    xt = torch.from_numpy(np.asarray(x,dtype=np.float32)).float()
    with torch.inference_mode():
        probs = model.predict_proba(xt, ecg_tensor.repeat(xt.shape[0],1,1))
        return probs.detach().cpu().numpy()[:,1]

# Try shap first
try:
    import shap
    print('Running SHAP KernelExplainer (may be slow)')
    background = np.zeros((1, tab_X.shape[1]), dtype=np.float32)
    explainer = shap.KernelExplainer(predict_from_transformed, background)
    shap_values = explainer.shap_values(tab_X, nsamples=200)
    # shap_values for binary is list; take index 1
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    # plot force_plot as a matplotlib figure
    plt.figure(figsize=(6,3))
    shap.summary_plot(sv, tab_X, feature_names=[f'f{i}' for i in range(tab_X.shape[1])], show=False)
    out = FIG / 'shap_force_example.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print('Saved SHAP summary to', out)
except Exception as e:
    print('SHAP not available or failed:', e)
    # fallback: simple leave-one-out importance by zeroing each transformed feature
    print('Computing leave-one-out feature importance fallback')
    baseline = predict_from_transformed(tab_X)[0]
    importances = []
    for i in range(tab_X.shape[1]):
        X2 = tab_X.copy()
        X2[:, i] = 0.0
        v = predict_from_transformed(X2)[0]
        # importance = decrease in prob when feature is zeroed
        importances.append(baseline - v)
    importances = np.array(importances)
    # plot
    plt.figure(figsize=(8,3))
    idx = np.argsort(importances)[::-1]
    names = [f'f{i}' for i in range(len(importances))]
    plt.barh([names[i] for i in idx], importances[idx])
    plt.xlabel('Change in predicted prob (baseline - zeroed)')
    plt.title('Tabular feature importance (zero-out fallback)')
    out = FIG / 'shap_force_example.png'
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print('Saved fallback importance to', out)
