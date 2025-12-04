#!/usr/bin/env python3
"""Compute gradient-based saliency for example ECG and save figure.
Saves to `figures/ecg_saliency.png`.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import MultiModalCVD, load_checkpoint
import joblib
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
FIG = ROOT / 'figures'
FIG.mkdir(exist_ok=True)
MODEL_PATH = ART / 'model.pt'
TRANSFORMER = ART / 'tabular_transformer.joblib'
META = ART / 'tabular_meta.json'
EX_ECG = ART / 'example_ecg_000.npy'

# load artifacts
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

# example inputs (use neutral tabular zeros)
if expected_cols:
    raw = {c:0 for c in expected_cols}
else:
    raw = {'age':35,'systolic':130,'cholesterol':200,'bmi':27.5,'smoker':0,'sex_male':1}

if transformer is not None and expected_cols:
    import pandas as pd
    tab_df = pd.DataFrame([raw]).reindex(columns=expected_cols, fill_value=0)
    tab_X = transformer.transform(tab_df)
else:
    tab_X = np.asarray(list(raw.values()))[None, :].astype(np.float32)

# load example ecg
ecg = np.load(EX_ECG).astype(np.float32)
# tensors
tab_tensor = torch.from_numpy(np.asarray(tab_X,dtype=np.float32)).float()
ecg_tensor = torch.from_numpy(ecg[None, None, :].astype(np.float32)).float()

# enable grad on ecg input
ecg_tensor.requires_grad_(True)
model.eval()
# forward and get logit for class 1
logits = model(tab_tensor, ecg_tensor)
# If logits shape (1,2), take logit for class 1
if logits.shape[-1] == 2:
    target = logits[0,1]
else:
    target = logits[0]

# backward
model.zero_grad()
target.backward()

# grad w.r.t. input
grads = ecg_tensor.grad.detach().cpu().numpy()[0,0,:]
# saliency as absolute gradient
saliency = np.abs(grads)
# smooth with small window
from scipy.ndimage import gaussian_filter1d
saliency_s = gaussian_filter1d(saliency, sigma=3)

# normalize for plotting
saliency_s = (saliency_s - saliency_s.min()) / (saliency_s.max() - saliency_s.min() + 1e-12)

# plot signal + saliency
plt.figure(figsize=(12,3))
ax1 = plt.gca()
ax1.plot(ecg, color='C0', label='ECG')
ax1.set_ylabel('ECG (a.u.)')
ax2 = ax1.twinx()
ax2.fill_between(np.arange(len(saliency_s)), 0, saliency_s, color='C1', alpha=0.5, label='Saliency')
ax2.set_ylabel('Saliency (norm)')
ax1.set_title('ECG + Gradient Saliency (example_ecg_000)')
ax1.set_xlim(0, len(ecg))
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
plt.tight_layout()
out = FIG / 'ecg_saliency.png'
plt.savefig(out, dpi=150)
print('Saved saliency to', out)
