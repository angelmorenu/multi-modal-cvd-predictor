#!/usr/bin/env python3
"""Fit a Platt-scaling logistic calibrator on validation logits if validation set exists.
Saves calibrator to `artifacts/calibrator.joblib` and a small JSON summary in `results/calibration_summary.json`.
"""
import os
import numpy as np
from pathlib import Path
import joblib
import json
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.model import MultiModalCVD, load_checkpoint
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts'
RES = ROOT / 'results'
RES.mkdir(exist_ok=True)
MODEL_PATH = ART / 'model.pt'
TRANSFORMER = ART / 'tabular_transformer.joblib'
META = ART / 'tabular_meta.json'

# validation arrays
TAB_X_PATH = ROOT / 'data' / 'processed' / 'tabular_val_X.npy'
TAB_Y_PATH = ROOT / 'data' / 'processed' / 'tabular_val_y.npy'
ECG_VAL_PATH = ROOT / 'data' / 'processed' / 'ecg_val.npy'

if not MODEL_PATH.exists():
    print('No model checkpoint; exiting')
    raise SystemExit(1)

if not TAB_X_PATH.exists() or not TAB_Y_PATH.exists() or not ECG_VAL_PATH.exists():
    print('Validation arrays missing; skipping calibration')
    raise SystemExit(0)

# load transformer/meta
transformer = joblib.load(TRANSFORMER) if TRANSFORMER.exists() else None
expected_cols = None
if META.exists():
    meta = json.load(open(META,'r'))
    num_cols = meta.get('num_cols',[]) or []
    cat_cols = meta.get('cat_cols',[]) or []
    expected_cols = list(num_cols)+list(cat_cols)

# infer tab_dim and build model
if transformer is not None and expected_cols:
    tab_dim = transformer.transform(pd.DataFrame({c:[0] for c in expected_cols})).shape[1]
else:
    tab_dim = 32

model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)
load_checkpoint(model, str(MODEL_PATH), map_location='cpu')

# load validation data
tab_X_val = np.load(TAB_X_PATH)
tab_y_val = np.load(TAB_Y_PATH)
ecg_val = np.load(ECG_VAL_PATH)

# ensure shapes consistent
N = min(tab_X_val.shape[0], tab_y_val.shape[0], ecg_val.shape[0])
print('Validation sizes (tab_X, tab_y, ecg):', tab_X_val.shape[0], tab_y_val.shape[0], ecg_val.shape[0])
print('Using N =', N)
# Build transformed tabular inputs if transformer available
if transformer is not None and expected_cols:
    # assume raw tabular X not available, but the preprocessed file might already be the transformed X
    # check shape: if tab_X_val.shape[1] equals transformer output dim, use directly
    try:
        dummy = transformer.transform(pd.DataFrame({c:[0] for c in expected_cols}))
        trans_dim = dummy.shape[1]
    except Exception:
        trans_dim = tab_X_val.shape[1]
    if tab_X_val.shape[1] == trans_dim:
        tab_trans = tab_X_val
    else:
        # if tab_X_val appears raw, attempt transform
        try:
            raw_df = pd.DataFrame(tab_X_val)
            tab_trans = transformer.transform(raw_df)
        except Exception:
            tab_trans = tab_X_val
else:
    tab_trans = tab_X_val

# prepare tensors and get logits
import torch
model.eval()
logits_list = []
# ecg_val may be shape (N,T) or (N,1,T)
for i in range(N):
    ttab = torch.from_numpy(np.asarray(tab_trans[i:i+1],dtype=np.float32)).float()
    e = ecg_val[i]
    e = np.asarray(e)
    if e.ndim == 1:
        et = torch.from_numpy(e[None,None,:].astype(np.float32)).float()
    elif e.ndim == 2:
        # assume (T, C) or (C, T)
        if e.shape[0] < e.shape[1]:
            et = torch.from_numpy(e[None,:,:].astype(np.float32)).float()
        else:
            et = torch.from_numpy(e[None,None,:].astype(np.float32)).float()
    else:
        et = torch.from_numpy(e.astype(np.float32)).float()
    with torch.inference_mode():
        l = model(ttab, et)
        # take logit for class 1
        if l.shape[-1] == 2:
            logit = l.detach().cpu().numpy()[0,1]
        else:
            logit = l.detach().cpu().numpy()[0]
    logits_list.append(logit)

logits = np.array(logits_list)
probs = 1.0/(1.0+np.exp(-logits))

# Fit logistic regression on logits -> true labels (Platt scaling)
clf = LogisticRegression(solver='lbfgs')
clf.fit(logits.reshape(-1,1), tab_y_val[:N])
joblib.dump(clf, ART / 'calibrator.joblib')

# evaluate
cal_probs = clf.predict_proba(logits.reshape(-1,1))[:,1]
brier_before = brier_score_loss(tab_y_val[:N], probs[:N])
brier_after = brier_score_loss(tab_y_val[:N], cal_probs[:N])

summary = {
    'n_val': int(N),
    'brier_before': float(brier_before),
    'brier_after': float(brier_after)
}
with open(RES / 'calibration_summary.json','w') as f:
    json.dump(summary, f, indent=2)

print('Saved calibrator and summary')
