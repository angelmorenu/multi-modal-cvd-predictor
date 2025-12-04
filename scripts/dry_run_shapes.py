#!/usr/bin/env python3
import os
import json
import numpy as np
import joblib
import torch
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MultiModalCVD, load_checkpoint

ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model.pt")
TRANSFORMER_PATH = os.path.join(ART_DIR, "tabular_transformer.joblib")
META_PATH = os.path.join(ART_DIR, "tabular_meta.json")

print("Working dir:", os.getcwd())
# load transformer
transformer = None
if os.path.exists(TRANSFORMER_PATH):
    transformer = joblib.load(TRANSFORMER_PATH)
    print("Loaded transformer")
else:
    print("No transformer found at", TRANSFORMER_PATH)

# read expected cols
expected_cols = None
if os.path.exists(META_PATH):
    with open(META_PATH, "r") as f:
        meta = json.load(f)
    num_cols = meta.get("num_cols", []) or []
    cat_cols = meta.get("cat_cols", []) or []
    expected_cols = list(num_cols) + list(cat_cols)
    print("Expected raw cols:", expected_cols)

# infer tab_dim
tab_dim = 32
if transformer is not None:
    if expected_cols:
        dummy_df = {c: [0] for c in expected_cols}
        import pandas as pd
        df = pd.DataFrame(dummy_df)
        tab_dim = transformer.transform(df).shape[1]
    else:
        # fallback: try to inspect
        import pandas as pd
        cols = [f"f{i}" for i in range(8)]
        df = pd.DataFrame({c: [0] for c in cols})
        tab_dim = transformer.transform(df).shape[1]
print("Inferred tab_dim:", tab_dim)

# build model
model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)
if os.path.exists(MODEL_PATH):
    try:
        load_checkpoint(model, MODEL_PATH, map_location="cpu")
        print("Loaded model checkpoint")
    except Exception as e:
        print("Failed to load checkpoint:", e)

# build dummy inputs
tab_x = np.zeros((1, tab_dim), dtype=np.float32)
# synthetic ECG length
T = 2000
ecg = 0.1 * np.sin(2 * np.pi * 5 * np.linspace(0, 1, T))
ecg = ecg.astype(np.float32)

tab_tensor = torch.from_numpy(tab_x).float()
ecg_tensor = torch.from_numpy(ecg[None, None, :]).float()
print("tab_tensor", tab_tensor.shape)
print("ecg_tensor", ecg_tensor.shape)

with torch.inference_mode():
    probs = model.predict_proba(tab_tensor, ecg_tensor)
    print("probs shape", probs.shape)
    print("probs:", probs)

print("Done")
