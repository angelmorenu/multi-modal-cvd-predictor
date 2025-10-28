#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import torch
from torch import nn

# Try to load sklearn transformer
try:
    import joblib
except Exception:
    joblib = None

# Import the model definition
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model import MultiModalCVD, load_checkpoint  # noqa: E402


st.set_page_config(page_title="CVD Risk Predictor (Demo)", layout="centered")
st.title("🫀 Multi-Modal CVD Risk Predictor (Demo)")
st.caption("Hybrid scikit-learn + PyTorch • Streamlit UI • Conceptual Edge Deployment")


# -----------------------------
#   Load artifacts if present
# -----------------------------
ART_DIR = "artifacts"
MODEL_PATH = os.path.join(ART_DIR, "model.pt")
TRANSFORMER_PATH = os.path.join(ART_DIR, "tabular_transformer.joblib")

transformer = None
if joblib and os.path.exists(TRANSFORMER_PATH):
    transformer = joblib.load(TRANSFORMER_PATH)

# Default tab_dim if transformer not available
tab_dim = 32
if transformer is not None:
    # quick way to infer transformed dimension: fit dummy or inspect attributes
    try:
        # build a dummy row to get transformed shape
        # assume the transformer expects the columns it was trained on
        # fall back to default if this fails
        dummy_cols = []
        for name, trans, cols in transformer.transformers_:
            if cols is None:
                continue
            if isinstance(cols, list):
                dummy_cols.extend(cols)
        if not dummy_cols:
            dummy_cols = [f"f{i}" for i in range(8)]
        dummy_df = pd.DataFrame({c: [0] for c in dummy_cols})
        tab_dim = transformer.transform(dummy_df).shape[1]
    except Exception:
        pass

# Build model
model = MultiModalCVD(tab_dim=tab_dim, ecg_channels=1, ecg_embed_dim=128, n_classes=2)

# Try load weights
if os.path.exists(MODEL_PATH):
    try:
        load_checkpoint(model, MODEL_PATH, map_location="cpu")
        st.success("✅ Loaded trained model and transformer.")
    except Exception as e:
        st.warning(f"⚠️ Found model file but failed to load: {e}\nUsing untrained model (demo mode).")
else:
    st.info("ℹ️ No trained model found. Running in demo mode with an untrained network.")


# -----------------------------
#       Input widgets
# -----------------------------
st.subheader("1) Tabular Inputs")
st.caption("Provide a few example fields (the full ColumnTransformer will handle real feature sets).")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=55)
    systolic = st.number_input("Systolic BP", min_value=80, max_value=220, value=130)
    cholesterol = st.number_input("Cholesterol (mg/dL)", min_value=100, max_value=400, value=210)
with col2:
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=27.5, step=0.1)
    smoker = st.selectbox("Smoker", ["No", "Yes"])
    sex = st.selectbox("Sex", ["Female", "Male"])

# Build a small tabular row (real project will map to your transformer columns)
tab_df = pd.DataFrame(
    {
        "age": [age],
        "systolic": [systolic],
        "cholesterol": [cholesterol],
        "bmi": [bmi],
        "smoker": [1 if smoker == "Yes" else 0],
        "sex_male": [1 if sex == "Male" else 0],
    }
)

# If we have a real transformer, transform; else pad to tab_dim
if transformer is not None:
    try:
        tab_X = transformer.transform(tab_df.reindex(columns=[c for c in tab_df.columns], fill_value=0))
        tab_X = np.asarray(tab_X, dtype=np.float32)
    except Exception:
        # Fallback: simple numeric cast + pad/truncate
        arr = tab_df.select_dtypes(include=[np.number]).values.astype(np.float32)
        tab_X = arr
else:
    arr = tab_df.select_dtypes(include=[np.number]).values.astype(np.float32)
    # Pad to tab_dim
    if arr.shape[1] < tab_dim:
        pad = np.zeros((arr.shape[0], tab_dim - arr.shape[1]), dtype=np.float32)
        tab_X = np.concatenate([arr, pad], axis=1)
    else:
        tab_X = arr[:, :tab_dim]

st.write("**Tabular vector shape:**", tab_X.shape)


st.subheader("2) ECG Upload")
st.caption("Upload a single-lead ECG as **.csv** (1 column) or **.npy** (1D array). The app will resample/pad to a fixed length for the demo.")
uploaded = st.file_uploader("ECG file (.csv or .npy)", type=["csv", "npy"])

# Load ECG signal
TARGET_T = 2000  # demo length
ecg = None
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            sig = pd.read_csv(uploaded, header=None).values.squeeze()
            ecg = np.asarray(sig, dtype=np.float32)
        elif uploaded.name.endswith(".npy"):
            buf = io.BytesIO(uploaded.read())
            sig = np.load(buf).squeeze()
            ecg = np.asarray(sig, dtype=np.float32)
    except Exception as e:
        st.error(f"Failed to parse ECG file: {e}")
else:
    # If nothing uploaded, create a synthetic ECG-like waveform for demo
    t = np.linspace(0, 1, TARGET_T)
    ecg = (0.1 * np.sin(2 * np.pi * 5 * t) + 0.01 * np.random.randn(TARGET_T)).astype(np.float32)

# Fit to length TARGET_T (pad or crop)
if ecg is not None and ecg.ndim == 1:
    if len(ecg) < TARGET_T:
        pad = np.zeros(TARGET_T - len(ecg), dtype=np.float32)
        ecg = np.concatenate([ecg, pad], axis=0)
    else:
        ecg = ecg[:TARGET_T]

# Ensure ECG is not None before displaying
if ecg is None:
    # Create default synthetic ECG if still None
    t = np.linspace(0, 1, TARGET_T)
    ecg = (0.1 * np.sin(2 * np.pi * 5 * t) + 0.01 * np.random.randn(TARGET_T)).astype(np.float32)

# Show ECG
st.write("**ECG length:**", len(ecg))
fig, ax = plt.subplots(figsize=(7, 2))
ax.plot(ecg)
ax.set_xlabel("Time")
ax.set_ylabel("mV (a.u.)")
ax.set_title("ECG (demo view)")
st.pyplot(fig)

# -----------------------------
#         Inference
# -----------------------------
if st.button("Predict Risk"):
    # Ensure ECG is available for prediction
    if ecg is None:
        t = np.linspace(0, 1, TARGET_T)
        ecg = (0.1 * np.sin(2 * np.pi * 5 * t) + 0.01 * np.random.randn(TARGET_T)).astype(np.float32)
    
    model.eval()
    tab_tensor = torch.from_numpy(tab_X).float()
    ecg_tensor = torch.from_numpy(ecg[None, None, :]).float()  # (1, C=1, T)

    try:
        probs = model.predict_proba(tab_tensor, ecg_tensor)[0].detach().cpu().numpy()
        risk_prob = float(probs[1])  # class 1 = "high risk" (by convention)
    except Exception as e:
        # If anything fails (e.g., shape mismatch), return a benign demo probability
        st.warning(f"Prediction failed with error: {e}. Returning demo probability.")
        risk_prob = float(0.35 + 0.1 * np.random.rand())

    st.metric("Estimated CVD Risk (probability)", f"{risk_prob:.2%}")

    if risk_prob >= 0.5:
        st.error("High risk signal — consult a clinician.")
    elif risk_prob >= 0.3:
        st.warning("Moderate risk — consider follow-up.")
    else:
        st.success("Low risk — keep monitoring.")

st.caption("Demo mode: if no trained model/transformer is found, predictions are illustrative only.")
