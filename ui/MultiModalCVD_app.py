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
import json
from datetime import datetime
from pathlib import Path

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

# If available, we'll try to read the expected raw tabular column order from metadata
EXPECTED_TAB_COLS = None

# Default tab_dim if transformer not available (use 12 for compact demo vector)
tab_dim = 12
if transformer is not None:
    # quick way to infer transformed dimension: fit dummy or inspect attributes
    try:
        # Prefer reading saved metadata that lists the original training columns
        meta_path = os.path.join(ART_DIR, "tabular_meta.json")
        expected_cols = None
        if os.path.exists(meta_path):
            try:
                meta = json.load(open(meta_path, "r", encoding="utf-8"))
                num_cols = meta.get("num_cols", []) or []
                cat_cols = meta.get("cat_cols", []) or []
                # Build expected raw columns in the same order as training
                expected_cols = list(num_cols) + list(cat_cols)
            except Exception:
                expected_cols = None

        # If we have expected columns, keep them in global var and use them;
        # otherwise fall back to inspecting transformer
        if expected_cols:
            EXPECTED_TAB_COLS = expected_cols
            dummy_df = pd.DataFrame({c: [0] for c in expected_cols})
        else:
            # fallback: try to inspect transformer internals (best-effort)
            dummy_cols = []
            try:
                for name, trans, cols in transformer.transformers_:
                    if cols is None:
                        continue
                    if isinstance(cols, list):
                        dummy_cols.extend(cols)
                if not dummy_cols:
                    dummy_cols = [f"f{i}" for i in range(8)]
            except Exception:
                dummy_cols = [f"f{i}" for i in range(8)]
            dummy_df = pd.DataFrame({c: [0] for c in dummy_cols})

        tab_dim = transformer.transform(dummy_df).shape[1]
    except Exception:
        pass

# --- Logging and explanation helpers (defined early so UI can call them) ---
LOG_PATH = Path("results") / "predictions_log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def append_log(entry: dict):
    """Append a JSON line to `results/predictions_log.jsonl`.

    This runs during Streamlit interactions; errors are reported to the UI.
    """
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        st.warning(f"Failed to write prediction log: {e}")


def heuristic_demo_risk(tab_row: pd.DataFrame, ecg_signal: np.ndarray) -> float:
    """Compute a simple deterministic demo risk (0..1) from tabular row and ECG.

    This provides consistent, interpretable demo outputs when the trained model
    is missing or raises an error.
    """
    try:
        # normalize a few key features if present
        age = float(tab_row.get("age", [55])[0])
        systolic = float(tab_row.get("systolic", [120])[0])
        cholesterol = float(tab_row.get("cholesterol", [200])[0])
        bmi = float(tab_row.get("bmi", [25.0])[0])
        smoker = float(tab_row.get("smoker", [0])[0])

        # simple standardized scores (roughly scaled 0..1)
        a = (age - 30.0) / 50.0
        s = (systolic - 100.0) / 60.0
        c = (cholesterol - 150.0) / 150.0
        b = (bmi - 18.0) / 22.0

        # ECG signal energy / variability heuristic
        ecg = np.asarray(ecg_signal, dtype=np.float32)
        rms = float(np.sqrt(np.mean(ecg ** 2))) if ecg.size > 0 else 0.0
        # map rms to a small contribution
        e = min(max((rms - 0.02) / 0.2, 0.0), 1.0)

        score = 0.35 * a + 0.25 * s + 0.2 * c + 0.1 * b + 0.1 * smoker + 0.1 * e
        # logistic squashing and clamp
        prob = 1.0 / (1.0 + np.exp(-3.0 * (score - 0.5)))
        return float(min(max(prob, 0.0), 1.0))
    except Exception:
        return 0.35


# Compatibility wrapper for Streamlit image parameter deprecation
def _st_image(path, **kwargs):
    """Display an image using `use_container_width` when supported,
    otherwise fall back to `use_column_width` for older Streamlit versions.
    """
    try:
        # Preferred newer argument
        st.image(path, use_container_width=True, **kwargs)
    except TypeError:
        # Older Streamlit: use_column_width
        st.image(path, use_column_width=True, **kwargs)

def show_explanations(thumbnail: bool = True):
    """Display SHAP and ECG saliency images if they exist.

    If `thumbnail` is True, show smaller images side-by-side.
    """
    shap_path = Path("figures") / "shap_force_example.png"
    saliency_path = Path("figures") / "ecg_saliency.png"
    if thumbnail:
        cols = st.columns(2)
        if shap_path.exists():
            with cols[0]:
                st.caption("Tabular feature explanation (SHAP)")
                _st_image(str(shap_path))
        else:
            with cols[0]:
                st.info("No SHAP image found: figures/shap_force_example.png")
        if saliency_path.exists():
            with cols[1]:
                st.caption("ECG saliency map")
                _st_image(str(saliency_path))
        else:
            with cols[1]:
                st.info("No ECG saliency image found: figures/ecg_saliency.png")
    else:
        shown = False
        if shap_path.exists():
            st.subheader("Tabular feature explanation (SHAP)")
            _st_image(str(shap_path))
            shown = True
        if saliency_path.exists():
            st.subheader("ECG saliency map")
            _st_image(str(saliency_path))
            shown = True
        if not shown:
            st.info("Explanation images not found (look for figures/shap_force_example.png and figures/ecg_saliency.png).")

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
        # Ensure we pass the transformer the same raw columns used at training.
        cols_to_use = EXPECTED_TAB_COLS if EXPECTED_TAB_COLS is not None else [c for c in tab_df.columns]
        tab_df_in = tab_df.reindex(columns=cols_to_use, fill_value=0)
        tab_X = transformer.transform(tab_df_in)
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

# Quick example loader button
if st.button("Use example ECG (demo)"):
    try:
        example_path = os.path.join(ART_DIR, "example_ecg_000.npy")
        if os.path.exists(example_path):
            ecg = np.load(example_path).astype(np.float32)
            # mark that an example was loaded so the UI can display the filename
            st.session_state['example_ecg_loaded'] = True
            st.session_state['example_ecg_name'] = os.path.basename(example_path)
            st.success("Loaded example ECG into the session. Scroll down and press Predict.")
        else:
            st.error("Example ECG not found in artifacts/example_ecg_000.npy")
    except Exception as e:
        st.error(f"Failed to load example ECG: {e}")

# Load ECG signal
TARGET_T = 2000  # demo length
ecg = None
if uploaded is not None:
    try:
        if uploaded.name.endswith(".csv"):
            sig = pd.read_csv(uploaded, header=None).values
            ecg = np.asarray(sig)
        elif uploaded.name.endswith(".npy"):
            buf = io.BytesIO(uploaded.read())
            sig = np.load(buf)
            ecg = np.asarray(sig)
    except Exception as e:
        st.error(f"Failed to parse ECG file: {e}")
else:
    # If nothing uploaded, create a synthetic ECG-like waveform for demo
    t = np.linspace(0, 1, TARGET_T)
    ecg = (0.1 * np.sin(2 * np.pi * 5 * t) + 0.01 * np.random.randn(TARGET_T)).astype(np.float32)

# Coerce uploaded arrays to a 1D single-lead signal. Many users upload files
# that contain shape (T, C) or (C, T) or have extra singleton dimensions.
if ecg is not None:
    try:
        ecg = np.asarray(ecg)
        # If array has singleton dimensions, squeeze them
        if ecg.ndim == 0:
            ecg = ecg.reshape(-1).astype(np.float32)
        elif ecg.ndim > 1:
            # If any dimension equals 1, squeeze safely
            if 1 in ecg.shape:
                ecg = ecg.squeeze()
            else:
                # Ambiguous multi-channel data: choose the first channel as a single-lead
                # Prefer treating rows as timepoints when rows >= cols
                if ecg.shape[0] >= ecg.shape[1]:
                    st.warning(f"Uploaded ECG has shape {ecg.shape}; using column 0 as single-lead signal.")
                    ecg = ecg[:, 0]
                else:
                    st.warning(f"Uploaded ECG has shape {ecg.shape}; using row 0 as single-lead signal.")
                    ecg = ecg[0, :]
        ecg = np.asarray(ecg, dtype=np.float32)
    except Exception as e:
        st.error(f"Failed to coerce ECG to 1D signal: {e}")
        ecg = None

# Show where example came from (helpful when pressing button)
if 'ecg' in locals() and isinstance(ecg, np.ndarray):
    if uploaded is not None:
        src_info = 'uploaded file'
    elif st.session_state.get('example_ecg_loaded', False):
        src_info = f"example: {st.session_state.get('example_ecg_name')}"
    else:
        src_info = 'session/example or synthetic'
else:
    src_info = None

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
if st.session_state.get('example_ecg_loaded', False):
    st.write("**ECG file:**", st.session_state.get('example_ecg_name'))
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

    # Allow forcing a demo-high-risk for presentation/testing when model is missing
    force_high = False
    if not os.path.exists(MODEL_PATH):
        force_high = st.checkbox("Force demo high risk (show 100%)", value=False)

    # Let user choose to prefer model or heuristic fallback (useful when debugging)
    use_model_pref = st.checkbox("Use saved model for prediction if available", value=True)

    # Softening controls for demo: temperature scaling and optional clipping
    st.markdown("**Demo safeguards (soften overly-confident logits)**")
    temperature = float(st.slider("Logit temperature (T) — higher = softer probabilities", 0.1, 10.0, 1.0, step=0.1))
    do_clip = st.checkbox("Clip logits to limit extreme values", value=False)
    clip_limit = None
    if do_clip:
        clip_limit = float(st.number_input("Clip limit (abs max) for logits", min_value=1.0, max_value=100.0, value=10.0, step=1.0))

    used_model = False
    probs = None
    orig_logits = None
    adjusted_logits = None
    logits_stats = None
    try:
        if os.path.exists(MODEL_PATH) and use_model_pref:
            # Run forward pass to obtain raw logits
            with torch.no_grad():
                logits_tensor = model.forward(tab_tensor, ecg_tensor)
            orig_logits = logits_tensor[0].detach().cpu().numpy()

            # Optionally clip extreme logits
            adjusted = orig_logits.copy()
            if do_clip and clip_limit is not None:
                adjusted = np.clip(adjusted, -abs(clip_limit), abs(clip_limit))

            # Apply temperature scaling (divide logits by T)
            if temperature is not None and temperature > 0:
                adjusted = adjusted / float(temperature)

            # Softmax in a numerically stable way
            exps = np.exp(adjusted - np.max(adjusted))
            probs = exps / (exps.sum() + 1e-12)
            adjusted_logits = adjusted.tolist()
            logits_stats = {
                "orig_max": float(orig_logits.max()),
                "orig_min": float(orig_logits.min()),
                "orig_mean": float(orig_logits.mean()),
                "orig_std": float(orig_logits.std()),
                "diff_1_minus_0": float(orig_logits[1] - orig_logits[0]) if orig_logits.size > 1 else None,
            }
            used_model = True
            risk_prob = float(probs[1]) if probs.size > 1 else float(probs[0])
        else:
            # no model artifact present — use heuristic fallback
            risk_prob = heuristic_demo_risk(tab_df, ecg)
    except Exception as e:
        # If anything fails (e.g., shape mismatch), show the error and use deterministic heuristic
        st.warning(f"Prediction failed with error: {e}. Using deterministic demo fallback.")
        risk_prob = heuristic_demo_risk(tab_df, ecg)

    if force_high:
        risk_prob = 1.0

    # Visible debug info (helpful when predictions look wrong)
    st.markdown("**Debug — model / fallback diagnostics**")
    debug_info = {
        "model_path_exists": os.path.exists(MODEL_PATH),
        "used_model": bool(used_model),
        "forced_high": bool(force_high),
        "tab_X_shape": tab_X.shape if 'tab_X' in locals() else None,
        "ecg_shape": ecg.shape if ecg is not None else None,
        "risk_prob_raw": float(risk_prob),
        "raw_probs": probs.tolist() if (probs is not None) else None,
        "orig_logits": orig_logits.tolist() if orig_logits is not None else None,
        "adjusted_logits": adjusted_logits,
        "logits_stats": logits_stats,
        "temperature_used": float(temperature),
        "clip_limit_used": float(clip_limit) if clip_limit is not None else None,
    }
    st.json(debug_info)

    # Show last-layer weight/bias stats (best-effort) and logit stats when model used
    if used_model:
        try:
            try:
                last_lin = model.fusion.mlp[-1]
                w = last_lin.weight.detach().cpu().numpy()
                b = last_lin.bias.detach().cpu().numpy()
                st.write("Final linear weight max/mean:", float(w.max()), float(w.mean()))
                st.write("Final linear bias:", b.tolist())
            except Exception:
                pass
            if logits_stats is not None:
                st.write("Logits stats (orig):", logits_stats)
                st.write("Adjusted logits (post-clip/temp):", adjusted_logits)
        except Exception:
            pass

    st.metric("Estimated CVD Risk (probability)", f"{risk_prob:.2%}")

    # Safety disclaimer
    st.info("Model output is uncalibrated and for demonstration only — not medical advice.")
    if risk_prob >= 0.99:
        st.warning("Extremely high model confidence (>=99%). This may indicate overconfidence or model bias.")
    if risk_prob <= 0.01:
        st.warning("Extremely low model confidence (<=1%). This may indicate overconfidence or model bias.")

    if risk_prob >= 0.5:
        st.error("High risk signal — consult a clinician.")
    elif risk_prob >= 0.3:
        st.warning("Moderate risk — consider follow-up.")
    else:
        st.success("Low risk — keep monitoring.")

    # --- Logging (append JSONL) ---
    log_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "inputs": {
            "age": int(age),
            "systolic": int(systolic),
            "cholesterol": int(cholesterol),
            "bmi": float(bmi),
            "smoker": bool(smoker == "Yes"),
            "sex": str(sex),
            "uploaded_ecg": uploaded.name if uploaded is not None else None,
        },
        "model": {
            "model_path_exists": os.path.exists(MODEL_PATH),
            "transformer_present": transformer is not None,
        },
        "prediction": {
            "risk_prob": float(risk_prob)
        }
    }
    append_log(log_entry)

    # --- Explanations panel ---
    with st.expander("Explanations and interpretability (if available)"):
        show_explanations()

st.caption("Demo mode: if no trained model/transformer is found, predictions are illustrative only.")

