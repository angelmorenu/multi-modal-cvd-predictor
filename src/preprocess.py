#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing scaffolding for Multi-Modal CVD project.

- Tabular preprocessing (scikit-learn ColumnTransformer)
- Clinical/hospital preprocessing (tabular-like)
- ECG feature extraction (optional wfdb/neurokit2; falls back to simple stats)
- Train/val/test splits
- Pipeline persistence (joblib)
- CLI usage for each modality

Author: Angel Morenu
Course: EEE 6778 Applied Machine Learning II (Fall 2025)
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Use importlib to avoid static import errors when joblib is not installed
import importlib
import importlib.util

_spec = importlib.util.find_spec("joblib")
if _spec is not None:
    joblib = importlib.import_module("joblib")
else:
    joblib = None
    warnings.warn("joblib not found; pipeline saving/loading disabled.")

# Optional ECG deps (handled gracefully)
_wfdb_spec = importlib.util.find_spec("wfdb")
if _wfdb_spec is not None:
    wfdb = importlib.import_module("wfdb")
else:
    wfdb = None

_nk_spec = importlib.util.find_spec("neurokit2")
if _nk_spec is not None:
    nk = importlib.import_module("neurokit2")
else:
    nk = None


# ---------- Configuration ----------

@dataclass
class Paths:
    raw_dir: str = "data"
    processed_dir: str = "data/processed"
    artifacts_dir: str = "artifacts"  # transformers, encoders, etc.

    def ensure(self):
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        os.makedirs(self.artifacts_dir, exist_ok=True)


# ---------- Utilities ----------

def load_csv_safely(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def split_train_val_test(
    df: pd.DataFrame,
    y_col: str,
    test_size: float = 0.2,
    val_size: float = 0.1,
    stratify: bool = True,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (train, val, test) with preserved y distribution if stratify=True.
    """
    y = df[y_col]
    strat = y if stratify else None
    df_train, df_temp = train_test_split(
        df, test_size=test_size + val_size, stratify=strat, random_state=random_state
    )
    # Adjust val proportion relative to remaining temp
    val_ratio = val_size / (test_size + val_size)
    y_temp = df_temp[y_col]
    strat_temp = y_temp if stratify else None
    df_val, df_test = train_test_split(
        df_temp, test_size=1 - val_ratio, stratify=strat_temp, random_state=random_state
    )
    return df_train, df_val, df_test


def save_artifact(obj, path: str):
    if joblib is None:
        warnings.warn("joblib not installed; cannot save artifacts.")
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)


def load_artifact(path: str):
    if joblib is None:
        raise RuntimeError("joblib not installed; cannot load artifacts.")
    return joblib.load(path)


# ---------- Tabular / Hospital Pipelines ----------

def infer_feature_types(df: pd.DataFrame, y_col: str) -> Tuple[List[str], List[str]]:
    """
    Heuristic: numeric = number dtype; categorical = object/category/bool.
    Excludes the target column.
    """
    features = [c for c in df.columns if c != y_col]
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    return num_cols, cat_cols


def build_tabular_pipeline(
    num_cols: List[str],
    cat_cols: List[str],
) -> ColumnTransformer:
    """
    Returns a ColumnTransformer:
      - Numeric: impute median + standardize
      - Categorical: impute most_frequent + one-hot
    """
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, num_cols),
            ("cat", cat_pipe, cat_cols),
        ]
    )
    return ct


def preprocess_tabular(
    csv_path: str,
    y_col: str,
    out_dir: str = "data/processed",
    artifacts_dir: str = "artifacts",
    random_state: int = 42,
) -> None:
    """
    Fit a ColumnTransformer on train set; transform splits; save arrays + artifacts.
    Outputs:
      - {out_dir}/tabular_{split}_X.npy
      - {out_dir}/tabular_{split}_y.npy
      - {artifacts_dir}/tabular_transformer.joblib
      - {artifacts_dir}/tabular_meta.json
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    df = load_csv_safely(csv_path)
    if y_col not in df.columns:
        raise KeyError(f"Target column '{y_col}' not in CSV columns.")

    # Split
    tr, va, te = split_train_val_test(df, y_col=y_col, stratify=True, random_state=random_state)

    # Infer features
    num_cols, cat_cols = infer_feature_types(tr, y_col=y_col)

    # Build & fit transformer
    transformer = build_tabular_pipeline(num_cols, cat_cols)
    X_tr = transformer.fit_transform(tr.drop(columns=[y_col]))
    X_va = transformer.transform(va.drop(columns=[y_col]))
    X_te = transformer.transform(te.drop(columns=[y_col]))

    # Ensure arrays are dense (ColumnTransformer already outputs dense via sparse_output=False)
    X_tr = np.asarray(X_tr)
    X_va = np.asarray(X_va)
    X_te = np.asarray(X_te)

    y_tr = tr[y_col].to_numpy()
    y_va = va[y_col].to_numpy()
    y_te = te[y_col].to_numpy()

    # Save arrays
    np.save(os.path.join(out_dir, "tabular_train_X.npy"), X_tr)
    np.save(os.path.join(out_dir, "tabular_train_y.npy"), y_tr)
    np.save(os.path.join(out_dir, "tabular_val_X.npy"), X_va)
    np.save(os.path.join(out_dir, "tabular_val_y.npy"), y_va)
    np.save(os.path.join(out_dir, "tabular_test_X.npy"), X_te)
    np.save(os.path.join(out_dir, "tabular_test_y.npy"), y_te)

    # Save transformer + meta
    if joblib:
        save_artifact(transformer, os.path.join(artifacts_dir, "tabular_transformer.joblib"))
    meta = {
        "y_col": y_col,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
        "n_train": len(tr),
        "n_val": len(va),
        "n_test": len(te),
    }
    with open(os.path.join(artifacts_dir, "tabular_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] Tabular preprocessing complete. Arrays saved to: {out_dir}")
    print(f"[OK] Transformer + meta saved to: {artifacts_dir}")


# For hospital data, we can reuse the same logic (it’s also tabular-like)
def preprocess_hospital(
    csv_path: str,
    y_col: Optional[str] = None,
    out_dir: str = "data/processed",
    artifacts_dir: str = "artifacts",
    random_state: int = 42,
) -> None:
    """
    Same behavior as preprocess_tabular. If y_col is None, only fit/transform features and save X arrays.
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)

    df = load_csv_safely(csv_path)

    if y_col is not None:
        tr, va, te = split_train_val_test(df, y_col=y_col, stratify=True, random_state=random_state)
        num_cols, cat_cols = infer_feature_types(tr, y_col=y_col)
        transformer = build_tabular_pipeline(num_cols, cat_cols)

        X_tr = transformer.fit_transform(tr.drop(columns=[y_col]))
        X_va = transformer.transform(va.drop(columns=[y_col]))
        X_te = transformer.transform(te.drop(columns=[y_col]))

        # Ensure arrays are dense
        X_tr = np.asarray(X_tr)
        X_va = np.asarray(X_va)
        X_te = np.asarray(X_te)

        y_tr = tr[y_col].to_numpy()
        y_va = va[y_col].to_numpy()
        y_te = te[y_col].to_numpy()

        np.save(os.path.join(out_dir, "hospital_train_X.npy"), X_tr)
        np.save(os.path.join(out_dir, "hospital_train_y.npy"), y_tr)
        np.save(os.path.join(out_dir, "hospital_val_X.npy"), X_va)
        np.save(os.path.join(out_dir, "hospital_val_y.npy"), y_va)
        np.save(os.path.join(out_dir, "hospital_test_X.npy"), X_te)
        np.save(os.path.join(out_dir, "hospital_test_y.npy"), y_te)

        if joblib:
            save_artifact(transformer, os.path.join(artifacts_dir, "hospital_transformer.joblib"))

        meta = {"y_col": y_col, "num_cols": num_cols, "cat_cols": cat_cols}
        with open(os.path.join(artifacts_dir, "hospital_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] Hospital preprocessing (with target) complete. Saved to: {out_dir}")
    else:
        # Feature-only transform (fit on all data)
        # Infer with heuristic using all columns (no target to exclude)
        all_cols = list(df.columns)
        num_cols = [c for c in all_cols if pd.api.types.is_numeric_dtype(df[c])]
        cat_cols = [c for c in all_cols if c not in num_cols]

        transformer = build_tabular_pipeline(num_cols, cat_cols)
        X = transformer.fit_transform(df)
        
        # Ensure array is dense
        X = np.asarray(X)

        np.save(os.path.join(out_dir, "hospital_all_X.npy"), X)
        if joblib:
            save_artifact(transformer, os.path.join(artifacts_dir, "hospital_transformer.joblib"))
        meta = {"y_col": None, "num_cols": num_cols, "cat_cols": cat_cols}
        with open(os.path.join(artifacts_dir, "hospital_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        print(f"[OK] Hospital preprocessing (no target) complete. Saved features to: {out_dir}")


# ---------- ECG Feature Extraction (basic scaffold) ----------

def extract_ecg_features_simple(signal: np.ndarray, fs: int = 500) -> dict:
    """
    Minimal, dependency-light ECG summary features (works even without wfdb/neurokit2).
    """
    sig = np.asarray(signal)
    feats = {
        "ecg_mean": float(np.mean(sig)),
        "ecg_std": float(np.std(sig)),
        "ecg_min": float(np.min(sig)),
        "ecg_max": float(np.max(sig)),
        "ecg_median": float(np.median(sig)),
        "ecg_ptp": float(np.ptp(sig)),
    }
    return feats


def extract_ecg_features_neurokit(signal: np.ndarray, fs: int = 500) -> dict:
    """
    If neurokit2 is available, compute richer features (R-peaks, heart rate, etc.).
    """
    if nk is None:
        return extract_ecg_features_simple(signal, fs=fs)
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs)
        _, info = nk.ecg_peaks(cleaned, sampling_rate=fs)
        rate = nk.ecg_rate(info["ECG_R_Peaks"], sampling_rate=fs)
        feats = {
            "hr_mean": float(np.mean(rate)) if len(rate) else np.nan,
            "hr_std": float(np.std(rate)) if len(rate) else np.nan,
            "n_beats": int(len(info.get("ECG_R_Peaks", []))),
        }
        feats.update(extract_ecg_features_simple(signal, fs))
        return feats
    except Exception as e:
        warnings.warn(f"neurokit2 ECG feature extraction failed, falling back to simple: {e}")
        return extract_ecg_features_simple(signal, fs=fs)


def preprocess_ecg_directory(
    ecg_dir: str,
    out_dir: str = "data/processed",
    fs: int = 500,
    use_neurokit: bool = True,
) -> None:
    """
    Walk a directory of ECG samples and produce a feature table.
    Assumes per-record files or arrays you can load; customize as needed.

    Output:
      - {out_dir}/ecg_features.csv
    """
    os.makedirs(out_dir, exist_ok=True)

    records = []
    for root, _, files in os.walk(ecg_dir):
        for f in files:
            path = os.path.join(root, f)
            rec_id = os.path.splitext(f)[0]

            # Example loading logic—customize for your PTB-XL layout.
            signal = None
            if f.endswith(".npy"):
                signal = np.load(path)
            elif f.endswith(".csv"):
                arr = pd.read_csv(path, header=None).values.squeeze()
                signal = np.asarray(arr, dtype=float)
            elif wfdb is not None and (f.endswith(".dat") or f.endswith(".hea")):
                # WFDB typically requires record name without extension
                try:
                    base = path.replace(".dat", "").replace(".hea", "")
                    record = wfdb.rdrecord(base)
                    signal = record.p_signal[:, 0]  # take first lead
                except Exception:
                    signal = None

            if signal is None:
                continue

            feats = extract_ecg_features_neurokit(signal, fs=fs) if use_neurokit else extract_ecg_features_simple(signal, fs=fs)
            feats["record_id"] = rec_id
            records.append(feats)

    if not records:
        warnings.warn(f"No ECG features extracted from: {ecg_dir}")
        return

    df_feats = pd.DataFrame(records).set_index("record_id")
    out_path = os.path.join(out_dir, "ecg_features.csv")
    df_feats.to_csv(out_path)
    print(f"[OK] ECG features saved: {out_path}")


# ---------- CLI ----------

def _add_common_args(p: argparse.ArgumentParser):
    p.add_argument("--out", dest="out_dir", default="data/processed", help="Output directory for processed arrays/tables.")
    p.add_argument("--artifacts", dest="artifacts_dir", default="artifacts", help="Directory to save transformers and metadata.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocessing CLI for Multi-Modal CVD project.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # Tabular
    p_tab = sub.add_parser("tabular", help="Preprocess tabular dataset (e.g., cardiovascular risk CSV).")
    p_tab.add_argument("--csv", required=True, help="Path to CSV file.")
    p_tab.add_argument("--y", required=True, help="Target column name.")
    _add_common_args(p_tab)

    # Hospital
    p_hosp = sub.add_parser("hospital", help="Preprocess hospital/clinical dataset.")
    p_hosp.add_argument("--csv", required=True, help="Path to CSV file.")
    p_hosp.add_argument("--y", required=False, help="Optional target column (if available).")
    _add_common_args(p_hosp)

    # ECG
    p_ecg = sub.add_parser("ecg", help="Extract features from ECG directory (npy/csv/WFDB).")
    p_ecg.add_argument("--dir", required=True, help="Path to ECG directory.")
    p_ecg.add_argument("--fs", type=int, default=500, help="Sampling rate (Hz).")
    p_ecg.add_argument("--simple", action="store_true", help="Use simple stats only (skip neurokit2).")
    p_ecg.add_argument("--out", dest="out_dir", default="data/processed", help="Output directory for feature table.")

    args = parser.parse_args(argv)

    if args.cmd == "tabular":
        preprocess_tabular(csv_path=args.csv, y_col=args.y, out_dir=args.out_dir, artifacts_dir=args.artifacts_dir)
    elif args.cmd == "hospital":
        preprocess_hospital(csv_path=args.csv, y_col=args.y, out_dir=args.out_dir, artifacts_dir=args.artifacts_dir)
    elif args.cmd == "ecg":
        preprocess_ecg_directory(ecg_dir=args.dir, out_dir=args.out_dir, fs=getattr(args, "fs", 500), use_neurokit=not args.simple)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()