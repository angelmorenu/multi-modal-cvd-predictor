#!/usr/bin/env python3
"""Train and evaluate tabular baselines with stratified CV.

Produces JSON output in `results/tabular_baselines.json` containing per-fold
and aggregate metrics (ROC AUC, PR AUC, accuracy).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Helper functions to load data and run CV
def load_data(processed_dir: str):
    """Load and align tabular splits (train/val/test). If X and y lengths differ,
    truncate X to the length of y for that split (warn).
    Returns concatenated X, y across available splits.
    """
    splits = ["train", "val", "test"]
    X_list = []
    y_list = []
    for s in splits:
        Xp = Path(processed_dir) / f"tabular_{s}_X.npy"
        yp = Path(processed_dir) / f"tabular_{s}_y.npy"
        if Xp.exists() and yp.exists():
            X = np.load(Xp)
            y = np.load(yp).astype(int)
            if X.shape[0] != y.shape[0]:
                n = min(X.shape[0], y.shape[0])
                print(f"[WARN] split={s}: X rows={X.shape[0]} != y rows={y.shape[0]}; truncating to {n} rows")
                X = X[:n]
                y = y[:n]
            X_list.append(X)
            y_list.append(y)
    if not X_list:
        raise SystemExit(f"No tabular splits found in {processed_dir}")
    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all

# Run CV for a given estimator and return results
def run_cv(X, y, estimator, cv, scoring=None, n_jobs=1):
    res = cross_validate(estimator, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=False)
    return res

# Main fu=nction to load data, run CV for baselines, and save results
def main():
    processed_dir = os.environ.get("PROCESSED_DIR", "data/processed")
    out_dir = Path(os.environ.get("OUT_DIR", "results"))
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(processed_dir)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "accuracy": "accuracy",
    }

    # Logistic Regression pipeline
    logreg = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000))])
    rf = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))])

    print("Running CV for LogisticRegression...")
    lr_res = run_cv(X, y, logreg, cv=cv, scoring=scoring, n_jobs=1)

    print("Running CV for RandomForest...")
    rf_res = run_cv(X, y, rf, cv=cv, scoring=scoring, n_jobs=1)

# Helper to summarize CV results into dict with per-fold and aggregate metrics
    def summarize(res):
        out = {}
        for k, v in res.items():
            if k.startswith("test_"):
                name = k.replace("test_", "")
                out[name] = {
                    "folds": [float(x) for x in v.tolist()],
                    "mean": float(v.mean()),
                    "std": float(v.std()),
                }
        return out

    results = {
        "logistic_regression": summarize(lr_res),
        "random_forest": summarize(rf_res),
        "n_samples": int(X.shape[0]),
    }

    out_path = out_dir / "tabular_baselines.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved baseline results to {out_path}")


if __name__ == "__main__":
    main()
