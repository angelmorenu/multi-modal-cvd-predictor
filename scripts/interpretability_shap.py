#!/usr/bin/env python3
"""Generate a SHAP bar summary image for tabular features or a placeholder.

This script attempts to compute SHAP values using a small RandomForest trained
on the processed tabular data. If `shap` is not installed, it creates a simple
placeholder image explaining how to enable SHAP.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.ensemble import RandomForestClassifier

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
figs = root / "figures"
figs.mkdir(exist_ok=True)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def run_feature_importance():
    """Train a RandomForest and plot feature importances as a fallback when SHAP is unavailable."""
    X = np.load(processed / "tabular_train_X.npy")
    y = np.load(processed / "tabular_train_y.npy")
    m = min(X.shape[0], y.shape[0])
    X, y = X[:m], y[:m]

    clf = RandomForestClassifier(n_estimators=200, random_state=0)
    clf.fit(X, y)
    imp = clf.feature_importances_
    idx = np.argsort(imp)[::-1]

    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(range(len(imp)), imp[idx])
    ax.set_xticks(range(len(imp)))
    ax.set_xticklabels([f"f{i}" for i in idx], rotation=45)
    ax.set_ylabel("Feature importance (MDI)")
    fig.tight_layout()
    fig.savefig(figs / "shap_force_example.png", dpi=150)
    print("Wrote fallback feature-importance image: figures/shap_force_example.png")

def run_shap():
    X = np.load(processed / "tabular_train_X.npy")
    y = np.load(processed / "tabular_train_y.npy")
    m = min(X.shape[0], y.shape[0])
    X, y = X[:m], y[:m]

    clf = RandomForestClassifier(n_estimators=100, random_state=0)
    clf.fit(X, y)

    explainer = shap.TreeExplainer(clf)
    # explain a few samples
    shap_vals = explainer.shap_values(X[:50])
    # mean absolute shap value per feature (class 1)
    mean_abs = np.abs(shap_vals[1]).mean(axis=0)

    fig, ax = plt.subplots(figsize=(6,3))
    idx = np.argsort(mean_abs)[::-1]
    ax.bar(range(len(mean_abs)), mean_abs[idx])
    ax.set_xticks(range(len(mean_abs)))
    ax.set_xticklabels([f"f{i}" for i in idx], rotation=45)
    ax.set_ylabel("Mean |SHAP value|")
    fig.tight_layout()
    fig.savefig(figs / "shap_force_example.png", dpi=150)
    print("Wrote figures/shap_force_example.png")

if __name__ == '__main__':
    if SHAP_AVAILABLE:
        try:
            run_shap()
        except Exception as e:
            print("SHAP failed at runtime; falling back to feature importance:", e)
            run_feature_importance()
    else:
        print("shap not available; writing RandomForest feature-importance fallback image")
        run_feature_importance()
