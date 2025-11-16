#!/usr/bin/env python3
"""Generate unimodal (tabular, ECG) and fusion predictions and update metrics.

This script trains quick baselines on the processed arrays, aligns mismatched
splits by truncation to the minimum length, and writes per-run prediction
artifacts to `results/` and a `results/metric_summary.json` file.

It's intentionally lightweight so it can run in the course environment without
heavy deep-learning dependencies.
"""
import os
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_recall_curve, auc, brier_score_loss, confusion_matrix
import json

root = Path(__file__).resolve().parents[1]
processed = root / "data" / "processed"
results = root / "results"
figs = root / "figures"
results.mkdir(exist_ok=True)
figs.mkdir(exist_ok=True)

def align(X, y):
    m = min(X.shape[0], y.shape[0])
    if X.shape[0] != y.shape[0]:
        print(f"Truncating to min length {m} (X: {X.shape[0]}, y: {y.shape[0]})")
    return X[:m], y[:m]

def probs_from_clf(clf, X):
    try:
        p = clf.predict_proba(X)[:, 1]
    except Exception:
        # fallback to decision_function
        try:
            s = clf.decision_function(X)
            p = (s - s.min()) / (s.max() - s.min())
        except Exception:
            p = np.zeros(X.shape[0])
    return p

def compute_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true))>1 else 0.5
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc),
        "pr_auc": float(pr_auc),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "sensitivity": float(sens),
        "specificity": float(spec),
    }

def run_tabular():
    X_train = np.load(processed / "tabular_train_X.npy")
    y_train = np.load(processed / "tabular_train_y.npy")
    X_test = np.load(processed / "tabular_test_X.npy")
    y_test = np.load(processed / "tabular_test_y.npy")

    X_train, y_train = align(X_train, y_train)
    X_test, y_test = align(X_test, y_test)

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = probs_from_clf(clf, X_test)

    np.save(results / "tabular_y_true.npy", y_test)
    np.save(results / "tabular_y_pred.npy", y_pred)
    np.save(results / "tabular_y_prob.npy", y_prob)

    metrics = compute_metrics(y_test, y_pred, y_prob)
    return metrics

def run_ecg():
    # Simple ECG baseline: extract summary features and train logistic regression
    ecg_train = np.load(processed / "ecg_train.npy")
    ecg_val = np.load(processed / "ecg_val.npy")
    y_train = np.load(processed / "tabular_train_y.npy")
    y_val = np.load(processed / "tabular_val_y.npy")

    # Align
    ecg_train, y_train = align(ecg_train, y_train)
    ecg_val, y_val = align(ecg_val, y_val)

    def feat(X):
        # per-signal: mean, std, max, min
        return np.vstack([X.mean(axis=1), X.std(axis=1), X.max(axis=1), X.min(axis=1)]).T

    Xtr = feat(ecg_train)
    Xte = feat(ecg_val)

    clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=0)
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    y_prob = probs_from_clf(clf, Xte)

    np.save(results / "ecg_y_true.npy", y_val)
    np.save(results / "ecg_y_pred.npy", y_pred)
    np.save(results / "ecg_y_prob.npy", y_prob)

    metrics = compute_metrics(y_val, y_pred, y_prob)
    return metrics

def run_fusion():
    # Try to load fusion model from artifacts; if not available, train a quick stacked logistic on summary features
    try:
        from src.model import MultiModalCVD, load_checkpoint
        import torch
        model_path = root / "artifacts" / "model.pt"
        if model_path.exists():
            print("Found fusion checkpoint; attempting to run inference on test set")
            # Build test arrays aligning to tabular_test and ecg_val if lengths mismatch
            X_tab_test = np.load(processed / "tabular_test_X.npy")
            y_test = np.load(processed / "tabular_test_y.npy")
            # For ECG, prefer ecg_val if length matches y_test; otherwise try to align
            ecg_candidates = np.load(processed / "ecg_val.npy")
            # Align sizes
            minlen = min(X_tab_test.shape[0], ecg_candidates.shape[0], y_test.shape[0])
            X_tab_test = X_tab_test[:minlen]
            ecg = ecg_candidates[:minlen]
            y_test = y_test[:minlen]

            # load model
            model = MultiModalCVD(tab_dim=X_tab_test.shape[1], ecg_channels=1, ecg_embed_dim=128, n_classes=2)
            load_checkpoint(model, str(model_path), map_location="cpu")
            model.eval()
            import torch
            with torch.no_grad():
                tab_t = torch.from_numpy(X_tab_test.astype(np.float32))
                ecg_t = torch.from_numpy(ecg.astype(np.float32))[:, None, :]
                probs = model.predict_proba(tab_t, ecg_t)
                probs = probs.detach().cpu().numpy()[:,1]
                # naive threshold 0.5
                y_pred = (probs >= 0.5).astype(int)
            np.save(results / "fusion_y_true.npy", y_test)
            np.save(results / "fusion_y_pred.npy", y_pred)
            np.save(results / "fusion_y_prob.npy", probs)
            metrics = compute_metrics(y_test, y_pred, probs)
            return metrics
    except Exception as e:
        print("Fusion checkpoint not usable or missing, falling back to stacked logistic. Error:", e)

    # Fallback: concatenate tabular and ECG summary features and train a logistic
    Xtr_tab = np.load(processed / "tabular_train_X.npy")
    ytr = np.load(processed / "tabular_train_y.npy")
    Xtr_tab, ytr = align(Xtr_tab, ytr)
    ecg_tr = np.load(processed / "ecg_train.npy")
    ecg_tr, ytr = align(ecg_tr, ytr)
    def ecg_feat(X):
        return np.vstack([X.mean(axis=1), X.std(axis=1), X.max(axis=1), X.min(axis=1)]).T
    Xtr = np.hstack([Xtr_tab, ecg_feat(ecg_tr)])

    Xte_tab = np.load(processed / "tabular_test_X.npy")
    yte = np.load(processed / "tabular_test_y.npy")
    ecg_val = np.load(processed / "ecg_val.npy")
    # align all test arrays to the same minimum length
    minlen = min(Xte_tab.shape[0], ecg_val.shape[0], yte.shape[0])
    if minlen < max(Xte_tab.shape[0], ecg_val.shape[0], yte.shape[0]):
        print(f"Aligning test splits to min length {minlen} (tab:{Xte_tab.shape[0]}, ecg:{ecg_val.shape[0]}, y:{yte.shape[0]})")
    Xte_tab = Xte_tab[:minlen]
    ecg_val = ecg_val[:minlen]
    yte = yte[:minlen]
    Xte = np.hstack([Xte_tab, ecg_feat(ecg_val)])

    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(Xtr, ytr)
    y_pred = clf.predict(Xte)
    y_prob = probs_from_clf(clf, Xte)

    np.save(results / "fusion_y_true.npy", yte)
    np.save(results / "fusion_y_pred.npy", y_pred)
    np.save(results / "fusion_y_prob.npy", y_prob)

    metrics = compute_metrics(yte, y_pred, y_prob)
    return metrics

def main():
    print("Running tabular baseline...")
    tab_metrics = run_tabular()
    print(json.dumps({"tabular": tab_metrics}, indent=2))

    print("Running ECG baseline...")
    ecg_metrics = run_ecg()
    print(json.dumps({"ecg": ecg_metrics}, indent=2))

    print("Running fusion (checkpoint or stacked fallback)...")
    fusion_metrics = run_fusion()
    print(json.dumps({"fusion": fusion_metrics}, indent=2))

    summary = {"tabular": tab_metrics, "ecg": ecg_metrics, "fusion": fusion_metrics}
    with open(results / "metric_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("Wrote results/metric_summary.json and per-run npy files in results/")

if __name__ == '__main__':
    main()
