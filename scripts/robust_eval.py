#!/usr/bin/env python3
"""Robust evaluation: repeated stratified CV + bootstrap CIs for metrics.

This script trains LogisticRegression and RandomForest using repeated
StratifiedKFold, collects out-of-fold probabilities, and computes bootstrap
95% confidence intervals for AUROC, AUPRC, and Brier score. It also computes
DeLong's test to compare model AUCs using pooled mean probabilities across
repeats. Results are saved to `results/robust_eval.json` and
`results/robust_eval_predictions.npz`.

Note: for small datasets this procedure may overfit; see README for caveats.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import time

# DeLong's test implementation adapted from 
# https://github.com/yandexdataschool/roc_auc_comparison/blob/master/roc_auc_comparison/delong.py
# This is used to compare the AUROC of the two models on the same dataset. It accounts for the 
# correlation between the two sets of predictions since they are made on the same samples.
def compute_midrank(x):
    """Fast midrank computation used by DeLong's test."""
    x = np.asarray(x)
    order = np.argsort(x)
    ranks = np.empty(len(x), dtype=float)
    sorted_x = x[order]
    n = len(x)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        mid = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = mid
        i = j
    return ranks

# The rest of the DeLong's test implementation is adapted from the same source and is used 
# to compute the AUCs and their covariance, which are then used to compute the p-value for 
# the difference in AUCs between the two models.
def fast_delong(predictions_sorted_transposed, label_1_count):
    """Fast DeLong implementation.

    predictions_sorted_transposed: shape (n_classifiers, n_examples)
    label_1_count: number of positive examples in the sorted label array.
    """
    m = int(label_1_count)
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]

    k = predictions_sorted_transposed.shape[0]
    tx = np.empty((k, m), dtype=float)
    ty = np.empty((k, n), dtype=float)
    tz = np.empty((k, m + n), dtype=float)
    for r in range(k):
        tx[r] = compute_midrank(positive_examples[r])
        ty[r] = compute_midrank(negative_examples[r])
        tz[r] = compute_midrank(predictions_sorted_transposed[r])

    aucs = (tz[:, :m].sum(axis=1) - m * (m + 1) / 2.0) / (m * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m

    sx = np.cov(v01)
    sy = np.cov(v10)
    if np.ndim(sx) == 0:
        sx = np.array([[sx]])
    if np.ndim(sy) == 0:
        sy = np.array([[sy]])
    delong_cov = sx / m + sy / n
    return aucs, delong_cov


def delong_roc_test(y_true, preds_one, preds_two):
    """Return AUCs and two-sided p-value for difference in AUCs."""
    y_true = np.asarray(y_true).astype(int)
    preds_one = np.asarray(preds_one)
    preds_two = np.asarray(preds_two)
    order = np.argsort(-y_true)
    y_sorted = y_true[order]
    preds_sorted = np.vstack([preds_one[order], preds_two[order]])
    label_1_count = int(y_sorted.sum())
    aucs, cov = fast_delong(preds_sorted, label_1_count)
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    if var <= 0:
        return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "auc_diff": float(diff), "p_value": 1.0}
    z = abs(diff) / np.sqrt(var)
    from math import erfc, sqrt
    p_value = erfc(z / sqrt(2.0))
    return {"auc_1": float(aucs[0]), "auc_2": float(aucs[1]), "auc_diff": float(diff), "p_value": float(p_value)}


def load_data(processed_dir: str):
    Xs = []
    ys = []
    for s in ["train", "val", "test"]:
        xp = Path(processed_dir) / f"tabular_{s}_X.npy"
        yp = Path(processed_dir) / f"tabular_{s}_y.npy"
        if xp.exists() and yp.exists():
            X = np.load(xp)
            y = np.load(yp).astype(int)
            n = min(len(X), len(y))
            if len(X) != len(y):
                print(f"[WARN] Split={s}: X rows={len(X)} != y rows={len(y)}; truncating to {n}")
                X = X[:n]
                y = y[:n]
            Xs.append(X)
            ys.append(y)
    if not Xs:
        raise SystemExit("No processed tabular data found in " + processed_dir)
    X_all = np.concatenate(Xs, axis=0)
    y_all = np.concatenate(ys, axis=0)
    return X_all, y_all


def bootstrap_metric(y, probs, metric_fn, n_bootstrap=1000, seed=0):
    rng = np.random.RandomState(seed)
    n = len(y)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        try:
            v = metric_fn(y[idx], probs[idx])
        except Exception:
            v = float('nan')
        stats.append(v)
    arr = np.array(stats)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    return float(np.nanmean(arr)), float(lo), float(hi)


def bootstrap_repeat_metric(repeat_metric_values, n_bootstrap=1000, seed=0):
    """Bootstrap CIs over repeated CV runs.

    This resamples the per-repeat metric values rather than samples, which is
    more appropriate for repeated CV summaries on small datasets.
    """
    rng = np.random.RandomState(seed)
    vals = np.asarray(repeat_metric_values, dtype=float)
    n = len(vals)
    stats = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        stats.append(float(np.nanmean(vals[idx])))
    arr = np.asarray(stats)
    lo, hi = np.nanpercentile(arr, [2.5, 97.5])
    return float(np.nanmean(arr)), float(lo), float(hi)


def compute_metrics(y, probs):
    roc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float('nan')
    pr = average_precision_score(y, probs)
    brier = brier_score_loss(y, probs)
    return dict(roc_auc=roc, pr_auc=pr, brier=brier)


def main():
    processed_dir = os.environ.get('PROCESSED_DIR', 'data/processed')
    out_dir = Path(os.environ.get('OUT_DIR', 'results'))
    out_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(processed_dir)
    n_samples = X.shape[0]
    if n_samples < 10:
        print('[WARN] Very small sample size:', n_samples)

    n_splits = int(os.environ.get('N_SPLITS', 5))
    n_repeats = int(os.environ.get('N_REPEATS', 5))
    n_boot = int(os.environ.get('N_BOOT', 1000))

    models = {
        'logistic_regression': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=2000))]),
        'random_forest': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(n_estimators=200, random_state=42))]),
    }

    results = {}
    pooled_predictions = {}
    t0 = time.time()
    for name, model in models.items():
        print('Evaluating model:', name)
        per_repeat_metrics = []
        all_probs = []
        all_y = []
        for r in range(n_repeats):
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + r)
            probs_oof = np.zeros(n_samples, dtype=float)
            y_oof = np.zeros(n_samples, dtype=int)
            # We'll do out-of-fold predictions by splitting indices over the concatenated data
            for train_idx, test_idx in skf.split(X, y):
                Xtr, Xte = X[train_idx], X[test_idx]
                ytr, yte = y[train_idx], y[test_idx]
                model.fit(Xtr, ytr)
                p = model.predict_proba(Xte)[:, 1]
                probs_oof[test_idx] = p
                y_oof[test_idx] = yte
            # compute metrics for this repeat
            m = compute_metrics(y_oof, probs_oof)
            per_repeat_metrics.append(m)
            all_probs.append(probs_oof)
            all_y.append(y_oof)

        repeat_probs = np.vstack(all_probs)
        repeat_y = np.vstack(all_y)
        mean_probs = np.mean(repeat_probs, axis=0)
        mean_y = repeat_y[0]

        # compute mean metrics across repeats
        mean_metrics = {}
        for k in per_repeat_metrics[0].keys():
            vals = np.array([m[k] for m in per_repeat_metrics])
            mean_metrics[k] = dict(mean=float(np.nanmean(vals)), std=float(np.nanstd(vals)))

        repeat_boot = {
            'roc_auc': bootstrap_repeat_metric([m['roc_auc'] for m in per_repeat_metrics], n_bootstrap=n_boot, seed=0),
            'pr_auc': bootstrap_repeat_metric([m['pr_auc'] for m in per_repeat_metrics], n_bootstrap=n_boot, seed=1),
            'brier': bootstrap_repeat_metric([m['brier'] for m in per_repeat_metrics], n_bootstrap=n_boot, seed=2),
        }

        pooled_boot = {
            'roc_auc': bootstrap_metric(mean_y, mean_probs, roc_auc_score, n_bootstrap=n_boot, seed=3),
            'pr_auc': bootstrap_metric(mean_y, mean_probs, average_precision_score, n_bootstrap=n_boot, seed=4),
            'brier': bootstrap_metric(mean_y, mean_probs, brier_score_loss, n_bootstrap=n_boot, seed=5),
        }

        pooled_predictions[name] = {
            'y_true': mean_y.tolist(),
            'repeat_probs': repeat_probs.tolist(),
            'mean_probs': mean_probs.tolist(),
        }

        results[name] = {
            'n_samples': int(n_samples),
            'n_splits': int(n_splits),
            'n_repeats': int(n_repeats),
            'per_repeat_metrics': per_repeat_metrics,
            'mean_metrics': mean_metrics,
            'repeat_bootstrap_ci': {
                'roc_auc': {
                    'mean': repeat_boot['roc_auc'][0],
                    'ci_2.5': repeat_boot['roc_auc'][1],
                    'ci_97.5': repeat_boot['roc_auc'][2],
                },
                'pr_auc': {
                    'mean': repeat_boot['pr_auc'][0],
                    'ci_2.5': repeat_boot['pr_auc'][1],
                    'ci_97.5': repeat_boot['pr_auc'][2],
                },
                'brier': {
                    'mean': repeat_boot['brier'][0],
                    'ci_2.5': repeat_boot['brier'][1],
                    'ci_97.5': repeat_boot['brier'][2],
                },
            },
            'pooled_predictions_bootstrap_ci': {
                'roc_auc': {
                    'mean': pooled_boot['roc_auc'][0],
                    'ci_2.5': pooled_boot['roc_auc'][1],
                    'ci_97.5': pooled_boot['roc_auc'][2],
                },
                'pr_auc': {
                    'mean': pooled_boot['pr_auc'][0],
                    'ci_2.5': pooled_boot['pr_auc'][1],
                    'ci_97.5': pooled_boot['pr_auc'][2],
                },
                'brier': {
                    'mean': pooled_boot['brier'][0],
                    'ci_2.5': pooled_boot['brier'][1],
                    'ci_97.5': pooled_boot['brier'][2],
                },
            },
        }

    model_names = list(models.keys())
    if len(model_names) >= 2:
        first, second = model_names[0], model_names[1]
        y_true = np.array(pooled_predictions[first]['y_true'], dtype=int)
        comp = delong_roc_test(y_true, pooled_predictions[first]['mean_probs'], pooled_predictions[second]['mean_probs'])
        results['comparisons'] = {
            f'{first}_vs_{second}': comp,
            'note': 'DeLong test computed on pooled mean probabilities across repeats.'
        }

    np.savez_compressed(
        out_dir / 'robust_eval_predictions.npz',
        **{f'{name}_y_true': np.array(payload['y_true'], dtype=int) for name, payload in pooled_predictions.items()},
        **{f'{name}_mean_probs': np.array(payload['mean_probs'], dtype=float) for name, payload in pooled_predictions.items()},
        **{f'{name}_repeat_probs': np.array(payload['repeat_probs'], dtype=float) for name, payload in pooled_predictions.items()},
    )

    t1 = time.time()
    results['_runtime_seconds'] = t1 - t0
    out_path = out_dir / 'robust_eval.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print('Saved robust evaluation to', out_path)


if __name__ == '__main__':
    main()
