#!/usr/bin/env python3
"""Compute metrics for available runs and write results/metric_summary.json and LaTeX table.
Usage:
  python scripts/metric_summary.py --runs tabular ecg fusion
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix


def summarize(y_true, y_prob, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-9)
    spec = tn / (tn + fp + 1e-9)
    return dict(
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred)),
        roc_auc=float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true))>1 else float('nan'),
        sensitivity=float(sens),
        specificity=float(spec),
    )


def load_run(run):
    base = Path('results')
    # try results/{run}_*.npy then fallback to results/*
    y_true_path = base / f"{run}_y_true.npy"
    y_prob_path = base / f"{run}_y_prob.npy"
    y_pred_path = base / f"{run}_y_pred.npy"
    fallback_true = base / 'y_true.npy'
    fallback_prob = base / 'y_prob.npy'
    fallback_pred = base / 'y_pred.npy'

    if y_true_path.exists() and y_prob_path.exists() and y_pred_path.exists():
        return np.load(y_true_path), np.load(y_prob_path), np.load(y_pred_path)
    # fallback to generic
    if fallback_true.exists() and fallback_prob.exists() and fallback_pred.exists():
        print(f"[warn] run={run}: using fallback generic results/y_*.npy")
        return np.load(fallback_true), np.load(fallback_prob), np.load(fallback_pred)
    # not available
    print(f"[skip] run={run}: missing result files")
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', default=['tabular', 'ecg', 'fusion'])
    args = p.parse_args()

    rows = {}
    for run in args.runs:
        data = load_run(run)
        if data is None:
            continue
        yt, yp, yhat = data
        if len(yt) == 0:
            print(f"[skip] run={run}: empty arrays")
            continue
        rows[run] = summarize(yt, yp, yhat)

    out = Path('results')
    out.mkdir(exist_ok=True)
    with open(out / 'metric_summary.json', 'w') as f:
        json.dump(rows, f, indent=2)

    # Print LaTeX table snippet with numbers formatted
    def fmt(x):
        if x is None:
            return 'NA'
        try:
            return f"{float(x):.2f}"
        except Exception:
            return str(x)

    print('\nLaTeX table snippet:')
    print('\\begin{table}[!t]')
    print('\\centering')
    print('\\caption{Unimodal vs. Multimodal Performance (Held-Out Test)}')
    print('\\label{tab:uni-vs-multi}')
    print('\\begin{tabular}{lccccc}')
    print('\\toprule')
    print('\\textbf{Model} & Acc & F1 & ROC AUC & Sens & Spec \\\\')
    print('\\midrule')
    order = ['tabular','ecg','fusion']
    names = {'tabular':'Tabular-only','ecg':'ECG-only','fusion':'\\textbf{Fusion (Ours)}'}
    for r in order:
        if r not in rows:
            vals = ['0.00']*5
        else:
            vals = [fmt(rows[r]['accuracy']), fmt(rows[r]['f1']), fmt(rows[r]['roc_auc']), fmt(rows[r]['sensitivity']), fmt(rows[r]['specificity'])]
        line = f"{names[r]} & {vals[0]} & {vals[1]} & {vals[2]} & {vals[3]} & {vals[4]} \\\\"
        if r=='fusion':
            # bold fusion row
            line = line.replace('Fusion (Ours)', '\\textbf{Fusion (Ours)}')
        print(line)
    print('\\bottomrule')
    print('\\end{tabular}')
    print('\\end{table}')


if __name__ == '__main__':
    main()
