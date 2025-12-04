#!/usr/bin/env python3
"""Create a 2x2 performance dashboard: ROC, PR, Confusion, Prob hist
Usage:
  python scripts/perf_dashboard.py --run fusion
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, ConfusionMatrixDisplay


def load_run(run):
    base = Path('results')
    y_true = base / f"{run}_y_true.npy"
    y_prob = base / f"{run}_y_prob.npy"
    y_pred = base / f"{run}_y_pred.npy"
    fallback = (base / 'y_true.npy', base / 'y_prob.npy', base / 'y_pred.npy')
    if y_true.exists() and y_prob.exists() and y_pred.exists():
        return np.load(y_true), np.load(y_prob), np.load(y_pred)
    if fallback[0].exists() and fallback[1].exists():
        print(f"[warn] using fallback generic results/y_*.npy for run={run}")
        return np.load(fallback[0]), np.load(fallback[1]), np.load(fallback[2])
    raise FileNotFoundError("Missing result files")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', default='fusion')
    args = p.parse_args()

    yt, yp, yhat = load_run(args.run)

    fig, ax = plt.subplots(2,2, figsize=(10,8))

    # ROC
    fpr, tpr, _ = roc_curve(yt, yp)
    ax[0,0].plot(fpr, tpr, label=f'AUC={auc(fpr,tpr):.2f}')
    ax[0,0].plot([0,1],[0,1],'--', color='gray')
    ax[0,0].set_title('ROC')
    ax[0,0].legend()

    # PR
    prec, rec, _ = precision_recall_curve(yt, yp)
    ax[0,1].plot(rec, prec)
    ax[0,1].set_title('PR')
    ax[0,1].set_xlabel('Recall')
    ax[0,1].set_ylabel('Precision')

    # Confusion
    ConfusionMatrixDisplay.from_predictions(yt, (yp>=0.5).astype(int), ax=ax[1,0], colorbar=False)
    ax[1,0].set_title('Confusion')

    # Prob hist
    ax[1,1].hist(yp, bins=20, edgecolor='black')
    ax[1,1].set_title('Prob. Histogram')

    fig.tight_layout()
    out = Path('figures')
    out.mkdir(exist_ok=True)
    outp = out / f'perf_dashboard_{args.run}.png'
    fig.savefig(outp, dpi=300)
    print('Saved', outp)

if __name__ == '__main__':
    main()
