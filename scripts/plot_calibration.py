#!/usr/bin/env python3
"""Plot calibration curve and probability histogram for a run.
Usage:
  python scripts/plot_calibration.py --run fusion
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve


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

    figdir = Path('figures')
    figdir.mkdir(exist_ok=True)

    prob_true, prob_pred = calibration_curve(yt, yp, n_bins=10, strategy='quantile')
    plt.figure()
    plt.plot(prob_pred, prob_true, marker='o')
    plt.plot([0,1],[0,1],'--', color='gray')
    plt.xlabel('Mean predicted prob')
    plt.ylabel('Fraction of positives')
    plt.title(f'Calibration ({args.run})')
    plt.tight_layout()
    out1 = figdir / f'calibration_curve_{args.run}.png'
    plt.savefig(out1, dpi=300)
    print('Saved', out1)
    plt.close()

    plt.figure()
    plt.hist(yp, bins=20, edgecolor='black')
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.title(f'Prediction Probability Histogram ({args.run})')
    plt.tight_layout()
    out2 = figdir / f'prob_hist_{args.run}.png'
    plt.savefig(out2, dpi=300)
    print('Saved', out2)

if __name__ == '__main__':
    main()
