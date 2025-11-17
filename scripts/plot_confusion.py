#!/usr/bin/env python3
"""Plot a confusion matrix from numpy prediction/true arrays.
Usage examples:
  python scripts/plot_confusion.py --pred results/fusion_y_pred.npy --true results/fusion_y_true.npy --out figures/confusion_matrix.png
  python scripts/plot_confusion.py --run fusion --out figures/confusion_matrix.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay


def load_args():
    p = argparse.ArgumentParser()
    p.add_argument('--pred', help='Path to y_pred numpy .npy file')
    p.add_argument('--true', help='Path to y_true numpy .npy file')
    p.add_argument('--run', help='Run name (fusion, tabular, ecg). If set, loads results/<run>_y_*.npy')
    p.add_argument('--out', help='Output filename (PNG). Default: figures/confusion_matrix.png', default='figures/confusion_matrix.png')
    return p.parse_args()


def main():
    args = load_args()
    if args.run:
        base = Path('results')
        y_true_path = base / f"{args.run}_y_true.npy"
        y_pred_path = base / f"{args.run}_y_pred.npy"
    else:
        y_true_path = Path(args.true) if args.true else Path('results/fusion_y_true.npy')
        y_pred_path = Path(args.pred) if args.pred else Path('results/fusion_y_pred.npy')

    if not y_true_path.exists() or not y_pred_path.exists():
        raise FileNotFoundError(f"Missing result files: {y_true_path} or {y_pred_path}")

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)

    fig, ax = plt.subplots(figsize=(6,5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix')

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outp, dpi=300, bbox_inches='tight')
    print('Saved', outp)


if __name__ == '__main__':
    main()
