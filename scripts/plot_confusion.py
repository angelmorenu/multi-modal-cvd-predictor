#!/usr/bin/env python3
# Plot a confusion matrix PNG for the paper.
# Usage:
#   python scripts/plot_confusion.py --y_true results/y_true.npy --y_pred results/y_pred.npy --out figures/confusion_matrix.png

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--y_true', type=str, default='results/y_true.npy')
    p.add_argument('--y_pred', type=str, default='results/y_pred.npy')
    p.add_argument('--out',    type=str, default='figures/confusion_matrix.png')
    args = p.parse_args()

    y_true = np.load(args.y_true)
    y_pred = np.load(args.y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative','Positive'])

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(4,3), dpi=200)
    disp.plot(values_format='d', cmap='Blues', ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(args.out)
    print(f"Saved {args.out}")

if __name__ == "__main__":
    main()
