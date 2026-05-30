"""Threshold and calibration analysis for saved predictions.

Usage:
    python3 scripts/threshold_analysis.py --probs results/y_prob.npy --truth results/y_true.npy

Produces: printed metrics at 0.5 and at F1-optimal threshold, PR AUC, ROC AUC.
"""
import argparse
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    auc,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)


def analyze(probs_path, truth_path):
    yp = np.load(probs_path)
    yt = np.load(truth_path)

    print(f"samples: {len(yt)}")
    print(f"positive count: {int(yt.sum())} negative count: {int((1-yt).sum())}")
    print("prob min/max/mean: {:.4f}/{:.4f}/{:.4f}".format(yp.min(), yp.max(), yp.mean()))

    pred_05 = (yp >= 0.5).astype(int)
    acc_05 = accuracy_score(yt, pred_05)
    f1_05 = f1_score(yt, pred_05, zero_division=0)
    cm_05 = confusion_matrix(yt, pred_05)

    print("\nMetrics at threshold 0.5:")
    print(f"Accuracy: {acc_05:.4f}")
    print(f"F1: {f1_05:.4f}")
    print("Confusion matrix:\n", cm_05)

    try:
        roc = roc_auc_score(yt, yp)
    except Exception:
        roc = float("nan")
    pr_precision, pr_recall, _ = precision_recall_curve(yt, yp)
    pr_auc = auc(pr_recall, pr_precision)
    print("\nROC AUC: {:.4f}".format(roc))
    print("PR AUC: {:.4f}".format(pr_auc))

    ths = np.linspace(0.0, 1.0, 1001)
    best_t = 0.5
    best_f1 = -1
    for t in ths:
        f1 = f1_score(yt, (yp >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    pred_best = (yp >= best_t).astype(int)
    acc_best = accuracy_score(yt, pred_best)
    f1_best = f1_score(yt, pred_best, zero_division=0)
    cm_best = confusion_matrix(yt, pred_best)

    print("\nBest threshold by F1: {:.3f}".format(best_t))
    print("Accuracy at best threshold: {:.4f}".format(acc_best))
    print("F1 at best threshold: {:.4f}".format(f1_best))
    print("Confusion matrix at best threshold:\n", cm_best)
    print("\nPrediction positive fraction at best threshold: {:.4f}".format(pred_best.mean()))

    print("\nClassification report (best threshold):")
    print(classification_report(yt, pred_best, digits=4, zero_division=0))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--probs', default='results/y_prob.npy')
    p.add_argument('--truth', default='results/y_true.npy')
    args = p.parse_args()
    analyze(args.probs, args.truth)
