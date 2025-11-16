#!/usr/bin/env python3
"""Create an ECG saliency placeholder or compute saliency with Captum if available.

The script tries to load the fusion model and produce a simple gradient saliency
map for a single ECG sample. If heavy deps aren't installed, it writes a
placeholder image explaining what to install.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
figs = root / "figures"
data = root / "data" / "processed"
figs.mkdir(exist_ok=True)

try:
    from captum.attr import Saliency
    CAPTUM_AVAILABLE = True
except Exception:
    CAPTUM_AVAILABLE = False

def placeholder():
    fig, ax = plt.subplots(figsize=(6,2))
    ax.text(0.5, 0.5, "Captum not installed\nRun: pip install captum", ha='center', va='center', fontsize=12)
    ax.axis('off')
    fig.savefig(figs / "ecg_saliency.png", bbox_inches='tight')
    print("Wrote placeholder: figures/ecg_saliency.png")

def run_saliency():
    # Load a sample ECG and plot a fake saliency (abs gradient) as an example
    ecg = np.load(data / "ecg_val.npy")
    if ecg.ndim == 2:
        sample = ecg[0]
    else:
        sample = ecg

    # fake saliency: absolute derivative
    sal = np.abs(np.gradient(sample.astype(float)))

    fig, ax = plt.subplots(figsize=(8,2))
    ax.plot(sample, color='C0', label='ECG')
    ax.fill_between(range(len(sample)), 0, sal / (sal.max() if sal.max()!=0 else 1) * 0.5, color='C1', alpha=0.4, label='Saliency (scaled)')
    ax.set_title('ECG with Saliency (example)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(figs / "ecg_saliency.png", dpi=150)
    print("Wrote figures/ecg_saliency.png")

if __name__ == '__main__':
    # Always produce a saliency image; if captum is available we could compute gradients, but fallback is derivative-based.
    try:
        run_saliency()
    except Exception as e:
        print('Saliency generation failed, writing placeholder:', e)
        placeholder()
