#!/usr/bin/env python3
"""Generate example single-lead ECG files for the demo UI.

Loads `data/ptbxl_records/record_000.npy`, coerces to 1D (chooses first channel
if multi-channel), resamples (linear interp) to TARGET_T=2000, saves both NPY
and CSV into `artifacts/` so the UI can load them.
"""
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "ptbxl_records" / "record_000.npy"
OUT_DIR = ROOT / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

TARGET_T = 2000

def load_and_coerce(path: Path):
    arr = np.load(path)
    arr = np.asarray(arr)
    # If 2D, try to pick the first channel sensibly
    if arr.ndim > 1:
        if arr.shape[0] >= arr.shape[1]:
            sig = arr[:, 0]
        else:
            sig = arr[0, :]
    else:
        sig = arr
    sig = sig.astype(np.float32)
    return sig


def resample_signal(sig: np.ndarray, target_len: int):
    if len(sig) == target_len:
        return sig
    # linear interpolation resample
    old_idx = np.linspace(0, 1, num=len(sig))
    new_idx = np.linspace(0, 1, num=target_len)
    res = np.interp(new_idx, old_idx, sig).astype(np.float32)
    return res


def main():
    print("Loading:", SRC)
    sig = load_and_coerce(SRC)
    print("Original shape:", sig.shape)
    sig2 = resample_signal(sig, TARGET_T)
    print("Resampled shape:", sig2.shape)

    out_npy = OUT_DIR / "example_ecg_000.npy"
    out_csv = OUT_DIR / "example_ecg_000.csv"
    np.save(out_npy, sig2)
    np.savetxt(out_csv, sig2, delimiter=',')
    print("Saved:", out_npy, out_csv)

if __name__ == '__main__':
    main()
