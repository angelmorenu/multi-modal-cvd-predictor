#!/usr/bin/env python3
"""Standardize an external ECG dataset for validation.

This script is intentionally flexible because PTBDB/CPSC mirrors often ship in
different layouts (WFDB records, .npy arrays, or CSV exports). It scans an input
directory, loads ECG signals, aligns them to a common length, and writes:

- `external_ecg_<dataset_name>_signals.npy`
- `external_ecg_<dataset_name>_labels.npy` (if labels are available)
- `external_ecg_<dataset_name>_manifest.csv`

It supports optional WFDB loading if `wfdb` is installed. For WFDB datasets,
labels can be provided via a CSV file with columns `record_id,label`.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np

# Optional import of wfdb for WFDB record reading; not a hard dependency
@dataclass
class Record:
    record_id: str
    path: str
    label: Optional[int]
    num_channels: int
    num_samples: int

# Helper function to load labels from a CSV file with columns record_id,label
def load_labels_csv(labels_csv: Optional[str]) -> dict:
    if not labels_csv:
        return {}
    mapping = {}
    with open(labels_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get("record_id") or row.get("record") or row.get("id")
            value = row.get("label") or row.get("target") or row.get("y")
            if key is None or value is None:
                continue
            mapping[str(key)] = int(value)
    return mapping

# Optional function to load WFDB records if wfdb is installed; returns signal and fields
def maybe_load_wfdb(record_path: Path):
    try:
        import wfdb  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("wfdb is not installed; cannot read WFDB records") from exc

    base = record_path.with_suffix("")
    signal, fields = wfdb.rdsamp(str(base))
    return signal, fields

# Helper function to load ECG signal from .npy, .csv, or .hea (WFDB) files
def load_signal(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr[:1].astype(np.float32)
    if path.suffix.lower() == ".csv":
        arr = np.loadtxt(path, delimiter=",")
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
            arr = arr.T
        return arr[:1].astype(np.float32)
    if path.suffix.lower() == ".mat":
        # Try scipy.io.loadmat first (common v4/v6 .mat files), then fall back to h5py for v7.3
        try:
            from scipy.io import loadmat

            mat = loadmat(path)
            # Heuristic: pick the first numeric ndarray value present
            arr = None
            for v in mat.values():
                if isinstance(v, np.ndarray):
                    # skip MATLAB meta-entries
                    if v.size == 0:
                        continue
                    arr = v
                    break
            if arr is None:
                raise ValueError("no numeric array found in .mat file")
            if arr.ndim == 1:
                arr = arr[None, :]
            elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            else:
                # collapse higher dims if present
                if arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
            return arr[:1].astype(np.float32)
        except Exception:
            try:
                import h5py

                with h5py.File(path, "r") as f:
                    def find_dataset(group):
                        for k in group:
                            obj = group[k]
                            if isinstance(obj, h5py.Dataset):
                                return obj[()]
                            if isinstance(obj, h5py.Group):
                                res = find_dataset(obj)
                                if res is not None:
                                    return res
                        return None

                    arr = find_dataset(f)
                    if arr is None:
                        raise ValueError(".mat HDF5 contains no datasets")
                    arr = np.asarray(arr)
                    if arr.ndim == 1:
                        arr = arr[None, :]
                    elif arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                        arr = arr.T
                    return arr[:1].astype(np.float32)
            except Exception as exc:
                raise ValueError(f"Failed to read .mat file: {exc}")
    if path.suffix.lower() == ".hea":
        signal, _ = maybe_load_wfdb(path)
        signal = np.asarray(signal)
        if signal.ndim == 1:
            signal = signal[None, :]
        elif signal.ndim == 2:
            signal = signal.T
        return signal[:1].astype(np.float32)
    raise ValueError(f"Unsupported ECG file type: {path}")

# Helper function to align ECG signal to a target length by padding or cropping
def align_signal(signal: np.ndarray, ecg_len: int) -> np.ndarray:
    if signal.ndim == 1:
        signal = signal[None, :]
    if signal.shape[1] < ecg_len:
        pad = np.zeros((signal.shape[0], ecg_len - signal.shape[1]), dtype=signal.dtype)
        return np.concatenate([signal, pad], axis=1)
    return signal[:, :ecg_len]

# Helper function to discover ECG records in the input directory with supported extensions
def discover_records(input_root: Path) -> List[Path]:
    exts = {".npy", ".csv", ".hea"}
    records = []
    for path in input_root.rglob("*"):
        if path.suffix.lower() in exts:
            if path.suffix.lower() == ".dat":
                continue
            records.append(path)
    return sorted(records)

# Helper function to infer label from parent directory name (e.g., "positive" -> 1, "negative" -> 0)
def infer_label_from_parent(path: Path) -> Optional[int]:
    parent = path.parent.name.lower()
    if parent in {"1", "pos", "positive", "abnormal", "disease", "diseased"}:
        return 1
    elif parent in {"0", "neg", "negative", "normal", "healthy"}:
        return 0
    return None


def infer_label_from_header(path: Path) -> Optional[int]:
    if path.suffix.lower() != ".hea":
        path = path.with_suffix(".hea")
    if not path.exists():
        return None

    try:
        text = path.read_text(errors="ignore").lower()
    except Exception:
        return None

    # CPSC/PhysioNet style diagnosis line, e.g. "#Dx: 426783006" or multiple codes
    # For binary external validation, treat pure normal rhythm as 0 and any other diagnosis as 1.
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("#dx:"):
            codes_part = line.split(":", 1)[1].strip()
            codes = {c.strip() for c in codes_part.split(",") if c.strip()}
            if not codes:
                return None
            if codes == {"426783006"}:  # normal sinus rhythm
                return 0
            return 1

    if "myocardial infarction" in text:
        return 1
    if "acute infarction" in text:
        return 1
    if "healthy control" in text or "normal sinus rhythm" in text:
        return 0
    return None

# Dry-run function to check if ECG and label files can be loaded and if model forward pass works
def build_dataset(input_root: Path, dataset_name: str, ecg_len: int, labels_csv: Optional[str] = None):
    label_map = load_labels_csv(labels_csv)
    records = discover_records(input_root)
    if not records:
        raise SystemExit(f"No ECG records found under {input_root}")

    signals = []
    labels = []
    manifest: List[Record] = []

    for record_path in records:
        try:
            signal = load_signal(record_path)
        except Exception as exc:
            print(f"[WARN] Skipping {record_path}: {exc}")
            continue

        aligned = align_signal(signal, ecg_len)
        record_id = record_path.stem
        label = label_map.get(record_id)
        if label is None:
            label = infer_label_from_header(record_path)
        if label is None:
            label = infer_label_from_parent(record_path)
        signals.append(aligned)
        labels.append(-1 if label is None else int(label))
        manifest.append(
            Record(
                record_id=record_id,
                path=str(record_path),
                label=label,
                num_channels=int(aligned.shape[0]),
                num_samples=int(aligned.shape[1]),
            )
        )

    if not signals:
        raise SystemExit("No ECG signals could be loaded.")

    return np.asarray(signals, dtype=np.float32), np.asarray(labels, dtype=np.int64), manifest

# Helper function to save manifest CSV with columns record_id, path, label, num_channels, num_samples
def save_manifest(manifest: Iterable[Record], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["record_id", "path", "label", "num_channels", "num_samples"])
        writer.writeheader()
        for rec in manifest:
            writer.writerow(
                {
                    "record_id": rec.record_id,
                    "path": rec.path,
                    "label": rec.label if rec.label is not None else "",
                    "num_channels": rec.num_channels,
                    "num_samples": rec.num_samples,
                }
            )

# Helper function to load data from processed_dir and concatenate all splits into X, y arrays
def main():
    parser = argparse.ArgumentParser(description="Prepare external ECG data for validation")
    parser.add_argument("--input-root", required=True, help="Root directory containing PTBDB/CPSC records")
    parser.add_argument("--dataset-name", required=True, choices=["ptbdb", "cpsc", "other"], help="Dataset label")
    parser.add_argument("--ecg-len", type=int, default=2000, help="Target ECG length after padding/cropping")
    parser.add_argument("--labels-csv", default=None, help="Optional CSV with columns record_id,label")
    parser.add_argument("--out-dir", default="data/external", help="Output directory")
    args = parser.parse_args()

    input_root = Path(args.input_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    signals, labels, manifest = build_dataset(input_root, args.dataset_name, args.ecg_len, labels_csv=args.labels_csv)

    signals_path = out_dir / f"external_ecg_{args.dataset_name}_signals.npy"
    labels_path = out_dir / f"external_ecg_{args.dataset_name}_labels.npy"
    manifest_path = out_dir / f"external_ecg_{args.dataset_name}_manifest.csv"

    np.save(signals_path, signals)
    np.save(labels_path, labels)
    save_manifest(manifest, manifest_path)

    label_count = int(np.sum(labels >= 0))
    print(f"Saved signals: {signals_path}")
    print(f"Saved labels:  {labels_path} (labeled={label_count}/{len(labels)})")
    print(f"Saved manifest: {manifest_path}")


if __name__ == "__main__":
    main()
