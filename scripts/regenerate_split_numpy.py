#!/usr/bin/env python3
"""
Regenerate train/val/test numpy arrays from stratified patient-level split assignments.

This script reads patient IDs from data/splits/{train,val,test}.csv and remaps
the original tabular_train_X.npy, tabular_train_y.npy, ecg_train.npy to match.

This fixes a critical issue where old preprocessing created a single "train" file
containing all 32 patients (despite being named "train"), and the actual model
training randomly split this all-patient set, causing potential test leakage.

Usage:
    python scripts/regenerate_split_numpy.py --processed data/processed --splits data/splits
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def regenerate_splits(processed_dir, splits_dir, output_dir=None):
    """
    Regenerate train/val/test numpy arrays from split assignments.
    
    Args:
        processed_dir: Path to data/processed (contains original numpy files)
        splits_dir: Path to data/splits (contains train.csv, val.csv, test.csv)
        output_dir: Where to save regenerated numpy files (defaults to processed_dir)
    """
    if output_dir is None:
        output_dir = processed_dir
    
    processed = Path(processed_dir)
    splits = Path(splits_dir)
    output = Path(output_dir)
    
    # Load manifest to map patient_id -> record index
    manifest_path = processed / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    manifest = pd.read_csv(manifest_path)
    patient_to_idx = {pid: idx for idx, pid in enumerate(manifest["patient_id"].values)}
    print(f"Loaded manifest: {len(patient_to_idx)} patients")
    
    # Load original (all-patient) numpy arrays
    orig_X = np.load(processed / "tabular_train_X.npy")
    orig_y = np.load(processed / "tabular_train_y.npy")
    print(f"Loaded original arrays: X shape {orig_X.shape}, y shape {orig_y.shape}")
    
    # Check if ECG exists
    ecg_path = processed / "ecg_train.npy"
    orig_ecg = None
    if ecg_path.exists():
        orig_ecg = np.load(ecg_path)
        print(f"Loaded original ECG: shape {orig_ecg.shape}")
    
    # Process each split
    split_data = {}
    for split_name in ["train", "val", "test"]:
        split_file = splits / f"{split_name}.csv"
        if not split_file.exists():
            print(f"WARNING: {split_file} not found, skipping split '{split_name}'")
            continue
        
        split_df = pd.read_csv(split_file)
        patient_ids = split_df["patient_id"].values
        
        # Map patient IDs to indices
        indices = []
        for pid in patient_ids:
            if pid not in patient_to_idx:
                raise ValueError(f"Patient {pid} not found in manifest")
            indices.append(patient_to_idx[pid])
        
        indices = np.array(indices)
        print(f"\n{split_name.upper()}: {len(indices)} patients, indices {indices}")
        
        # Extract rows for this split
        split_data[split_name] = {
            "X": orig_X[indices],
            "y": orig_y[indices],
        }
        if orig_ecg is not None:
            split_data[split_name]["ecg"] = orig_ecg[indices]
    
    # Save regenerated arrays
    print("\n" + "="*60)
    print("Saving regenerated arrays...")
    print("="*60)
    
    for split_name, data in split_data.items():
        X = data["X"]
        y = data["y"]
        
        X_path = output / f"tabular_{split_name}_X.npy"
        y_path = output / f"tabular_{split_name}_y.npy"
        
        np.save(X_path, X)
        np.save(y_path, y)
        print(f"{split_name}: X saved to {X_path.name} (shape {X.shape})")
        print(f"{split_name}: y saved to {y_path.name} (shape {y.shape})")
        
        # Label distribution
        n_pos = (y == 1).sum()
        n_neg = (y == 0).sum()
        print(f"           Label dist: {n_pos} positive ({n_pos/len(y):.1%}), {n_neg} negative ({n_neg/len(y):.1%})")
        
        if "ecg" in data:
            ecg = data["ecg"]
            ecg_path = output / f"ecg_{split_name}.npy"
            np.save(ecg_path, ecg)
            print(f"{split_name}: ECG saved to {ecg_path.name} (shape {ecg.shape})")
    
    print("\n✓ Regeneration complete. Data leakage risk eliminated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Regenerate train/val/test numpy arrays from split assignments."
    )
    parser.add_argument(
        "--processed",
        type=str,
        default="data/processed",
        help="Path to data/processed (contains original numpy files)"
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="data/splits",
        help="Path to data/splits (contains train.csv, val.csv, test.csv)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for regenerated files (defaults to --processed)"
    )
    
    args = parser.parse_args()
    regenerate_splits(args.processed, args.splits, args.output_dir)
