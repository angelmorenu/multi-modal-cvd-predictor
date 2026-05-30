#!/usr/bin/env python3
"""
Verify preprocessed data is ready for model training.
"""

import numpy as np
import json
from pathlib import Path

def check_preprocessing_outputs():
    """Check all preprocessing outputs are valid."""
    
    project_root = Path(__file__).parent.parent
    processed_dir = project_root / "data" / "processed"
    artifacts_dir = project_root / "artifacts"
    
    print("🔍 Checking Preprocessing Outputs\n")
    print("=" * 60)
    
    # Check tabular preprocessing
    print("\n📊 Tabular Data (Cardiovascular)")
    tabular_files = [
        "tabular_train_X.npy",
        "tabular_train_y.npy",
        "tabular_val_X.npy",
        "tabular_val_y.npy",
        "tabular_test_X.npy",
        "tabular_test_y.npy",
    ]
    
    all_exist = True
    for fname in tabular_files:
        path = processed_dir / fname
        if path.exists():
            arr = np.load(path)
            print(f"  ✅ {fname:25s} shape: {arr.shape}")
        else:
            print(f"  ❌ {fname:25s} NOT FOUND")
            all_exist = False
    
    # Check metadata
    meta_path = artifacts_dir / "tabular_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"\n  📋 Metadata:")
        print(f"     Target: {meta['y_col']}")
        print(f"     Numeric features: {len(meta['num_cols'])} columns")
        print(f"     Categorical features: {len(meta['cat_cols'])} columns")
        print(f"     Train/Val/Test split: {meta['n_train']}/{meta['n_val']}/{meta['n_test']}")
    else:
        print(f"  ❌ tabular_meta.json NOT FOUND")
        all_exist = False
    
    # Check transformer
    trans_path = artifacts_dir / "tabular_transformer.joblib"
    if trans_path.exists():
        print(f"  ✅ Transformer saved: {trans_path.name}")
    else:
        print(f"  ❌ Transformer NOT FOUND")
        all_exist = False
    
    print("\n" + "=" * 60)

    # Decide return status but defer splits creation until after we attempt to ensure splits exist
    if all_exist:
        print("\n✅ All preprocessing outputs are ready!")
        print("\n📝 Next steps:")
        print("   1. Load data in your notebook:")
        print("      X_train = np.load('data/processed/tabular_train_X.npy')")
        print("      y_train = np.load('data/processed/tabular_train_y.npy')")
        print("\n   2. Train your models (scikit-learn, PyTorch, etc.)")
        print("\n   3. Use saved transformer for new data:")
        print("      import joblib")
        print("      transformer = joblib.load('artifacts/tabular_transformer.joblib')")
        print("      X_new = transformer.transform(df_new)")
        retval = True
    else:
        print("\n⚠️  Some preprocessing outputs are missing.")
        print("\n   Run preprocessing again:")
        print("   python src/preprocess.py tabular --csv data/cardio.csv --y target")
        retval = False

    # If splits are missing, attempt to create patient-level manifest and splits.
    splits_dir = project_root / "data" / "splits"
    manifest_path = project_root / "data" / "processed" / "manifest.csv"
    if not splits_dir.exists():
        print("\n➡️  No `data/splits/` found. Attempting to create splits from available processed data...")
        # If a manifest exists, call prepare_splits.py
        if manifest_path.exists():
            print(f"  Found manifest at {manifest_path}; running prepare_splits.py")
            import subprocess
            subprocess.check_call(["python", str(project_root / "scripts" / "prepare_splits.py"), "--input-manifest", str(manifest_path), "--out-dir", str(splits_dir)])
            print("  Splits created.")
        else:
            # Try to synthesize a manifest from processed arrays (fallback)
            print("  No manifest found. Creating synthetic manifest from processed arrays (fallback).")
            proc = project_root / "data" / "processed"
            # Prefer tabular_train_y.npy to infer counts
            fallback_n = None
            tab_y = proc / "tabular_train_y.npy"
            ecg_tr = proc / "ecg_train.npy"
            if tab_y.exists():
                try:
                    import numpy as _np
                    arr = _np.load(tab_y)
                    fallback_n = int(len(arr))
                except Exception:
                    fallback_n = None
            elif ecg_tr.exists():
                try:
                    import numpy as _np
                    arr = _np.load(ecg_tr)
                    fallback_n = int(arr.shape[0])
                except Exception:
                    fallback_n = None

            if fallback_n is None:
                print("  Could not infer number of records to build manifest. Please provide a manifest at data/processed/manifest.csv and re-run.")
            else:
                print(f"  Synthesizing manifest with {fallback_n} pseudo-patients.")
                rows = ["patient_id,label"]
                # If label file exists, load labels
                label_vals = None
                if tab_y.exists():
                    try:
                        import numpy as _np
                        label_vals = _np.load(tab_y)
                        if len(label_vals) != fallback_n:
                            label_vals = None
                    except Exception:
                        label_vals = None

                for i in range(fallback_n):
                    pid = f"p_{i:06d}"
                    lbl = int(label_vals[i]) if label_vals is not None else 0
                    rows.append(f"{pid},{lbl}")
                manifest_path.parent.mkdir(parents=True, exist_ok=True)
                manifest_path.write_text("\n".join(rows))
                print(f"  Wrote synthetic manifest to {manifest_path}")
                import subprocess
                subprocess.check_call(["python", str(project_root / "scripts" / "prepare_splits.py"), "--input-manifest", str(manifest_path), "--out-dir", str(splits_dir)])
                print("  Splits created from synthetic manifest.")

    return retval



if __name__ == "__main__":
    check_preprocessing_outputs()
