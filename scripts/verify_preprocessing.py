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
        return True
    else:
        print("\n⚠️  Some preprocessing outputs are missing.")
        print("\n   Run preprocessing again:")
        print("   python src/preprocess.py tabular --csv data/cardio.csv --y target")
        return False


if __name__ == "__main__":
    check_preprocessing_outputs()
