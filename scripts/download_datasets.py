#!/usr/bin/env python3
"""
Dataset downloader for Multi-Modal CVD project.
Downloads Kaggle datasets and optional PhysioNet ECG datasets.

Setup:
1. Install kaggle: pip install kaggle
2. Get API credentials: https://www.kaggle.com/docs/api
3. Place kaggle.json in ~/.kaggle/

Usage:
    python scripts/download_datasets.py --all
    python scripts/download_datasets.py --cardio
    python scripts/download_datasets.py --hospital
    python scripts/download_datasets.py --ecg
    python scripts/download_datasets.py --ptbdb
    python scripts/download_datasets.py --cpsc
    python scripts/download_datasets.py --all-external
"""
# This script provides a unified interface to download all necessary datasets for the project.
# It uses the Kaggle CLI for downloading public datasets and wfdb for PhysioNet ECG datasets. 
# It also includes a fallback option to create small sample datasets for testing if downloads
# fail or if the user does not have Kaggle access. The downloaded files are organized into a consistent 
# directory structure under `data/raw/`.
import os
import sys
import argparse
import subprocess
from pathlib import Path

# Check for Kaggle CLI availability
def check_kaggle_installed():
    """Check if kaggle CLI is available."""
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def check_wfdb_installed():
    try:
        import wfdb  # noqa: F401
        return True
    except Exception:
        return False


def download_cardio(data_dir: Path):
    """Download cardiovascular disease dataset."""
    print("\n📥 Downloading Cardiovascular Diseases dataset...")
    dataset = "mexwell/cardiovascular-diseases"
    out_dir = data_dir / "raw" / "cardio"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"],
            check=True
        )
        
        # Find the CSV and move it to data root
        csv_files = list(out_dir.glob("*.csv"))
        if csv_files:
            target = data_dir / "cardio.csv"
            csv_files[0].rename(target)
            print(f"✅ Saved to: {target}")
        else:
            print("⚠️  No CSV found in downloaded files")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")


def download_hospital(data_dir: Path):
    """Download hospital admissions dataset."""
    print("\n📥 Downloading Hospital Admissions dataset...")
    dataset = "ashishsahani/hospital-admissions-data"
    out_dir = data_dir / "raw" / "hospital"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"],
            check=True
        )
        
        # Find the CSV and move it to data root
        csv_files = list(out_dir.glob("*.csv"))
        if csv_files:
            target = data_dir / "hospital_admissions.csv"
            csv_files[0].rename(target)
            print(f"✅ Saved to: {target}")
        else:
            print("⚠️  No CSV found in downloaded files")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")


def download_ecg(data_dir: Path):
    """Download PTB-XL ECG dataset."""
    print("\n📥 Downloading PTB-XL ECG dataset (this is large ~800MB)...")
    dataset = "khyeh0719/ptb-xl-dataset-reformatted"
    out_dir = data_dir / "raw" / "ptbxl"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", str(out_dir), "--unzip"],
            check=True
        )
        
        # Create a symlink or move to expected location
        target = data_dir / "ptbxl_records"
        if not target.exists():
            target.symlink_to(out_dir, target_is_directory=True)
        print(f"✅ Saved to: {target}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Download failed: {e}")


def download_physionet_ecg(database_name: str, data_dir: Path, target_subdir: str):
    """Download a PhysioNet ECG dataset using wfdb's downloader.

    Database names are the PhysioNet/WFDB identifiers. We default to:
    - ptbdb: PTB Diagnostic ECG Database
    - cpsc2018: CPSC 2018 Challenge ECG dataset
    """
    print(f"\n📥 Downloading PhysioNet ECG dataset: {database_name}...")
    if not check_wfdb_installed():
        print("❌ wfdb is not installed in this environment.")
        print("   Install with: pip install wfdb")
        return

    try:
        import wfdb  # type: ignore
    except Exception as exc:
        print(f"❌ Unable to import wfdb: {exc}")
        return

    out_dir = data_dir / "raw" / target_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Download all records available for the database.
        wfdb.dl_database(database_name, dl_dir=str(out_dir), keep_subdirs=True)
        print(f"✅ Saved PhysioNet files to: {out_dir}")
        print(f"   Next: run scripts/prepare_external_ecg.py with --input-root {out_dir}")
    except Exception as e:
        print(f"❌ Download failed for {database_name}: {e}")


def download_ptbdb(data_dir: Path):
    download_physionet_ecg("ptbdb", data_dir, "ptbdb")


def download_cpsc(data_dir: Path):
    download_physionet_ecg("cpsc2018", data_dir, "cpsc2018")


def create_sample_data(data_dir: Path):
    """Create small sample CSV files for testing (if download fails)."""
    print("\n🔧 Creating sample test data...")
    
    import pandas as pd
    import numpy as np
    
    # Sample cardiovascular data
    np.random.seed(42)
    n_samples = 100
    cardio_df = pd.DataFrame({
        'age': np.random.randint(30, 80, n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples),
        'height': np.random.randint(150, 195, n_samples),
        'weight': np.random.randint(50, 120, n_samples),
        'ap_hi': np.random.randint(90, 180, n_samples),  # systolic BP
        'ap_lo': np.random.randint(60, 120, n_samples),  # diastolic BP
        'cholesterol': np.random.choice([1, 2, 3], n_samples),
        'gluc': np.random.choice([1, 2, 3], n_samples),
        'smoke': np.random.choice([0, 1], n_samples),
        'alco': np.random.choice([0, 1], n_samples),
        'active': np.random.choice([0, 1], n_samples),
        'target': np.random.choice([0, 1], n_samples)  # CVD present
    })
    cardio_path = data_dir / "cardio.csv"
    cardio_df.to_csv(cardio_path, index=False)
    print(f"✅ Created sample: {cardio_path}")
    
    # Sample hospital data
    hosp_df = pd.DataFrame({
        'patient_id': range(1, n_samples + 1),
        'admission_type': np.random.choice(['Emergency', 'Elective', 'Urgent'], n_samples),
        'diagnosis': np.random.choice(['MI', 'HF', 'Arrhythmia', 'Other'], n_samples),
        'length_of_stay': np.random.randint(1, 15, n_samples),
        'num_procedures': np.random.randint(0, 5, n_samples),
        'outcome': np.random.choice([0, 1], n_samples)  # 0=discharged, 1=readmitted
    })
    hosp_path = data_dir / "hospital_admissions.csv"
    hosp_df.to_csv(hosp_path, index=False)
    print(f"✅ Created sample: {hosp_path}")
    
    # Sample ECG directory with dummy signals
    ecg_dir = data_dir / "ptbxl_records"
    ecg_dir.mkdir(exist_ok=True)
    for i in range(5):
        # Create simple sine wave as dummy ECG
        signal = np.sin(np.linspace(0, 10 * np.pi, 5000)) + np.random.normal(0, 0.1, 5000)
        ecg_path = ecg_dir / f"record_{i:03d}.npy"
        np.save(ecg_path, signal)
    print(f"✅ Created sample ECG records in: {ecg_dir}")

# Main entry point to parse args and trigger downloads or sample data creation
def main():
    parser = argparse.ArgumentParser(description="Download CVD project datasets from Kaggle")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--cardio", action="store_true", help="Download cardiovascular dataset")
    parser.add_argument("--hospital", action="store_true", help="Download hospital dataset")
    parser.add_argument("--ecg", action="store_true", help="Download ECG dataset")
    parser.add_argument("--ptbdb", action="store_true", help="Download PhysioNet PTB Diagnostic ECG database")
    parser.add_argument("--cpsc", action="store_true", help="Download PhysioNet CPSC 2018 ECG database")
    parser.add_argument("--all-external", action="store_true", help="Download all external ECG sources (PTBDB + CPSC)")
    parser.add_argument("--sample", action="store_true", help="Create sample test data instead of downloading")
    parser.add_argument("--data-dir", default="data", help="Data directory (default: data)")
    
    args = parser.parse_args()
    
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / args.data_dir
    data_dir.mkdir(exist_ok=True)
    
    # Check if we should create sample data instead
    if args.sample:
        create_sample_data(data_dir)
        return
    
    # Check for Kaggle CLI
    if not check_kaggle_installed():
        print("❌ Kaggle CLI not found!")
        print("\nTo install:")
        print("  pip install kaggle")
        print("\nThen setup API credentials:")
        print("  1. Go to https://www.kaggle.com/settings/account")
        print("  2. Click 'Create New Token' under API section")
        print("  3. Place kaggle.json in ~/.kaggle/")
        print("\nOr use --sample flag to create test data:")
        print("  python scripts/download_datasets.py --sample")
        sys.exit(1)
    
    # Download requested datasets
    if args.all or args.cardio:
        download_cardio(data_dir)
    
    if args.all or args.hospital:
        download_hospital(data_dir)
    
    if args.all or args.ecg:
        download_ecg(data_dir)

    if args.all_external or args.ptbdb:
        download_ptbdb(data_dir)

    if args.all_external or args.cpsc:
        download_cpsc(data_dir)
    
    if not any([args.all, args.cardio, args.hospital, args.ecg, args.ptbdb, args.cpsc, args.all_external]):
        parser.print_help()
        print("\n💡 Tip: Use --sample to quickly create test data without Kaggle")


if __name__ == "__main__":
    main()
