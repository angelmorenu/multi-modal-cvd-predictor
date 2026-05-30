#!/usr/bin/env bash
set -euo pipefail

python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_001/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_002/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_003/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_004/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_005/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_006/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_007/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_008/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_009/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_010/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_011/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_012/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_013/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_014/artifacts --epochs 1 --batch_size 8 --seed 42
python src/train.py --processed_dir data/processed --artifacts_dir results/repro/fold_015/artifacts --epochs 1 --batch_size 8 --seed 42
