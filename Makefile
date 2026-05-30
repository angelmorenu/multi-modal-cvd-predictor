PYTHON ?= .venv/bin/python
PIP ?= .venv/bin/pip
IMAGE_NAME ?= multi-modal-cvd-predictor
CONTAINER_PORT ?= 8501

.PHONY: help install test dry-run eval train ui docker-build docker-run

help:
	@echo "Targets: install, test, dry-run, eval, train, ui, docker-build, docker-run"

install:
	$(PIP) install -r requirements.txt

test:
	$(PYTHON) -m pytest -q tests/test_preprocess.py tests/test_model_shapes.py tests/test_train_smoke.py tests/test_uncertainty_training.py

dry-run:
	$(PYTHON) scripts/dry_run_shapes.py

eval:
	$(PYTHON) src/eval.py --proc data/processed --art artifacts --out-dir results --ecg-len 2000 --device cpu

train:
	$(PYTHON) -m experiments.train --processed data/processed --epochs 1 --batch-size 8 --device cpu --augment --random-crop-len 2000 --noise-std 0.01 --scale-min 0.95 --scale-max 1.05

ui:
	streamlit run ui/MultiModalCVD_app.py --server.port $(CONTAINER_PORT)

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p $(CONTAINER_PORT):$(CONTAINER_PORT) $(IMAGE_NAME)
