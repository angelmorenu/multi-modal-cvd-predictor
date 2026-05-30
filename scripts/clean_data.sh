#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data"

APPLY=false

usage() {
	cat <<'EOF'
Usage:
	scripts/clean_data.sh            # dry-run (default)
	scripts/clean_data.sh --apply    # actually delete stale large files

What it removes:
	- data/external/challenge-2020-1.0.2.zip
	- data/external/CPSC2018.rar
	- data/external/.DS_Store
	- data/external/cpsc_extracted/
	- data/external/cpsc_physionet/ (only if empty)

What it keeps:
	- data/processed/*
	- data/raw/*
	- data/external/external_ecg_*.npy and manifests
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
	usage
	exit 0
fi

if [[ "${1:-}" == "--apply" ]]; then
	APPLY=true
fi

if [[ ! -d "$DATA_DIR" ]]; then
	echo "[ERROR] Data directory not found: $DATA_DIR"
	exit 1
fi

FILES=(
	"$DATA_DIR/external/challenge-2020-1.0.2.zip"
	"$DATA_DIR/external/CPSC2018.rar"
	"$DATA_DIR/external/.DS_Store"
)

DIRS=(
	"$DATA_DIR/external/cpsc_extracted"
)

echo "Repository: $ROOT_DIR"
echo "Mode: $([[ "$APPLY" == true ]] && echo "APPLY" || echo "DRY-RUN")"
echo

echo "Candidates:"
for path in "${FILES[@]}"; do
	if [[ -e "$path" ]]; then
		size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
		echo "  [FILE] $path ($size)"
	fi
done

for path in "${DIRS[@]}"; do
	if [[ -d "$path" ]]; then
		size=$(du -sh "$path" 2>/dev/null | awk '{print $1}')
		echo "  [DIR ] $path ($size)"
	fi
done

EMPTY_CPSC_DIR="$DATA_DIR/external/cpsc_physionet"
if [[ -d "$EMPTY_CPSC_DIR" ]]; then
	if [[ -z "$(ls -A "$EMPTY_CPSC_DIR")" ]]; then
		echo "  [DIR ] $EMPTY_CPSC_DIR (empty)"
	else
		echo "  [KEEP] $EMPTY_CPSC_DIR (not empty)"
	fi
fi

if [[ "$APPLY" != true ]]; then
	echo
	echo "Dry-run only. Re-run with --apply to delete listed stale files."
	exit 0
fi

echo
echo "Deleting..."
for path in "${FILES[@]}"; do
	if [[ -e "$path" ]]; then
		rm -f "$path"
		echo "  removed $path"
	fi
done

for path in "${DIRS[@]}"; do
	if [[ -d "$path" ]]; then
		rm -rf "$path"
		echo "  removed $path"
	fi
done

if [[ -d "$EMPTY_CPSC_DIR" && -z "$(ls -A "$EMPTY_CPSC_DIR")" ]]; then
	rmdir "$EMPTY_CPSC_DIR"
	echo "  removed empty $EMPTY_CPSC_DIR"
fi

echo
echo "Done. Current data usage:"
du -sh "$DATA_DIR"/* 2>/dev/null | sort -rh
