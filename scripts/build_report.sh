#!/usr/bin/env bash
# Build script for Deliverable 3
# Usage: ./scripts/build_report.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
REPORT="Report/Project Deliverables 3.tex"
OUTDIR="Report"
mkdir -p "$OUTDIR"
# Run pdflatex twice to resolve references
pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
# If bibtex required (refs.bib exists), run bibtex sequence
if [ -f "$OUTDIR/refs.bib" ]; then
  (cd "$OUTDIR" && bibtex "Project Deliverables 3")
  pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
  pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
fi
echo "Built PDF at $OUTDIR/Project Deliverables 3.pdf"
