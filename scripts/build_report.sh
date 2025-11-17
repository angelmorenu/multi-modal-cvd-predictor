#!/usr/bin/env bash
# Build script for Deliverable 3
# Usage: ./scripts/build_report.sh
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
# Accept optional filename (relative to Report/) or use the clean default
REPORT_FILE=${1:-"Project_Deliverables_3_clean.tex"}
REPORT="Report/$REPORT_FILE"
OUTDIR="Report"
mkdir -p "$OUTDIR"
# Run pdflatex twice to resolve references
pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
# If bibtex required (refs.bib exists), run bibtex sequence
if [ -f "$OUTDIR/refs.bib" ]; then
  (cd "$OUTDIR" && bibtex "$(basename "$REPORT" .tex)")
  pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
  pdflatex -interaction=nonstopmode -output-directory "$OUTDIR" "$REPORT"
fi
echo "Built PDF at $OUTDIR/$(basename "$REPORT" .tex).pdf"
