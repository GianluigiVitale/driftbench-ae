#!/bin/bash
# Build ARTIFACT_APPENDIX.pdf from LaTeX source
# Usage: ./build_pdf.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TEX_FILE="$SCRIPT_DIR/ARTIFACT_APPENDIX.tex"
OUTPUT_DIR="$SCRIPT_DIR/build"

# Check for pdflatex
if ! command -v pdflatex &> /dev/null; then
    echo "pdflatex not found. Installing texlive..."
    apt-get update -qq 2>/dev/null || sudo apt-get update -qq
    apt-get install -y -qq texlive-latex-base texlive-latex-recommended texlive-latex-extra > /dev/null 2>&1 || \
    sudo apt-get install -y -qq texlive-latex-base texlive-latex-recommended texlive-latex-extra > /dev/null 2>&1
    echo "✓ texlive installed"
fi

# Create build directory
mkdir -p "$OUTPUT_DIR"

echo "Building ARTIFACT_APPENDIX.pdf..."

# Run pdflatex twice (for cross-references and hyperlinks)
pdflatex -output-directory="$OUTPUT_DIR" -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1
pdflatex -output-directory="$OUTPUT_DIR" -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1

# Copy PDF to main directory
cp "$OUTPUT_DIR/ARTIFACT_APPENDIX.pdf" "$SCRIPT_DIR/ARTIFACT_APPENDIX.pdf"

echo "✓ PDF built: $SCRIPT_DIR/ARTIFACT_APPENDIX.pdf"

# Clean up build artifacts (keep PDF)
echo "Cleaning build artifacts..."
rm -rf "$OUTPUT_DIR"

echo "Done."
