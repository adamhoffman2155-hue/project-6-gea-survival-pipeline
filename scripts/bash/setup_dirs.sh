#!/usr/bin/env bash
# setup_dirs.sh — Create directory structure for the GEA survival pipeline
# Usage: bash scripts/bash/setup_dirs.sh

set -euo pipefail

echo "Setting up GEA Survival Pipeline directory structure..."

# Data directories
mkdir -p data/raw
mkdir -p data/processed

# Results directories
mkdir -p results/figures
mkdir -p results/models

# Log directory
mkdir -p logs

# Workflow directories (for Snakemake)
mkdir -p workflow/rules
mkdir -p workflow/envs

echo "Directory structure created:"
find data results logs workflow -type d | sort | sed 's/^/  /'
echo ""
echo "Ready to run the pipeline."
echo "  1. Download data:  bash scripts/bash/download_tcga.sh"
echo "  2. Run pipeline:   snakemake --cores 4"
echo "  3. Launch dashboard: streamlit run dashboard/app.py"
