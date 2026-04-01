"""
GEA Survival Risk Stratifier — Snakemake Pipeline

Full DAG: download → preprocess → build_features → model → figures
"""

import os
from pathlib import Path

# Configuration
configfile: "config/config.yaml"

# Directories
RAW_DIR = config["paths"]["raw_data"]
PROCESSED_DIR = config["paths"]["processed_data"]
RESULTS_DIR = config["paths"]["results"]
FIGURES_DIR = config["paths"]["figures"]
DB_PATH = config["paths"]["database"]

# Create directories
Path(RAW_DIR).mkdir(parents=True, exist_ok=True)
Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
Path(FIGURES_DIR).mkdir(parents=True, exist_ok=True)

# ===== RULES =====

rule all:
    """Final target: all figures and model summary."""
    input:
        figures=expand(f"{FIGURES_DIR}/{{fig}}.png", fig=["km_curves_msi", "forest_plot_cox", "tmb_distribution", "ddr_burden_distribution"]),
        model_summary=f"{RESULTS_DIR}/cox_model_summary.csv",
        risk_scores=f"{RESULTS_DIR}/risk_scores.csv"


rule generate_synthetic_data:
    """Generate synthetic TCGA-STAD data for portfolio demonstration."""
    output:
        clinical=f"{RAW_DIR}/TCGA-STAD_clinical.json",
        mutations=f"{RAW_DIR}/TCGA-STAD_mutations.csv",
        msi=f"{RAW_DIR}/TCGA-STAD_msi_status.csv",
        immune=f"{RAW_DIR}/TCGA-STAD_immune_subtypes.csv"
    shell:
        """
        python scripts/python/generate_synthetic_data.py
        """


rule preprocess:
    """Preprocess clinical and mutation data, load into DuckDB."""
    input:
        clinical=f"{RAW_DIR}/TCGA-STAD_clinical.json",
        mutations=f"{RAW_DIR}/TCGA-STAD_mutations.csv",
        msi=f"{RAW_DIR}/TCGA-STAD_msi_status.csv",
        immune=f"{RAW_DIR}/TCGA-STAD_immune_subtypes.csv"
    output:
        db=DB_PATH
    shell:
        """
        python scripts/python/preprocess.py {input.clinical} {input.mutations} {input.msi} {input.immune} {output.db}
        """


rule build_feature_matrix:
    """Construct multi-omic feature matrix: MSI, TMB, DDR burden, immune subtype."""
    input:
        db=DB_PATH,
        mutations=f"{RAW_DIR}/TCGA-STAD_mutations.csv"
    output:
        features=f"{PROCESSED_DIR}/feature_matrix.csv"
    shell:
        """
        python scripts/python/build_feature_matrix.py {input.db} {input.mutations} {output.features}
        """


rule survival_model:
    """Fit Cox PH and Kaplan-Meier survival models."""
    input:
        features=f"{PROCESSED_DIR}/feature_matrix.csv"
    output:
        model=f"{RESULTS_DIR}/cox_model.pkl",
        summary=f"{RESULTS_DIR}/cox_model_summary.csv",
        risk_scores=f"{RESULTS_DIR}/risk_scores.csv"
    shell:
        """
        python scripts/python/survival_model.py {input.features} {RESULTS_DIR}
        """


rule generate_figures:
    """Generate publication-quality figures."""
    input:
        features=f"{PROCESSED_DIR}/feature_matrix.csv",
        model=f"{RESULTS_DIR}/cox_model.pkl"
    output:
        km=f"{FIGURES_DIR}/km_curves_msi.png",
        forest=f"{FIGURES_DIR}/forest_plot_cox.png",
        tmb=f"{FIGURES_DIR}/tmb_distribution.png",
        ddr=f"{FIGURES_DIR}/ddr_burden_distribution.png"
    shell:
        """
        python scripts/python/figures.py {input.features} {input.model} {FIGURES_DIR}
        """


rule run_tests:
    """Run pytest unit tests."""
    output:
        test_report="results/test_report.txt"
    shell:
        """
        pytest tests/ -v > {output.test_report} 2>&1 || true
        """
