#!/usr/bin/env python3
"""
Generate publication-quality figures: KM curves, forest plots, TMB distributions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
import pickle
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["font.size"] = 10


def generate_km_curves(
    feature_matrix_csv: str,
    output_dir: str = "results/figures"
):
    """Generate Kaplan-Meier survival curves stratified by MSI status."""
    logger.info("Generating KM curves...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(feature_matrix_csv)
    df = df.dropna(subset=["os_days", "os_event", "msi_binary"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    kmf = KaplanMeierFitter()
    
    # MSI-H
    msi_h = df[df["msi_binary"] == 1]
    kmf.fit(msi_h["os_days"], msi_h["os_event"], label=f"MSI-H (n={len(msi_h)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color="steelblue")
    
    # MSS
    msi_s = df[df["msi_binary"] == 0]
    kmf.fit(msi_s["os_days"], msi_s["os_event"], label=f"MSS (n={len(msi_s)})")
    kmf.plot_survival_function(ax=ax, ci_show=True, linewidth=2, color="coral")
    
    ax.set_xlabel("Time (days)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Survival Probability", fontsize=12, fontweight="bold")
    ax.set_title("Kaplan-Meier Survival Curves by MSI Status", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "km_curves_msi.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved KM curves to {Path(output_dir) / 'km_curves_msi.png'}")
    plt.close()


def generate_forest_plot(
    cox_model_pkl: str,
    output_dir: str = "results/figures"
):
    """Generate forest plot of hazard ratios from Cox model."""
    logger.info("Generating forest plot...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(cox_model_pkl, "rb") as f:
        cph = pickle.load(f)
    
    # Extract hazard ratios and CIs
    summary = cph.summary
    summary["exp_coef"] = np.exp(summary["coef"])
    summary["exp_ci_lower"] = np.exp(summary["coef lower 95%"])
    summary["exp_ci_upper"] = np.exp(summary["coef upper 95%"])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(summary))
    
    ax.scatter(summary["exp_coef"], y_pos, s=100, color="darkblue", zorder=3)
    
    for i, (idx, row) in enumerate(summary.iterrows()):
        ax.plot([row["exp_ci_lower"], row["exp_ci_upper"]], [i, i], "b-", linewidth=2, zorder=2)
    
    ax.axvline(x=1, color="red", linestyle="--", linewidth=2, label="HR=1 (No effect)")
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(summary.index)
    ax.set_xlabel("Hazard Ratio (95% CI)", fontsize=12, fontweight="bold")
    ax.set_title("Cox Model: Hazard Ratios", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="x")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "forest_plot_cox.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved forest plot to {Path(output_dir) / 'forest_plot_cox.png'}")
    plt.close()


def generate_tmb_distribution(
    feature_matrix_csv: str,
    output_dir: str = "results/figures"
):
    """Generate TMB distribution by MSI status."""
    logger.info("Generating TMB distribution plot...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(feature_matrix_csv)
    df = df.dropna(subset=["tmb", "msi_status"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    msi_labels = ["MSS", "MSI-H"]
    msi_data = [df[df["msi_status"] == "MSS"]["tmb"], df[df["msi_status"] == "MSI-H"]["tmb"]]
    
    parts = ax.violinplot(msi_data, positions=[0, 1], showmeans=True, showmedians=True)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(msi_labels)
    ax.set_ylabel("TMB (mutations/Mb)", fontsize=12, fontweight="bold")
    ax.set_title("Tumor Mutational Burden by MSI Status", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "tmb_distribution.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved TMB distribution to {Path(output_dir) / 'tmb_distribution.png'}")
    plt.close()


def generate_ddr_burden_plot(
    feature_matrix_csv: str,
    output_dir: str = "results/figures"
):
    """Generate DDR burden distribution."""
    logger.info("Generating DDR burden plot...")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(feature_matrix_csv)
    df = df.dropna(subset=["ddr_burden", "msi_status"])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    for msi_status in ["MSS", "MSI-H"]:
        data = df[df["msi_status"] == msi_status]["ddr_burden"]
        ax.hist(data, bins=15, alpha=0.6, label=f"{msi_status} (n={len(data)})")
    
    ax.set_xlabel("DDR Mutation Burden", fontsize=12, fontweight="bold")
    ax.set_ylabel("Frequency", fontsize=12, fontweight="bold")
    ax.set_title("DDR Gene Mutation Burden Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(Path(output_dir) / "ddr_burden_distribution.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved DDR burden plot to {Path(output_dir) / 'ddr_burden_distribution.png'}")
    plt.close()


if __name__ == "__main__":
    import sys
    
    feature_matrix_csv = sys.argv[1] if len(sys.argv) > 1 else "data/processed/feature_matrix.csv"
    cox_model_pkl = sys.argv[2] if len(sys.argv) > 2 else "results/cox_model.pkl"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "results/figures"
    
    generate_km_curves(feature_matrix_csv, output_dir)
    generate_forest_plot(cox_model_pkl, output_dir)
    generate_tmb_distribution(feature_matrix_csv, output_dir)
    generate_ddr_burden_plot(feature_matrix_csv, output_dir)
    
    logger.info("Figure generation complete!")
