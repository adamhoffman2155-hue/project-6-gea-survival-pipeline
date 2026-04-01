#!/usr/bin/env python3
"""
Survival modeling: Cox Proportional Hazards and Kaplan-Meier analysis.

Fits multivariate Cox model and stratified KM curves.
Outputs model summary, concordance index, and hazard ratios.
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test
import pickle

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fit_survival_models(
    feature_matrix_csv: str,
    output_dir: str = "results"
) -> dict:
    """
    Fit Cox PH and KM models.
    
    Returns:
        Dictionary with fitted models and summaries
    """
    logger.info(f"Loading feature matrix from {feature_matrix_csv}...")
    df = pd.read_csv(feature_matrix_csv)
    
    # Prepare data for modeling
    # Ensure no NaNs in critical columns
    df = df.dropna(subset=["os_days", "os_event", "msi_binary", "tmb", "ddr_burden"])
    
    logger.info(f"Modeling cohort: {len(df)} cases")
    logger.info(f"Event rate: {df['os_event'].mean():.2%}")
    logger.info(f"Median follow-up: {df['os_days'].median():.0f} days")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # ===== KAPLAN-MEIER ANALYSIS =====
    logger.info("\n=== Kaplan-Meier Analysis ===")
    
    kmf = KaplanMeierFitter()
    
    # KM by MSI status
    logger.info("Fitting KM stratified by MSI status...")
    msi_h = df[df["msi_binary"] == 1]
    msi_s = df[df["msi_binary"] == 0]
    
    kmf.fit(msi_h["os_days"], msi_h["os_event"], label="MSI-H")
    msi_h_summary = kmf.survival_function_
    
    kmf.fit(msi_s["os_days"], msi_s["os_event"], label="MSS")
    msi_s_summary = kmf.survival_function_
    
    # Log-rank test
    results = logrank_test(msi_h["os_days"], msi_s["os_days"], msi_h["os_event"], msi_s["os_event"])
    logger.info(f"Log-rank test (MSI-H vs MSS): p-value = {results.p_value:.4f}")
    
    # ===== COX PROPORTIONAL HAZARDS =====
    logger.info("\n=== Cox Proportional Hazards Model ===")
    
    # Prepare data for Cox model
    cox_data = df[["os_days", "os_event", "msi_binary", "tmb", "ddr_burden", "immune_subtype_code", "age_at_diagnosis"]].copy()
    cox_data.columns = ["T", "E", "msi_status", "tmb", "ddr_burden", "immune_subtype", "age"]
    
    # Standardize continuous variables
    cox_data["tmb"] = (cox_data["tmb"] - cox_data["tmb"].mean()) / cox_data["tmb"].std()
    cox_data["ddr_burden"] = (cox_data["ddr_burden"] - cox_data["ddr_burden"].mean()) / cox_data["ddr_burden"].std()
    cox_data["age"] = (cox_data["age"] - cox_data["age"].mean()) / cox_data["age"].std()
    
    # Fit Cox model
    cph = CoxPHFitter()
    cph.fit(cox_data, duration_col="T", event_col="E", show_progress=False)
    
    logger.info(cph.summary)
    
    # Concordance index
    c_index = cph.concordance_index_
    logger.info(f"Concordance Index: {c_index:.3f}")
    
    # Save model summary
    summary_csv = Path(output_dir) / "cox_model_summary.csv"
    cph.summary.to_csv(summary_csv)
    logger.info(f"Saved Cox model summary to {summary_csv}")
    
    # Save fitted model
    model_pkl = Path(output_dir) / "cox_model.pkl"
    with open(model_pkl, "wb") as f:
        pickle.dump(cph, f)
    logger.info(f"Saved fitted Cox model to {model_pkl}")
    
    # Save KM summaries
    msi_h_summary.to_csv(Path(output_dir) / "km_msi_h.csv")
    msi_s_summary.to_csv(Path(output_dir) / "km_mss.csv")
    
    # Risk scores
    risk_scores = cph.predict_partial_hazard(cox_data.drop(columns=["T", "E"]))
    risk_scores_df = pd.DataFrame({
        "case_id": df["case_id"],
        "risk_score": risk_scores,
        "risk_percentile": (risk_scores.rank(pct=True) * 100).round(1)
    })
    risk_scores_df.to_csv(Path(output_dir) / "risk_scores.csv", index=False)
    logger.info(f"Saved risk scores to {Path(output_dir) / 'risk_scores.csv'}")
    
    return {
        "cox_model": cph,
        "kmf": kmf,
        "concordance_index": c_index,
        "risk_scores": risk_scores_df,
        "cox_data": cox_data,
        "df": df
    }


if __name__ == "__main__":
    import sys
    
    feature_matrix_csv = sys.argv[1] if len(sys.argv) > 1 else "data/processed/feature_matrix.csv"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "results"
    
    results = fit_survival_models(feature_matrix_csv, output_dir)
    logger.info("Survival modeling complete!")
