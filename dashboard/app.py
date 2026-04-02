#!/usr/bin/env python3
"""
Streamlit Risk Stratifier Dashboard

Interactive tool for GEA survival risk prediction.
Input patient molecular profile, receive risk estimate and survival probabilities.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="GEA Survival Risk Stratifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    :root {
        --primary-color: #0084D1;
        --background-color: #1a1a1a;
        --secondary-background-color: #262626;
        --text-color: #f0f0f0;
    }
    body {
        background-color: var(--background-color);
        color: var(--text-color);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.title("🔬 Gastroesophageal Adenocarcinoma Survival Risk Stratifier")
st.markdown("**Predict chemotherapy response and survival in gastroesophageal adenocarcinoma patients**")

# Disclaimer
st.warning("""
⚠️ **DISCLAIMER**: This tool is for educational and research purposes only. 
It should NOT be used for clinical decision-making or patient care. 
Survival estimates are based on public The Cancer Genome Atlas data and portfolio demonstration modeling.
""")

# Load model and data
@st.cache_resource
def load_model_and_data():
    """Load fitted Cox model and cohort statistics."""
    try:
        with open("results/cox_model.pkl", "rb") as f:
            cox_model = pickle.load(f)
        
        risk_scores = pd.read_csv("results/risk_scores.csv")
        feature_matrix = pd.read_csv("data/processed/feature_matrix.csv")
        
        return cox_model, risk_scores, feature_matrix
    except FileNotFoundError:
        st.error("Model files not found. Please run the pipeline first.")
        return None, None, None


cox_model, risk_scores, feature_matrix = load_model_and_data()

if cox_model is None:
    st.stop()

# Sidebar inputs
st.sidebar.header("📋 Patient Molecular Profile")

msi_status = st.sidebar.selectbox(
    "Microsatellite Instability Status",
    options=["Microsatellite Stable", "Microsatellite Instability High"],
    help="Microsatellite Instability indicates mismatch repair deficiency"
)

tmb = st.sidebar.slider(
    "Tumor Mutational Burden (mutations per megabase)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.5,
    help="Total number of somatic mutations per megabase of sequenced genome"
)

ddr_burden = st.sidebar.slider(
    "DNA Damage Repair Gene Mutation Count",
    min_value=0,
    max_value=10,
    value=2,
    help="Count of pathogenic mutations in DNA Damage Repair genes (BRCA1, BRCA2, ATM, ATR, and others)"
)

immune_subtype = st.sidebar.selectbox(
    "Tumor Immune Subtype",
    options=["C1: Wound Healing", "C2: Interferon-Gamma Dominant", "C3: Inflammatory", 
             "C4: Lymphoid Depleted", "C5: Immunologically Quiet"],
    help="The Cancer Genome Atlas immune classification based on tumor microenvironment"
)

age = st.sidebar.slider(
    "Patient Age at Diagnosis (years)",
    min_value=30,
    max_value=90,
    value=65,
    step=1,
    help="Age in years at time of cancer diagnosis"
)

# Predict button
if st.sidebar.button("🔮 Calculate Risk", key="predict_btn"):
    
    # Prepare input data (must match Cox model training)
    immune_mapping = {
        "C1: Wound Healing": 1,
        "C2: Interferon-Gamma Dominant": 2,
        "C3: Inflammatory": 3,
        "C4: Lymphoid Depleted": 4,
        "C5: Immunologically Quiet": 5
    }
    
    msi_binary = 1 if msi_status == "Microsatellite Instability High" else 0
    immune_code = immune_mapping[immune_subtype]
    
    # Standardize using cohort statistics
    tmb_mean = feature_matrix["tmb"].mean()
    tmb_std = feature_matrix["tmb"].std()
    tmb_scaled = (tmb - tmb_mean) / tmb_std
    
    ddr_mean = feature_matrix["ddr_burden"].mean()
    ddr_std = feature_matrix["ddr_burden"].std()
    ddr_scaled = (ddr_burden - ddr_mean) / ddr_std
    
    age_mean = feature_matrix["age_at_diagnosis"].mean()
    age_std = feature_matrix["age_at_diagnosis"].std()
    age_scaled = (age - age_mean) / age_std
    
    # Create input DataFrame
    input_data = pd.DataFrame({
        "msi_status": [msi_binary],
        "tmb": [tmb_scaled],
        "ddr_burden": [ddr_scaled],
        "immune_subtype": [immune_code],
        "age": [age_scaled]
    })
    
    # Predict risk score
    risk_score = cox_model.predict_partial_hazard(input_data).values[0]
    
    # Calculate percentile relative to cohort
    risk_percentile = (risk_scores["risk_score"] < risk_score).sum() / len(risk_scores) * 100
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Risk Score",
            value=f"{risk_score:.2f}",
            delta="Higher values indicate worse prognosis"
        )
    
    with col2:
        st.metric(
            label="Risk Percentile",
            value=f"{risk_percentile:.1f}%",
            delta="Relative to The Cancer Genome Atlas cohort"
        )
    
    with col3:
        risk_category = "High Risk" if risk_percentile > 66 else ("Intermediate Risk" if risk_percentile > 33 else "Low Risk")
        st.metric(
            label="Risk Stratification Category",
            value=risk_category
        )
    
    # Survival probability estimates
    st.subheader("📊 Estimated Overall Survival Probabilities")
    
    timepoints = [365, 730, 1095]  # 12, 24, 36 months
    timepoint_labels = ["12 months", "24 months", "36 months"]
    
    # Simplified survival estimate (based on risk percentile)
    base_survival = {365: 0.75, 730: 0.55, 1095: 0.40}
    
    survival_probs = {}
    for tp in timepoints:
        # Adjust based on risk percentile
        adjustment = 1 - (risk_percentile / 100) * 0.5
        survival_probs[tp] = max(0.1, base_survival[tp] * adjustment)
    
    col1, col2, col3 = st.columns(3)
    for i, (tp, label) in enumerate(zip(timepoints, timepoint_labels)):
        with [col1, col2, col3][i]:
            st.metric(
                label=f"Survival at {label}",
                value=f"{survival_probs[tp]:.1%}"
            )
    
    # Feature importance (coefficients)
    st.subheader("🔍 Feature Contributions to Risk Score")
    
    cox_summary = cox_model.summary
    cox_summary["exp_coef"] = np.exp(cox_summary["coef"])
    
    feature_importance = pd.DataFrame({
        "Feature": cox_summary.index,
        "Coefficient": cox_summary["coef"].values,
        "Hazard Ratio": cox_summary["exp_coef"].values,
        "p-value": cox_summary["p"].values
    })
    
    st.dataframe(feature_importance.set_index("Feature"), use_container_width=True)
    st.caption("Hazard Ratio > 1 indicates increased risk; < 1 indicates decreased risk")
    
    # Cohort comparison
    st.subheader("📈 Comparison to The Cancer Genome Atlas Gastroesophageal Adenocarcinoma Cohort")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Your Patient Profile:**")
        st.write(f"- Microsatellite Instability Status: {msi_status}")
        st.write(f"- Tumor Mutational Burden: {tmb:.1f} mutations per megabase")
        st.write(f"- DNA Damage Repair Gene Mutations: {ddr_burden}")
        st.write(f"- Immune Subtype: {immune_subtype}")
        st.write(f"- Age at Diagnosis: {age} years")
    
    with col2:
        st.write("**Cohort Statistics:**")
        st.write(f"- Total Cases: {len(feature_matrix)}")
        st.write(f"- Median Tumor Mutational Burden: {feature_matrix['tmb'].median():.1f} mutations per megabase")
        st.write(f"- Median DNA Damage Repair Gene Mutations: {feature_matrix['ddr_burden'].median():.0f}")
        st.write(f"- Microsatellite Instability High Prevalence: {(feature_matrix['msi_binary'] == 1).sum() / len(feature_matrix):.1%}")

# Footer
st.markdown("---")
st.markdown("""
**About this tool:**
- Built with The Cancer Genome Atlas Gastroesophageal Adenocarcinoma public data
- Cox Proportional Hazards survival model
- For educational and research purposes only
- Not validated for clinical use or patient care
- Survival estimates are illustrative and based on portfolio demonstration data

**Disclaimer:** This is a portfolio project. Real clinical applications require independent validation, regulatory approval, and clinical trial data.
""")
