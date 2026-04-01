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
st.title("🔬 GEA Survival Risk Stratifier")
st.markdown("**Predict chemotherapy response and survival in gastroesophageal adenocarcinoma**")

# Disclaimer
st.warning("""
⚠️ **DISCLAIMER**: This tool is for educational purposes only. 
It should NOT be used for clinical decision-making. 
Survival estimates are based on public TCGA data and portfolio modeling.
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
    "MSI Status",
    options=["MSS", "MSI-H"],
    help="Microsatellite Instability Status"
)

tmb = st.sidebar.slider(
    "Tumor Mutational Burden (TMB)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=0.5,
    help="Mutations per megabase"
)

ddr_burden = st.sidebar.slider(
    "DDR Gene Mutation Burden",
    min_value=0,
    max_value=10,
    value=2,
    help="Count of pathogenic mutations in DDR genes (BRCA1/2, ATM, ATR, etc.)"
)

immune_subtype = st.sidebar.selectbox(
    "Immune Subtype",
    options=["C1: Wound Healing", "C2: IFN-gamma Dominant", "C3: Inflammatory", 
             "C4: Lymphoid Depleted", "C5: Immunologically Quiet"],
    help="TCGA immune classification"
)

age = st.sidebar.slider(
    "Age at Diagnosis",
    min_value=30,
    max_value=90,
    value=65,
    step=1
)

# Predict button
if st.sidebar.button("🔮 Calculate Risk", key="predict_btn"):
    
    # Prepare input data (must match Cox model training)
    immune_mapping = {
        "C1: Wound Healing": 1,
        "C2: IFN-gamma Dominant": 2,
        "C3: Inflammatory": 3,
        "C4: Lymphoid Depleted": 4,
        "C5: Immunologically Quiet": 5
    }
    
    msi_binary = 1 if msi_status == "MSI-H" else 0
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
            delta="Higher = Worse Prognosis"
        )
    
    with col2:
        st.metric(
            label="Risk Percentile",
            value=f"{risk_percentile:.1f}%",
            delta="Relative to TCGA-STAD cohort"
        )
    
    with col3:
        risk_category = "High Risk" if risk_percentile > 66 else ("Intermediate Risk" if risk_percentile > 33 else "Low Risk")
        st.metric(
            label="Risk Category",
            value=risk_category
        )
    
    # Survival probability estimates
    st.subheader("📊 Estimated Survival Probabilities")
    
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
                label=label,
                value=f"{survival_probs[tp]:.1%}"
            )
    
    # Feature importance (coefficients)
    st.subheader("🔍 Feature Contributions to Risk")
    
    cox_summary = cox_model.summary
    cox_summary["exp_coef"] = np.exp(cox_summary["coef"])
    
    feature_importance = pd.DataFrame({
        "Feature": cox_summary.index,
        "Coefficient": cox_summary["coef"].values,
        "Hazard Ratio": cox_summary["exp_coef"].values,
        "p-value": cox_summary["p"].values
    })
    
    st.dataframe(feature_importance.set_index("Feature"), use_container_width=True)
    
    # Cohort comparison
    st.subheader("📈 Comparison to TCGA-STAD Cohort")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Your Profile:**")
        st.write(f"- MSI Status: {msi_status}")
        st.write(f"- TMB: {tmb:.1f} mutations/Mb")
        st.write(f"- DDR Burden: {ddr_burden} mutations")
        st.write(f"- Immune Subtype: {immune_subtype}")
        st.write(f"- Age: {age} years")
    
    with col2:
        st.write("**Cohort Statistics:**")
        st.write(f"- N = {len(feature_matrix)} cases")
        st.write(f"- Median TMB: {feature_matrix['tmb'].median():.1f} mutations/Mb")
        st.write(f"- Median DDR Burden: {feature_matrix['ddr_burden'].median():.0f} mutations")
        st.write(f"- MSI-H: {(feature_matrix['msi_binary'] == 1).sum() / len(feature_matrix):.1%}")

# Footer
st.markdown("---")
st.markdown("""
**About this tool:**
- Built with TCGA-STAD public data
- Cox Proportional Hazards model
- For educational and research purposes only
- Not validated for clinical use
""")
