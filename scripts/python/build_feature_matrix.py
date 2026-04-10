#!/usr/bin/env python3
"""
Build multi-omic feature matrix: MSI, TMB, DDR burden, immune subtype.

Constructs the feature matrix for survival modeling by joining
molecular and clinical data. DDR gene list is loaded from config/config.yaml.
"""

import pandas as pd
import duckdb
import logging
import yaml
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fallback DDR gene set used only if config can't be loaded.
_FALLBACK_DDR_GENES = [
    "BRCA1", "BRCA2", "ATM", "ATR", "PALB2",
    "RAD51", "MLH1", "MSH2", "MSH6", "POLE",
]


def load_ddr_genes(config_path: str = "config/config.yaml") -> List[str]:
    """Load DDR gene list from YAML config with graceful fallback."""
    cfg_file = Path(config_path)
    if not cfg_file.exists():
        logger.warning(f"Config {config_path} not found; using fallback DDR gene list")
        return list(_FALLBACK_DDR_GENES)

    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f) or {}

    genes = cfg.get("ddr_genes")
    if not genes or not isinstance(genes, list):
        logger.warning(f"ddr_genes missing or invalid in {config_path}; using fallback")
        return list(_FALLBACK_DDR_GENES)

    logger.info(f"Loaded {len(genes)} DDR genes from {config_path}")
    return list(genes)


def build_feature_matrix(
    db_path: str,
    mutations_csv: str,
    output_csv: str = "data/processed/feature_matrix.csv",
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Build multi-omic feature matrix.

    Features:
    - MSI status (binary: MSI-H vs MSS)
    - TMB (total mutations per megabase)
    - DDR burden (count of pathogenic mutations in DDR genes from config)
    - Immune subtype (categorical)
    """
    logger.info("Building feature matrix...")

    ddr_genes = load_ddr_genes(config_path)

    # Load mutations for TMB and DDR burden calculation
    df_mutations = pd.read_csv(mutations_csv)

    # Calculate TMB (mutations per megabase)
    # Assume ~3 billion bp genome
    genome_size_mb = 3000

    tmb_data = []
    for case_id in df_mutations["case_id"].unique():
        case_mutations = df_mutations[df_mutations["case_id"] == case_id]
        n_mutations = len(case_mutations)
        tmb = n_mutations / genome_size_mb

        # DDR burden using config-driven gene list
        ddr_mutations = case_mutations[case_mutations["gene_symbol"].isin(ddr_genes)]
        ddr_burden = int((ddr_mutations["is_pathogenic"] == 1).sum())

        tmb_data.append({
            "case_id": case_id,
            "tmb": tmb,
            "ddr_burden": ddr_burden,
            "n_mutations": n_mutations,
        })

    df_tmb = pd.DataFrame(tmb_data)
    logger.info(f"Calculated TMB and DDR burden for {len(df_tmb)} cases")
    logger.info(f"  Median TMB: {df_tmb['tmb'].median():.2f} mutations/Mb")
    logger.info(f"  Median DDR burden: {df_tmb['ddr_burden'].median():.0f} mutations")

    # Query feature matrix from DuckDB
    conn = duckdb.connect(db_path)

    query = """
    SELECT
        c.case_id,
        c.age_at_diagnosis,
        c.tumor_stage,
        c.treatment_received,
        c.treatment_type,
        c.os_days,
        c.os_event,
        m.msi_status,
        m.msi_score,
        i.immune_subtype,
        i.immune_score
    FROM clinical c
    LEFT JOIN msi_status m ON c.case_id = m.case_id
    LEFT JOIN immune_subtypes i ON c.case_id = i.case_id
    """

    df_features = conn.execute(query).df()
    conn.close()

    # Join with TMB and DDR burden
    df_features = df_features.merge(df_tmb, on="case_id", how="left")

    # Encode MSI status
    df_features["msi_binary"] = (df_features["msi_status"] == "MSI-H").astype(int)

    # Create DDR burden quartiles
    df_features["ddr_quartile"] = pd.qcut(
        df_features["ddr_burden"], q=4,
        labels=["Q1", "Q2", "Q3", "Q4"],
        duplicates="drop",
    )

    # Encode immune subtype
    immune_mapping = {
        "C1: Wound Healing": 1,
        "C2: IFN-gamma Dominant": 2,
        "C3: Inflammatory": 3,
        "C4: Lymphoid Depleted": 4,
        "C5: Immunologically Quiet": 5,
    }
    df_features["immune_subtype_code"] = df_features["immune_subtype"].map(immune_mapping)

    # Handle missing values
    df_features["immune_subtype_code"] = df_features["immune_subtype_code"].fillna(
        df_features["immune_subtype_code"].median()
    )
    df_features["ddr_burden"] = df_features["ddr_burden"].fillna(df_features["ddr_burden"].median())
    df_features["tmb"] = df_features["tmb"].fillna(df_features["tmb"].median())

    # Verify no NaNs in critical columns
    critical_cols = ["case_id", "os_days", "os_event", "msi_binary", "tmb", "ddr_burden"]
    for col in critical_cols:
        n_missing = df_features[col].isna().sum()
        if n_missing > 0:
            logger.warning(f"Column {col} has {n_missing} missing values")

    # Save
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv, index=False)

    logger.info(f"Feature matrix saved to {output_csv}")
    logger.info(f"Shape: {df_features.shape}")
    logger.info(f"Columns: {list(df_features.columns)}")

    return df_features


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.duckdb"
    mutations_csv = sys.argv[2] if len(sys.argv) > 2 else "data/raw/TCGA-STAD_mutations.csv"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "data/processed/feature_matrix.csv"
    config_path = sys.argv[4] if len(sys.argv) > 4 else "config/config.yaml"

    build_feature_matrix(db_path, mutations_csv, output_csv, config_path)
