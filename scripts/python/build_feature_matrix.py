#!/usr/bin/env python3
"""
Build multi-omic feature matrix: MSI, TMB, DDR burden, immune subtype.

Constructs the feature matrix for survival modeling by joining
molecular and clinical data. DDR gene list is sourced from config/config.yaml.
"""

import pandas as pd
import duckdb
import logging
import os
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Fallback DDR gene set used only if config.yaml is missing.
_DEFAULT_DDR_GENES = [
    "BRCA1", "BRCA2", "ATM", "ATR", "PALB2", "RAD51",
    "MLH1", "MSH2", "MSH6", "POLE",
]


def load_ddr_genes(config_path: Optional[str] = None) -> List[str]:
    """Load the DDR gene list from config/config.yaml.

    Falls back to a sensible default if the file or key is missing, and
    logs a warning so the caller notices the config is being ignored.
    """
    if config_path is None:
        # Default: config/config.yaml relative to the project root (two up from this script)
        project_root = Path(__file__).resolve().parents[2]
        config_path = project_root / "config" / "config.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning("Config file not found at %s; using default DDR gene list", config_path)
        return list(_DEFAULT_DDR_GENES)

    try:
        import yaml
    except ImportError:
        logger.warning("pyyaml not installed; using default DDR gene list")
        return list(_DEFAULT_DDR_GENES)

    with open(config_path, "r") as fh:
        cfg = yaml.safe_load(fh) or {}

    genes = cfg.get("ddr_genes")
    if not genes:
        logger.warning("config.yaml has no ddr_genes key; using default list")
        return list(_DEFAULT_DDR_GENES)

    logger.info("Loaded %d DDR genes from %s", len(genes), config_path)
    return list(genes)


def build_feature_matrix(
    db_path: str,
    mutations_csv: str,
    output_csv: str = "data/processed/feature_matrix.csv",
    config_path: Optional[str] = None,
) -> pd.DataFrame:
    """Build multi-omic feature matrix (MSI, TMB, DDR burden, immune subtype)."""
    ddr_genes = load_ddr_genes(config_path)
    logger.info("Using DDR gene set: %s", ddr_genes)

    logger.info("Building feature matrix...")

    df_mutations = pd.read_csv(mutations_csv)

    # TMB: mutations per megabase (assume ~3000 Mb genome)
    genome_size_mb = 3000

    tmb_data = []
    for case_id in df_mutations["case_id"].unique():
        case_mutations = df_mutations[df_mutations["case_id"] == case_id]
        n_mutations = len(case_mutations)
        tmb = n_mutations / genome_size_mb

        ddr_mutations = case_mutations[case_mutations["gene_symbol"].isin(ddr_genes)]
        ddr_burden = len(ddr_mutations[ddr_mutations["is_pathogenic"] == 1])

        tmb_data.append({
            "case_id": case_id,
            "tmb": tmb,
            "ddr_burden": ddr_burden,
            "n_mutations": n_mutations,
        })

    df_tmb = pd.DataFrame(tmb_data)
    logger.info("Computed TMB and DDR burden for %d cases", len(df_tmb))
    logger.info("  Median TMB: %.2f mutations/Mb", df_tmb["tmb"].median())
    logger.info("  Median DDR burden: %.0f mutations", df_tmb["ddr_burden"].median())

    # Query clinical + MSI + immune subtype from DuckDB
    conn = duckdb.connect(db_path, read_only=True)
    query = """
        SELECT
            c.case_id,
            c.age_at_diagnosis,
            c.tumor_stage,
            c.treatment_received,
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

    df_features = df_features.merge(df_tmb, on="case_id", how="left")

    # Encode MSI status
    df_features["msi_binary"] = (df_features["msi_status"] == "MSI-H").astype(int)

    # DDR burden quartiles
    df_features["ddr_quartile"] = pd.qcut(
        df_features["ddr_burden"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop"
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

    # Impute missing values using medians (no chained-assignment inplace warning)
    for col in ("immune_subtype_code", "ddr_burden", "tmb"):
        median_val = df_features[col].median()
        df_features[col] = df_features[col].fillna(median_val)

    # Verify critical columns
    critical_cols = ["case_id", "os_days", "os_event", "msi_binary", "tmb", "ddr_burden"]
    for col in critical_cols:
        n_missing = df_features[col].isna().sum()
        if n_missing > 0:
            logger.warning("Column %s has %d missing values", col, n_missing)

    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_csv, index=False)

    logger.info("Feature matrix saved to %s", output_csv)
    logger.info("Shape: %s", df_features.shape)
    logger.info("Columns: %s", list(df_features.columns))

    return df_features


if __name__ == "__main__":
    import sys

    db_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/features.duckdb"
    mutations_csv = sys.argv[2] if len(sys.argv) > 2 else "data/raw/TCGA-STAD_mutations.csv"
    output_csv = sys.argv[3] if len(sys.argv) > 3 else "data/processed/feature_matrix.csv"
    config_path = sys.argv[4] if len(sys.argv) > 4 else None

    build_feature_matrix(db_path, mutations_csv, output_csv, config_path)
