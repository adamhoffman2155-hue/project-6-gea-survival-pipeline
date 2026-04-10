#!/usr/bin/env python3
"""
Data preprocessing: cleaning, validation, and DuckDB ingestion.

Processes raw clinical and mutation data, applies quality filters,
and loads into DuckDB feature store.
"""

import json
import pandas as pd
import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_clinical_data(clinical_json: str) -> pd.DataFrame:
    """
    Load and preprocess clinical data from JSON.

    Returns:
        DataFrame with columns: case_id, age_at_diagnosis, tumor_stage,
                               treatment_received, treatment_type, os_days, os_event
    """
    logger.info(f"Loading clinical data from {clinical_json}...")

    with open(clinical_json, "r") as f:
        cases = json.load(f)

    logger.info(f"Loaded {len(cases)} cases")

    records = []
    for case in cases:
        case_id = case.get("case_id")

        # Extract diagnosis info
        diagnoses = case.get("diagnoses", [])
        if not diagnoses:
            logger.warning(f"Case {case_id} has no diagnoses, skipping")
            continue

        primary_diagnosis = diagnoses[0]
        age_at_diagnosis = primary_diagnosis.get("age_at_diagnosis")
        tumor_stage = primary_diagnosis.get("tumor_stage", "Unknown")

        # Check for treatment
        treatments = primary_diagnosis.get("treatments", [])
        treatment_received = 1 if len(treatments) > 0 else 0
        treatment_type = treatments[0].get("treatment_type") if treatments else None

        # Survival data
        follow_up = case.get("follow_up", {})
        os_days = follow_up.get("days_to_last_follow_up", 0)
        vital_status = follow_up.get("vital_status", "Alive")
        os_event = 1 if vital_status == "Dead" else 0

        # Quality filters
        if os_days < 30:
            logger.warning(f"Case {case_id} has insufficient follow-up ({os_days} days), skipping")
            continue

        if age_at_diagnosis is None or age_at_diagnosis < 0:
            logger.warning(f"Case {case_id} has invalid age, skipping")
            continue

        records.append({
            "case_id": case_id,
            "age_at_diagnosis": age_at_diagnosis,
            "tumor_stage": tumor_stage,
            "treatment_received": treatment_received,
            "treatment_type": treatment_type,
            "os_days": os_days,
            "os_event": os_event,
        })

    df_clinical = pd.DataFrame(records)
    logger.info(
        f"Preprocessed {len(df_clinical)} clinical records "
        f"(removed {len(cases) - len(df_clinical)} invalid records)"
    )

    logger.info("=== DATA CLEANING LOG ===")
    logger.info(f"Initial cases: {len(cases)}")
    logger.info(f"Final clinical records: {len(df_clinical)}")
    if len(df_clinical) > 0:
        logger.info(f"Records with treatment: {df_clinical['treatment_received'].sum()}")
        logger.info(f"Event rate (deaths): {df_clinical['os_event'].mean():.2%}")
        logger.info(f"Median follow-up: {df_clinical['os_days'].median():.0f} days")
    logger.info("========================")

    return df_clinical


def preprocess_mutations(mutations_csv: str) -> pd.DataFrame:
    """Load and preprocess mutation data."""
    logger.info(f"Loading mutations from {mutations_csv}...")

    df_mutations = pd.read_csv(mutations_csv)

    # Filter to pathogenic mutations
    df_pathogenic = df_mutations[df_mutations["is_pathogenic"] == 1].copy()

    logger.info(f"Loaded {len(df_mutations)} mutations, {len(df_pathogenic)} pathogenic")

    return df_pathogenic


def create_and_load_duckdb(
    db_path: str,
    clinical_df: pd.DataFrame,
    mutations_df: pd.DataFrame,
    msi_df: pd.DataFrame,
    immune_df: pd.DataFrame,
):
    """
    Create schema and load data into DuckDB in a single connection.

    Using one connection ensures tables exist before inserts and that
    the pandas DataFrames remain resolvable as DuckDB table references.
    """
    logger.info(f"Creating DuckDB schema and loading data at {db_path}...")

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(db_path)

    try:
        # Drop existing tables and view for idempotent re-runs
        conn.execute("DROP VIEW IF EXISTS molecular_features")
        conn.execute("DROP TABLE IF EXISTS clinical")
        conn.execute("DROP TABLE IF EXISTS mutations")
        conn.execute("DROP TABLE IF EXISTS msi_status")
        conn.execute("DROP TABLE IF EXISTS immune_subtypes")

        # Register DataFrames so DuckDB can reference them by name
        conn.register("clinical_df", clinical_df)
        conn.register("mutations_df", mutations_df)
        conn.register("msi_df", msi_df)
        conn.register("immune_df", immune_df)

        # Create tables directly from DataFrames (schema inferred from columns)
        conn.execute("CREATE TABLE clinical AS SELECT * FROM clinical_df")
        conn.execute("CREATE TABLE mutations AS SELECT * FROM mutations_df")
        conn.execute("CREATE TABLE msi_status AS SELECT * FROM msi_df")
        conn.execute("CREATE TABLE immune_subtypes AS SELECT * FROM immune_df")

        # Create the feature-store view
        conn.execute(
            """
            CREATE VIEW molecular_features AS
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
        )

        # Verify row counts
        n_clinical = conn.execute("SELECT COUNT(*) FROM clinical").fetchone()[0]
        n_mutations = conn.execute("SELECT COUNT(*) FROM mutations").fetchone()[0]
        n_msi = conn.execute("SELECT COUNT(*) FROM msi_status").fetchone()[0]
        n_immune = conn.execute("SELECT COUNT(*) FROM immune_subtypes").fetchone()[0]

        logger.info(f"Loaded {n_clinical} clinical records")
        logger.info(f"Loaded {n_mutations} mutations")
        logger.info(f"Loaded {n_msi} MSI status records")
        logger.info(f"Loaded {n_immune} immune subtype records")
    finally:
        conn.close()

    logger.info("DuckDB schema created and data loaded")


if __name__ == "__main__":
    import sys

    clinical_json = sys.argv[1] if len(sys.argv) > 1 else "data/raw/TCGA-STAD_clinical.json"
    mutations_csv = sys.argv[2] if len(sys.argv) > 2 else "data/raw/TCGA-STAD_mutations.csv"
    msi_csv = sys.argv[3] if len(sys.argv) > 3 else "data/raw/TCGA-STAD_msi_status.csv"
    immune_csv = sys.argv[4] if len(sys.argv) > 4 else "data/raw/TCGA-STAD_immune_subtypes.csv"
    db_path = sys.argv[5] if len(sys.argv) > 5 else "data/processed/features.duckdb"

    # Preprocess
    df_clinical = preprocess_clinical_data(clinical_json)
    df_mutations = preprocess_mutations(mutations_csv)
    df_msi = pd.read_csv(msi_csv)
    df_immune = pd.read_csv(immune_csv)

    # Single-connection schema creation + load
    create_and_load_duckdb(db_path, df_clinical, df_mutations, df_msi, df_immune)

    logger.info("Preprocessing complete!")
