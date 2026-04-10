#!/usr/bin/env python3
"""
Data preprocessing: cleaning, validation, and DuckDB ingestion.

Processes raw clinical and mutation data, applies quality filters,
and loads into DuckDB feature store. Schema is created on the same
connection that inserts the data, so the tables are guaranteed to
exist at insert time.
"""

import json
import pandas as pd
import duckdb
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def preprocess_clinical_data(clinical_json: str) -> pd.DataFrame:
    """Load and preprocess clinical data from JSON."""
    logger.info("Loading clinical data from %s...", clinical_json)

    with open(clinical_json, "r") as f:
        cases = json.load(f)

    logger.info("Loaded %d cases", len(cases))

    records = []
    for case in cases:
        case_id = case.get("case_id")
        diagnoses = case.get("diagnoses", [])
        if not diagnoses:
            logger.warning("Case %s has no diagnoses, skipping", case_id)
            continue

        primary_diagnosis = diagnoses[0]
        age_at_diagnosis = primary_diagnosis.get("age_at_diagnosis")
        tumor_stage = primary_diagnosis.get("tumor_stage", "Unknown")

        treatments = primary_diagnosis.get("treatments", [])
        treatment_received = 1 if len(treatments) > 0 else 0
        treatment_type = treatments[0].get("treatment_type") if treatments else None

        follow_up = case.get("follow_up", {})
        os_days = follow_up.get("days_to_last_follow_up", 0)
        vital_status = follow_up.get("vital_status", "Alive")
        os_event = 1 if vital_status == "Dead" else 0

        # Quality filters
        if os_days < 30:
            logger.warning("Case %s has insufficient follow-up (%s days), skipping", case_id, os_days)
            continue
        if age_at_diagnosis is None or age_at_diagnosis < 0:
            logger.warning("Case %s has invalid age, skipping", case_id)
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
        "Preprocessed %d clinical records (removed %d invalid)",
        len(df_clinical),
        len(cases) - len(df_clinical),
    )

    logger.info("=== DATA CLEANING LOG ===")
    logger.info("Initial cases: %d", len(cases))
    logger.info("Final clinical records: %d", len(df_clinical))
    if len(df_clinical) > 0:
        logger.info("Records with treatment: %d", df_clinical["treatment_received"].sum())
        logger.info("Event rate (deaths): %.2f%%", 100 * df_clinical["os_event"].mean())
        logger.info("Median follow-up: %.0f days", df_clinical["os_days"].median())
    logger.info("========================")

    return df_clinical


def preprocess_mutations(mutations_csv: str) -> pd.DataFrame:
    """Load and filter mutations to pathogenic variants."""
    logger.info("Loading mutations from %s...", mutations_csv)
    df_mutations = pd.read_csv(mutations_csv)
    df_pathogenic = df_mutations[df_mutations["is_pathogenic"] == 1].copy()
    logger.info("Loaded %d mutations, %d pathogenic", len(df_mutations), len(df_pathogenic))
    return df_pathogenic


SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS clinical (
        case_id VARCHAR,
        age_at_diagnosis INTEGER,
        tumor_stage VARCHAR,
        treatment_received INTEGER,
        treatment_type VARCHAR,
        os_days INTEGER,
        os_event INTEGER,
        PRIMARY KEY (case_id)
    );

    CREATE TABLE IF NOT EXISTS mutations (
        case_id VARCHAR,
        gene_symbol VARCHAR,
        variant_classification VARCHAR,
        is_pathogenic INTEGER,
        tumor_f DOUBLE
    );

    CREATE TABLE IF NOT EXISTS msi_status (
        case_id VARCHAR,
        msi_status VARCHAR,
        msi_score DOUBLE,
        PRIMARY KEY (case_id)
    );

    CREATE TABLE IF NOT EXISTS immune_subtypes (
        case_id VARCHAR,
        immune_subtype VARCHAR,
        immune_score DOUBLE,
        PRIMARY KEY (case_id)
    );

    CREATE OR REPLACE VIEW molecular_features AS
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
    LEFT JOIN immune_subtypes i ON c.case_id = i.case_id;
"""


def build_duckdb_feature_store(
    clinical_df: pd.DataFrame,
    mutations_df: pd.DataFrame,
    msi_df: pd.DataFrame,
    immune_df: pd.DataFrame,
    db_path: str,
):
    """Create schema and load all tables in a single connection.

    Previously schema creation and insertion used separate connections, which
    could produce confusing 'table does not exist' errors on fresh databases.
    Doing both in one connection guarantees ordering.
    """
    logger.info("Building DuckDB feature store at %s...", db_path)
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    conn = duckdb.connect(db_path)
    try:
        # Create schema
        for stmt in SCHEMA_SQL.strip().split(";"):
            stmt = stmt.strip()
            if stmt:
                conn.execute(stmt)
        logger.info("Schema created")

        # Register DataFrames so SELECT * FROM df_name works reliably
        conn.register("clinical_df", clinical_df)
        conn.register("mutations_df", mutations_df)
        conn.register("msi_df", msi_df)
        conn.register("immune_df", immune_df)

        # Truncate + insert
        conn.execute("DELETE FROM clinical")
        conn.execute("INSERT INTO clinical SELECT * FROM clinical_df")

        conn.execute("DELETE FROM mutations")
        conn.execute("INSERT INTO mutations SELECT * FROM mutations_df")

        conn.execute("DELETE FROM msi_status")
        conn.execute("INSERT INTO msi_status SELECT * FROM msi_df")

        conn.execute("DELETE FROM immune_subtypes")
        conn.execute("INSERT INTO immune_subtypes SELECT * FROM immune_df")

        n_clinical = conn.execute("SELECT COUNT(*) FROM clinical").fetchone()[0]
        n_mutations = conn.execute("SELECT COUNT(*) FROM mutations").fetchone()[0]
        n_msi = conn.execute("SELECT COUNT(*) FROM msi_status").fetchone()[0]
        n_immune = conn.execute("SELECT COUNT(*) FROM immune_subtypes").fetchone()[0]

        logger.info("Loaded %d clinical records", n_clinical)
        logger.info("Loaded %d mutations", n_mutations)
        logger.info("Loaded %d MSI status records", n_msi)
        logger.info("Loaded %d immune subtype records", n_immune)
    finally:
        conn.close()


if __name__ == "__main__":
    import sys

    clinical_json = sys.argv[1] if len(sys.argv) > 1 else "data/raw/TCGA-STAD_clinical.json"
    mutations_csv = sys.argv[2] if len(sys.argv) > 2 else "data/raw/TCGA-STAD_mutations.csv"
    msi_csv = sys.argv[3] if len(sys.argv) > 3 else "data/raw/TCGA-STAD_msi_status.csv"
    immune_csv = sys.argv[4] if len(sys.argv) > 4 else "data/raw/TCGA-STAD_immune_subtypes.csv"
    db_path = sys.argv[5] if len(sys.argv) > 5 else "data/processed/features.duckdb"

    df_clinical = preprocess_clinical_data(clinical_json)
    df_mutations = preprocess_mutations(mutations_csv)
    df_msi = pd.read_csv(msi_csv)
    df_immune = pd.read_csv(immune_csv)

    build_duckdb_feature_store(df_clinical, df_mutations, df_msi, df_immune, db_path)
    logger.info("Preprocessing complete!")
