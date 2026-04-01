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
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def preprocess_clinical_data(clinical_json: str) -> pd.DataFrame:
    """
    Load and preprocess clinical data from JSON.
    
    Returns:
        DataFrame with columns: case_id, age_at_diagnosis, tumor_stage, 
                               os_days, os_event, treatment_received
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
            "os_event": os_event
        })
    
    df_clinical = pd.DataFrame(records)
    logger.info(f"Preprocessed {len(df_clinical)} clinical records (removed {len(cases) - len(df_clinical)} invalid records)")
    
    # Print cleaning log
    logger.info("=== DATA CLEANING LOG ===")
    logger.info(f"Initial cases: {len(cases)}")
    logger.info(f"Final clinical records: {len(df_clinical)}")
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


def create_duckdb_schema(db_path: str):
    """Create DuckDB schema for feature store."""
    logger.info(f"Creating DuckDB schema at {db_path}...")
    
    conn = duckdb.connect(db_path)
    
    # Clinical table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clinical (
            case_id VARCHAR,
            age_at_diagnosis INTEGER,
            tumor_stage VARCHAR,
            treatment_received INTEGER,
            treatment_type VARCHAR,
            os_days INTEGER,
            os_event INTEGER,
            PRIMARY KEY (case_id)
        )
    """)
    
    # Mutations table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mutations (
            case_id VARCHAR,
            gene_symbol VARCHAR,
            variant_classification VARCHAR,
            is_pathogenic INTEGER,
            tumor_f DOUBLE
        )
    """)
    
    # MSI status table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS msi_status (
            case_id VARCHAR,
            msi_status VARCHAR,
            msi_score DOUBLE,
            PRIMARY KEY (case_id)
        )
    """)
    
    # Immune subtypes table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS immune_subtypes (
            case_id VARCHAR,
            immune_subtype VARCHAR,
            immune_score DOUBLE,
            PRIMARY KEY (case_id)
        )
    """)
    
    # Feature matrix view
    conn.execute("""
        CREATE VIEW IF NOT EXISTS molecular_features AS
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
    """)
    
    conn.close()
    logger.info("DuckDB schema created successfully")


def load_data_to_duckdb(
    clinical_df: pd.DataFrame,
    mutations_df: pd.DataFrame,
    msi_df: pd.DataFrame,
    immune_df: pd.DataFrame,
    db_path: str
):
    """Load preprocessed data into DuckDB."""
    logger.info(f"Loading data into DuckDB at {db_path}...")
    
    conn = duckdb.connect(db_path)
    
    # Load tables
    conn.execute("DELETE FROM clinical")
    conn.execute("INSERT INTO clinical SELECT * FROM clinical_df")
    
    conn.execute("DELETE FROM mutations")
    conn.execute("INSERT INTO mutations SELECT * FROM mutations_df")
    
    conn.execute("DELETE FROM msi_status")
    conn.execute("INSERT INTO msi_status SELECT * FROM msi_df")
    
    conn.execute("DELETE FROM immune_subtypes")
    conn.execute("INSERT INTO immune_subtypes SELECT * FROM immune_df")
    
    # Verify
    n_clinical = conn.execute("SELECT COUNT(*) FROM clinical").fetchall()[0][0]
    n_mutations = conn.execute("SELECT COUNT(*) FROM mutations").fetchall()[0][0]
    n_msi = conn.execute("SELECT COUNT(*) FROM msi_status").fetchall()[0][0]
    n_immune = conn.execute("SELECT COUNT(*) FROM immune_subtypes").fetchall()[0][0]
    
    logger.info(f"Loaded {n_clinical} clinical records")
    logger.info(f"Loaded {n_mutations} mutations")
    logger.info(f"Loaded {n_msi} MSI status records")
    logger.info(f"Loaded {n_immune} immune subtype records")
    
    conn.close()


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
    
    # Create schema and load
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    create_duckdb_schema(db_path)
    load_data_to_duckdb(df_clinical, df_mutations, df_msi, df_immune, db_path)
    
    logger.info("Preprocessing complete!")
