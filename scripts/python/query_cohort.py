#!/usr/bin/env python3
"""Query the DuckDB feature store for cohort selection and summary statistics.

Demonstrates SQL-based data pipelines for bioinformatics feature stores.
Uses DuckDB for efficient columnar queries on the molecular feature matrix.
"""

import argparse
import logging
import os
import sys

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)


def get_connection(db_path: str) -> duckdb.DuckDBPyConnection:
    """Connect to the DuckDB feature store."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DuckDB database not found: {db_path}")
    return duckdb.connect(db_path, read_only=True)


def query_full_cohort(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Select the full analysis cohort with all molecular features.
    
    Returns patients with stomach primary site and documented treatment.
    """
    query = """
    SELECT 
        m.case_id,
        m.msi_status,
        m.tmb,
        m.ddr_burden,
        m.immune_subtype,
        c.os_days,
        c.os_event,
        c.age_at_diagnosis,
        c.gender,
        c.stage
    FROM molecular_features m
    JOIN clinical c ON m.case_id = c.case_id
    WHERE c.primary_site = 'Stomach'
      AND c.treatment_type IS NOT NULL
      AND c.os_days > 0
    ORDER BY c.os_days DESC
    """
    df = con.execute(query).fetchdf()
    logger.info(f"Full cohort: {len(df)} patients")
    return df


def query_msi_summary(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Summary statistics stratified by MSI status."""
    query = """
    SELECT 
        m.msi_status,
        COUNT(*) AS n_patients,
        ROUND(AVG(m.tmb), 2) AS mean_tmb,
        ROUND(AVG(m.ddr_burden), 2) AS mean_ddr_burden,
        ROUND(AVG(c.os_days), 0) AS mean_os_days,
        ROUND(AVG(CAST(c.os_event AS FLOAT)), 3) AS event_rate,
        ROUND(AVG(c.age_at_diagnosis), 1) AS mean_age
    FROM molecular_features m
    JOIN clinical c ON m.case_id = c.case_id
    WHERE c.primary_site = 'Stomach'
      AND c.treatment_type IS NOT NULL
    GROUP BY m.msi_status
    ORDER BY m.msi_status
    """
    df = con.execute(query).fetchdf()
    logger.info(f"MSI summary:\n{df.to_string(index=False)}")
    return df


def query_ddr_high_patients(con: duckdb.DuckDBPyConnection, threshold: int = 2) -> pd.DataFrame:
    """Identify patients with high DDR mutation burden.
    
    Args:
        con: DuckDB connection
        threshold: Minimum DDR mutations to classify as 'high burden'
    """
    query = f"""
    SELECT 
        m.case_id,
        m.msi_status,
        m.tmb,
        m.ddr_burden,
        m.immune_subtype,
        c.os_days,
        c.os_event
    FROM molecular_features m
    JOIN clinical c ON m.case_id = c.case_id
    WHERE m.ddr_burden >= {threshold}
      AND c.primary_site = 'Stomach'
    ORDER BY m.ddr_burden DESC, m.tmb DESC
    """
    df = con.execute(query).fetchdf()
    logger.info(f"DDR-high patients (burden >= {threshold}): {len(df)}")
    return df


def query_immune_subtype_survival(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Median survival by immune subtype."""
    query = """
    SELECT 
        m.immune_subtype,
        COUNT(*) AS n_patients,
        ROUND(MEDIAN(c.os_days), 0) AS median_os_days,
        ROUND(AVG(CAST(c.os_event AS FLOAT)), 3) AS event_rate,
        ROUND(AVG(m.tmb), 2) AS mean_tmb
    FROM molecular_features m
    JOIN clinical c ON m.case_id = c.case_id
    WHERE c.primary_site = 'Stomach'
      AND c.treatment_type IS NOT NULL
      AND m.immune_subtype IS NOT NULL
    GROUP BY m.immune_subtype
    ORDER BY median_os_days DESC
    """
    df = con.execute(query).fetchdf()
    logger.info(f"Immune subtype survival:\n{df.to_string(index=False)}")
    return df


def query_feature_correlations(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Pairwise feature summary for correlation analysis."""
    query = """
    SELECT 
        ROUND(CORR(m.tmb, m.ddr_burden), 3) AS tmb_ddr_corr,
        ROUND(CORR(m.tmb, c.os_days), 3) AS tmb_os_corr,
        ROUND(CORR(m.ddr_burden, c.os_days), 3) AS ddr_os_corr,
        COUNT(*) AS n_patients
    FROM molecular_features m
    JOIN clinical c ON m.case_id = c.case_id
    WHERE c.primary_site = 'Stomach'
      AND c.treatment_type IS NOT NULL
    """
    df = con.execute(query).fetchdf()
    logger.info(f"Feature correlations:\n{df.to_string(index=False)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Query the GEA survival feature store")
    parser.add_argument("--db", default="data/processed/features.duckdb", help="Path to DuckDB database")
    parser.add_argument("--query", choices=["cohort", "msi", "ddr", "immune", "correlations", "all"],
                        default="all", help="Which query to run")
    parser.add_argument("--output", default="results", help="Output directory for CSV results")
    parser.add_argument("--ddr-threshold", type=int, default=2, help="DDR burden threshold")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    os.makedirs(args.output, exist_ok=True)

    try:
        con = get_connection(args.db)
    except FileNotFoundError:
        logger.error(f"Database not found at {args.db}. Run the pipeline first: snakemake --cores 4")
        sys.exit(1)

    queries = {
        "cohort": ("full_cohort.csv", lambda: query_full_cohort(con)),
        "msi": ("msi_summary.csv", lambda: query_msi_summary(con)),
        "ddr": ("ddr_high_patients.csv", lambda: query_ddr_high_patients(con, args.ddr_threshold)),
        "immune": ("immune_subtype_survival.csv", lambda: query_immune_subtype_survival(con)),
        "correlations": ("feature_correlations.csv", lambda: query_feature_correlations(con)),
    }

    to_run = queries.keys() if args.query == "all" else [args.query]

    for name in to_run:
        filename, func = queries[name]
        logger.info(f"\n{'='*50}\nRunning query: {name}\n{'='*50}")
        df = func()
        out_path = os.path.join(args.output, filename)
        df.to_csv(out_path, index=False)
        logger.info(f"Saved to {out_path}")

    con.close()
    logger.info("\nAll queries complete.")


if __name__ == "__main__":
    main()
