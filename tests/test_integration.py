"""
End-to-end integration tests that exercise the DuckDB-backed DAG:

    generate_synthetic_data -> preprocess -> build_feature_matrix

This is the regression net for the 5-col-schema vs 10-col-CSV insert
mismatch that broke preprocess.py on real data. Skipped if duckdb isn't
installed.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PY_SCRIPTS = PROJECT_ROOT / "scripts" / "python"


@pytest.fixture(scope="module")
def synthetic_pipeline_outputs(tmp_path_factory):
    """Run the three Python stages end-to-end in a tmp workspace."""
    duckdb = pytest.importorskip("duckdb")
    work = tmp_path_factory.mktemp("project6_e2e")
    raw = work / "raw"
    processed = work / "processed"
    raw.mkdir(); processed.mkdir()

    # Stage 1: generate synthetic data
    env_paths = {
        "clinical": raw / "TCGA-STAD_clinical.json",
        "mutations": raw / "TCGA-STAD_mutations.csv",
        "msi": raw / "TCGA-STAD_msi_status.csv",
        "immune": raw / "TCGA-STAD_immune_subtypes.csv",
    }
    subprocess.run(
        [sys.executable, str(PY_SCRIPTS / "generate_synthetic_data.py")],
        check=True, cwd=PROJECT_ROOT,
        # The synthetic-data script writes to data/raw/; copy into tmp after.
    )
    src_raw = PROJECT_ROOT / "data" / "raw"
    for key, dest in env_paths.items():
        src_name = {
            "clinical": "TCGA-STAD_clinical.json",
            "mutations": "TCGA-STAD_mutations.csv",
            "msi": "TCGA-STAD_msi_status.csv",
            "immune": "TCGA-STAD_immune_subtypes.csv",
        }[key]
        dest.write_bytes((src_raw / src_name).read_bytes())

    # Stage 2: preprocess -> DuckDB
    db_path = processed / "features.duckdb"
    subprocess.run(
        [sys.executable, str(PY_SCRIPTS / "preprocess.py"),
         str(env_paths["clinical"]), str(env_paths["mutations"]),
         str(env_paths["msi"]), str(env_paths["immune"]),
         str(db_path)],
        check=True, cwd=PROJECT_ROOT,
    )
    assert db_path.is_file()

    # Stage 3: feature matrix
    feat_path = processed / "feature_matrix.csv"
    subprocess.run(
        [sys.executable, str(PY_SCRIPTS / "build_feature_matrix.py"),
         str(db_path), str(env_paths["mutations"]), str(feat_path)],
        check=True, cwd=PROJECT_ROOT,
    )
    assert feat_path.is_file()

    return {"db": db_path, "features": feat_path}


def test_duckdb_tables_populated(synthetic_pipeline_outputs):
    """Regression guard for the preprocess INSERT bug."""
    import duckdb
    conn = duckdb.connect(str(synthetic_pipeline_outputs["db"]))
    try:
        for table in ("clinical", "mutations", "msi_status", "immune_subtypes"):
            n = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            assert n > 0, f"table {table} is empty"
        # mutations table specifically: schema is 5 columns; ensure that the
        # rows we loaded respect it (no silent width drift).
        # PRAGMA table_info returns (cid, name, type, notnull, default, pk)
        cols = [r[1] for r in conn.execute(
            "PRAGMA table_info('mutations')"
        ).fetchall()]
        assert cols == ["case_id", "gene_symbol", "variant_classification",
                        "is_pathogenic", "tumor_f"]
    finally:
        conn.close()


def test_feature_matrix_has_expected_columns(synthetic_pipeline_outputs):
    df = pd.read_csv(synthetic_pipeline_outputs["features"])
    required = {
        "case_id", "os_days", "os_event", "msi_binary", "tmb",
        "ddr_burden", "ddr_quartile", "immune_subtype_code",
    }
    missing = required - set(df.columns)
    assert not missing, f"missing columns: {missing}"
    assert df["os_event"].isin([0, 1]).all()
    assert df["msi_binary"].isin([0, 1]).all()
    assert (df["tmb"] >= 0).all()
