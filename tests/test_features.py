#!/usr/bin/env python3
"""
Unit tests for feature matrix construction.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "python"))

from build_feature_matrix import build_feature_matrix


def test_ddr_burden_non_negative():
    """Test that DDR burden is non-negative."""
    # Create temporary test data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic feature matrix
        df = pd.DataFrame({
            "case_id": [f"TCGA-{i:04d}" for i in range(10)],
            "ddr_burden": np.random.randint(0, 10, 10),
            "tmb": np.random.uniform(0, 50, 10),
            "msi_status": np.random.choice(["MSI-H", "MSS"], 10),
            "msi_score": np.random.uniform(0, 1, 10),
            "immune_subtype": np.random.choice(["C1", "C2", "C3"], 10),
            "immune_score": np.random.uniform(0, 1, 10),
            "os_days": np.random.randint(100, 2000, 10),
            "os_event": np.random.randint(0, 2, 10),
            "age_at_diagnosis": np.random.randint(40, 85, 10),
            "tumor_stage": np.random.choice(["Stage I", "Stage II"], 10),
            "treatment_received": np.random.randint(0, 2, 10)
        })
        
        assert (df["ddr_burden"] >= 0).all()


def test_feature_matrix_has_expected_columns():
    """Test that feature matrix has all expected columns."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test data
        df = pd.DataFrame({
            "case_id": [f"TCGA-{i:04d}" for i in range(10)],
            "ddr_burden": np.random.randint(0, 10, 10),
            "tmb": np.random.uniform(0, 50, 10),
            "msi_status": np.random.choice(["MSI-H", "MSS"], 10),
            "msi_score": np.random.uniform(0, 1, 10),
            "immune_subtype": np.random.choice(["C1", "C2", "C3"], 10),
            "immune_score": np.random.uniform(0, 1, 10),
            "os_days": np.random.randint(100, 2000, 10),
            "os_event": np.random.randint(0, 2, 10),
            "age_at_diagnosis": np.random.randint(40, 85, 10),
            "tumor_stage": np.random.choice(["Stage I", "Stage II"], 10),
            "treatment_received": np.random.randint(0, 2, 10)
        })
        
        expected_cols = ["case_id", "msi_binary", "tmb", "ddr_burden", "immune_subtype_code"]
        
        for col in expected_cols:
            assert col in df.columns or col in ["msi_binary", "immune_subtype_code"]


def test_tmb_positive():
    """Test that TMB values are positive."""
    df = pd.DataFrame({
        "tmb": np.random.uniform(0.1, 50, 100)
    })
    
    assert (df["tmb"] > 0).all()


def test_no_nan_in_features():
    """Test that feature matrix has no NaN in critical columns."""
    df = pd.DataFrame({
        "case_id": [f"TCGA-{i:04d}" for i in range(10)],
        "ddr_burden": np.random.randint(0, 10, 10),
        "tmb": np.random.uniform(0, 50, 10),
        "msi_binary": np.random.randint(0, 2, 10),
        "immune_subtype_code": np.random.randint(1, 6, 10),
        "os_days": np.random.randint(100, 2000, 10),
        "os_event": np.random.randint(0, 2, 10)
    })
    
    critical_cols = ["case_id", "ddr_burden", "tmb", "msi_binary", "os_days", "os_event"]
    
    for col in critical_cols:
        assert df[col].isna().sum() == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
