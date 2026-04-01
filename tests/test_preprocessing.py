#!/usr/bin/env python3
"""
Unit tests for preprocessing module.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts" / "python"))

from preprocess import preprocess_clinical_data


def test_clinical_data_loading():
    """Test that clinical data loads without errors."""
    import json
    
    # Create temporary test data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_cases = [
            {
                "case_id": "TCGA-TEST-0001",
                "primary_site": "Stomach",
                "diagnoses": [{
                    "age_at_diagnosis": 65,
                    "tumor_stage": "Stage III",
                    "treatments": [{"treatment_type": "Chemotherapy"}]
                }],
                "follow_up": {"days_to_last_follow_up": 500, "vital_status": "Dead"}
            }
        ]
        json.dump(test_cases, f)
        temp_file = f.name
    
    try:
        df = preprocess_clinical_data(temp_file)
        assert len(df) > 0
        assert "case_id" in df.columns
        assert "os_days" in df.columns
        assert "os_event" in df.columns
    finally:
        Path(temp_file).unlink()


def test_no_missing_values_in_critical_columns():
    """Test that critical columns have no NaN values after preprocessing."""
    import json
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_cases = [
            {
                "case_id": f"TCGA-TEST-{i:04d}",
                "primary_site": "Stomach",
                "diagnoses": [{
                    "age_at_diagnosis": 50 + i,
                    "tumor_stage": "Stage II",
                    "treatments": []
                }],
                "follow_up": {"days_to_last_follow_up": 300 + i * 10, "vital_status": "Alive"}
            }
            for i in range(10)
        ]
        json.dump(test_cases, f)
        temp_file = f.name
    
    try:
        df = preprocess_clinical_data(temp_file)
        
        critical_cols = ["case_id", "os_days", "os_event", "age_at_diagnosis"]
        for col in critical_cols:
            assert df[col].isna().sum() == 0, f"Column {col} has NaN values"
    finally:
        Path(temp_file).unlink()


def test_os_days_positive():
    """Test that overall survival days are positive."""
    import json
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_cases = [
            {
                "case_id": "TCGA-TEST-0001",
                "primary_site": "Stomach",
                "diagnoses": [{
                    "age_at_diagnosis": 65,
                    "tumor_stage": "Stage I",
                    "treatments": []
                }],
                "follow_up": {"days_to_last_follow_up": 100, "vital_status": "Alive"}
            }
        ]
        json.dump(test_cases, f)
        temp_file = f.name
    
    try:
        df = preprocess_clinical_data(temp_file)
        assert (df["os_days"] > 0).all()
    finally:
        Path(temp_file).unlink()


def test_os_event_binary():
    """Test that os_event is binary (0 or 1)."""
    import json
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        test_cases = [
            {
                "case_id": "TCGA-TEST-0001",
                "primary_site": "Stomach",
                "diagnoses": [{
                    "age_at_diagnosis": 65,
                    "tumor_stage": "Stage I",
                    "treatments": []
                }],
                "follow_up": {"days_to_last_follow_up": 100, "vital_status": "Dead"}
            }
        ]
        json.dump(test_cases, f)
        temp_file = f.name
    
    try:
        df = preprocess_clinical_data(temp_file)
        assert df["os_event"].isin([0, 1]).all()
    finally:
        Path(temp_file).unlink()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
