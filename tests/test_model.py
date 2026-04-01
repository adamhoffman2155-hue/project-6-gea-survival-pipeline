#!/usr/bin/env python3
"""
Unit tests for survival modeling.
"""

import pytest
import pandas as pd
import numpy as np
from lifelines import CoxPHFitter


def test_risk_score_in_valid_range():
    """Test that risk scores are positive floats."""
    # Create synthetic data
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        "T": np.random.randint(100, 2000, n),
        "E": np.random.randint(0, 2, n),
        "msi_status": np.random.randint(0, 2, n),
        "tmb": np.random.normal(0, 1, n),
        "ddr_burden": np.random.normal(0, 1, n),
        "immune_subtype": np.random.randint(1, 6, n),
        "age": np.random.normal(0, 1, n)
    })
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col="T", event_col="E", show_progress=False)
    
    # Predict risk scores
    risk_scores = cph.predict_partial_hazard(data.drop(columns=["T", "E"]))
    
    # Check that all risk scores are positive
    assert (risk_scores > 0).all()
    assert isinstance(risk_scores.values[0], (float, np.floating))


def test_cox_model_loads_without_error():
    """Test that Cox model can be fitted without error."""
    np.random.seed(42)
    n = 50
    
    data = pd.DataFrame({
        "T": np.random.randint(100, 2000, n),
        "E": np.random.randint(0, 2, n),
        "msi_status": np.random.randint(0, 2, n),
        "tmb": np.random.normal(0, 1, n),
        "ddr_burden": np.random.normal(0, 1, n),
        "immune_subtype": np.random.randint(1, 6, n),
        "age": np.random.normal(0, 1, n)
    })
    
    cph = CoxPHFitter()
    
    try:
        cph.fit(data, duration_col="T", event_col="E", show_progress=False)
        assert cph is not None
    except Exception as e:
        pytest.fail(f"Cox model fitting failed: {e}")


def test_concordance_index_valid():
    """Test that concordance index is between 0 and 1."""
    np.random.seed(42)
    n = 100
    
    data = pd.DataFrame({
        "T": np.random.randint(100, 2000, n),
        "E": np.random.randint(0, 2, n),
        "msi_status": np.random.randint(0, 2, n),
        "tmb": np.random.normal(0, 1, n),
        "ddr_burden": np.random.normal(0, 1, n),
        "immune_subtype": np.random.randint(1, 6, n),
        "age": np.random.normal(0, 1, n)
    })
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col="T", event_col="E", show_progress=False)
    
    c_index = cph.concordance_index_
    
    assert 0 <= c_index <= 1


def test_risk_score_shape():
    """Test that risk score output has correct shape."""
    np.random.seed(42)
    n = 50
    
    data = pd.DataFrame({
        "T": np.random.randint(100, 2000, n),
        "E": np.random.randint(0, 2, n),
        "msi_status": np.random.randint(0, 2, n),
        "tmb": np.random.normal(0, 1, n),
        "ddr_burden": np.random.normal(0, 1, n),
        "immune_subtype": np.random.randint(1, 6, n),
        "age": np.random.normal(0, 1, n)
    })
    
    cph = CoxPHFitter()
    cph.fit(data, duration_col="T", event_col="E", show_progress=False)
    
    risk_scores = cph.predict_partial_hazard(data.drop(columns=["T", "E"]))
    
    assert len(risk_scores) == n


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
