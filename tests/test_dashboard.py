"""
Smoke test for dashboard/app.py.

Ensures the dashboard module imports without a running Streamlit
server, and the pure helper `survival_probabilities_from_cox`
returns monotone-non-increasing probabilities over time when
invoked against a freshly-fitted Cox model.

Skipped when lifelines / streamlit aren't importable (GitHub Actions
installs both via requirements.txt).
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = PROJECT_ROOT / "dashboard" / "app.py"


def _have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None


requires_streamlit_and_lifelines = pytest.mark.skipif(
    not (_have("streamlit") and _have("lifelines")),
    reason="streamlit + lifelines required",
)


@requires_streamlit_and_lifelines
def test_survival_probabilities_from_cox_is_monotone(tmp_path):
    """Fit a small Cox model, load the dashboard's helper, and check
    that survival probabilities decrease over time (never increase)."""
    # Run the dashboard app.py under a guard so the sidebar widgets
    # don't actually render. We only want the module's top-level defs.
    # Streamlit happily imports when no Streamlit server is attached —
    # calls like `st.title` become no-ops.
    spec = importlib.util.spec_from_file_location("dashboard_app", DASHBOARD)
    module = importlib.util.module_from_spec(spec)
    # Prevent `st.stop()` from killing the pytest process when no model
    # is present by chdir-ing into a directory where load_model_and_data()
    # fails gracefully — we only need the helper symbol.
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        try:
            spec.loader.exec_module(module)
        except SystemExit:
            # st.stop() raises; swallow so we can still inspect the
            # symbols that were defined before that call.
            pass
        helper = getattr(module, "survival_probabilities_from_cox", None)
        assert helper is not None, "helper was not defined"
    finally:
        os.chdir(cwd)

    # Fit a tiny Cox model on synthetic data
    import numpy as np
    from lifelines import CoxPHFitter
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "T": rng.integers(30, 1800, size=n).astype(float),
        "E": rng.integers(0, 2, size=n).astype(int),
        "msi_status": rng.integers(0, 2, size=n).astype(float),
        "tmb": rng.normal(0, 1, size=n),
        "ddr_burden": rng.normal(0, 1, size=n),
        "immune_subtype": rng.integers(1, 6, size=n).astype(float),
        "age": rng.normal(0, 1, size=n),
    })
    cph = CoxPHFitter()
    cph.fit(df, duration_col="T", event_col="E", show_progress=False)

    input_data = pd.DataFrame({
        "msi_status": [1.0], "tmb": [0.5], "ddr_burden": [0.0],
        "immune_subtype": [2.0], "age": [0.2],
    })
    probs = helper(cph, input_data, [180, 365, 730, 1095])
    series = [probs[t] for t in sorted(probs)]
    # Monotone non-increasing; survival curves never go up.
    for a, b in zip(series, series[1:]):
        assert b <= a + 1e-9, f"survival rose from {a:.3f} to {b:.3f}"
    assert all(0.0 <= v <= 1.0 for v in series)
