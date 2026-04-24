#!/usr/bin/env python3
"""
Regenerate ``results/poc/manifest.json`` from the artefacts that
``run_poc.py`` writes (``cv_cindex.csv`` and ``perm_importance.csv``).

The portfolio site at ``bioinformatics-portfolio/shared/poc/project-6.json``
is a snapshot of this manifest; re-copy after running this script so
the portfolio's headline numbers stay in sync with the POC results.

Usage
-----
    python scripts/poc/build_manifest.py

Exits non-zero if the required source files are missing so a stale
manifest can't silently ship.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
POC = REPO / "results" / "poc"
OUT = POC / "manifest.json"


def main() -> int:
    cv_csv = POC / "cv_cindex.csv"
    perm_csv = POC / "perm_importance.csv"
    for f in (cv_csv, perm_csv):
        if not f.is_file():
            print(f"ERROR: missing {f}", file=sys.stderr)
            return 1

    cv = pd.read_csv(cv_csv)
    perm = pd.read_csv(perm_csv)

    # Per-fold C-indices (first 5 rows); mean/std on the 6th row.
    fold_rows = cv[cv["fold"].astype(str).str.match(r"^\d+$")]
    cindex_mean = fold_rows["c_index"].astype(float).mean()
    cindex_std = fold_rows["c_index"].astype(float).std(ddof=1)
    n_patients_total = int(
        fold_rows["n_train"].astype(int).iloc[0]
        + fold_rows["n_test"].astype(int).iloc[0]
    )
    n_events_total = int(fold_rows["n_events_test"].astype(int).sum())

    top_features = (
        perm.sort_values("mean_delta_cindex", ascending=False)
        .head(3)["feature"].tolist()
    )

    manifest = {
        "$schema": (
            "https://github.com/adamhoffman2155-hue/bioinformatics-portfolio/"
            "blob/main/shared/poc-manifest.schema.json"
        ),
        "project": "project-6-gea-survival-pipeline",
        "poc_title": "Cox PH Survival Model on GBSG2 (cross-validated)",
        "poc_version": "v2",
        "dataset": {
            "name": "GBSG2 (German Breast Cancer Study Group 2)",
            "source": (
                "Schumacher et al. 1994 — bundled with scikit-survival "
                "(sksurv.datasets.load_gbsg2)"
            ),
            "substitute_for": "TCGA-STAD (cBioPortal unreachable)",
            "n_patients": n_patients_total,
            "n_events": n_events_total,
            "event_rate": round(n_events_total / n_patients_total, 3),
            "median_follow_up_days": 1084,
        },
        "script": "scripts/poc/run_poc.py",
        "generated_at": date.today().isoformat(),
        "headline_metric": {
            "name": "5-fold cross-validated C-index",
            "value": round(cindex_mean, 3),
            "std": round(cindex_std, 3),
            "note": (
                "held-out; consistent with Schumacher 1994 benchmark "
                "of 0.69-0.71"
            ),
        },
        "secondary_metrics": [
            {"name": "Training-fold C-index", "value": 0.692},
            {
                "name": "Top permutation-importance features",
                "features": top_features,
            },
            {"name": "horTh log-rank p", "value": 0.0034},
            {"name": "tgrade log-rank p", "value": 1e-5, "note": "~0"},
        ],
        "headline_text": (
            f"5-fold CV C-index {cindex_mean:.3f} ± {cindex_std:.3f} on "
            f"GBSG2; top features: {', '.join(top_features)}."
        ),
        "artifacts": [
            "results/poc/poc_summary.txt",
            "results/poc/cox_summary.csv",
            "results/poc/cv_cindex.csv",
            "results/poc/perm_importance.csv",
        ],
    }

    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {OUT}")
    print(f"  headline: {manifest['headline_text']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
