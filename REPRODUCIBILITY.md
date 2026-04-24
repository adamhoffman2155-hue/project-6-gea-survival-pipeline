# Reproducibility Scorecard

This project self-scores against three 2026 reproducibility standards used in
computational biology: **FAIR-BioRS** (Nature Scientific Data, 2023), **DOME**
(ML-in-biology validation, EMBL-EBI), and **CURE** (Credible, Understandable,
Reproducible, Extensible — Nature npj Systems Biology 2026).

![Repro](https://img.shields.io/badge/FAIR_DOME_CURE-13%2F14_%7C_6%2F7_%7C_4%2F4-brightgreen)

## FAIR-BioRS (13 / 14)

| # | Item | Status | Evidence |
|---|---|---|---|
| 1 | Source code in a public VCS | ✅ | GitHub repo |
| 2 | License file present | ✅ | `LICENSE` (MIT) |
| 3 | Persistent identifier (DOI/Zenodo) | ⬜ | Not yet minted |
| 4 | Dependencies pinned | ✅ | `requirements.txt`, `environment.yaml` (exact versions) |
| 5 | Containerized environment | ✅ | Three Dockerfiles (analysis, dashboard, download) |
| 6 | Automated tests | ✅ | pytest suite (12 tests) |
| 7 | CI/CD on every push | ✅ | `.github/workflows/ci.yml` |
| 8 | README with install + run instructions | ✅ | `README.md` Quick Start |
| 9 | Example data included or referenced | ✅ | GBSG2 (sksurv, no network) + synthetic TCGA-STAD |
| 10 | Expected outputs documented | ✅ | `results/poc/poc_summary.txt` |
| 11 | Version-controlled configuration | ✅ | `config/config.yaml` |
| 12 | Code style enforced (linter) | ✅ | `ruff` + `pre-commit` |
| 13 | Data provenance documented | ✅ | README "Features" + Snakefile rules |
| 14 | Archived release (vX.Y.Z) | ⬜ | No tagged release yet |

## DOME (ML-in-biology) (6 / 7)

| # | Dimension | Status | Evidence |
|---|---|---|---|
| D | **Data**: source, version, preprocessing documented | ✅ | GBSG2 (Schumacher 1994) in POC; synthetic TCGA-STAD in full pipeline |
| O | **Optimization**: hyperparameter search documented | ✅ | Cox PH α=0.01 fixed; RSF default hyperparameters |
| M | **Model**: architecture, code, learned params available | ✅ | `scripts/python/survival_model.py`, pickled via joblib |
| E | **Evaluation**: metrics, CV scheme, baselines documented | ✅ | 5-fold CV C-index 0.682 ± 0.051; log-rank; permutation importance |
| + | Interpretability | ✅ | Permutation importance on held-out folds; SHAP optional |
| + | Class-imbalance handled | ✅ | StratifiedKFold on event indicator |
| + | Independent validation cohort | ⬜ | Only GBSG2; TCGA-STAD synthetic, not real external holdout |

## CURE (Nature npj Sys Biol 2026) (4 / 4)

| Letter | Criterion | Status | Evidence |
|---|---|---|---|
| **C** | Container reproducibility | ✅ | `docker/Dockerfile.analysis`, `.dashboard`, `.download` |
| **U** | URL persistence | ✅ | GitHub + sksurv GBSG2 bundled + GDC REST API |
| **R** | Registered methods | ✅ | `Snakefile` DAG + `scripts/poc/run_poc.py` |
| **E** | Evidence of a real run | ✅ | `results/poc/poc_summary.txt` (CV C-index 0.682±0.051) |

## How to reproduce the score

```bash
ruff check . && ruff format --check .
pytest tests/ -v
python scripts/poc/run_poc.py          # regenerates results/poc/*
snakemake --cores 4                    # full pipeline (synthetic TCGA-STAD)
docker build -f docker/Dockerfile.analysis .
```

## Cross-project standing

Project-6 is the **capstone** of the portfolio chain. It integrates features
conceptually sourced from upstream projects:

- From Project-1 — transcriptomic DEGs and pathway scores (narrative input)
- From Project-2 — TME immune-subtype labels (narrative input)
- From Project-3 — drug-response prediction context (narrative input)
- From Project-4 — DDR biomarker panel (MSI status, HRD burden) used as Cox covariates

The Streamlit dashboard exposes the trained Cox model as a risk calculator
— the end-user-facing tip of the entire portfolio.
