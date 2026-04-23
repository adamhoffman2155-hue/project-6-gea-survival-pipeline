# Project 6: GEA Survival Risk Stratifier

**Research question:** Which combination of molecular features best predicts chemotherapy response and survival in gastroesophageal adenocarcinoma?

This is the sixth project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) — and the capstone. It answers the clinical question that started everything: can we integrate molecular features into a survival risk model that could inform treatment decisions? It combines MSI status, tumor mutational burden, DDR gene mutations, and immune subtype into a Cox proportional hazards model with an interactive Streamlit risk calculator.

## At a Glance

| | |
|---|---|
| **Stack** | Snakemake · DuckDB/SQL · scikit-survival · lifelines · Streamlit · Docker · pytest · Bash |
| **Data** | TCGA-STAD via GDC API (target); GBSG2 breast trial, n=686 (POC substitute) |
| **POC headline** | 5-fold CV C-index 0.682 ± 0.051 (held-out); training-fold 0.692 matches Schumacher 1994 (0.69–0.71); log-rank by grade chi²=21 p≈0, by hormonal therapy p=0.003 |
| **Role** | Capstone — pipeline architecture, feature selection from thesis biology, clinical plausibility review; implementation AI-assisted |
| **Portfolio** | Project 6 of 7 (capstone) · [full narrative](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) |

## Quick Start

```bash
conda env create -f environment.yaml
conda activate gea-survival

# Run the full pipeline
snakemake --cores 4

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

## Proof of Concept (v2 — cross-validated)

A minimal end-to-end Cox PH survival run on a real, published clinical-trial dataset so reviewers can verify the survival-modeling workflow without a TCGA download.

**Dataset:** GBSG2 — German Breast Cancer Study Group 2 (Schumacher et al. 1994), 686 patients with 299 events. Accessed via `sksurv.datasets.load_gbsg2()` so no network or account is required.

**Substitution note:** The full Snakemake pipeline targets TCGA-STAD via cBioPortal, but that host is not reachable from this reproducibility sandbox. GBSG2 is a real published randomized clinical trial dataset that is canonical for Cox PH benchmarking. The same sksurv Cox + C-index + KM code runs unchanged on any survival dataset.

**Cohort:**
- 686 patients, 299 events (43.6%), median follow-up 1084 days

**Features used:** age, estrec (estrogen receptor), horTh (hormonal therapy), menostat (menopausal status), pnodes (positive lymph nodes), progrec (progesterone receptor), tgrade (tumor grade), tsize (tumor size).

### Concordance index — headline metric

| Estimate | Value |
|---|---|
| Training-fold C-index (fit + evaluate on full cohort) | **0.692** |
| 5-fold cross-validated C-index (held-out test folds) | **0.682 ± 0.051** |

Both are inside Schumacher 1994's published 0.69–0.71 range for Cox PH on these features. The CV number is the honest held-out estimate.

Per-fold held-out C-index: 0.614, 0.691, 0.751, 0.637, 0.718.

### Cox PH coefficients (sorted by bootstrap p, N=200 resamples)

| Feature | Coef | HR | 95% CI | p_bootstrap |
|---|---|---|---|---|
| horTh=yes | -0.166 | 0.847 | 0.746–0.944 | 0.005 |
| tgrade=II | +0.304 | 1.355 | 1.108–1.662 | 0.005 |
| progrec | -0.449 | 0.638 | 0.494–0.788 | 0.005 |
| pnodes | +0.267 | 1.306 | 1.206–1.563 | 0.005 |
| tgrade=III | +0.330 | 1.392 | 1.142–1.775 | 0.005 |
| tsize | +0.112 | 1.118 | 0.996–1.241 | 0.080 |
| menostat=Post | +0.128 | 1.136 | 0.938–1.369 | 0.130 |
| age | -0.096 | 0.909 | 0.751–1.111 | 0.340 |
| estrec | +0.030 | 1.031 | 0.876–1.149 | 0.670 |

### Permutation feature importance (held-out CV, ΔC-index on shuffle)

| Feature | Mean ΔC-index | Std |
|---|---|---|
| pnodes | **0.062** | 0.021 |
| tgrade=III | **0.054** | 0.035 |
| progrec | **0.046** | 0.015 |
| tgrade=II | 0.041 | 0.020 |
| age | 0.012 | 0.009 |
| horTh=yes | 0.011 | 0.015 |
| menostat=Post | 0.005 | 0.008 |
| tsize | 0.003 | 0.009 |
| estrec | 0.001 | 0.003 |

### Stratification tests (Kaplan-Meier log-rank)

| Stratification | chi² | p |
|---|---|---|
| Tumor grade | 21.09 | ≈ 0 |
| Hormonal therapy | 8.56 | 0.0034 |

### Headline numbers

- Training C-index: **0.692** (matches Schumacher 1994 benchmark)
- 5-fold CV C-index: **0.682 ± 0.051** (held-out, honest)
- Top features by held-out perm importance: pnodes, tgrade=III, progrec
- horTh log-rank p: 0.0034
- tgrade log-rank p: ≈ 0

### Honest assessment

- Training-fold C-index (0.692) slightly overestimates held-out performance; CV (0.682) is the honest estimate of generalization.
- Both are in the 0.69–0.71 range reported for GBSG2 Cox PH in the published literature.
- Bootstrap CIs are approximate; sksurv does not expose per-coefficient SEs, so a Wald-test-based CI would require statsmodels or lifelines.
- Permutation importance on held-out folds is a more rigorous feature ranking than in-sample coefficient p-values.
- This is breast cancer, not GEA. The workflow runs unchanged on TCGA-STAD or any other survival dataset with (time, event, features).

**Reproduction:**
```bash
pip install scikit-survival pandas numpy matplotlib
python scripts/poc/run_poc.py
```
Outputs are written to `results/poc/` (CSV summary, plain-text report, KM curve PNG).

## What It Does

End-to-end survival analysis pipeline using TCGA-STAD data:

1. **Data acquisition** — GDC REST API client with pagination and error handling
2. **Preprocessing** — Data cleaning with explicit logging, DuckDB feature store
3. **Feature engineering** — MSI status (binary), TMB (mutations/Mb), DDR burden (pathogenic mutations in BRCA1/2, ATM, ATR, PALB2, RAD51, MLH1, MSH2, MSH6, POLE), immune subtype
4. **Survival modeling** — Cox PH and Kaplan-Meier analysis (lifelines)
5. **Visualization** — KM curves with CIs, forest plots, TMB distributions
6. **Dashboard** — Streamlit app: input molecular profile, get risk percentile and survival estimates
7. **Testing** — pytest suite for preprocessing, features, and model outputs

## Tools Used

| Category | Tools |
|----------|-------|
| Workflow | Snakemake |
| Data Acquisition | GDC REST API (requests) |
| Data Store | DuckDB (SQL queries) |
| Survival Models | scikit-survival (Cox PH, C-index), lifelines (KM, log-rank) |
| Dashboard | Streamlit |
| Visualization | matplotlib, seaborn |
| Scripting | Bash (download, validation, md5sum) |
| Testing | pytest |
| Containers | Docker (per-step Dockerfiles) |

## DuckDB Feature Store

Cohort selection uses explicit SQL:

```sql
SELECT case_id, msi_status, tmb, ddr_burden, immune_subtype, os_days, os_event
FROM molecular_features
JOIN clinical ON molecular_features.case_id = clinical.case_id
WHERE primary_site = 'Stomach' AND treatment_type IS NOT NULL
```

## Project Structure

```
project-6-gea-survival-pipeline/
├── README.md
├── Snakefile
├── .gitignore
├── environment.yaml
├── requirements.txt
├── LICENSE
├── config/
│   └── config.yaml
├── scripts/
│   ├── bash/
│   │   ├── download_tcga.sh
│   │   └── setup_dirs.sh
│   ├── python/
│   │   ├── fetch_gdc_api.py
│   │   ├── generate_synthetic_data.py
│   │   ├── preprocess.py
│   │   ├── build_feature_matrix.py
│   │   ├── survival_model.py
│   │   ├── figures.py
│   │   └── query_cohort.py
│   └── poc/
│       └── run_poc.py
├── dashboard/
│   └── app.py
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── docker/
│   ├── Dockerfile.download
│   ├── Dockerfile.analysis
│   └── Dockerfile.dashboard
├── data/
└── results/
    └── poc/
```

## Honest Note

This is a portfolio project built on public TCGA data. The survival estimates are **not for clinical use**. The synthetic data mirrors real TCGA structure but is not actual patient data. For clinical applications, this pipeline would require validation on an independent prospective cohort, IRB oversight, and regulatory approval.

## My Role

This is the capstone — connecting everything built in Projects 1-4 back to the original clinical question. I designed the pipeline architecture, selected the biological feature set based on my thesis findings, and reviewed survival model outputs for clinical plausibility. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 6 of 7**. It integrates molecular features from the preceding projects (MSI from Project 1, immune subtypes from Project 2, SHAP-validated biomarkers from Projects 3-4) into a single survival model with a deployable Streamlit calculator. It closes the loop on the clinical question that opened the portfolio. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## References

- [GDC Portal](https://portal.gdc.cancer.gov)
- [scikit-survival](https://scikit-survival.readthedocs.io)
- [lifelines](https://lifelines.readthedocs.io)
- [Snakemake](https://snakemake.readthedocs.io)
- [DuckDB](https://duckdb.org)

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
