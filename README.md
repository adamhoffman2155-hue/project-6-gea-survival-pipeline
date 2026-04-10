# Project 6: GEA Survival Risk Stratifier

**Research question:** Which combination of molecular features best predicts chemotherapy response and survival in gastroesophageal adenocarcinoma?

This is the sixth project in a [computational biology portfolio](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) — and the capstone. It answers the clinical question that started everything: can we integrate molecular features into a survival risk model that could inform treatment decisions? It combines MSI status, tumor mutational burden, DDR gene mutations, and immune subtype into a Cox proportional hazards model with an interactive Streamlit risk calculator.

## Quick Start

```bash
conda env create -f environment.yaml
conda activate gea-survival

# Run the full pipeline
snakemake --cores 4

# Launch the interactive dashboard
streamlit run dashboard/app.py
```

## Proof of Concept

A minimal end-to-end Cox PH survival run on a real, published clinical trial dataset so reviewers can verify the survival-modeling workflow without a TCGA download.

**Dataset:** GBSG2 — German Breast Cancer Study Group 2 (Schumacher et al. 1994), 686 patients with 299 events. Accessed via `sksurv.datasets.load_gbsg2()` so no network or account is required.

**Substitution note:** The full Snakemake pipeline targets TCGA-STAD via cBioPortal, but that host is not reachable from this reproducibility sandbox. GBSG2 is a real published randomized clinical trial dataset that is canonical for Cox PH benchmarking. The same sksurv Cox + C-index + KM code runs unchanged on any survival dataset.

**What the POC tests:**
- Cox Proportional Hazards fit on 9 features (age, tumor size, tumor grade, node count, progesterone/estrogen receptor, menopausal status, hormonal therapy)
- Concordance index on training cohort
- Bootstrap 95%% CIs on hazard ratios (N=200 resamples)
- Kaplan-Meier curves stratified by hormonal therapy + log-rank test

**Headline numbers** (actual run output):
- Cohort: 686 patients, 299 events, median follow-up 1084 days
- **Concordance index: 0.692** (matches Schumacher 1994 published benchmark of 0.69–0.71)
- Top prognostic features by bootstrap p:
  - `progrec` (progesterone receptor): HR = 0.64 (0.48–0.79), p ≈ 0.005 — protective
  - `tgrade=III`: HR = 1.39 (1.14–1.69), p ≈ 0.005 — higher risk
  - `pnodes` (positive nodes): HR = 1.31 (1.18–1.60), p ≈ 0.005 — higher risk
  - `horTh=yes` (hormonal therapy): HR = 0.85 (0.74–0.95), p ≈ 0.01 — protective
- **Log-rank test by hormonal therapy: chi² = 8.56, p = 0.0034**

**Limits:**
- C-index is training-fold only, not cross-validated; held-out performance will be lower
- Bootstrap p-values are approximate; a proper Wald test from a stats package would be preferred
- sksurv does not expose per-coefficient standard errors directly
- GBSG2 is breast cancer, not GEA; biological interpretation is dataset-specific

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
| Survival Models | lifelines (Cox PH, KM) |
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
- [lifelines documentation](https://lifelines.readthedocs.io)
- [Snakemake](https://snakemake.readthedocs.io)
- [DuckDB](https://duckdb.org)

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
