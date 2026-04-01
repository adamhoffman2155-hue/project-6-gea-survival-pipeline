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
├── Snakefile
├── config/
│   └── config.yaml
├── workflow/
│   ├── rules/                     # download, preprocess, features, model, figures
│   └── envs/                      # Conda envs per step
├── scripts/
│   ├── bash/
│   │   ├── download_tcga.sh       # GDC download + md5 validation
│   │   └── setup_dirs.sh
│   └── python/
│       ├── fetch_gdc_api.py       # GDC REST API client
│       ├── generate_synthetic_data.py
│       ├── preprocess.py          # Cleaning + DuckDB ingestion
│       ├── build_feature_matrix.py
│       ├── survival_model.py      # Cox PH + KM
│       └── figures.py
├── dashboard/
│   └── app.py                     # Streamlit risk calculator
├── tests/
│   ├── test_preprocessing.py
│   ├── test_features.py
│   └── test_model.py
├── docker/
│   ├── Dockerfile.download
│   ├── Dockerfile.analysis
│   └── Dockerfile.dashboard
├── notebooks/
│   └── exploratory_analysis.ipynb
├── data/
├── results/
├── requirements.txt
├── environment.yaml
└── README.md
```

## Honest Note

This is a portfolio project built on public TCGA data. The survival estimates are **not for clinical use**. The synthetic data mirrors real TCGA structure but is not actual patient data. For clinical applications, this pipeline would require validation on an independent prospective cohort, IRB oversight, and regulatory approval.

## My Role

This is the capstone — connecting everything built in Projects 1–4 back to the original clinical question. I designed the pipeline architecture, selected the biological feature set based on my thesis findings, and reviewed survival model outputs for clinical plausibility. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 6 of 7** — currently in development. It integrates molecular features from the preceding projects (MSI from Project 1, immune subtypes from Project 2, SHAP-validated biomarkers from Projects 3–4) into a single survival model with a deployable Streamlit calculator. It closes the loop on the clinical question that opened the portfolio. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## References

- [GDC Portal](https://portal.gdc.cancer.gov)
- [lifelines documentation](https://lifelines.readthedocs.io)
- [Snakemake](https://snakemake.readthedocs.io)
- [DuckDB](https://duckdb.org)

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
