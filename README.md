# GEA Survival Risk Stratifier

**End-to-end survival analysis pipeline for gastroesophageal adenocarcinoma (GEA) chemotherapy response prediction using TCGA-STAD data.**

This project answers a clinical question from wet-lab cancer research: **Which molecular features predict chemotherapy response and survival in GEA patients?** It combines multi-omic data (MSI status, tumor mutational burden, DNA damage repair gene mutations, immune subtype) with survival modeling to stratify patients by risk.

## Quick Start

```bash
# 1. Install dependencies
conda env create -f environment.yaml
conda activate gea-survival

# 2. Run the full pipeline
snakemake --cores 4

# 3. Launch the interactive dashboard
streamlit run dashboard/app.py
```

## What This Demonstrates

This portfolio project showcases production-grade bioinformatics engineering across multiple domains:

**Workflow Orchestration**: Snakemake DAG with modular rules for download → preprocess → feature engineering → modeling → visualization. Each step is containerized and reproducible.

**Data Engineering**: DuckDB feature store with explicit SQL queries for cohort selection, feature joins, and summary statistics. Demonstrates SQL-based data pipelines and efficient columnar storage.

**GDC API Integration**: REST API client (requests library) with pagination, error handling, and manifest validation. Handles real-world data acquisition challenges.

**Bash Scripting**: Directory scaffolding, file validation with md5sum, and standard Unix text processing tools (awk, cut, sort).

**Survival Modeling**: Cox Proportional Hazards and Kaplan-Meier analysis using lifelines. Multivariate risk stratification with concordance index validation.

**Visualization**: Publication-quality matplotlib/seaborn figures (KM curves with confidence intervals, forest plots of hazard ratios, distribution plots).

**Streamlit Dashboard**: Interactive risk calculator with patient molecular profile input, real-time risk percentile computation, survival probability estimates, and feature importance display.

**Testing**: pytest unit tests for preprocessing edge cases, feature construction correctness, and model output validation.

**Docker**: Containerized environments for download, analysis, and dashboard steps with reproducible dependencies.

## Biological Context

**Gastroesophageal Adenocarcinoma (GEA)** is an aggressive cancer with poor prognosis. Chemotherapy response varies widely among patients, and identifying predictive biomarkers is critical for precision oncology.

**MSI-H (Microsatellite Instability-High)**: Indicates mismatch repair deficiency; associated with better immunotherapy response but variable chemotherapy outcomes.

**TMB (Tumor Mutational Burden)**: Higher mutation load correlates with immune infiltration and can predict immunotherapy benefit.

**DDR Mutations (DNA Damage Repair)**: Pathogenic mutations in BRCA1/2, ATM, ATR, and other DDR genes predict sensitivity to PARP inhibitors and platinum-based chemotherapy.

**Immune Subtype**: TCGA immune classification (C1–C5) reflects tumor microenvironment composition and predicts treatment response.

This pipeline integrates these features into a Cox model to compute individual risk scores and survival probabilities.

## Project Structure

```
project-6-gea-survival-pipeline/
├── Snakefile                          # Full pipeline DAG
├── config/
│   └── config.yaml                    # Pipeline parameters and feature toggles
├── workflow/
│   ├── rules/                         # Modular Snakemake rules
│   └── envs/                          # Conda environments per step
├── scripts/
│   ├── bash/
│   │   ├── download_tcga.sh           # GDC manifest download + validation
│   │   └── setup_dirs.sh              # Directory scaffolding
│   └── python/
│       ├── fetch_gdc_api.py           # GDC REST API client
│       ├── generate_synthetic_data.py # Synthetic TCGA data for portfolio
│       ├── preprocess.py              # Data cleaning + DuckDB ingestion
│       ├── build_feature_matrix.py    # Multi-omic feature construction
│       ├── survival_model.py          # Cox PH + KM fitting
│       └── figures.py                 # Publication-quality plots
├── dashboard/
│   └── app.py                         # Streamlit risk stratifier
├── tests/
│   ├── test_preprocessing.py          # Preprocessing validation
│   ├── test_features.py               # Feature matrix correctness
│   └── test_model.py                  # Model output validation
├── docker/
│   ├── Dockerfile.download            # GDC API + bash tools
│   ├── Dockerfile.analysis            # lifelines + duckdb
│   └── Dockerfile.dashboard           # streamlit
├── notebooks/
│   └── exploratory_analysis.ipynb     # EDA and data cleaning decisions
├── data/
│   ├── raw/                           # Downloaded TCGA data
│   └── processed/                     # DuckDB feature store
├── results/
│   ├── cox_model.pkl                  # Fitted Cox model
│   ├── cox_model_summary.csv          # Model coefficients + p-values
│   ├── risk_scores.csv                # Patient risk percentiles
│   └── figures/                       # Publication-quality plots
├── requirements.txt
├── environment.yaml
└── README.md
```

## Key Implementation Details

### GDC API (scripts/python/fetch_gdc_api.py)

Uses the GDC REST API (https://api.gdc.cancer.gov) to query TCGA-STAD for clinical data and somatic mutations. Implements pagination (size/from parameters) and error handling for robust data acquisition.

### DuckDB Feature Store (scripts/python/query_cohort.py)

Loads cleaned data into DuckDB at `data/processed/features.duckdb`. Cohort selection uses explicit SQL:

```sql
SELECT case_id, msi_status, tmb, ddr_burden, immune_subtype, os_days, os_event
FROM molecular_features
JOIN clinical ON molecular_features.case_id = clinical.case_id
WHERE primary_site = 'Stomach' AND treatment_type IS NOT NULL
```

### Feature Matrix (scripts/python/build_feature_matrix.py)

Constructs multi-omic features:
- **MSI status**: Binary (MSI-H vs MSS)
- **TMB**: Total somatic mutations per megabase
- **DDR burden**: Count of pathogenic mutations in DDR gene set (BRCA1, BRCA2, ATM, ATR, PALB2, RAD51, MLH1, MSH2, MSH6, POLE)
- **Immune subtype**: From published TCGA immune classifications

### Survival Modeling (scripts/python/survival_model.py)

Fits Cox Proportional Hazards model with MSI status, TMB, DDR burden, immune subtype, and age as covariates. Outputs concordance index, hazard ratios with 95% CIs, and individual risk scores.

### Streamlit Dashboard (dashboard/app.py)

Interactive risk calculator with:
- Sidebar inputs for patient molecular profile
- Real-time risk percentile computation
- Estimated survival probabilities at 12, 24, 36 months
- Feature importance display (hazard ratios)
- Cohort comparison statistics
- Educational disclaimer

## Running the Pipeline

### Full Pipeline with Snakemake

```bash
snakemake --cores 4
```

This executes all steps: synthetic data generation → preprocessing → feature engineering → survival modeling → figure generation.

### Individual Steps

```bash
# Generate synthetic data
python scripts/python/generate_synthetic_data.py

# Preprocess and load to DuckDB
python scripts/python/preprocess.py data/raw/TCGA-STAD_clinical.json data/raw/TCGA-STAD_mutations.csv data/raw/TCGA-STAD_msi_status.csv data/raw/TCGA-STAD_immune_subtypes.csv data/processed/features.duckdb

# Build feature matrix
python scripts/python/build_feature_matrix.py data/processed/features.duckdb data/raw/TCGA-STAD_mutations.csv data/processed/feature_matrix.csv

# Fit survival models
python scripts/python/survival_model.py data/processed/feature_matrix.csv results

# Generate figures
python scripts/python/figures.py data/processed/feature_matrix.csv results/cox_model.pkl results/figures

# Run tests
pytest tests/ -v

# Launch dashboard
streamlit run dashboard/app.py
```

## Testing

```bash
pytest tests/ -v
```

Tests validate:
- Preprocessing: No NaN values in critical columns, positive OS days, binary event encoding
- Features: Non-negative DDR burden, expected columns, positive TMB, no NaN in features
- Model: Risk scores are positive floats, concordance index between 0–1, correct output shape

## Docker Containers

Each step has a dedicated container for reproducibility:

```bash
# Build containers
docker build -f docker/Dockerfile.download -t gea-download .
docker build -f docker/Dockerfile.analysis -t gea-analysis .
docker build -f docker/Dockerfile.dashboard -t gea-dashboard .

# Run dashboard
docker run -p 8501:8501 gea-dashboard streamlit run dashboard/app.py
```

## Honest Note

**This is a portfolio project on public TCGA data.** Survival estimates should **NOT** be used for clinical decision-making. The synthetic data used here mirrors real TCGA structure but is not actual patient data. The Cox model is trained on a small cohort and has not been validated on independent data.

For clinical applications, this pipeline would require:
- Validation on an independent prospective cohort
- Regulatory approval (FDA 510(k) or De Novo)
- Clinical trial data
- Institutional review board oversight

## References

- GDC Portal: https://portal.gdc.cancer.gov
- TCGA-STAD Project: https://www.cancer.gov/about-cancer/understanding/what-is-cancer
- lifelines Documentation: https://lifelines.readthedocs.io
- Snakemake: https://snakemake.readthedocs.io
- DuckDB: https://duckdb.org

## Author

Adam Hoffman  
Bioinformatics & Computational Biology  
M.Sc. Cancer Research, McGill University

---

*Built as a portfolio project demonstrating production-grade bioinformatics engineering: Snakemake orchestration, DuckDB SQL pipelines, survival modeling, Streamlit dashboards, Docker containerization, and comprehensive testing.*
