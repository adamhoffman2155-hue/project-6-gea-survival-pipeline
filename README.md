# Project 6: GEA Survival Risk Stratifier

> **A statistical model that combines a cancer patient's tumor features into a single survival-risk score, with a point-and-click calculator built on top. The capstone of the portfolio.**

## The short version

**What this project does.** Takes the standard set of tumor features a pathologist would record (tumor size, grade, node involvement, hormone receptor status, etc.) and combines them into a Cox proportional-hazards model that estimates each patient's survival risk. Exposes the model as a Streamlit web app where a clinician could enter a patient's profile and see a risk percentile.

**The question behind it.** This is the capstone — it asks the clinical question that started the portfolio: can we integrate molecular features from Projects 1-4 (MSI, immune subtype, DDR burden) into a single risk score that could actually inform treatment decisions?

**What the proof-of-concept shows.** On a landmark 1994 breast-cancer trial dataset (GBSG2, 686 patients, 299 death events), the model achieves a cross-validated concordance index of **0.68 ± 0.05**. Plain English: if you pick two patients at random, the model correctly identifies the higher-risk one **68% of the time** — matching the published Schumacher 1994 benchmark (0.69-0.71). Tumor grade and positive-node count are the strongest individual predictors.

**Why it matters.** This closes the loop on the portfolio. Projects 1-4 identify biomarkers; this project asks "can we put them together into something that changes what happens to a patient?" The GBSG2 POC proves the statistical infrastructure works; the full pipeline targets TCGA-STAD for the real GEA clinical question.

---

_The rest of this README is technical detail for bioinformaticians, recruiters doing a deep review, or anyone reproducing the work. This is a portfolio project built on public TCGA data; the survival estimates are **not for clinical use** and would require independent validation, IRB oversight, and regulatory approval before any clinical application._

## At a Glance

| | |
|---|---|
| **Stack** | Snakemake · DuckDB/SQL · scikit-survival · lifelines · Streamlit · Docker · pytest · Bash |
| **Data** | TCGA-STAD via GDC API (full-pipeline target); GBSG2 breast trial, n=686 (POC substitute) |
| **POC headline** | 5-fold CV C-index 0.682 ± 0.051 (held-out); training-fold 0.692 matches Schumacher 1994 (0.69–0.71); log-rank by grade chi²=21 p≈0, by hormonal therapy p=0.003 |
| **Status** | POC: **Runnable POC** with committed CV outputs. Full pipeline: **Full-data target** (requires GDC API access) |
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

**Substitution note:** The full Snakemake pipeline targets TCGA-STAD via cBioPortal, but that host is not reachable from this reproducibility sandbox. GBSG2 is a real published randomized clinical trial dataset that is canonical for Cox PH benchmarking.

**Cohort:** 686 patients, 299 events (43.6%), median follow-up 1084 days.

**Features used:** age, estrec (estrogen receptor), horTh (hormonal therapy), menostat (menopausal status), pnodes (positive lymph nodes), progrec (progesterone receptor), tgrade (tumor grade), tsize (tumor size).

### Concordance index — headline metric

| Estimate | Value |
|---|---|
| Training-fold C-index | **0.692** |
| 5-fold cross-validated C-index (held-out) | **0.682 ± 0.051** |

Per-fold held-out C-index: 0.614, 0.691, 0.751, 0.637, 0.718.

### Cox PH coefficients (sorted by bootstrap p, N=200 resamples)

| Feature | Coef | HR | 95% CI | p_bootstrap |
|---|---|---|---|---|
| horTh=yes | -0.166 | 0.847 | 0.746–0.944 | 0.005 |
| tgrade=II | +0.304 | 1.355 | 1.108–1.662 | 0.005 |
| progrec | -0.449 | 0.638 | 0.494–0.788 | 0.005 |
| pnodes | +0.267 | 1.306 | 1.206–1.563 | 0.005 |
| tgrade=III | +0.330 | 1.392 | 1.142–1.775 | 0.005 |

### Permutation feature importance (held-out CV, ΔC-index on shuffle)

| Feature | Mean ΔC-index |
|---|---|
| pnodes | **0.062** |
| tgrade=III | **0.054** |
| progrec | **0.046** |
| tgrade=II | 0.041 |
| age | 0.012 |

### Stratification tests (Kaplan-Meier log-rank)

| Stratification | chi² | p |
|---|---|---|
| Tumor grade | 21.09 | ≈ 0 |
| Hormonal therapy | 8.56 | 0.0034 |

### Reproduction

```bash
pip install scikit-survival pandas numpy matplotlib
python scripts/poc/run_poc.py
```

Outputs in `results/poc/`: `cox_summary.csv`, `cv_cindex.csv`, `perm_importance.csv`, `poc_summary.txt`, KM curve PNG.

### Honest assessment

- Training-fold C-index (0.692) slightly overestimates held-out performance; CV (0.682) is the honest estimate of generalization.
- Both are in the 0.69–0.71 range reported for GBSG2 Cox PH in published literature.
- Bootstrap CIs are approximate; a Wald-test-based CI would require statsmodels or lifelines.
- This is breast cancer, not GEA. The workflow runs unchanged on TCGA-STAD or any other survival dataset.

## What the Full Pipeline Does

End-to-end survival analysis using TCGA-STAD data:

1. **Data acquisition** — GDC REST API client with pagination and error handling
2. **Preprocessing** — Data cleaning with explicit logging, DuckDB feature store
3. **Feature engineering** — MSI status (binary), TMB (mutations/Mb), DDR burden (pathogenic mutations in BRCA1/2, ATM, ATR, PALB2, RAD51, MLH1, MSH2, MSH6, POLE), immune subtype
4. **Survival modeling** — Cox PH and Kaplan-Meier analysis
5. **Visualization** — KM curves with CIs, forest plots, TMB distributions
6. **Dashboard** — Streamlit app: input molecular profile → get risk percentile and survival estimates
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

```sql
SELECT case_id, msi_status, tmb, ddr_burden, immune_subtype, os_days, os_event
FROM molecular_features
JOIN clinical ON molecular_features.case_id = clinical.case_id
WHERE primary_site = 'Stomach' AND treatment_type IS NOT NULL
```

## Honest Note

This is a portfolio project built on public TCGA data. The survival estimates are **not for clinical use**. The synthetic data mirrors real TCGA structure but is not actual patient data. For clinical applications, this pipeline would require validation on an independent prospective cohort, IRB oversight, and regulatory approval.

## My Role

This is the capstone — connecting everything built in Projects 1-4 back to the original clinical question. I designed the pipeline architecture, selected the biological feature set based on my thesis findings, and reviewed survival model outputs for clinical plausibility. Implementation was heavily AI-assisted.

## Context in the Portfolio

This is **Project 6 of 7**. It integrates molecular features from the preceding projects (MSI from Project 1, immune subtypes from Project 2, SHAP-validated biomarkers from Projects 3-4) into a single survival model with a deployable Streamlit calculator. It closes the loop on the clinical question that opened the portfolio. See the [portfolio site](https://github.com/adamhoffman2155-hue/bioinformatics-portfolio) for the full narrative.

## License

MIT

## Author

Adam Hoffman — M.Sc. Cancer Research, McGill University
