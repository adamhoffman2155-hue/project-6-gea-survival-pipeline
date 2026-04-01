#!/usr/bin/env python3
"""
Generate realistic synthetic TCGA-STAD data for portfolio demonstration.

Creates clinical and mutation data that mirrors real TCGA structure
but is fully synthetic for reproducibility and speed.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(42)


def generate_synthetic_clinical_data(n_cases: int = 250, output_file: str = "data/raw/TCGA-STAD_clinical.json"):
    """Generate synthetic clinical data."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating {n_cases} synthetic clinical cases...")
    
    cases = []
    for i in range(n_cases):
        # Diagnoses with treatment
        diagnoses = []
        for j in range(np.random.randint(1, 3)):
            treatments = []
            if np.random.random() > 0.3:  # 70% received treatment
                treatments.append({
                    "treatment_type": np.random.choice(["Chemotherapy", "Radiation", "Surgery"]),
                    "days_to_treatment_start": np.random.randint(0, 365)
                })
            
            diagnoses.append({
                "diagnosis_id": f"diagnosis_{i}_{j}",
                "age_at_diagnosis": np.random.randint(40, 85),
                "primary_diagnosis": "Adenocarcinoma",
                "tissue_or_organ_of_origin": "Stomach",
                "tumor_stage": np.random.choice(["Stage I", "Stage II", "Stage III", "Stage IV"]),
                "treatments": treatments
            })
        
        # Survival data
        os_days = np.random.randint(30, 2000)
        os_event = 1 if np.random.random() > 0.4 else 0  # 60% event rate
        
        case = {
            "case_id": f"TCGA-STAD-{i:04d}",
            "primary_site": "Stomach",
            "disease_type": "Adenocarcinoma",
            "diagnoses": diagnoses,
            "demographic": {
                "gender": np.random.choice(["male", "female"]),
                "race": np.random.choice(["white", "black or african american", "asian"])
            },
            "follow_up": {
                "days_to_last_follow_up": os_days,
                "vital_status": "Dead" if os_event else "Alive"
            }
        }
        cases.append(case)
    
    with open(output_file, "w") as f:
        json.dump(cases, f, indent=2)
    
    logger.info(f"Wrote {len(cases)} synthetic clinical cases to {output_file}")
    return output_file


def generate_synthetic_mutation_data(n_cases: int = 250, output_file: str = "data/raw/TCGA-STAD_mutations.csv"):
    """Generate synthetic mutation data."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating synthetic mutation data for {n_cases} cases...")
    
    ddr_genes = ["BRCA1", "BRCA2", "ATM", "ATR", "PALB2", "RAD51", "MLH1", "MSH2", "MSH6", "POLE"]
    all_genes = ddr_genes + ["TP53", "KRAS", "CDH1", "ARID1A", "MYC", "EGFR", "MET"]
    
    mutations = []
    for case_idx in range(n_cases):
        case_id = f"TCGA-STAD-{case_idx:04d}"
        
        # Total mutations per case (TMB proxy)
        n_mutations = np.random.randint(20, 200)
        
        for mut_idx in range(n_mutations):
            gene = np.random.choice(all_genes)
            
            mutation = {
                "case_id": case_id,
                "gene_symbol": gene,
                "variant_classification": np.random.choice(["Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins", "Silent"]),
                "chromosome": str(np.random.randint(1, 23)),
                "start_position": np.random.randint(1000000, 100000000),
                "reference_allele": np.random.choice(["A", "T", "G", "C"]),
                "tumor_seq_allele1": np.random.choice(["A", "T", "G", "C"]),
                "tumor_seq_allele2": np.random.choice(["A", "T", "G", "C"]),
                "tumor_f": np.random.uniform(0.1, 1.0),
                "is_pathogenic": 1 if np.random.random() > 0.7 else 0
            }
            mutations.append(mutation)
    
    df_mutations = pd.DataFrame(mutations)
    df_mutations.to_csv(output_file, index=False)
    
    logger.info(f"Wrote {len(mutations)} synthetic mutations to {output_file}")
    return output_file


def generate_synthetic_immune_subtypes(n_cases: int = 250, output_file: str = "data/raw/TCGA-STAD_immune_subtypes.csv"):
    """Generate synthetic immune subtype assignments."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating synthetic immune subtypes for {n_cases} cases...")
    
    immune_subtypes = ["C1: Wound Healing", "C2: IFN-gamma Dominant", "C3: Inflammatory", "C4: Lymphoid Depleted", "C5: Immunologically Quiet"]
    
    data = []
    for i in range(n_cases):
        data.append({
            "case_id": f"TCGA-STAD-{i:04d}",
            "immune_subtype": np.random.choice(immune_subtypes),
            "immune_score": np.random.uniform(0, 1)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Wrote {len(data)} immune subtype assignments to {output_file}")
    return output_file


def generate_synthetic_msi_status(n_cases: int = 250, output_file: str = "data/raw/TCGA-STAD_msi_status.csv"):
    """Generate synthetic MSI status assignments."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating synthetic MSI status for {n_cases} cases...")
    
    data = []
    for i in range(n_cases):
        msi_status = np.random.choice(["MSI-H", "MSS"], p=[0.3, 0.7])  # ~30% MSI-H in GEA
        data.append({
            "case_id": f"TCGA-STAD-{i:04d}",
            "msi_status": msi_status,
            "msi_score": np.random.uniform(0, 1)
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)
    
    logger.info(f"Wrote {len(data)} MSI status assignments to {output_file}")
    return output_file


if __name__ == "__main__":
    generate_synthetic_clinical_data(n_cases=250)
    generate_synthetic_mutation_data(n_cases=250)
    generate_synthetic_immune_subtypes(n_cases=250)
    generate_synthetic_msi_status(n_cases=250)
    logger.info("Synthetic data generation complete!")
