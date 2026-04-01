#!/usr/bin/env python3
"""
GDC REST API client for TCGA-STAD data acquisition.

Fetches clinical data and somatic mutations from the GDC API.
Handles pagination and writes raw downloads to data/raw/.
"""

import requests
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GDC_API_BASE = "https://api.gdc.cancer.gov"
PAGE_SIZE = 200


def fetch_gdc_data(
    project_id: str = "TCGA-STAD",
    data_type: str = "clinical",
    output_dir: str = "data/raw"
) -> Dict[str, Any]:
    """
    Fetch data from GDC API with pagination.
    
    Args:
        project_id: GDC project ID (e.g., 'TCGA-STAD')
        data_type: 'clinical' or 'mutation'
        output_dir: Directory to write raw data
    
    Returns:
        Dictionary with metadata and file path
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    if data_type == "clinical":
        return _fetch_clinical_data(project_id, output_dir)
    elif data_type == "mutation":
        return _fetch_mutation_data(project_id, output_dir)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


def _fetch_clinical_data(project_id: str, output_dir: str) -> Dict[str, Any]:
    """Fetch clinical data (cases with survival info)."""
    logger.info(f"Fetching clinical data for {project_id}...")
    
    endpoint = f"{GDC_API_BASE}/cases"
    
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "primary_site", "value": ["Stomach"]}}
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "format": "JSON",
        "size": PAGE_SIZE,
        "from": 0,
        "expand": "diagnoses,diagnoses.treatments"
    }
    
    all_cases = []
    page = 0
    
    while True:
        params["from"] = page * PAGE_SIZE
        logger.info(f"  Fetching page {page + 1}...")
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            cases = data.get("data", {}).get("hits", [])
            if not cases:
                break
            
            all_cases.extend(cases)
            
            pagination = data.get("data", {}).get("pagination", {})
            total = pagination.get("total", 0)
            
            if len(all_cases) >= total:
                break
            
            page += 1
        except requests.RequestException as e:
            logger.error(f"Error fetching page {page}: {e}")
            break
    
    # Write to file
    output_file = os.path.join(output_dir, f"{project_id}_clinical.json")
    with open(output_file, "w") as f:
        json.dump(all_cases, f, indent=2)
    
    logger.info(f"Wrote {len(all_cases)} cases to {output_file}")
    
    return {
        "file": output_file,
        "count": len(all_cases),
        "data_type": "clinical"
    }


def _fetch_mutation_data(project_id: str, output_dir: str) -> Dict[str, Any]:
    """Fetch somatic mutation data (MAF files)."""
    logger.info(f"Fetching mutation data for {project_id}...")
    
    endpoint = f"{GDC_API_BASE}/files"
    
    filters = {
        "op": "and",
        "content": [
            {"op": "in", "content": {"field": "cases.project.project_id", "value": [project_id]}},
            {"op": "in", "content": {"field": "data_category", "value": ["Simple Nucleotide Variation"]}},
            {"op": "in", "content": {"field": "data_type", "value": ["Raw Simple Somatic Mutation"]}},
            {"op": "in", "content": {"field": "file_format", "value": ["MAF"]}}
        ]
    }
    
    params = {
        "filters": json.dumps(filters),
        "format": "JSON",
        "size": PAGE_SIZE,
        "from": 0,
        "expand": "cases"
    }
    
    all_files = []
    page = 0
    
    while True:
        params["from"] = page * PAGE_SIZE
        logger.info(f"  Fetching page {page + 1}...")
        
        try:
            response = requests.get(endpoint, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            files = data.get("data", {}).get("hits", [])
            if not files:
                break
            
            all_files.extend(files)
            
            pagination = data.get("data", {}).get("pagination", {})
            total = pagination.get("total", 0)
            
            if len(all_files) >= total:
                break
            
            page += 1
        except requests.RequestException as e:
            logger.error(f"Error fetching page {page}: {e}")
            break
    
    # Write to file
    output_file = os.path.join(output_dir, f"{project_id}_mutations_manifest.json")
    with open(output_file, "w") as f:
        json.dump(all_files, f, indent=2)
    
    logger.info(f"Found {len(all_files)} mutation files for {output_file}")
    
    return {
        "file": output_file,
        "count": len(all_files),
        "data_type": "mutation"
    }


if __name__ == "__main__":
    import sys
    
    data_type = sys.argv[1] if len(sys.argv) > 1 else "clinical"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/raw"
    
    result = fetch_gdc_data(data_type=data_type, output_dir=output_dir)
    print(json.dumps(result, indent=2))
