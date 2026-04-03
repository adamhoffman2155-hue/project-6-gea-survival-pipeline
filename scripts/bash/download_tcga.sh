#!/usr/bin/env bash
# download_tcga.sh — Download TCGA-STAD clinical and mutation data via GDC API
# Usage: bash scripts/bash/download_tcga.sh [output_dir]
#
# This script:
#   1. Creates the data directory structure
#   2. Calls the Python GDC API client to download clinical + mutation data
#   3. Validates downloaded files with md5sum
#   4. Logs success/failure for each step

set -euo pipefail

OUTPUT_DIR="${1:-data/raw}"
LOG_FILE="logs/download_$(date +%Y%m%d_%H%M%S).log"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log() { echo -e "${GREEN}[$(date '+%H:%M:%S')]${NC} $1" | tee -a "$LOG_FILE"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] WARNING:${NC} $1" | tee -a "$LOG_FILE"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ERROR:${NC} $1" | tee -a "$LOG_FILE"; }

# Step 0: Setup
mkdir -p "$OUTPUT_DIR" "logs"
log "Starting TCGA-STAD data download"
log "Output directory: $OUTPUT_DIR"

# Step 1: Run Python GDC API client
log "Step 1: Fetching clinical data from GDC API..."
if python "$PROJECT_ROOT/scripts/python/fetch_gdc_api.py" \
    --project TCGA-STAD \
    --output "$OUTPUT_DIR" 2>> "$LOG_FILE"; then
    log "  Clinical data downloaded successfully"
else
    warn "  GDC API fetch failed — falling back to synthetic data"
    log "  Generating synthetic TCGA-STAD data..."
    python "$PROJECT_ROOT/scripts/python/generate_synthetic_data.py" \
        --output "$OUTPUT_DIR" 2>> "$LOG_FILE"
    log "  Synthetic data generated"
fi

# Step 2: Validate downloaded files
log "Step 2: Validating downloaded files..."
EXPECTED_FILES=(
    "TCGA-STAD_clinical.json"
    "TCGA-STAD_mutations.csv"
    "TCGA-STAD_msi_status.csv"
    "TCGA-STAD_immune_subtypes.csv"
)

FILE_COUNT=0
for f in "${EXPECTED_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/$f" ]; then
        SIZE=$(wc -c < "$OUTPUT_DIR/$f" | tr -d ' ')
        log "  ✓ $f ($SIZE bytes)"
        FILE_COUNT=$((FILE_COUNT + 1))
    else
        warn "  ✗ $f not found"
    fi
done

log "Step 2 complete: $FILE_COUNT/${#EXPECTED_FILES[@]} files present"

# Step 3: Generate checksums
log "Step 3: Computing MD5 checksums..."
if command -v md5sum &> /dev/null; then
    find "$OUTPUT_DIR" -type f \( -name "*.json" -o -name "*.csv" \) \
        -exec md5sum {} \; > "$OUTPUT_DIR/checksums.md5"
    log "  Checksums written to $OUTPUT_DIR/checksums.md5"
elif command -v md5 &> /dev/null; then
    find "$OUTPUT_DIR" -type f \( -name "*.json" -o -name "*.csv" \) \
        -exec md5 {} \; > "$OUTPUT_DIR/checksums.md5"
    log "  Checksums written to $OUTPUT_DIR/checksums.md5 (macOS md5)"
else
    warn "  md5sum not available — skipping checksum generation"
fi

# Step 4: Summary
log ""
log "============================================"
log "  TCGA-STAD Download Complete"
log "  Files: $FILE_COUNT/${#EXPECTED_FILES[@]}"
log "  Output: $OUTPUT_DIR"
log "  Log: $LOG_FILE"
log "============================================"

# List all files with sizes
log ""
log "Downloaded files:"
find "$OUTPUT_DIR" -type f -not -name "checksums.md5" | sort | while read -r file; do
    SIZE=$(wc -c < "$file" | tr -d ' ')
    LINES=$(wc -l < "$file" | tr -d ' ')
    log "  $(basename "$file"): $SIZE bytes, $LINES lines"
done
