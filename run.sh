#!/usr/bin/env bash

# Enforce strict error boundaries so execution fails loudly if any script errors out
set -euo pipefail

# Safely extract positional target parameters with built-in relative path fallbacks
DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./models/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

echo "========================================================="
echo "  Executing AIgnition 3.0 Automated Evaluation Pipeline  "
echo "========================================================="

# Step 1: Execute features generation
echo "[*] Step 1 of 2: Compiling features from raw directory metrics..."
python3 src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out data/features.parquet

# Step 2: Run inference execution
echo "[*] Step 2 of 2: Initializing inference engine & multi-horizon ranges..."
python3 src/predict.py \
    --features data/features.parquet \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "========================================================="
echo "  Pipeline execution verified. Outputs written to disk.  "
echo "========================================================="