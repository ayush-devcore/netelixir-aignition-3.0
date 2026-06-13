#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${1:-./data}"
MODEL_PATH="${2:-./pickle/model.pkl}"
OUTPUT_PATH="${3:-./output/predictions.csv}"

echo "========================================================="
echo "  Executing AIgnition 3.0 Automated Evaluation Pipeline  "
echo "========================================================="

echo "[*] Step 1 of 2: Compiling features from raw directory metrics..."
python3 src/generate_features.py \
    --data-dir "$DATA_DIR" \
    --out features.parquet

echo "[*] Step 2 of 2: Initializing inference engine & multi-horizon ranges..."
python3 src/predict.py \
    --features features.parquet \
    --model "$MODEL_PATH" \
    --output "$OUTPUT_PATH"

echo "========================================================="
echo "  Pipeline execution verified. Outputs written to disk.  "
echo "========================================================="