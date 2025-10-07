#!/bin/bash
# --------------------------------------------
# run_graph_builder.sh
# Run graph_builder.py for one train directory
# Usage: ./run_graph_builder.sh 00
# --------------------------------------------

# Load the micromamba shell hook for bash
eval "$(micromamba shell hook --shell bash)"
# Activate the environment
micromamba activate pytorch

set -euo pipefail

IDX=$(printf "%02d" "${1:-0}")
IN_DIR="JetClass_Zenodo/train_${IDX}"
OUT_DIR="outputs"
OUT_FILE="${OUT_DIR}/train_${IDX}.h5"
TMP_FILE="${OUT_FILE}.tmp"

mkdir -p "$OUT_DIR"

# Skip if already exists
if [[ -f "$OUT_FILE" ]]; then
  echo "[SKIP] $OUT_FILE already exists."
  exit 0
fi

# Ensure ROOT files exist
shopt -s nullglob
ROOTS=("${IN_DIR}"/*.root)
shopt -u nullglob
if (( ${#ROOTS[@]} == 0 )); then
  echo "[WARN] No ROOT files found in ${IN_DIR}"
  exit 0
fi

echo "[INFO] Processing ${#ROOTS[@]} ROOT files from ${IN_DIR}"
echo "[INFO] Writing ${OUT_FILE}"

# Run your graph builder
ipython graph_builder.py -- \
  --roots "${IN_DIR}"/*.root \
  --out "$TMP_FILE" \
  --chunk_size 16384 \
  --compression lzf \
  --max_particles 100

# Atomic rename
mv -f "$TMP_FILE" "$OUT_FILE"
echo "[DONE] ${OUT_FILE}"

