#!/bin/bash
# Convert ONE shard of a too-big dataset (e.g. dolma3) into shard_<I>/text.lance
# under data/text/lance/<DATASET>/. Run as a Slurm ARRAY job — one task per shard:
#
#   sbatch --array=0-31%16 recipes/qwen3_pt/tools/convert_shard.sbatch \
#          allenai_dolma3_mix-6T-1025-7B 32 143855 text
#
#   args: <DATASET> <NUM_SHARDS> [EXPECTED_FILES] [TEXT_COLUMN]
#   shard index = $SLURM_ARRAY_TASK_ID ; %16 caps concurrency to spare JuiceFS IO.
#
# After ALL shards finish, build the dataset meta.json (with strict cover check):
#   python recipes/qwen3_pt/tools/finalize_shards.py --dataset-dir <lance>/<DATASET>
#
# Safety (same contract as convert_dataset.sbatch): unique staging, verify on
# staging before an atomic promote, refuse to overwrite, trap-clean on failure,
# and refuse to add shards once the dataset meta.json is already published.
#
#SBATCH --job-name=lance-shard
#SBATCH --partition=cpu
#SBATCH --account=proj_camera
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=72:00:00
#SBATCH --output=/home/b84417810/lance_shard_%x_%A_%a.log

set -euo pipefail

DATASET="${1:?Usage: sbatch --array=0-N%K convert_shard.sbatch <DATASET> <NUM_SHARDS> [EXPECTED_FILES] [TEXT_COLUMN]}"
NUM_SHARDS="${2:?need NUM_SHARDS}"
EXPECTED_FILES="${3:-}"
TEXT_COLUMN="${4:-}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?must be submitted as an array job (--array=0-$((NUM_SHARDS-1))%K)}"

[[ "$DATASET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid dataset name: '$DATASET'"; exit 1; }
[ "$SHARD_INDEX" -lt "$NUM_SHARDS" ] || { echo "ERROR: shard_index $SHARD_INDEX >= num_shards $NUM_SHARDS"; exit 1; }
PAD=$(printf "%02d" "$SHARD_INDEX")

DATA_ROOT=/mnt/data-alpha-sg-01/team-camera/shared/mvp-engine/data
INPUT="$DATA_ROOT/text/$DATASET"
DATASET_DIR="$DATA_ROOT/text/lance/$DATASET"
JOB="${SLURM_JOB_ID:-manual}"
STAGING="$DATA_ROOT/text/lance/.staging/${DATASET}.shard_${PAD}.${JOB}"
FINAL="$DATASET_DIR/shard_${PAD}"

echo "[$(date -u)] host=$(hostname) dataset=$DATASET shard=$SHARD_INDEX/$NUM_SHARDS job=$JOB"
[ -d "$INPUT" ] || { echo "ERROR: input not found: $INPUT"; exit 1; }
# Publish gate: once the dataset meta.json exists, the split is sealed — do not add shards.
[ -e "$DATASET_DIR/meta.json" ] && { echo "ERROR: dataset already finalized ($DATASET_DIR/meta.json); refusing"; exit 1; }
[ -e "$FINAL" ] && { echo "ERROR: shard already exists, refusing to overwrite: $FINAL"; exit 1; }

source /home/b84417810/miniconda3/etc/profile.d/conda.sh
conda activate mvp_engine
cd /home/b84417810/mvp_engine

trap 'echo "[$(date -u)] FAILED (rc=$?), cleaning staging $STAGING"; rm -rf "$STAGING"' ERR
rm -rf "$STAGING"
mkdir -p "$STAGING"

EXTRA=()
[ -n "$EXPECTED_FILES" ] && EXTRA+=(--expected-files "$EXPECTED_FILES")
[ -n "$TEXT_COLUMN" ]    && EXTRA+=(--text-column "$TEXT_COLUMN")

echo "[$(date -u)] converting shard $SHARD_INDEX of $NUM_SHARDS ..."
python recipes/qwen3_pt/tools/convert_parquet_to_lance.py \
    --input  "$INPUT" \
    --output "$STAGING/text.lance" \
    --num-shards "$NUM_SHARDS" --shard-index "$SHARD_INDEX" \
    "${EXTRA[@]}"

echo "[$(date -u)] === verifying STAGING (before promote) ==="
python - "$STAGING" <<'PY'
import json
import os
import sys

import lance

staging = sys.argv[1]
dataset = lance.dataset(os.path.join(staging, "text.lance"))
rows = dataset.count_rows()
assert rows > 0, "verify: shard lance is empty"
schema_names = {field.name for field in dataset.schema}
assert {"id", "info", "data", "tags"} <= schema_names, f"verify: schema missing columns: {sorted(schema_names)}"
row = dataset.head(1).to_pylist()[0]
assert isinstance(row["data"], str) and row["data"].strip(), "verify: first row has empty/non-str data"
manifest = json.load(open(os.path.join(staging, "shard_manifest.json"), encoding="utf-8"))
assert "files" in manifest and manifest["selected_files"] == len(manifest["files"]), "verify: bad shard_manifest"
print(f"verify OK: rows={rows:,}  id0={row['id']!r}  selected_files={manifest['selected_files']}")
PY

echo "[$(date -u)] verify passed; promoting shard -> $FINAL ..."
mkdir -p "$DATASET_DIR"
[ -e "$DATASET_DIR/meta.json" ] && { echo "ERROR: dataset finalized during conversion; refusing"; exit 1; }
[ -e "$FINAL" ] && { echo "ERROR: shard appeared during conversion (concurrent job?): $FINAL"; exit 1; }
mv -T "$STAGING" "$FINAL"
trap - ERR
echo "[$(date -u)] === DONE shard $SHARD_INDEX: $FINAL ==="
