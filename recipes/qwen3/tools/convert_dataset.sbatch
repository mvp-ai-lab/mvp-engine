#!/bin/bash
# Convert one raw dataset (parquet / arrow IPC / jsonl[.gz|.zst]) under
# data/text/<name> into an HKdata-style text.lance under data/text/lance/<name>/
# (Option A: text.lance + meta.json). Input format is auto-detected.
# (Output lives inside the world-writable data/text/ because we are not in the
#  slurm-lrc-users group that owns data/; move to a proper sibling once granted.)
#
# Safety (this is a long, expensive run — do not corrupt data):
#   * unique per-job staging dir; cleaned on any failure (trap)
#   * VERIFY on staging BEFORE promoting (count/schema/sample/meta)
#   * refuse to overwrite an existing FINAL (re-convert = remove it by hand first)
#   * promote = single rename into a path that did not exist (no rm -rf FINAL window)
#   * optional --expected-files / --expected-rows so a silent under-scan fails loudly
#
# Usage (CPU node, never login):
#   sbatch recipes/qwen3/tools/convert_dataset.sbatch <dataset_dirname> [expected_files] [expected_rows] [text_column]
#   e.g.  sbatch recipes/qwen3/tools/convert_dataset.sbatch bigcode_starcoderdata 863 206642239
#         sbatch recipes/qwen3/tools/convert_dataset.sbatch institutional_institutional-books-1.0 9831 983004 text_by_page_gen
#
#SBATCH --job-name=lance-convert
#SBATCH --partition=cpu
#SBATCH --account=proj_camera
#SBATCH --qos=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/b84417810/lance_convert_%x_%j.log

set -euo pipefail

DATASET="${1:?Usage: sbatch convert_dataset.sbatch <dataset_dirname> [expected_files] [expected_rows] [text_column]}"
EXPECTED_FILES="${2:-}"
EXPECTED_ROWS="${3:-}"
TEXT_COLUMN="${4:-}"

# Reject anything but a plain directory name (no /, .., spaces) to keep paths safe.
[[ "$DATASET" =~ ^[A-Za-z0-9._-]+$ ]] || { echo "ERROR: invalid dataset name: '$DATASET'"; exit 1; }

DATA_ROOT=/mnt/data-alpha-sg-01/team-camera/shared/mvp-engine/data
INPUT="$DATA_ROOT/text/$DATASET"
LANCE_ROOT="$DATA_ROOT/text/lance"
JOB="${SLURM_JOB_ID:-manual}"
STAGING="$LANCE_ROOT/.staging/${DATASET}.${JOB}"
FINAL="$LANCE_ROOT/$DATASET"

echo "[$(date -u)] host=$(hostname) dataset=$DATASET job=$JOB"
[ -d "$INPUT" ] || { echo "ERROR: input not found: $INPUT"; exit 1; }
[ -e "$FINAL" ] && { echo "ERROR: FINAL already exists, refusing to overwrite: $FINAL (remove it by hand to re-convert)"; exit 1; }

source /home/b84417810/miniconda3/etc/profile.d/conda.sh
conda activate mvp_engine
cd /home/b84417810/mvp_engine

# Unique staging; remove it on ANY failure so we never leave junk behind.
trap 'echo "[$(date -u)] FAILED (rc=$?), cleaning staging $STAGING"; rm -rf "$STAGING"' ERR
rm -rf "$STAGING"
mkdir -p "$STAGING"

# Optional args (only added when provided).
EXTRA=()
[ -n "$EXPECTED_FILES" ] && EXTRA+=(--expected-files "$EXPECTED_FILES")
[ -n "$EXPECTED_ROWS" ]  && EXTRA+=(--expected-rows "$EXPECTED_ROWS")
[ -n "$TEXT_COLUMN" ]    && EXTRA+=(--text-column "$TEXT_COLUMN")

echo "[$(date -u)] converting (streaming; text column -> data; row-index ids) ..."
python recipes/qwen3/tools/convert_parquet_to_lance.py \
    --input  "$INPUT" \
    --output "$STAGING/text.lance" \
    "${EXTRA[@]}"

echo "[$(date -u)] === verifying STAGING (before promote) ==="
python - "$STAGING" <<'PY'
import json
import os
import sys

import lance

staging = sys.argv[1]
lance_path = os.path.join(staging, "text.lance")
dataset = lance.dataset(lance_path)

rows = dataset.count_rows()
assert rows > 0, "verify: lance table is empty"
schema_names = {field.name for field in dataset.schema}
assert {"id", "info", "data", "tags"} <= schema_names, f"verify: schema missing columns: {sorted(schema_names)}"

row = dataset.head(1).to_pylist()[0]
assert isinstance(row["data"], str) and row["data"].strip(), "verify: first row has empty/non-str data"

meta = json.load(open(os.path.join(staging, "meta.json"), encoding="utf-8"))
assert meta.get("shards") == ["text.lance"], f"verify: bad meta.json: {meta}"

print(f"verify OK: rows={rows:,}  schema={[f'{f.name}:{f.type}' for f in dataset.schema]}")
print("data[:120]:", repr(row["data"][:120]))
PY

echo "[$(date -u)] verify passed; promoting staging -> final (single rename) ..."
mkdir -p "$(dirname "$FINAL")"
# Re-check right before promote: a concurrent job for the same dataset may have
# finished first; -T also stops mv from moving staging INTO an existing dir.
[ -e "$FINAL" ] && { echo "ERROR: FINAL appeared during conversion (concurrent job?): $FINAL"; exit 1; }
mv -T "$STAGING" "$FINAL"
trap - ERR
echo "[$(date -u)] === DONE: $FINAL ==="
