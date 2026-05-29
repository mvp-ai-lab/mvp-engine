#!/bin/bash -i
# Use the current working directory
#SBATCH -D ../slurm
# Reset environment for this job.
#Specify the partition
#SBATCH -p cpu
# Define job name
#SBATCH -J download
#SBATCH --mem-per-cpu=4GB

set -u

# TARGET_DIR="./data/Open-Bee/Honey-Data-1M"

# EXPECTED_TARS=2176

cd ..

source .venv/bin/activate

# This public preview dataset does not require an interactive Hugging Face login.
# If private datasets are used later, submit the job with HF_TOKEN in the environment.

# DATASETNAME='Open-Bee/Honey-Data-1M'

# count_tars() {
#   find "$TARGET_DIR" -maxdepth 2 -type f -name '*.tar' | wc -l
# }

# current_count=$(count_tars)
# attempt=0

# while [ "$current_count" -lt "$EXPECTED_TARS" ]; do
#   attempt=$((attempt + 1))
#   echo "Attempt $attempt: found $current_count/$EXPECTED_TARS .tar files. Running download..."

#   hf download "$DATASETNAME" \
#     --repo-type=dataset \
#     --local-dir "$TARGET_DIR" \
#     --max-workers 8

#   current_count=$(count_tars)
# done

export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=900

# hf download "$DATASETNAME" \
#     --repo-type=dataset \
#     --local-dir "$TARGET_DIR" \
#     --max-workers 8


# TARGET_DIR="./data/Open-Bee/Bee-Training-Data-Stage1"
# DATASETNAME='Open-Bee/Bee-Training-Data-Stage1'

# hf download "$DATASETNAME" \
#     --repo-type=dataset \
#     --local-dir "$TARGET_DIR" \
#     --max-workers 8

TARGET_DIR="${TARGET_DIR:-./data/LLaVA-OneVision-2-Data-viewer}"
DEMO_DIR="${DEMO_DIR:-./data/video_vlm/demo}"
DATASETNAME="mvp-lab/LLaVA-OneVision-2-Data"

hf download "$DATASETNAME" \
    --repo-type=dataset \
    --local-dir "$TARGET_DIR" \
    --max-workers 4 \
    viewer/spatial.parquet \
    viewer/caption_gt10min.parquet

python recipes/video_vlm/tools/convert_llava_onevision_viewer.py \
    --input-dir "$TARGET_DIR" \
    --output-dir "$DEMO_DIR" \
    --image-limit 2 \
    --video-limit 1

echo "Video VLM demo parquet written to: ${DEMO_DIR}/demo.parquet"
