#!/bin/bash -l
#SBATCH -p gpu
#SBATCH --gres=gpu:h200:1
#SBATCH -J video_vlm_demo
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:59:59
#SBATCH --output=slurmlog/slurm-%j.out
#SBATCH --error=slurmlog/slurm-%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

CONFIG="${CONFIG:-./recipes/video_vlm/configs/hevc_smoke.yaml}"
MASTER_PORT="${MASTER_PORT:-29501}"
ONEVISION_ENCODER_ROOT="${ONEVISION_ENCODER_ROOT:-${REPO_ROOT}/../OneVision-Encoder}"

export OMP_NUM_THREADS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${ONEVISION_ENCODER_ROOT}:${PYTHONPATH:-}"

echo "Launching video_vlm demo"
echo "Config: ${CONFIG}"
echo "ONEVISION_ENCODER_ROOT: ${ONEVISION_ENCODER_ROOT}"

torchrun \
  --nproc_per_node=1 \
  --master_port="${MASTER_PORT}" \
  -m mvp_engine.launch \
  --config "${CONFIG}"
