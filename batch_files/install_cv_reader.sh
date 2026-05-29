#!/bin/bash -l
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"
source .venv/bin/activate

ONEVISION_ENCODER_ROOT="${ONEVISION_ENCODER_ROOT:-${REPO_ROOT}/../OneVision-Encoder}"
CV_READER_DIR="${ONEVISION_ENCODER_ROOT}/llava_next/Compressed_Video_Reader"

echo "Installing cv_reader from: ${CV_READER_DIR}"
cd "${CV_READER_DIR}"
bash install.sh

python -c "from cv_reader import api; print('cv_reader api ok:', api)"
