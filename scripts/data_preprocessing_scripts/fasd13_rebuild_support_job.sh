#!/usr/bin/env bash
#SBATCH --job-name=fasd13-support
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# Rebuild the FASD13 few-shot support clips with the support region correctly
# bounded by the Nth event, and re-upload.
#
# Reads slices straight from the GCS mirrors, so the ~50 GB of audio does not
# need re-staging. Only the annotation CSVs (small) are pulled down.
set -euo pipefail
cd "${HOME}/alp-data"

GCS="gs://esp-data-274503/fasd13/v0.1.0/raw"
STAGE="${HOME}/alp-data-staging/fasd13_support"
SCRIPT="scripts/data_preprocessing_scripts/fasd13.py"

export CLOUDSDK_CONFIG="$(mktemp -d)"
export GCE_METADATA_MTLS_MODE=none

rm -rf "${STAGE}"; mkdir -p "${STAGE}/source" "${STAGE}/manifests" "${STAGE}/out"
echo "=== fetch annotations + manifests ==="
gsutil -m -q cp -r "${GCS}/source/*" "${STAGE}/source/"
gsutil -q cp "${GCS}/fasd13_all.csv" "${STAGE}/manifests/"

echo "=== cut support clips (bounded by the 5th event) ==="
uv run python "${SCRIPT}" support \
    --root "${STAGE}/source" --mirrors "${GCS}" --out-root "${STAGE}/out" \
    --manifest-dir "${STAGE}/manifests" --workers "${SLURM_CPUS_PER_TASK:-12}"

echo "=== upload ==="
gsutil -m -q rsync -d -r "${STAGE}/out/support_16k" "${GCS}/support_16k"
gsutil -m -q rsync -d -r "${STAGE}/out/support_32k" "${GCS}/support_32k"
gsutil -q cp "${STAGE}/manifests/fasd13_support.csv" "${GCS}/"
echo "Done."
