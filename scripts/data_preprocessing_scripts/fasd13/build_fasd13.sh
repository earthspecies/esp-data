#!/usr/bin/env bash
#SBATCH --job-name=fasd13-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build FASD13 (Zenodo 15843741) into gs://esp-data-ingestion/fasd13/v0.1.0/.
#
# Downloads the 13 sub-dataset zips (~16 GB) + LICENSE/appendix, verifies md5,
# extracts, writes 16 kHz + 32 kHz mono mirrors of all 109 recordings (~143 h),
# builds the WABAD-shaped manifests, cuts the deterministic 5 few-shot support
# clips per recording, and uploads everything.
#
#   sbatch scripts/data_preprocessing_scripts/fasd13/build_fasd13.sh
#
# Resampling is streamed in 30 s blocks, so memory is flat regardless of recording
# length (HG is 8 h of stereo per file). Needs ~110 GB free on NFS; set
# KEEP_STAGE=1 to retain the working tree after upload.
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

GCS="gs://esp-data-ingestion/fasd13/v0.1.0"
STAGE="${HOME}/esp-data-staging/fasd13"
DL="${STAGE}/zenodo"
WORK="${STAGE}/work"
MIRRORS="${STAGE}/mirrors"
MANIFESTS="${STAGE}/manifests"
SCRIPT="scripts/data_preprocessing_scripts/fasd13/build_fasd13.py"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"

# Ignore stale user ADC; use the node's attached SA (gsutil + gcsfs).
export CLOUDSDK_CONFIG="$(mktemp -d)"
export GCE_METADATA_MTLS_MODE=none
echo "Node: $(hostname)  stage=${STAGE}  workers=${WORKERS}"
echo "ok" | gsutil cp - "${GCS}/.auth_probe" && gsutil rm "${GCS}/.auth_probe" && echo "auth probe OK"

# 1. Download + verify + extract (resumable; cached files are checksum-checked).
echo "=== 1. download ==="
uv run python "${SCRIPT}" download --stage-dir "${DL}" --work "${WORK}"

# 2. 16k + 32k mono mirrors + durations.csv.
echo "=== 2. resample ==="
uv run python "${SCRIPT}" resample --root "${WORK}" --out-root "${MIRRORS}" --workers "${WORKERS}"

# 3. Manifests (per sub-dataset + fasd13_all.csv).
echo "=== 3. manifests ==="
uv run python "${SCRIPT}" manifests \
    --root "${WORK}" --durations-csv "${MIRRORS}/durations.csv" --out-dir "${MANIFESTS}"

# 4. Few-shot support clips (5 per recording, both rates) + fasd13_support.csv.
echo "=== 4. support clips ==="
uv run python "${SCRIPT}" support \
    --root "${WORK}" --mirrors "${MIRRORS}" --out-root "${MIRRORS}" \
    --manifest-dir "${MANIFESTS}" --workers "${WORKERS}"

# 5. Upload. Raw originals + annotations are kept for provenance.
echo "=== 5. upload ==="
gsutil -m -q rsync -r "${MIRRORS}/audio_16k"   "${GCS}/audio_16k"
gsutil -m -q rsync -r "${MIRRORS}/audio_32k"   "${GCS}/audio_32k"
gsutil -m -q rsync -r "${MIRRORS}/support_16k" "${GCS}/support_16k"
gsutil -m -q rsync -r "${MIRRORS}/support_32k" "${GCS}/support_32k"
gsutil -m -q rsync -r -x '.*\.wav$' "${WORK}" "${GCS}/raw"
gsutil -m -q cp "${MANIFESTS}"/*.csv "${GCS}/"
gsutil -q cp "${DL}/LICENSE.txt" "${DL}/fasd13_appendix.pdf" "${GCS}/"

echo "final listing:"; gsutil ls "${GCS}/"
if [ "${KEEP_STAGE:-0}" != "1" ]; then
  echo "removing working tree (KEEP_STAGE=1 to retain)"
  rm -rf "${WORK}" "${MIRRORS}"
fi
echo "Done."
