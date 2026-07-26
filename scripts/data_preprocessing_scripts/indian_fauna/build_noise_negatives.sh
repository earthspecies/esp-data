#!/usr/bin/env bash
#SBATCH --job-name=ien-noise-neg
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Extract reliable-negative ("no animal") noise clips from IEN Background and
# upload them to the training noise bank:
#   gs://foundation-model-data/audio_32k/noise/ien_background_negatives_10s/
# Reads the Background 32 kHz mirrors from esp-data-ingestion, filters to files
# whose every annotation category is abiotic/anthropogenic, tiles into 10 s clips.
#   sbatch scripts/data_preprocessing_scripts/indian_fauna/build_noise_negatives.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

SRC_GCS="gs://esp-data-ingestion/indian-fauna/v0.1.0/background/audio_32k"
DST_GCS="gs://foundation-model-data/audio_32k/noise/ien_background_negatives_10s"
STAGE="${HOME}/esp-data-staging/indian_fauna"
AUDIO="${STAGE}/noise_neg_src"     # local copy of the 32k mirrors
OUT="${STAGE}/noise_neg_clips"     # 10s clips to upload
ANNO="${STAGE}/background_annotations.csv"
SCRIPT="scripts/data_preprocessing_scripts/indian_fauna/build_noise_negatives.py"

export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil + gcsfs
echo "Node: $(hostname)"
echo "ok" | gsutil cp - "${DST_GCS}/.auth_probe" && gsutil rm "${DST_GCS}/.auth_probe" && echo "auth OK"

# 1. Pull the Background 32k mirrors locally (authoritative source).
mkdir -p "${AUDIO}"
gsutil -m -q rsync -r "${SRC_GCS}" "${AUDIO}"
echo "source mirrors: $(find "${AUDIO}" -name '*.wav' | wc -l)"

# 2. Filter to reliable negatives + tile into 10s clips.
rm -rf "${OUT}"; mkdir -p "${OUT}"
uv run python "${SCRIPT}" build \
    --anno-csv "${ANNO}" --audio-root "${AUDIO}" --out-dir "${OUT}" \
    --workers "${SLURM_CPUS_PER_TASK:-16}"

# 3. Upload clips + manifest to the noise dir.
gsutil -m -q rsync -r "${OUT}" "${DST_GCS}"
n_gcs="$(gsutil ls "${DST_GCS}/**" 2>/dev/null | grep -c '\.wav' || true)"
echo "clips on GCS: ${n_gcs}"
echo "Done."
