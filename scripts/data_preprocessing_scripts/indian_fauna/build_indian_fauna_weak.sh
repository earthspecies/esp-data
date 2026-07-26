#!/usr/bin/env bash
#SBATCH --job-name=ien-weak-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the IEN Type-B WEAK (Zenodo 18927866) and Background (18928201) datasets
# into gs://esp-data-ingestion/indian-fauna/v0.1.0/{typeB,background}/. Uses the
# by-state zips + CSVs already on NFS.
#   sbatch scripts/data_preprocessing_scripts/indian_fauna/build_indian_fauna_weak.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

STAGE="${HOME}/esp-data-staging/indian_fauna"
SCRIPT="scripts/data_preprocessing_scripts/indian_fauna/build_indian_fauna_weak.py"
GCS_B="gs://esp-data-ingestion/indian-fauna/v0.1.0/typeB"
GCS_BG="gs://esp-data-ingestion/indian-fauna/v0.1.0/background"

export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil + gcsfs
echo "Node: $(hostname)"
echo "ok" | gsutil cp - "${GCS_B}/.auth_probe" && gsutil rm "${GCS_B}/.auth_probe" && echo "auth OK"

extract() {  # prefix out_work
  local prefix="$1" work="$2"; mkdir -p "${work}"
  for z in "${STAGE}"/${prefix}_*.zip; do
    [ -e "${z}" ] || continue
    local state; state="$(basename "${z}" .zip | sed "s/^${prefix}_//")"
    if [ ! -d "${work}/${state}" ]; then
      echo "extracting ${state} ..."; mkdir -p "${work}/${state}"; unzip -o -q "${z}" -d "${work}/${state}/"
    fi
  done
  echo "  recordings: $(find "${work}" -type f | wc -l)"
}

# ── Type B ────────────────────────────────────────────────────────
WORK_B="${STAGE}/work_typeB"; MIR_B="${STAGE}/mirrors_typeB"
extract audio_typeB "${WORK_B}"
rm -rf "${MIR_B}"; mkdir -p "${MIR_B}"
uv run python "${SCRIPT}" resample --work "${WORK_B}" --out-root "${MIR_B}" --workers "${SLURM_CPUS_PER_TASK:-16}"
uv run python "${SCRIPT}" manifests --kind typeB \
    --meta-csv "${STAGE}/type_B_metadata.csv" \
    --durations-csv "${MIR_B}/durations.csv" \
    --out-dir "${MIR_B}/manifests" --name indian_fauna_weak
gsutil -m -q rsync -r "${MIR_B}/audio_16k" "${GCS_B}/audio_16k"
gsutil -m -q rsync -r "${MIR_B}/audio_32k" "${GCS_B}/audio_32k"
gsutil -m -q cp "${MIR_B}/manifests"/*.csv "${GCS_B}/"

# ── Background ────────────────────────────────────────────────────
WORK_BG="${STAGE}/work_background"; MIR_BG="${STAGE}/mirrors_background"
extract audio_background "${WORK_BG}"
rm -rf "${MIR_BG}"; mkdir -p "${MIR_BG}"
uv run python "${SCRIPT}" resample --work "${WORK_BG}" --out-root "${MIR_BG}" --workers "${SLURM_CPUS_PER_TASK:-16}"
uv run python "${SCRIPT}" manifests --kind background \
    --meta-csv "${STAGE}/background_metadata.csv" \
    --anno-csv "${STAGE}/background_annotations.csv" \
    --durations-csv "${MIR_BG}/durations.csv" \
    --out-dir "${MIR_BG}/manifests" --name indian_fauna_background
gsutil -m -q rsync -r "${MIR_BG}/audio_16k" "${GCS_BG}/audio_16k"
gsutil -m -q rsync -r "${MIR_BG}/audio_32k" "${GCS_BG}/audio_32k"
gsutil -m -q cp "${MIR_BG}/manifests"/*.csv "${GCS_BG}/"

echo "typeB listing:"; gsutil ls "${GCS_B}/"
echo "background listing:"; gsutil ls "${GCS_BG}/"
echo "Done."
