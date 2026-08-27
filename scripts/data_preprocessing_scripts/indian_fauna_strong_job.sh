#!/usr/bin/env bash
#SBATCH --job-name=ien-strong-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the IEN Type-A STRONG detection dataset (Zenodo 18743214) into
# gs://esp-data-ingestion/indian-fauna/v0.1.0/typeA/. Uses the by-state zips +
# CSVs already on NFS: extracts each audio_typeA_<state>.zip into work/<state>/,
# resamples every (extensionless) WAV -> 16k+32k mono mirrors + durations.csv,
# joins type_A_annotations.csv by key (state_date_hash) into per-file selection
# tables, uploads.
#   sbatch scripts/data_preprocessing_scripts/indian_fauna/build_indian_fauna_strong.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/alp-data"

GCS="gs://esp-data-ingestion/indian-fauna/v0.1.0/typeA"
STAGE="${HOME}/esp-data-staging/indian_fauna"
WORK="${STAGE}/work_typeA"
MIRRORS="${STAGE}/mirrors_typeA"
SCRIPT="scripts/data_preprocessing_scripts/indian_fauna_strong.py"

export CLOUDSDK_CONFIG="$(mktemp -d)"   # attached SA for gsutil + gcsfs
echo "Node: $(hostname)  work=${WORK}"
echo "ok" | gsutil cp - "${GCS}/.auth_probe" && gsutil rm "${GCS}/.auth_probe" && echo "auth probe OK"

# 1. Extract each Type-A state zip into work/<state>/.
mkdir -p "${WORK}"
for z in "${STAGE}"/audio_typeA_*.zip; do
  state="$(basename "${z}" .zip | sed 's/^audio_typeA_//')"
  if [ ! -d "${WORK}/${state}" ]; then
    echo "extracting ${state} ..."; mkdir -p "${WORK}/${state}"; unzip -o -q "${z}" -d "${WORK}/${state}/"
  fi
done
echo "recordings: $(find "${WORK}" -type f | wc -l)"

# 2. Resample -> 16k + 32k mono mirrors + durations.csv.
rm -rf "${MIRRORS}"; mkdir -p "${MIRRORS}"
uv run python "${SCRIPT}" resample \
    --work "${WORK}" --out-root "${MIRRORS}" --workers "${SLURM_CPUS_PER_TASK:-16}"

# 3. Build manifests.
uv run python "${SCRIPT}" manifests \
    --anno-csv "${STAGE}/type_A_annotations.csv" \
    --meta-csv "${STAGE}/type_A_metadata.csv" \
    --durations-csv "${MIRRORS}/durations.csv" \
    --out-dir "${MIRRORS}/manifests"

# 4. Upload audio mirrors + manifests.
gsutil -m -q rsync -r "${MIRRORS}/audio_16k" "${GCS}/audio_16k"
gsutil -m -q rsync -r "${MIRRORS}/audio_32k" "${GCS}/audio_32k"
gsutil -m -q cp "${MIRRORS}/manifests"/*.csv "${GCS}/"

echo "final listing:"; gsutil ls "${GCS}/"
echo "Done."
