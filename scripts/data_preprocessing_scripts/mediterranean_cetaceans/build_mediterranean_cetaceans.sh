#!/usr/bin/env bash
#SBATCH --job-name=medcet-build
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output="/home/%u/logs/%A_%x.log"
#SBATCH --qos=naturelm

# ───────────────────────────────────────────────────────────────────
# Build the Mediterranean cetacean detection dataset (Zenodo 17282717) into
# gs://esp-data-ingestion/mediterranean-cetaceans/v0.1.0/. Uses the 6 species
# zips already on NFS: extracts, writes 16k+32k mirrors of every full WAV
# (flattened to <species>/<deployment>/<stem>.wav) + durations.csv, then
# splits the Raven .txt events to per-file selection tables, and uploads.
#   sbatch scripts/data_preprocessing_scripts/mediterranean_cetaceans/build_mediterranean_cetaceans.sh
# ───────────────────────────────────────────────────────────────────
set -euo pipefail
cd "${HOME}/esp-data-dev"

GCS="gs://esp-data-ingestion/mediterranean-cetaceans/v0.1.0"
STAGE="${HOME}/esp-data-staging/med_cetacean"
WORK="${STAGE}/work"
MIRRORS="${STAGE}/mirrors"
SCRIPT="scripts/data_preprocessing_scripts/mediterranean_cetaceans/build_mediterranean_cetaceans.py"

# Ignore stale user ADC; use node attached SA (gsutil + gcsfs).
export CLOUDSDK_CONFIG="$(mktemp -d)"
echo "Node: $(hostname)  work=${WORK}"
echo "ok" | gsutil cp - "${GCS}/.auth_probe" && gsutil rm "${GCS}/.auth_probe" && echo "auth probe OK"

# 1. Extract all 6 species trees if not already present.
mkdir -p "${WORK}"
for z in "${STAGE}"/*.zip; do
  name="$(basename "${z}" .zip)"
  if [ ! -d "${WORK}/${name}" ]; then echo "extracting ${name} ..."; unzip -o -q "${z}" -d "${WORK}/"; fi
done
echo "full WAVs: $(find "${WORK}" -path '*3-complete-WAV*' -iname '*.WAV' | wc -l)"
echo "anno txts: $(find "${WORK}" -path '*1-annotation-tables*' -iname '*.txt' | wc -l)"

# 2. Resample every full WAV -> 16k + 32k mirrors + durations.csv.
rm -rf "${MIRRORS}"; mkdir -p "${MIRRORS}"
uv run python "${SCRIPT}" resample \
    --root "${WORK}" --out-root "${MIRRORS}" --workers "${SLURM_CPUS_PER_TASK:-16}"

# 3. Build manifests (split cross-file events to per-file selection tables).
uv run python "${SCRIPT}" manifests \
    --root "${WORK}" --durations-csv "${MIRRORS}/durations.csv" --out-dir "${MIRRORS}/manifests"

# 4. Upload audio mirrors + manifests.
gsutil -m -q rsync -r "${MIRRORS}/audio_16k" "${GCS}/audio_16k"
gsutil -m -q rsync -r "${MIRRORS}/audio_32k" "${GCS}/audio_32k"
gsutil -m -q cp "${MIRRORS}/manifests"/*.csv "${GCS}/"

echo "final listing:"; gsutil ls "${GCS}/"
echo "Done."
