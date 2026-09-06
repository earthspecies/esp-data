#!/usr/bin/env bash

#SBATCH --partition=cpu
#SBATCH --output="/home/%u/logs/filter_resampled_splits_%A.log"
#SBATCH --job-name="filter-resampled-splits"

# Prepare the new Xeno-canto (20260622) and iNaturalist (20260616) split CSVs:
#   * drop rows without a pre-resampled 32kHz path
#   * (iNaturalist only) rewrite absolute, mixed-bucket 32khz/16khz paths to a
#     single relative form so the loaders resolve them against one data_root
# Outputs are written next to the source layout under DATA_HOME with a `_v1`
# suffix (the source dumps are never overwritten). Run on a batch node so the
# streaming Polars job has headroom and does not OOM an interactive VM.

set -euo pipefail

export GOOGLE_APPLICATION_CREDENTIALS=${GOOGLE_APPLICATION_CREDENTIALS:-$HOME/.config/gcloud/application_default_credentials.json}
export CLOUDPATHLIB_FORCE_OVERWRITE_FROM_CLOUD=1

# Repo checkout (override REPO_DIR if yours lives elsewhere).
REPO_DIR=${REPO_DIR:-$HOME/code/alp-data}
cd "$REPO_DIR"
uv venv --python 3.12
uv sync

SRC_BUCKET=${SRC_BUCKET:-gs://esp-data-ingestion}
# DATA_HOME (see alp_data/io/paths.py); split_paths in the datasets point here.
DST_BUCKET=${DST_BUCKET:-gs://esp-data-274503}
SCRIPT=scripts/data_preprocessing_scripts/filter_resampled_splits.py

WORKDIR=$(mktemp -d)
trap 'rm -rf "$WORKDIR"' EXIT

# process <dataset> <version_dir> <src_basename> <normalize_columns>
# reads $SRC_BUCKET/<dataset>/<version_dir>/raw/<src_basename>.csv and writes
# $DST_BUCKET/<dataset>/<version_dir>/raw/<src_basename>_v1.csv
process() {
    local dataset="$1" version_dir="$2" src_basename="$3" normalize="$4" join_file="${5:-}"
    local src="$SRC_BUCKET/$dataset/$version_dir/raw/${src_basename}.csv"
    local dst="$DST_BUCKET/$dataset/$version_dir/raw/${src_basename}_v1.csv"
    local lin="$WORKDIR/${dataset}__${src_basename}.csv"
    local lout="$WORKDIR/${dataset}__${src_basename}_v1.csv"

    local join_args=()
    if [ -n "$join_file" ]; then
        join_args=(--join-file "$join_file" --join-columns playback_used)
    fi

    echo "== ${src} -> ${dst} =="
    gsutil cp "$src" "$lin"
    uv run python "$SCRIPT" \
        --input "$lin" \
        --output "$lout" \
        --filter-column 32khz_path \
        --normalize-columns "$normalize" \
        --require-true gbif_link_ok \
        "${join_args[@]}"
    gsutil cp "$lout" "$dst"
    rm -f "$lin" "$lout"
}

# Xeno-canto (20260622): 32khz/16khz paths are already relative -> no normalization.
# The playback_used column was added upstream only to the undated splits after
# this snapshot was cut, so join it in by xc_id from the undated all.csv.
XC_PLAYBACK_SRC="$WORKDIR/xeno_playback_src.csv"
gsutil cp "$SRC_BUCKET/xeno-canto/v0.1.0/raw/all.csv" "$XC_PLAYBACK_SRC"
for base in \
    train_20260622 \
    val_20260622 \
    all_20260622 \
    train_unseen_20260622 \
    val_unseen_20260622 \
    all_unseen_20260622; do
    process xeno-canto v0.1.0 "$base" "" "$XC_PLAYBACK_SRC"
done

# iNaturalist (20260616): 32khz/16khz paths are absolute, mixed-bucket -> normalize.
for base in \
    train_20260616 \
    train_unseen_20260616 \
    val_20260616 \
    val_unseen_20260616 \
    all_20260616 \
    all_unseen_20260616; do
    process inaturalist v0.1.0 "$base" "32khz_path,16khz_path"
done

echo "Done."
