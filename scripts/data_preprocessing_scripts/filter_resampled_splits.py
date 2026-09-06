"""Filter resampled dataset splits and normalize their audio-path columns.

This offline job prepares the Xeno-canto / iNaturalist split CSVs so the dataset
loaders can consume them without any read-time path handling:

1. Rows without a pre-resampled path (by default ``32khz_path``) are dropped,
   since that column is the default pre-resampled audio source and rows missing
   it cannot be served at that rate.
2. Selected path columns that hold absolute ``gs://`` URIs are rewritten to
   paths relative to the dataset's ``.../v0.1.0/raw/`` root. The iNaturalist
   dump stores ``32khz_path`` / ``16khz_path`` as absolute, mixed-bucket URIs
   (``esp-data-ingestion`` and ``esp-ml-datasets``); rewriting them to a single
   relative form lets the loader resolve them against one ``data_root``, so the
   dataset implementation stays identical to ``main``.

The work is done with the streaming Polars engine (``scan_csv`` -> ``sink_csv``)
and every column is read as text, so large split CSVs are processed without
loading them fully into memory. Run it on a batch node rather than on an
interactive VM (see ``filter_resampled_splits_job.sh``).
"""

import argparse

import polars as pl

_TRUE_VALUES = ("true", "1", "yes")


def build_lazyframe(
    input_path: str,
    *,
    filter_column: str,
    normalize_columns: list[str],
    require_true_columns: list[str],
    raw_root_marker: str,
    join_path: str | None = None,
    join_key: str = "xc_id",
    join_columns: list[str] | None = None,
) -> pl.LazyFrame:
    """Build the streaming query for one split CSV.

    Parameters
    ----------
    input_path : str
        Path to the source CSV (local path; read lazily).
    filter_column : str
        Column that must be present and non-empty for a row to be kept.
    normalize_columns : list[str]
        Columns whose absolute ``gs://`` values are rewritten to paths relative
        to `raw_root_marker`. Values that are not absolute URIs are left as-is.
    require_true_columns : list[str]
        Columns whose value must be truthy (one of ``"true"``, ``"1"``, ``"yes"``,
        case-insensitive) for a row to be kept. Used to drop rows that did not
        link to GBIF (``gbif_link_ok``).
    raw_root_marker : str
        Substring marking the dataset raw root (e.g. ``"/v0.1.0/raw/"``). The
        relative path is whatever follows the last occurrence of this marker.
    join_path : str | None
        Optional CSV supplying extra columns to left-join onto the split by
        `join_key` (e.g. a ``playback_used`` column added upstream after this
        snapshot was cut). None to join nothing.
    join_key : str
        Column to join on, by default ``"xc_id"``.
    join_columns : list[str] | None
        Columns to pull from `join_path` (in addition to `join_key`).

    Returns
    -------
    pl.LazyFrame
        The lazy query producing the normalized, filtered rows.

    Raises
    ------
    ValueError
        If `filter_column`, any of `normalize_columns`, any of
        `require_true_columns`, or the join columns are not present.
    """
    join_columns = join_columns or []
    # Read every column as text so no dtype inference happens (some rows carry
    # sentinels such as "Not found" in otherwise-numeric columns) and values are
    # preserved verbatim.
    lf = pl.scan_csv(input_path, infer_schema_length=0)
    available = set(lf.collect_schema().names())

    required = [filter_column, *require_true_columns]
    if join_path:
        required.append(join_key)
    for column in required:
        if column not in available:
            raise ValueError(
                f"Column {column!r} not found in {input_path}. "
                f"Available columns: {sorted(available)}"
            )

    if join_path:
        right = pl.scan_csv(join_path, infer_schema_length=0)
        right_cols = set(right.collect_schema().names())
        for column in [join_key, *join_columns]:
            if column not in right_cols:
                raise ValueError(
                    f"Join column {column!r} not found in {join_path}. "
                    f"Available columns: {sorted(right_cols)}"
                )
        right = right.select([join_key, *join_columns]).unique(subset=[join_key])
        lf = lf.join(right, on=join_key, how="left")

    for column in normalize_columns:
        if column not in available:
            raise ValueError(
                f"Normalize column {column!r} not found in {input_path}. "
                f"Available columns: {sorted(available)}"
            )
        lf = lf.with_columns(
            pl.when(pl.col(column).str.contains("://", literal=True))
            .then(pl.col(column).str.split(raw_root_marker).list.last())
            .otherwise(pl.col(column))
            .alias(column)
        )

    lf = lf.filter(
        pl.col(filter_column).is_not_null() & (pl.col(filter_column).str.strip_chars() != "")
    )

    for column in require_true_columns:
        lf = lf.filter(pl.col(column).str.strip_chars().str.to_lowercase().is_in(_TRUE_VALUES))

    return lf


def count_rows(path: str) -> int:
    """Return the number of parsed data rows in a CSV.

    Counts by materializing a single (streamed) column rather than ``pl.len()``,
    whose fast newline-count path miscounts rows containing quoted embedded
    newlines (captions/fieldNotes here do).

    Parameters
    ----------
    path : str
        Path to the CSV file (read lazily).

    Returns
    -------
    int
        Number of data rows (excluding the header).
    """
    lf = pl.scan_csv(path, infer_schema_length=0)
    first_column = lf.collect_schema().names()[0]
    return lf.select(pl.col(first_column)).collect().height


def main() -> None:
    """Filter and normalize a single split CSV from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Source CSV path (local).")
    parser.add_argument("--output", required=True, help="Destination CSV path (local).")
    parser.add_argument(
        "--filter-column",
        default="32khz_path",
        help="Column that must be non-empty for a row to be kept.",
    )
    parser.add_argument(
        "--normalize-columns",
        default="",
        help="Comma-separated columns whose absolute gs:// values are rewritten "
        "to paths relative to --raw-root-marker. Empty to normalize nothing.",
    )
    parser.add_argument(
        "--require-true",
        default="",
        help="Comma-separated columns whose value must be truthy (true/1/yes) for "
        "a row to be kept, e.g. gbif_link_ok to drop rows that did not link to "
        "GBIF. Empty to require nothing.",
    )
    parser.add_argument(
        "--raw-root-marker",
        default="/v0.1.0/raw/",
        help="Substring marking the dataset raw root; the relative path is what "
        "follows its last occurrence.",
    )
    parser.add_argument(
        "--join-file",
        default="",
        help="Optional CSV supplying extra columns to left-join by --join-key "
        "(e.g. a playback_used column added upstream after the snapshot).",
    )
    parser.add_argument("--join-key", default="xc_id", help="Column to join on.")
    parser.add_argument(
        "--join-columns",
        default="",
        help="Comma-separated columns to pull from --join-file.",
    )
    args = parser.parse_args()

    normalize_columns = [c.strip() for c in args.normalize_columns.split(",") if c.strip()]
    require_true_columns = [c.strip() for c in args.require_true.split(",") if c.strip()]
    join_columns = [c.strip() for c in args.join_columns.split(",") if c.strip()]

    lf = build_lazyframe(
        args.input,
        filter_column=args.filter_column,
        normalize_columns=normalize_columns,
        require_true_columns=require_true_columns,
        raw_root_marker=args.raw_root_marker,
        join_path=args.join_file or None,
        join_key=args.join_key,
        join_columns=join_columns,
    )
    lf.sink_csv(args.output)

    n_in = count_rows(args.input)
    n_out = count_rows(args.output)
    print(f"{args.input} -> {args.output}: kept {n_out}/{n_in} rows (dropped {n_in - n_out})")


if __name__ == "__main__":
    main()
