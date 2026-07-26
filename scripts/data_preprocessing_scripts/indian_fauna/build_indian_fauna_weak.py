"""Build the IEN Type-B WEAK and Background datasets (Zenodo 18927866 / 18928201).

Ramesh, Singh et al. 2026, "A large-scale crowd-sourced annotated acoustic dataset
of Indian fauna" (data CC-BY-NC-4.0).

- **Type B** = weak clip-level multi-label: per-file species presence, no time
  localization. The release stores one metadata row per (file, species); a physical
  audio file with N species appears under N rows sharing the same key
  (state_date_hash). We group by key -> one manifest row per file with a
  ``foreground_species`` list and a full-clip ``selection_table`` (one row per
  species spanning the whole recording, ``Presence`` column) — the ndege_zetu shape.
- **Background** = noise recordings (wind/rain/traffic/cattle/unknown). Grouped the
  same way, with the ``background_annotations.csv`` categories as the label.

Join key = ``media_file_name`` with the leading ``{seqID}_`` stripped (the audio zips
renumber that prefix). Shared ``resample`` stage with the Type-A build.

Two stages:
1. ``resample`` — extract audio zips, decode every (extensionless) WAV, write 16k+32k
   mono mirrors flattened to ``<state>/<key>.wav`` + ``durations.csv``. Run on Slurm.
2. ``manifests`` (``--kind {typeB,background}``) — group metadata by key -> per-file
   weak manifest.
"""

from __future__ import annotations

import argparse
import ast
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

LICENSE = "CC-BY-NC-4.0"
_ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "Presence",
]


def _key(media_file_name: str) -> str:
    """Return the join key ``state_date_hash`` (media_file_name minus leading seqID).

    Returns
    -------
    str
    """
    stem = Path(str(media_file_name)).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def _parse(s: str) -> dict:
    """Parse a Python-repr dict string (``taxa_info``); ``{}`` on failure.

    Returns
    -------
    dict
    """
    try:
        return ast.literal_eval(s) or {}
    except (ValueError, SyntaxError):
        return {}


# ── resample (shared with the Type-A build) ──────────────────────────────────
def _resample_one(args: tuple[str, str, str, str]) -> tuple[str, str, float, int, str]:
    """Write 16k + 32k mono mirrors of one recording; return (key, state, dur, sr, status).

    Returns
    -------
    tuple[str, str, float, int, str]
    """
    import librosa
    import numpy as np
    import soundfile as sf

    src, out_root, state, key = args
    try:
        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / float(sr)
        for tgt in (16000, 32000):
            y = (
                audio
                if sr == tgt
                else librosa.resample(
                    y=audio, orig_sr=sr, target_sr=tgt, scale=True, res_type="kaiser_best"
                )
            )
            out = Path(out_root) / f"audio_{tgt // 1000}k" / state / f"{key}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.clip(y, -1.0, 1.0), tgt, subtype="PCM_16")
        return key, state, dur, int(sr), "ok"
    except Exception as exc:  # noqa: BLE001 - crowd-sourced audio; tolerate + report
        return key, state, 0.0, 0, f"ERROR: {exc}"


def _wavs(work: Path) -> list[tuple[Path, str, str]]:
    """List extracted recordings under ``work/<state>/*`` -> (path, state, key).

    Returns
    -------
    list[tuple[Path, str, str]]
    """
    out = []
    for state_dir in sorted(p for p in work.iterdir() if p.is_dir()):
        state = state_dir.name
        for f in sorted(state_dir.rglob("*")):
            if f.is_file():
                out.append((f, state, _key(f.name)))
    return out


def resample(work: Path, out_root: Path, workers: int) -> None:
    """Write 16k + 32k mirrors for every recording and emit durations.csv."""
    out_root.mkdir(parents=True, exist_ok=True)
    wavs = _wavs(work)
    print(f"resampling {len(wavs)} recordings with {workers} workers ...", flush=True)
    jobs = [(str(f), str(out_root), state, key) for (f, state, key) in wavs]
    rows, errors, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_resample_one, j) for j in jobs]):
            key, state, dur, sr, status = fut.result()
            done += 1
            if status != "ok":
                errors += 1
                if errors <= 30:
                    print(f"  {status} ({state}/{key})", flush=True)
            else:
                rows.append((key, state, round(dur, 3), sr))
            if done % 200 == 0:
                print(f"  {done}/{len(jobs)} ...", flush=True)
    with open(out_root / "durations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["key", "state", "duration_sec", "orig_sr"])
        w.writerows(sorted(rows))
    print(f"done: {len(rows)} written, {errors} errors", flush=True)


# ── manifests ────────────────────────────────────────────────────────────────
def _weak_tsv(species: list[str], dur: float) -> str:
    """Serialise a full-clip weak selection table (one row per species).

    Returns
    -------
    str
    """
    df = pd.DataFrame(
        {
            "Selection": range(1, len(species) + 1),
            "Begin Time (s)": [0.0] * len(species),
            "End Time (s)": [round(dur, 3)] * len(species),
            "Low Freq (Hz)": [0.0] * len(species),
            "High Freq (Hz)": [0.0] * len(species),
            "Species": species,
            "Presence": ["Present"] * len(species),
        }
    )[_ST_COLUMNS]
    return df.to_csv(sep="\t", index=False)


def build_manifests(
    kind: str, meta_csv: Path, anno_csv: Path | None, durations_csv: Path, out_dir: Path, name: str
) -> None:
    """Group per-file weak labels by key -> manifest.

    Parameters
    ----------
    kind : str
        ``"typeB"`` (species presence from metadata) or ``"background"`` (noise
        categories from ``background_annotations.csv``).
    meta_csv : Path
        Metadata csv (Type B: one row per file×species; background: one per file).
    anno_csv : Path | None
        Background annotations csv (categories); unused for Type B.
    durations_csv, out_dir, name
        durations.csv from resample, output dir, dataset name (``source_dataset``).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = pd.read_csv(durations_csv, dtype={"key": str, "state": str})
    dur_by_key = {
        r["key"]: (r["state"], float(r["duration_sec"]), int(r["orig_sr"]))
        for _, r in dur.iterrows()
    }

    species_by_key: dict[str, list[str]] = defaultdict(list)
    class_by_key: dict[str, set] = defaultdict(set)
    labels: set[str] = set()

    if kind == "typeB":
        meta = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
        for _, r in meta.iterrows():
            k = _key(r["media_file_name"])
            ti = _parse(r["taxa_info"])
            sp = ti.get("canonicalName") or str(r.get("scientific_name", "")).strip()
            if sp and sp not in species_by_key[k]:
                species_by_key[k].append(sp)
                labels.add(sp)
            if ti.get("class"):
                class_by_key[k].add(ti["class"])
    else:  # background: categories come from annotations (Background / unknown sp. N)
        anno = pd.read_csv(anno_csv, dtype=str, keep_default_na=False)
        for _, r in anno.iterrows():
            k = _key(r["media_file_name"])
            cat = str(r.get("scientific_name", "")).strip()
            if cat and cat not in species_by_key[k]:
                species_by_key[k].append(cat)
                labels.add(cat)
        # ensure every metadata file is present even if it has no annotation row
        meta = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
        for _, r in meta.iterrows():
            species_by_key.setdefault(_key(r["media_file_name"]), [])

    rows = []
    for k, (state, d, orig_sr) in dur_by_key.items():
        sp = species_by_key.get(k, [])
        rows.append(
            {
                "sound_name": f"{k}.wav",
                "key": k,
                "state": state,
                "foreground_species": ", ".join(sp),  # comma-separated == species_list convention
                "class": "; ".join(sorted(class_by_key.get(k, set()))),
                "n_species": len(sp),
                "split": "all",
                "audio_duration": round(d, 3),
                "audio_duration_sec": round(d, 3),
                "orig_sample_rate": orig_sr,
                "audio_fp": f"audio_32k/{state}/{k}.wav",
                "16khz_path": f"audio_16k/{state}/{k}.wav",
                "32khz_path": f"audio_32k/{state}/{k}.wav",
                "source_dataset": name,
                "license": LICENSE,
                "selection_table": _weak_tsv(sp, d),
            }
        )
    n_no_audio = len(set(species_by_key) - set(dur_by_key))
    df = pd.DataFrame(rows)
    head = [
        "sound_name",
        "key",
        "state",
        "foreground_species",
        "class",
        "n_species",
        "split",
        "audio_duration",
        "audio_duration_sec",
        "orig_sample_rate",
        "audio_fp",
        "16khz_path",
        "32khz_path",
    ]
    df = df[head + [c for c in df.columns if c not in head]]
    out = out_dir / f"{name}_all.csv"
    df.to_csv(out, index=False)
    print(
        f"  all: {len(df)} recordings, {int(df['n_species'].sum())} label-instances, "
        f"{int((df['n_species'] == 0).sum())} empty, {n_no_audio} keys w/o audio -> {out.name}"
    )
    lab_out = out_dir / f"{name}_labels.csv"
    pd.DataFrame({"Species": sorted(labels)}).to_csv(lab_out, index=False)
    print(f"  labels: {len(labels)} -> {lab_out.name}")


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    pr = sub.add_parser("resample")
    pr.add_argument("--work", type=Path, required=True)
    pr.add_argument("--out-root", type=Path, required=True)
    pr.add_argument("--workers", type=int, default=16)
    pm = sub.add_parser("manifests")
    pm.add_argument("--kind", choices=["typeB", "background"], required=True)
    pm.add_argument("--meta-csv", type=Path, required=True)
    pm.add_argument("--anno-csv", type=Path, default=None)
    pm.add_argument("--durations-csv", type=Path, required=True)
    pm.add_argument("--out-dir", type=Path, required=True)
    pm.add_argument(
        "--name", required=True, help="e.g. indian_fauna_weak / indian_fauna_background"
    )
    args = p.parse_args()
    if args.stage == "resample":
        resample(args.work, args.out_root, args.workers)
    else:
        build_manifests(
            args.kind, args.meta_csv, args.anno_csv, args.durations_csv, args.out_dir, args.name
        )


if __name__ == "__main__":
    main()
