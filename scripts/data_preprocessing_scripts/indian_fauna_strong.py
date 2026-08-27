"""Build the India Ecoacoustics Network (IEN) Type-A STRONG detection dataset.

Ramesh, Singh et al. 2026, "A large-scale crowd-sourced annotated acoustic dataset
of Indian fauna" (bioRxiv 10.64898/2026.07.20.739496; data CC-BY-NC-4.0). Type-A =
strong labels: Raven-style time+frequency bounding boxes on full-length recordings,
518 species across 25 Indian states and four classes (Aves-dominant + Amphibia,
Insecta, Mammalia). WABAD-shaped ingest — one row per audio file + a
``selection_table`` (``Selection, Begin Time (s), End Time (s), Low Freq (Hz),
High Freq (Hz), Species``).

Join gotcha: the audio zips renumber the leading sequential ID, so the audio
filename and the ``media_file_name`` in the csv tables differ in that prefix but
share the rest. The canonical key is therefore ``media_file_name`` with the leading
``{seqID}_`` stripped (== ``state_date_hash``), applied to both audio files and the
annotation/metadata tables.

Two stages (mirrors build_mediterranean_cetaceans.py):
1. ``resample`` — extract each ``audio_typeA_<state>.zip``, decode every
   (extensionless) WAV, write 16 kHz + 32 kHz mono mirrors flattened to
   ``<state>/<key>.wav``, and a ``durations.csv``. Run on Slurm.
2. ``manifests`` — join ``type_A_annotations.csv`` to the audio by key; group boxes
   per file into selection tables; attach per-file taxonomy from ``taxa_info``;
   write the manifest.

Nyquist: crowd-sourced 8-384 kHz source -> 16k/32k stack. Per-class @16 kHz:
Aves 81% boxes fully in band, Insecta 79%, Mammalia 73% (15% fully ultrasonic = bats,
lost), Amphibia boxes are full-band (freq not meaningful) — but 100% of boxes have
their low edge in band, so time-presence detection is viable across all classes;
freq-bbox usable for birds/insects/mammals with a ``High Freq (Hz) <= 16000`` filter.
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
SOURCE_DATASET = "indian_fauna_strong"
ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
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
    """Parse a Python-repr dict string (``taxa_info`` / ``media_info``); ``{}`` on fail.

    Returns
    -------
    dict
    """
    try:
        return ast.literal_eval(s) or {}
    except (ValueError, SyntaxError):
        return {}


def _num(v: str) -> str:
    """Return a numeric/date string, blanking ``NA``/``None``/empty.

    Returns
    -------
    str
    """
    v = str(v).strip()
    return "" if v in ("", "NA", "nan", "None") else v


# ── resample ────────────────────────────────────────────────────────────────
def _resample_one(args: tuple[str, str, str, str]) -> tuple[str, str, float, int, str]:
    """Write 16k + 32k mono mirrors of one recording; return (key, state, dur, sr, status).

    Parameters
    ----------
    args : tuple[str, str, str, str]
        ``(src, out_root, state, key)``.

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
def _selection_tsv(events: list[dict]) -> str:
    """Serialise per-file boxes into a WABAD-shaped Raven TSV blob.

    Returns
    -------
    str
    """
    events = sorted(events, key=lambda e: e["b"])
    df = pd.DataFrame(
        {
            "Selection": range(1, len(events) + 1),
            "Begin Time (s)": [round(e["b"], 4) for e in events],
            "End Time (s)": [round(e["e"], 4) for e in events],
            "Low Freq (Hz)": [round(e["lo"], 1) for e in events],
            "High Freq (Hz)": [round(e["hi"], 1) for e in events],
            "Species": [e["sp"] for e in events],
        }
    )[ST_COLUMNS]
    return df.to_csv(sep="\t", index=False)


def build_manifests(anno_csv: Path, meta_csv: Path, durations_csv: Path, out_dir: Path) -> None:
    """Join annotations to audio by key -> per-file selection tables -> manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = pd.read_csv(durations_csv, dtype={"key": str, "state": str})
    dur_by_key = {
        r["key"]: (r["state"], float(r["duration_sec"]), int(r["orig_sr"]))
        for _, r in dur.iterrows()
    }

    # per-file target species + taxonomy from metadata
    meta = pd.read_csv(meta_csv, dtype=str, keep_default_na=False)
    meta_by_key: dict[str, dict] = {}
    for _, r in meta.iterrows():
        ti = _parse(r["taxa_info"])
        # Type-A metadata has no scientific_name column; the per-file target species
        # is the taxa_info GBIF match (canonicalName, else the raw scientificName).
        target = str(r.get("scientific_name", "")).strip() or ti.get("canonicalName") or ""
        # Location/time context columns named to match context_builder defaults
        # (locality / latitudeDecimal / longitudeDecimal / eventDate) so the
        # with-context soundscape task works verbatim; "NA"/blank left empty.
        locality = (
            str(r.get("archive_name", "")).replace("audio_type_A_", "").replace(".zip", "").strip()
        )
        meta_by_key[_key(r["media_file_name"])] = {
            "target_species": target,
            "class": ti.get("class") or "",
            "canonical_name": ti.get("canonicalName") or "",
            "rank": ti.get("rank") or "",
            "locality": locality,
            "latitudeDecimal": _num(r.get("latitude", "")),
            "longitudeDecimal": _num(r.get("longitude", "")),
            "eventDate": _num(r.get("recording_date", "")),
        }

    # group annotation boxes by key
    anno = pd.read_csv(anno_csv, dtype=str, keep_default_na=False)
    events_by_key: dict[str, list[dict]] = defaultdict(list)
    labels: set[str] = set()
    for _, r in anno.iterrows():
        k = _key(r["media_file_name"])
        sp = str(r.get("scientific_name", "")).strip()
        try:
            lo, hi = float(r.get("low_freq") or 0), float(r.get("high_freq") or 0)
            b, e = float(r.get("begin_time") or 0), float(r.get("end_time") or 0)
        except ValueError:
            continue
        if sp:
            labels.add(sp)
        events_by_key[k].append({"b": max(0.0, b), "e": e, "lo": lo, "hi": hi, "sp": sp})

    rows, n_no_audio = [], 0
    for k, (state, d, orig_sr) in dur_by_key.items():
        evs = [e for e in events_by_key.get(k, []) if e["b"] < d + 0.5]  # clip stray boxes
        md = meta_by_key.get(k, {})
        rows.append(
            {
                "sound_name": f"{k}.wav",
                "key": k,
                "state": state,
                "target_species": md.get("target_species", ""),
                "class": md.get("class", ""),
                "locality": md.get("locality", state),
                "latitudeDecimal": md.get("latitudeDecimal", ""),
                "longitudeDecimal": md.get("longitudeDecimal", ""),
                "eventDate": md.get("eventDate", ""),
                "split": "all",
                "audio_duration": round(d, 3),
                "orig_sample_rate": orig_sr,
                "audio_fp": f"audio_32k/{state}/{k}.wav",
                "16khz_path": f"audio_16k/{state}/{k}.wav",
                "32khz_path": f"audio_32k/{state}/{k}.wav",
                "n_events": len(evs),
                "source_dataset": SOURCE_DATASET,
                "license": LICENSE,
                "selection_table": _selection_tsv(evs),
            }
        )
    # annotations whose key has no matching audio (should be ~0)
    n_no_audio = len(set(events_by_key) - set(dur_by_key))

    df = pd.DataFrame(rows)
    head = [
        "sound_name",
        "key",
        "state",
        "target_species",
        "class",
        "locality",
        "latitudeDecimal",
        "longitudeDecimal",
        "eventDate",
        "split",
        "audio_duration",
        "orig_sample_rate",
        "audio_fp",
        "16khz_path",
        "32khz_path",
        "n_events",
    ]
    df = df[head + [c for c in df.columns if c not in head]]
    out = out_dir / "indian_fauna_strong_all.csv"
    df.to_csv(out, index=False)
    n_evt = int(df["n_events"].sum())
    cls = df.groupby("class").size().to_dict()
    print(
        f"  all: {len(df)} recordings, {n_evt} boxes, {int((df['n_events'] == 0).sum())} empty, "
        f"{n_no_audio} anno-keys w/o audio -> {out.name}"
    )
    print(f"  class distribution (files): {cls}")
    lab_out = out_dir / "indian_fauna_strong_labels.csv"
    pd.DataFrame({"Species": sorted(labels)}).to_csv(lab_out, index=False)
    print(f"  labels: {len(labels)} species -> {lab_out.name}")


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    pr = sub.add_parser("resample")
    pr.add_argument("--work", type=Path, required=True, help="dir of work/<state>/ extracted audio")
    pr.add_argument("--out-root", type=Path, required=True)
    pr.add_argument("--workers", type=int, default=16)
    pm = sub.add_parser("manifests")
    pm.add_argument("--anno-csv", type=Path, required=True)
    pm.add_argument("--meta-csv", type=Path, required=True)
    pm.add_argument("--durations-csv", type=Path, required=True)
    pm.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.stage == "resample":
        resample(args.work, args.out_root, args.workers)
    else:
        build_manifests(args.anno_csv, args.meta_csv, args.durations_csv, args.out_dir)


if __name__ == "__main__":
    main()
