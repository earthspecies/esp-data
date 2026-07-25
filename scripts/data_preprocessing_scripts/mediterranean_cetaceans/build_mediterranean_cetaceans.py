"""Build the Mediterranean cetacean PAM SED dataset (Zenodo 17282717).

Jankauskaite et al. 2025, "Multi-platform low-cost cetacean PAM" (CC-BY-4.0).
Strong/detection ingest (WABAD-shaped) from the Raven ``.txt`` selection tables
on the FULL-LENGTH recordings — complementary to the existing weak clip-level
SuperWhales entry. Western Mediterranean HydroMoth (192 kHz etc.), 5 species +
unidentified Delphinidae, with time+frequency boxes and a ``signalType`` call
type (clicks / whistles).

Recording granularity = each ~1-minute WAV file under
``<Species>/3-complete-WAV/deployment-<ID>/``. Events are cross-file: within-file
times come directly from the ``Begin/End Date Time`` columns minus the file's
filename timestamp (verified), and multi-file events are split into one
selection row per file they cover.

Two stages (mirrors build_delphinid_whistles.py):
1. ``resample`` — decode every full WAV, write 16 kHz + 32 kHz mirrors flattened
   to ``<species>/<deployment>/<stem>.wav``, and a ``durations.csv``. Run on Slurm.
2. ``manifests`` — parse the ``.txt`` tables; using durations + per-deployment
   file lists, split events to per-file selection tables; write the manifest.

Nyquist caveat: 192 kHz source, most boxes straddle/exceed the 16 kHz stack
Nyquist (clicks 13-59 kHz, whistles up to ~27 kHz) — time detection + species
presence usable; freq-bbox non-viable; call-type degraded for clicks.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

LICENSE = "CC-BY-4.0"
SOURCE_DATASET = "mediterranean_cetaceans"
_TS = re.compile(r"(\d{8})_(\d{6})")
ST_COLUMNS = [
    "Selection",
    "Begin Time (s)",
    "End Time (s)",
    "Low Freq (Hz)",
    "High Freq (Hz)",
    "Species",
    "signalType",
]


def _file_ts(name: str) -> dt.datetime | None:
    """Parse the YYYYMMDD_HHMMSS timestamp from a WAV filename.

    Returns
    -------
    datetime | None
    """
    m = _TS.search(name)
    return dt.datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S") if m else None


def _parse_dt(s: str) -> dt.datetime | None:
    """Parse a Raven 'Begin/End Date Time' string.

    Returns
    -------
    datetime | None
    """
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return dt.datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _rel(species: str, deployment: str, stem: str) -> str:
    """Return the flattened relative key ``<species>/<deployment>/<stem>``.

    Returns
    -------
    str
    """
    return f"{species}/{deployment}/{stem}"


def _evs(b: float, e: float, d: float, meta: tuple) -> list[dict]:
    """Build clipped per-file event dicts from ``meta = (lo, hi, sp, sig)``.

    A pipe-joined ``species`` string (e.g. ``"A | B"``, used when two species
    co-occur in one box) is split into one event per species so labels stay
    clean binomials.

    Returns
    -------
    list[dict]
    """
    lo, hi, sp, sig = meta
    species = [s.strip() for s in str(sp).split("|") if s.strip()] or [str(sp).strip()]
    return [
        {"b": max(0.0, b), "e": min(e, d), "lo": lo, "hi": hi, "sp": s, "sig": sig} for s in species
    ]


# ── resample ────────────────────────────────────────────────────────────────
def _resample_one(args: tuple[str, str, str]) -> tuple[str, float, str]:
    """Write 16k + 32k WAV mirrors of one full recording; return its duration.

    Parameters
    ----------
    args : tuple[str, str, str]
        ``(src_wav, out_root, rel)``.

    Returns
    -------
    tuple[str, float, str]
        ``(rel, duration_sec, status)``.
    """
    import librosa
    import numpy as np
    import soundfile as sf

    src, out_root, rel = args
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
            out = Path(out_root) / f"audio_{tgt // 1000}k" / f"{rel}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            sf.write(out, np.clip(y, -1.0, 1.0), tgt, subtype="PCM_16")
        return rel, dur, "ok"
    except Exception as exc:  # noqa: BLE001
        return rel, 0.0, f"ERROR: {exc}"


def _wavs(root: Path) -> list[tuple[Path, str, str, str]]:
    """List full-length WAVs -> (path, species, deployment, stem).

    Returns
    -------
    list[tuple[Path, str, str, str]]
    """
    out = []
    for w in root.rglob("*.WAV"):
        parts = w.parts
        if "3-complete-WAV" not in parts:
            continue
        # <root>/<Species>/3-complete-WAV/deployment-<ID>/<file>.WAV
        i = parts.index("3-complete-WAV")
        species = parts[i - 1]
        deployment = parts[i + 1].replace("deployment-", "")
        out.append((w, species, deployment, w.stem))
    return out


def resample(root: Path, out_root: Path, workers: int) -> None:
    """Write 16k + 32k mirrors for every full WAV and emit durations.csv.

    Raises
    ------
    SystemExit
        If any file fails.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    wavs = _wavs(root)
    print(f"resampling {len(wavs)} full WAVs with {workers} workers ...", flush=True)
    jobs = [(str(w), str(out_root), _rel(sp, dp, st)) for (w, sp, dp, st) in wavs]
    durations, errors, done = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_resample_one, j) for j in jobs]):
            rel, dur, status = fut.result()
            done += 1
            if status != "ok":
                errors += 1
                print(f"  {status} ({rel})", flush=True)
            else:
                durations.append((rel, dur))
            if done % 50 == 0:
                print(f"  {done}/{len(jobs)} ...", flush=True)
    with open(out_root / "durations.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rel", "duration_sec"])
        w.writerows(sorted(durations))
    print(f"done: {done} written, {errors} errors", flush=True)
    if errors:
        raise SystemExit(f"{errors} files failed")


# ── manifests ────────────────────────────────────────────────────────────────
def _selection_tsv(events: list[dict]) -> str:
    """Serialise per-file events into a WABAD-shaped Raven TSV blob.

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
            "signalType": [e["sig"] for e in events],
        }
    )[ST_COLUMNS]
    return df.to_csv(sep="\t", index=False)


def build_manifests(root: Path, durations_csv: Path, out_dir: Path) -> None:
    """Parse Raven .txt -> per-file selection tables -> manifest.

    Parameters
    ----------
    root : Path
        Extracted tree root (holds ``<Species>/{1-annotation-tables,3-complete-WAV}``).
    durations_csv : Path
        ``durations.csv`` (rel, duration_sec) from the resample stage.
    out_dir : Path
        Destination for the manifest CSVs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = {r["rel"]: float(r["duration_sec"]) for r in csv.DictReader(open(durations_csv))}
    # per (species, deployment): ordered file stems (by timestamp) with durations
    dep_files: dict[tuple, list[str]] = defaultdict(list)
    for rel in dur:
        sp, dp, stem = rel.split("/", 2)
        dep_files[(sp, dp)].append(stem)
    for k in dep_files:
        dep_files[k].sort(key=lambda s: _file_ts(s) or dt.datetime.min)

    # accumulate per-file events (keyed by rel)
    events_per_rel: dict[str, list[dict]] = defaultdict(list)
    labels: set[str] = set()
    sigtypes: set[str] = set()
    for txt in root.rglob("*.txt"):
        if "1-annotation-tables" not in txt.parts:
            continue
        species_dir = txt.parts[txt.parts.index("1-annotation-tables") - 1]
        with open(txt, encoding="latin-1", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                bf, ef = r.get("Begin File", ""), r.get("End File", "")
                bts, ets = _file_ts(bf), _file_ts(ef)
                bdt = _parse_dt(r.get("Begin Date Time", ""))
                edt = _parse_dt(r.get("End Date Time", ""))
                if not (bts and ets and bdt and edt):
                    continue
                sp = str(r.get("species", species_dir)).strip() or species_dir
                sig = str(r.get("signalType", "")).strip()
                try:
                    lo, hi = float(r.get("Low Freq (Hz)", 0)), float(r.get("High Freq (Hz)", 0))
                except ValueError:
                    lo, hi = 0.0, 0.0
                labels.update(s.strip() for s in str(sp).split("|") if s.strip())
                sigtypes.add(sig)
                dep = r.get("deploymentID", "").strip()
                wf_begin = (bdt - bts).total_seconds()
                wf_end = (edt - ets).total_seconds()
                b_stem, e_stem = Path(bf).stem, Path(ef).stem
                meta = (lo, hi, sp, sig)
                if b_stem == e_stem:  # single-file event
                    rel = _rel(species_dir, dep, b_stem)
                    if rel in dur:
                        events_per_rel[rel].extend(_evs(wf_begin, wf_end, dur[rel], meta))
                else:  # multi-file: split across files
                    files = dep_files.get((species_dir, dep), [])
                    if b_stem not in files or e_stem not in files:
                        continue
                    i0, i1 = files.index(b_stem), files.index(e_stem)
                    for k in range(i0, i1 + 1):
                        rel = _rel(species_dir, dep, files[k])
                        d = dur.get(rel, 0.0)
                        b = wf_begin if k == i0 else 0.0
                        e = wf_end if k == i1 else d
                        events_per_rel[rel].extend(_evs(b, e, d, meta))

    rows = []
    for rel, d in dur.items():
        sp_dir, dep, stem = rel.split("/", 2)
        evs = events_per_rel.get(rel, [])
        rows.append(
            {
                "sound_name": f"{stem}.WAV",
                "species": sp_dir,
                "deployment_id": dep,
                "split": "all",
                "audio_duration": round(d, 3),
                "audio_fp": f"audio_32k/{rel}.wav",
                "16khz_path": f"audio_16k/{rel}.wav",
                "32khz_path": f"audio_32k/{rel}.wav",
                "n_events": len(evs),
                "source_dataset": SOURCE_DATASET,
                "license": LICENSE,
                "selection_table": _selection_tsv(evs),
            }
        )
    df = pd.DataFrame(rows)
    head = [
        "sound_name",
        "species",
        "deployment_id",
        "split",
        "audio_duration",
        "audio_fp",
        "16khz_path",
        "32khz_path",
        "n_events",
    ]
    df = df[head + [c for c in df.columns if c not in head]]
    out = out_dir / "mediterranean_cetaceans_all.csv"
    df.to_csv(out, index=False)
    n_evt = int(df["n_events"].sum())
    print(
        f"  all: {len(df)} recordings, {n_evt} events, {int((df['n_events'] == 0).sum())} empty "
        f"-> {out.name}  species={df.groupby('species').size().to_dict()}"
    )
    lab_out = out_dir / "mediterranean_cetaceans_labels.csv"
    pd.DataFrame({"Species": sorted(labels)}).to_csv(lab_out, index=False)
    print(f"  labels: {len(labels)} species; signalTypes={sorted(sigtypes)}")


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    pr = sub.add_parser("resample")
    pr.add_argument("--root", type=Path, required=True)
    pr.add_argument("--out-root", type=Path, required=True)
    pr.add_argument("--workers", type=int, default=16)
    pm = sub.add_parser("manifests")
    pm.add_argument("--root", type=Path, required=True)
    pm.add_argument("--durations-csv", type=Path, required=True)
    pm.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    if args.stage == "resample":
        resample(args.root, args.out_root, args.workers)
    else:
        build_manifests(args.root, args.durations_csv, args.out_dir)


if __name__ == "__main__":
    main()
