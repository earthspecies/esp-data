"""Extract reliable-negative ("no animal") noise clips from IEN Background.

The IEN Background record (Zenodo 18928201) is recordings the annotators set aside
as noise-dominated. Most categories are abiotic/anthropogenic (Background, Wind, Rain,
Vehicle, Human, Thunder, Aeroplane, …) but a minority contain animals (`unknown sp. N`,
`Unknown insect`, `Sambar deer alarm call?`, `Cattle`). For the training ``noise_bank``
(mixed in as negatives + used to mask open-detection tasks) the pool must be free of
animal sound, so a file qualifies only if EVERY one of its annotation categories is in
the abiotic/anthropogenic allowlist.

Qualifying files' 32 kHz mono mirrors are tiled into non-overlapping 10 s PCM16 clips
(a short remainder >= MIN_TAIL is kept; files < MIN_TAIL are dropped) named
``ien_bg_<key>_<idx>.wav`` for upload to the noise dir.

Human / Human Voice are KEPT as anthropogenic noise (consistent with the existing
urbansound / wham / audioset noise dirs).

Two stages: nothing to resample (source is already 32 kHz mono); a single ``build``
stage does the filter + crop + manifest. Run on Slurm — reads the GCS 32 kHz mirrors.
"""

from __future__ import annotations

import argparse
import ast
import csv
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

csv.field_size_limit(sys.maxsize)

CLIP_SEC = 10.0
MIN_TAIL_SEC = 3.0
SR = 32000

# Normalized-substring rules. A category is BLOCK if it names an animal / unknown
# vocalization; ALLOW if it names abiotic or anthropogenic noise; otherwise OTHER
# (treated as block — unrecognised, so excluded to stay reliable).
_BLOCK = [
    "unknown",
    "unidentified",
    "insect",
    "deer",
    "cattle",
    "alarm",
    "signal",
    " sp",
    "sp.",
    "sp ",
    "abc",
]
_ALLOW = [
    "backgrou",  # prefix matches "background" and the dataset's misspelled variant
    "noise",
    "rain",
    "wind",
    "water",
    "thunder",
    "vehicle",
    "vehicular",
    "train horn",
    "bike",
    "aeroplane",
    "anthropogenic",
    "recorder",
    "leaf litter",
    "leaf",
    "not an organism",
    "human",
]


def _norm(s: str) -> str:
    """Lowercase and strip a category to letters/spaces for keyword matching.

    Returns
    -------
    str
    """
    return re.sub(r"[^a-z ]", " ", str(s).lower()).strip()


def classify(category: str) -> str:
    """Classify a raw category string as ``allow`` / ``block`` / ``other``.

    A ``background``/``noise`` tag always allows (even though "noise" would match
    nothing in the blocklist); an explicit biotic/unknown keyword blocks.

    Returns
    -------
    str
    """
    n = _norm(category)
    if not n:
        return "other"
    is_noise = "backgrou" in n or "noise" in n
    if not is_noise and any(b.strip() in n for b in _BLOCK):
        return "block"
    if any(a in n for a in _ALLOW):
        return "allow"
    return "other"


def _parse(s: str) -> dict:
    """Parse a Python-repr dict string; ``{}`` on failure.

    Returns
    -------
    dict
    """
    try:
        return ast.literal_eval(s) or {}
    except (ValueError, SyntaxError):
        return {}


def _key(media_file_name: str) -> str:
    """Return the join key ``state_date_hash`` (media_file_name minus leading seqID).

    Returns
    -------
    str
    """
    stem = Path(str(media_file_name)).stem
    return stem.split("_", 1)[1] if "_" in stem else stem


def qualifying_keys(anno_csv: Path) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """Return ({key: categories} that pass, {key: categories} that are blocked).

    A key passes iff every one of its categories classifies as ``allow``.

    Returns
    -------
    tuple[dict[str, list[str]], dict[str, list[str]]]
    """
    a = pd.read_csv(anno_csv, dtype=str, keep_default_na=False)
    cats: dict[str, list[str]] = defaultdict(list)
    for _, r in a.iterrows():
        cats[_key(r["media_file_name"])].append(str(r["scientific_name"]).strip())
    passed, blocked = {}, {}
    for k, cs in cats.items():
        if all(classify(c) == "allow" for c in cs):
            passed[k] = cs
        else:
            blocked[k] = cs
    return passed, blocked


def _clip_one(args: tuple[str, str, str, str]) -> tuple[str, int, str]:
    """Tile one 32 kHz mono mirror into 10 s clips; return (key, n_clips, status).

    Returns
    -------
    tuple[str, int, str]
    """
    import numpy as np
    import soundfile as sf

    src, out_dir, key, state = args
    try:
        audio, sr = sf.read(src, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:  # mirrors are already 32k; guard anyway
            import librosa

            audio = librosa.resample(y=audio, orig_sr=sr, target_sr=SR, res_type="kaiser_best")
        step = int(CLIP_SEC * SR)
        tail = int(MIN_TAIL_SEC * SR)
        n = 0
        for i in range(0, max(len(audio), 1), step):
            seg = audio[i : i + step]
            if len(seg) < tail:
                break
            out = Path(out_dir) / f"ien_bg_{key}_{n:03d}.wav"
            sf.write(out, np.clip(seg, -1.0, 1.0), SR, subtype="PCM_16")
            n += 1
        return key, n, "ok"
    except Exception as exc:  # noqa: BLE001 - crowd-sourced audio; tolerate + report
        return key, 0, f"ERROR: {exc}"


def build(anno_csv: Path, audio_root: Path, out_dir: Path, workers: int) -> None:
    """Filter to reliable negatives, tile their mirrors into 10 s clips, write manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    passed, blocked = qualifying_keys(anno_csv)
    print(f"reliable-negative files: {len(passed)} (blocked {len(blocked)})", flush=True)
    blk_cats = sorted({c for cs in blocked.values() for c in cs if classify(c) != "allow"})
    print(f"  excluded-because-of categories: {blk_cats}", flush=True)

    # index every mirror by key
    idx: dict[str, tuple[str, str]] = {}
    for w in audio_root.rglob("*.wav"):
        idx[w.stem] = (str(w), w.parent.name)
    jobs = [(idx[k][0], str(out_dir), k, idx[k][1]) for k in passed if k in idx]
    print(f"  matched audio for {len(jobs)}/{len(passed)} keys", flush=True)

    rows, total, errs = [], 0, 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(_clip_one, j) for j in jobs]):
            key, n, status = fut.result()
            if status != "ok":
                errs += 1
                print(f"  {status} ({key})", flush=True)
                continue
            total += n
            rows.append({"key": key, "n_clips": n, "categories": "; ".join(passed.get(key, []))})
    pd.DataFrame(rows).to_csv(out_dir / "manifest.csv", index=False)
    print(f"done: {total} clips from {len(rows)} files, {errs} errors -> {out_dir}", flush=True)


def main() -> None:
    """Entry point — see module docstring."""
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="stage", required=True)
    pf = sub.add_parser("filter", help="dry-run: print the allow/block split only")
    pf.add_argument("--anno-csv", type=Path, required=True)
    pb = sub.add_parser("build")
    pb.add_argument("--anno-csv", type=Path, required=True)
    pb.add_argument("--audio-root", type=Path, required=True)
    pb.add_argument("--out-dir", type=Path, required=True)
    pb.add_argument("--workers", type=int, default=16)
    args = p.parse_args()
    if args.stage == "filter":
        passed, blocked = qualifying_keys(args.anno_csv)
        print(f"reliable-negative files: {len(passed)}  blocked: {len(blocked)}")
        for k, cs in blocked.items():
            bad = [c for c in cs if classify(c) != "allow"]
            print(f"  BLOCK {k}: {cs}  (because: {bad})")
    else:
        build(args.anno_csv, args.audio_root, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
