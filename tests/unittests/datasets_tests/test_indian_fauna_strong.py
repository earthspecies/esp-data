"""
Unit tests for IndianFaunaStrong dataset.

Run with:
    pytest -q test_indian_fauna_strong.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alp_data import DatasetConfig
from alp_data.datasets import IndianFaunaStrong
from alp_data.utils import create_hash


EXPECTED_LEN_ALL = 1702
EXPECTED_LEN_SMALL = 1702          # split 'all'
FIRST_ITEM_AUDIO_SHA256 = (
    "9820645e5fc74f70943c7336fe143c0dc2748f10dbbbf5d5ef85a8933785c6c2"
)
MANIFEST_SHA256 = "5a21c6cc34739de4d60e8b72b555627874e180e1d3c82b675e5758a7553dc4d5"

# Shared by every strongly-labeled soundscape dataset in this family.
CORE_ST_COLUMNS = ["Begin Time (s)", "End Time (s)", "Low Freq (Hz)", "High Freq (Hz)", "Species"]
EXTRA_ST_COLUMNS = ["Selection"]
AUDIO_FP_PREFIX = 'audio_32k/'
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def ds_all() -> IndianFaunaStrong:
    """Full manifest (metadata-level assertions; avoid bulk audio here)."""
    return IndianFaunaStrong(split="all", backend="pandas")


@pytest.fixture(scope="module")
def ds() -> IndianFaunaStrong:
    """Smallest split, used for every audio-level test."""
    return IndianFaunaStrong(split='all', sample_rate=32000, backend="pandas")


def test_lengths(ds_all: IndianFaunaStrong, ds: IndianFaunaStrong):
    assert len(ds_all) == EXPECTED_LEN_ALL
    assert len(ds) == EXPECTED_LEN_SMALL


def test_available_splits(ds_all: IndianFaunaStrong):
    assert "all" in ds_all.available_splits
    assert set(ds_all.available_splits) == {'all'}


def test_invalid_split_raises():
    with pytest.raises(LookupError):
        IndianFaunaStrong(split="not_a_split")


def test_check_audio(ds: IndianFaunaStrong):
    """Audio integrity on deterministic indices."""
    for idx in (0, 1, 2):
        item = ds[idx]
        audio = item["audio"]
        assert isinstance(audio, np.ndarray), f"[{idx}] audio is not a numpy array"
        assert audio.dtype == np.float32, f"[{idx}] dtype is {audio.dtype}"
        assert audio.ndim == 1, f"[{idx}] not mono, shape={audio.shape}"
        assert audio.size >= 10, f"[{idx}] too short"
        assert not np.any(np.isnan(audio)), f"[{idx}] contains NaN"
        assert not np.all(audio == 0), f"[{idx}] all zeros"
        assert item["sample_rate"] == 32000


def test_presampled_columns_exist(ds: IndianFaunaStrong):
    assert "16khz_path" in ds.columns
    assert "32khz_path" in ds.columns
    assert ds.available_sample_rates == [16000, 32000]


def test_load_presampled_16khz(ds: IndianFaunaStrong):
    """16 kHz reads the 16 kHz mirror: exactly half the samples of the 32 kHz read."""
    ds_16 = IndianFaunaStrong(split='all', sample_rate=16000, backend="pandas")
    item_16 = ds_16[0]
    assert item_16["sample_rate"] == 16000
    # Independently resampled mirrors, so allow a sample of rounding slack.
    assert abs(ds[0]["audio"].size - 2 * item_16["audio"].size) <= 2


def test_audio_fp_target(ds_all: IndianFaunaStrong):
    """`audio_fp` should point where this dataset's originals actually live."""
    assert ds_all._data.unwrap["audio_fp"].str.startswith(AUDIO_FP_PREFIX).all()


def test_annotation_columns(ds_all: IndianFaunaStrong):
    """The family contract: Species is the annotation column."""
    assert ds_all.annotation_columns == ["Species"]


def test_selection_table_schema(ds: IndianFaunaStrong):
    """Raven-shaped table with the columns this family guarantees."""
    for idx in (0, 1, 2):
        item = ds[idx]
        st = item["selection_table"]
        assert isinstance(st, pd.DataFrame), f"[{idx}] not a DataFrame"
        missing = set(CORE_ST_COLUMNS) - set(st.columns)
        assert not missing, f"[{idx}] missing {sorted(missing)}"
        for col in EXTRA_ST_COLUMNS:
            assert col in st.columns, f"[{idx}] missing {col}"
        if len(st):
            assert (st["End Time (s)"] >= st["Begin Time (s)"]).all()
            duration = item["audio"].size / item["sample_rate"]
            assert st["End Time (s)"].max() <= duration + 1.0
            assert st["Species"].astype(str).str.len().gt(0).all()


def test_n_events_matches_table(ds: IndianFaunaStrong):
    for idx in (0, 1, 2):
        item = ds[idx]
        assert len(item["selection_table"]) == int(item["n_events"])


def _windowed(ds: IndianFaunaStrong, idx: int, start: float, end: float) -> dict:
    """Process a row as if a caller had attached window columns."""
    row = dict(ds._data[idx])
    row["window_start_sec"], row["window_end_sec"] = start, end
    return ds._process(row)


def test_windowed_read_returns_only_the_window(ds: IndianFaunaStrong):
    item = _windowed(ds, 0, 5.0, 15.0)
    assert abs(item["audio"].size / item["sample_rate"] - 10.0) < 0.05
    assert item["audio"].size < ds[0]["audio"].size


def test_windowed_selection_table_is_window_relative(ds: IndianFaunaStrong):
    """
    Events must be filtered to the window and shifted onto the returned audio.

    Regression guard: these datasets previously clipped with
    `Begin Time (s) < audio_dur` against the *window* length while leaving times
    recording-absolute, which silently dropped every event in a non-zero window.
    """
    full_st = ds[0]["selection_table"]
    candidates = full_st[full_st["Begin Time (s)"] >= 2.0]
    if candidates.empty:
        pytest.skip("first recording has no event beyond 2 s")
    event = candidates.iloc[0]
    start = float(event["Begin Time (s)"]) - 2.0
    st = _windowed(ds, 0, start, start + 6.0)["selection_table"]

    assert len(st) > 0, "window was built around an event, so it cannot be empty"
    assert (st["Begin Time (s)"] >= 0).all()
    assert (st["End Time (s)"] <= 6.0 + 1e-6).all()
    assert np.isclose(st["Begin Time (s)"], 2.0, atol=1e-6).any(), (
        "the event the window was built around is not at its expected offset"
    )
    assert len(st) < len(full_st), "no events were filtered out of the window"


def test_unwindowed_selection_table_stays_absolute(ds: IndianFaunaStrong):
    item = ds[0]
    st = item["selection_table"]
    assert len(st) == int(item["n_events"]), "unwindowed read dropped events"


def test_backends_agree(ds: IndianFaunaStrong):
    ds_polars = IndianFaunaStrong(split='all', sample_rate=32000, backend="polars")
    assert len(ds_polars) == len(ds)
    assert np.array_equal(ds_polars[0]["audio"], ds[0]["audio"])


def test_from_config():
    config = DatasetConfig(
        dataset_name='indian_fauna_strong', split='all', sample_rate=32000, backend="pandas"
    )
    ds_cfg, meta = IndianFaunaStrong.from_config(config)
    assert isinstance(ds_cfg, IndianFaunaStrong)
    assert meta == {}
    assert len(ds_cfg) == EXPECTED_LEN_SMALL


def test_output_take_and_give(ds: IndianFaunaStrong):
    ds_mapped = IndianFaunaStrong(
        split='all',
        sample_rate=32000,
        backend="pandas",
        output_take_and_give={"audio": "waveform", "selection_table": "events"},
    )
    item = ds_mapped[0]
    assert set(item.keys()) == {"waveform", "events"}
    assert isinstance(item["waveform"], np.ndarray)


def test_reference_item_stability(ds: IndianFaunaStrong):
    """
    Canonical item (index 0 of the smallest split) is bitwise-stable.

    Catches sample-rate, channel-handling, dtype and split-ordering changes.
    Recompute and update the constants if a dataset revision is intentional.
    """
    audio = ds[0]["audio"]
    h = create_hash(audio.tobytes())
    assert h == FIRST_ITEM_AUDIO_SHA256, (
        f"First item's audio hash changed.\nGot    {h}\nExpect {FIRST_ITEM_AUDIO_SHA256}"
    )
    csv_bytes = (
        ds._data.unwrap.sort_index(axis=0).sort_index(axis=1).to_csv(index=True).encode("utf-8")
    )
    h = create_hash(csv_bytes)
    assert h == MANIFEST_SHA256, (
        f"Manifest hash changed.\nGot    {h}\nExpect {MANIFEST_SHA256}"
    )


def test_str(ds_all: IndianFaunaStrong):
    text = str(ds_all)
    assert 'indian_fauna_strong' in text
    assert "0.1.0" in text
