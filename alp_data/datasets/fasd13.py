"""FASD13 dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import AnyPathT, anypath, audio_stereo_to_mono, read_audio

# Staged in the ingestion bucket. Moves under DATA_HOME (as WABAD and most
# datasets resolve) when this lands; the loader picks the new location up
# from this one constant.
_RAW_ROOT = "gs://esp-data-ingestion/fasd13/v0.1.0"

# Sub-dataset codes, in the order of the FASD13 summary table.
SUBDATASET_CODES = (
    "AS",
    "CC",
    "GS",
    "HA",
    "HG",
    "HW",
    "JS",
    "KD",
    "MS",
    "PM",
    "RG",
    "RS",
    "RW",
)


@register_dataset
class FASD13(Dataset):
    """FASD13 Dataset

    Description
    -----------
    FASD13 ("Fewshot Animal Sound Detection 13") is the evaluation benchmark
    accompanying *Synthetic data enables context-aware bioacoustic sound event
    detection* (Hoffman et al., 2025). It gathers 13 sub-datasets, 109
    recordings and roughly 143 hours of audio spanning anurans, birds,
    primates, whales, insects, spiders and gunshots, recorded through
    terrestrial and underwater passive acoustic monitoring, on-body recorders,
    substrate-borne recording and in the lab.

    Each entry is a full recording plus a selection table, in the same shape as
    the WABAD dataset. The benchmark is deliberately evaluation-only: it
    complements, and shares no audio with, the synthetic *training* corpus
    released by the same paper (DRASDIC).

    Two of the 13 sub-datasets (`CC`, Carrion Crow, and `JS`, Jumping Spider)
    are published for the first time in FASD13; the other 11 are re-releases.
    Note the extreme duration imbalance: `HG` (72 h) and `GS` (38 h) make up
    77 % of the benchmark between them, so cap or stratify per sub-dataset
    before pooling.

    Labels
    ------
    Every recording is annotated for a single, *nameless* target category. In
    few-shot detection the target is defined by the support examples rather
    than by a name, so the selection table's `Label` column is the constant
    `"target"`, and the row-level `detection_target` column records what kind
    of category the sub-dataset annotates (Species, Call Type, Sound Type,
    Production Mechanism, or Species+Life Stage).

    Selection tables carry `Selection`, `Begin Time (s)`, `End Time (s)`, `Q`,
    `Label` and `event_index`. `Q` is one of `POS`, `UNK` or `NEG`. Per the
    official protocol, `UNK` events are neither positives nor negatives and
    must be masked out of scoring. There are no frequency bounds -- FASD13 is
    time-only.

    N-shot protocol
    ---------------
    Following Nolasco et al. 2023, events are ordered by end time and `UNK`
    events are excluded from shot selection. An N-shot system is given the
    audio up through the end of the Nth event and must detect events in the
    remainder, which is the only region scored. Each row carries
    `n_shots_available` and `shot_end_times` (the end times of the first five
    non-`UNK` events), so any episode with N <= 5 can be derived from the
    selection table without re-reading the annotations. Together with the
    windowed reads described below, that is everything an episode builder
    needs, so episode construction itself stays outside this class.

    Windowed reads
    --------------
    A row carrying `window_start_sec` and `window_end_sec` reads only that
    segment, streamed from cloud storage rather than downloaded whole, and its
    selection table is re-based onto the window: non-overlapping events are
    dropped and the rest are shifted and clipped so their times line up with
    the returned audio. `event_index` is preserved, so a caller can still tell
    which of the recording's events a row refers to.

    This is what makes the benchmark usable without loading whole recordings.
    Several sub-datasets ship 8-hour files -- one `HG` item is ~3.7 GB of
    float32 at 32 kHz -- so cutting an N-shot support clip or chunking a query
    region has to happen at read time.

    Pre-resampled Audio
    -------------------
    Three views of every recording are stored. `audio_fp` is the original, as
    Zenodo ships it, at its native rate and channel count; `16khz_path` and
    `32khz_path` are pre-resampled mono mirrors. When `sample_rate` is 16000 or
    32000 the matching mirror is read directly with no on-the-fly resampling.
    Any other rate -- including `None`, which returns the audio at its native
    rate -- reads the original, resampling from it with librosa's `kaiser_best`
    method where needed. 32 kHz is the default.

    Native rates span 8 kHz to 187.5 kHz. What the mirrors cost you, measured
    rather than assumed:

    - Very little bandwidth, despite the rates. `CC` (46.875 / 187.5 kHz),
      `KD` (96 kHz) and `MS` (44.1 kHz) can represent content above the
      mirrors' Nyquist, but at annotated events under 0.3 % of their energy
      sits above 16 kHz -- 0.00 % for the single 187.5 kHz `CC` file. For
      detection the 32 kHz mirror is a faithful view of every sub-dataset.
    - `GS` (8 kHz) and `HG` (9.6 kHz) are *upsampled* into the mirrors, and
      they are 77 % of the benchmark by duration, so a 32 kHz read of most of
      FASD13 carries nothing above ~4-5 kHz. Nothing is lost by it, but do not
      mistake the rate for bandwidth.
    - `AS` and `HG` are stereo at source (21 recordings, 72.2 h). Reads are
      mono-averaged whatever the view, as in WABAD; the originals keep their
      channels for anyone who needs them.

    So read the originals for provenance, exact source bytes and channels --
    not in the expectation of extra bandwidth.

    Every `HA` recording is a FLAC bitstream carrying a `.wav` extension. Two
    of them (`Hawaii_UHH_494_S04_20190418_203000.wav` and
    `Hawaii_UHH_627_S02_20220323_100400.wav`) hold frames libsndfile cannot
    decode, so their originals are published transcoded to plain PCM by ffmpeg,
    which reads them in full. `HA` is 32 kHz mono at source, so this is a
    container change only: both transcodes are bit-identical to the published
    32 kHz mirror. The files exactly as Zenodo ships them are kept under
    `raw/HA/` for provenance.

    Licensing
    ---------
    Licensing is per sub-dataset rather than uniform: CC-BY-1.0, CC-BY-4.0,
    CC-BY-SA-4.0, CC-BY-NC-4.0, CC-BY-NC-SA-4.0, public domain, and one custom
    citation-required entry (`MS`). It is carried per row in the `license` and
    `citation` columns; see also `LICENSE.txt` alongside the manifests.

    References
    ----------
    [Zenodo](https://zenodo.org/records/15843741)
    [arXiv](https://arxiv.org/abs/2503.00296)

    """

    info = DatasetInfo(
        name="fasd13",
        owner="david; benjamin",
        split_paths={
            "all": f"{_RAW_ROOT}/fasd13_all.csv",
            **{code: f"{_RAW_ROOT}/fasd13_{code}.csv" for code in SUBDATASET_CODES},
        },
        version="0.1.0",
        description="FASD13 few-shot bioacoustic sound event detection benchmark: 13 "
        "sub-datasets, 109 recordings and roughly 143 hours of audio, each recording "
        "annotated with the onsets and offsets of a single predetermined target "
        "category. Spans anurans, birds, primates, whales, insects, spiders and "
        "gunshots. Evaluation-only benchmark from Hoffman et al. 2025.",
        sources=["https://zenodo.org/records/15843741", "https://arxiv.org/abs/2503.00296"],
        license="Mixed per sub-dataset; see the per-row license/citation columns.",
    )

    _sample_rate_paths: dict[int, str] = {16000: "16khz_path", 32000: "32khz_path"}
    _originals_path_column = "audio_fp"

    def __init__(
        self,
        split: str = "all",
        output_take_and_give: dict[str, str] | None = None,
        sample_rate: int | None = 32000,
        data_root: str | AnyPathT | None = None,
        backend: BackendType = "polars",
        streaming: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        split : str
            Split to load (key in info.split_paths): `"all"` or one of the 13
            sub-dataset codes (`AS`, `CC`, `GS`, `HA`, `HG`, `HW`, `JS`, `KD`,
            `MS`, `PM`, `RG`, `RS`, `RW`).
        output_take_and_give : dict[str, str] | None
            Optional mapping of original → new output keys (filters columns as well).
        sample_rate : int | None
            If set, audio is resampled to this rate. 16000 and 32000 read the
            pre-resampled mirrors directly.
        data_root : str | AnyPathT | None
            Optional root directory to prepend to each row's audio path.
        backend : BackendType, optional
            The backend to use ("pandas" or "polars"), by default "polars"
        streaming : bool, optional
            Whether to use streaming mode, by default False
        """
        super().__init__(output_take_and_give, backend=backend, streaming=streaming)
        self.split = split
        self._data = None
        self.annotation_columns = ["Label"]
        self.sample_rate = sample_rate

        self._load()

        if data_root is None:
            self.data_root = anypath(self.info.split_paths[self.split]).parent
        else:
            self.data_root = anypath(data_root)

    @property
    def columns(self) -> list[str]:
        return list(self._data.columns) if self._data is not None else []

    @property
    def available_splits(self) -> list[str]:
        return list(self.info.split_paths.keys())

    @property
    def available_sample_rates(self) -> list[int]:
        """Return pre-resampled sample rates whose path columns exist in the data."""
        return [sr for sr, col in self._sample_rate_paths.items() if col in self._data.columns]

    def _load(self) -> None:
        if self.split not in self.info.split_paths:
            raise LookupError(
                f"Invalid split: {self.split}. Expected one of {list(self.info.split_paths.keys())}"
            )
        location = self.info.split_paths[self.split]
        self._data = self._backend_class.from_csv(
            location,
            streaming=self._streaming,
            keep_default_na=False,
            na_values=[""],
        )

    def __len__(self) -> int:
        """Return the number of recordings in the dataset.

        Returns
        -------
        int
            Number of recordings in the current split.

        Raises
        ------
        RuntimeError
            If no split has been loaded yet.
        NotImplementedError
            If the dataset is in streaming mode.
        """
        if self._data is None:
            raise RuntimeError("No split has been loaded yet. Call _load() first.")
        if self._streaming:
            raise NotImplementedError(
                "Length is not available in streaming mode.Iterate over the dataset instead."
            )
        return len(self._data)

    @staticmethod
    def _window_selection_table(
        st: pd.DataFrame, window_start: float, duration: float
    ) -> pd.DataFrame:
        """Re-base a recording-absolute selection table onto a windowed read.

        Events that do not overlap the window are dropped; the rest are shifted
        so their times are relative to the start of the returned audio, and
        clipped to it. `event_index` is left untouched so a caller can still
        tell which of the recording's events a row refers to, which is what the
        N-shot protocol needs.

        Parameters
        ----------
        st : pd.DataFrame
            Selection table with recording-absolute `Begin Time (s)` and
            `End Time (s)`.
        window_start : float
            Start of the window, in seconds from the start of the recording.
        duration : float
            Duration of the audio actually returned, in seconds. Used instead
            of the requested window end so that a window running past the end
            of the file is handled correctly.

        Returns
        -------
        pd.DataFrame
            The windowed selection table, with times relative to the window.
        """
        window_end = window_start + duration
        st = st[(st["End Time (s)"] > window_start) & (st["Begin Time (s)"] < window_end)].copy()
        st["Begin Time (s)"] = (st["Begin Time (s)"] - window_start).clip(lower=0.0)
        st["End Time (s)"] = (st["End Time (s)"] - window_start).clip(upper=duration)
        return st

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

        When the row carries `window_start_sec` and `window_end_sec`, only that
        segment is read and the selection table is re-based onto it. See the
        class docstring.

        Parameters
        ----------
        row : dict[str, Any]
            A dictionary representing a single row of the dataset.

        Returns
        -------
        dict[str, Any]
            The processed row.
        """
        use_presampled = False
        if self.sample_rate is not None and self.sample_rate in self._sample_rate_paths:
            path_column = self._sample_rate_paths[self.sample_rate]
            if path_column in row and row[path_column] is not None and row[path_column] != "":
                audio_path = anypath(self.data_root) / row[path_column]
                use_presampled = True

        if not use_presampled:
            audio_path = anypath(self.data_root) / row[self._originals_path_column]

        window_start = row.get("window_start_sec")
        window_end = row.get("window_end_sec")

        if window_start is not None and window_end is not None:
            audio, sr = read_audio(
                audio_path, start_time=float(window_start), end_time=float(window_end)
            )
        else:
            audio, sr = read_audio(audio_path)
        audio = audio_stereo_to_mono(audio, mono_method="average").astype(np.float32)

        if not use_presampled and self.sample_rate is not None and sr != self.sample_rate:
            audio = librosa.resample(
                y=audio,
                orig_sr=sr,
                target_sr=self.sample_rate,
                scale=True,
                res_type="kaiser_best",
            )
            sr = self.sample_rate

        st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
        audio_duration = len(audio) / float(sr)
        if window_start is not None and window_end is not None:
            st = self._window_selection_table(st, float(window_start), audio_duration)
        else:
            # Same guard WABAD and DCLDE2026 apply: drop events that begin past
            # the end of the audio. A no-op on the shipped manifests -- no event
            # in the benchmark overhangs its recording -- but it keeps the
            # unwindowed path identical to its siblings.
            st = st[st["Begin Time (s)"] < audio_duration].copy()

        row["audio"] = audio
        row["sample_rate"] = sr
        row["selection_table"] = st

        if self.output_take_and_give:
            item = {}
            for old_key, new_key in self.output_take_and_give.items():
                item[new_key] = row[old_key]
            return item

        return row

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a specific sample from the dataset.

        Parameters
        ----------
        idx : int
            Index of the sample to get.

        Returns
        -------
        dict[str, Any]
            A dictionary containing the audio, sample rate, selection table and
            the recording's metadata columns.
        """
        row = self._data[idx]
        return self._process(row)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Iterate over samples in the dataset.

        Yields
        -------
        dict[str, Any]
            Each sample in the dataset.
        """
        for row in self._data:
            yield self._process(row)

    @classmethod
    def from_config(cls, dataset_config: DatasetConfig) -> tuple["FASD13", dict[str, Any]]:
        """Create a Dataset instance from a configuration dictionary.

        Parameters
        ----------
        dataset_config : DatasetConfig
            Configuration dictionary containing dataset parameters.

        Returns
        -------
        tuple[Dataset, dict[str, Any]]
            A tuple containing the dataset instance and metadata.
            If the dataset_config contains transformations, they will be applied
            and the metadata will be returned as dict, otherwise an empty dict.
        """
        cfg = dataset_config.model_dump(exclude={"dataset_name", "transformations"})
        ds = cls(
            split=cfg["split"],
            output_take_and_give=cfg["output_take_and_give"],
            data_root=cfg["data_root"],
            sample_rate=cfg["sample_rate"],
            backend=cfg["backend"],
            streaming=cfg["streaming"],
        )

        if dataset_config.transformations:
            meta = ds.apply_transformations(dataset_config.transformations)
            return ds, meta

        return ds, {}

    def get_available_labels(self, anno_column: str | None = "Label") -> list[str]:
        """Return the label vocabulary of the selection tables.

        FASD13 has a single, nameless positive class per recording, so this is
        always `["target"]`. Use the `detection_target` column for the kind of
        category each sub-dataset annotates.

        Parameters
        ----------
        anno_column : str | None
            Annotation column to read; only `"Label"` is meaningful here.

        Returns
        -------
        list[str]
            A sorted list of all the available labels for anno_column.
        """
        available_labels = set()
        for row in self._data:
            st = pd.read_csv(StringIO(row["selection_table"]), sep="\t")
            if anno_column in st.columns:
                available_labels.update(st[anno_column].astype(str).tolist())
        return sorted(available_labels)

    def __str__(self) -> str:
        base = f"{self.info.name} (v{self.info.version})"
        return (
            f"{base}\n"
            f"Sources: {self.info.sources}\n"
            f"License: {self.info.license}\n"
            f"Available splits: {', '.join(self.info.split_paths.keys())}"
        )
