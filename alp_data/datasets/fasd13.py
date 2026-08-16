"""FASD13 dataset"""

from __future__ import annotations

from io import StringIO
from typing import Any, Iterator

import librosa
import numpy as np
import pandas as pd

from alp_data import Dataset, DatasetConfig, DatasetInfo, register_dataset
from alp_data.backends import BackendType
from alp_data.io import DATA_HOME, AnyPathT, anypath, audio_stereo_to_mono, read_audio

_RAW_ROOT = f"{DATA_HOME}/fasd13/v0.1.0/raw"

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
    selection table without re-reading the annotations.

    Pre-resampled Audio
    -------------------
    Pre-resampled mono audio is available at 16 kHz and 32 kHz. When
    `sample_rate` matches one of these rates, the pre-resampled files are
    loaded directly (no on-the-fly resampling). Native rates range from 16 kHz
    to 96 kHz and some sources are stereo, so the mirrors are the only
    mono-normalised view; 32 kHz is the default and the recommended rate.

    Native-rate audio is not mirrored. The `audio_fp` column points at the
    32 kHz mirror, so any other target rate is resampled from 32 kHz on the fly
    using librosa's `kaiser_best` method, and the `native_sample_rate` column
    is informational only.

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
        owner="david",
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

    def _process(self, row: dict[str, Any]) -> dict[str, Any]:
        """Process a single row of the dataset.

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
