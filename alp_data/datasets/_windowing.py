"""Shared helpers for datasets that support windowed reads.

Several strong-detection datasets accept `window_start_sec` / `window_end_sec`
on a row and read only that segment (see `alp_data.datasets.dclde2026`, whose
manifest ships those columns). Their selection tables hold recording-absolute
times, so a windowed read has to re-base the table onto the returned audio or
the two disagree.
"""

from __future__ import annotations

import pandas as pd

BEGIN = "Begin Time (s)"
END = "End Time (s)"


def window_selection_table(st: pd.DataFrame, window_start: float, duration: float) -> pd.DataFrame:
    """Re-base a recording-absolute selection table onto a windowed read.

    Events that do not overlap the window are dropped; the rest are shifted so
    their times are relative to the start of the returned audio, and clipped to
    it. All other columns are left untouched, so identity columns such as
    `Selection` or `event_index` still say which of the recording's events a
    row refers to.

    Parameters
    ----------
    st : pd.DataFrame
        Selection table with recording-absolute `Begin Time (s)` and
        `End Time (s)`. A table missing those columns is returned unchanged.
    window_start : float
        Start of the window, in seconds from the start of the recording.
    duration : float
        Duration of the audio actually returned, in seconds. Used in place of
        the requested window end so a window running past the end of the file
        is handled correctly.

    Returns
    -------
    pd.DataFrame
        The windowed selection table, with times relative to the window.
    """
    if BEGIN not in st.columns or END not in st.columns:
        return st
    window_end = window_start + duration
    st = st[(st[END] > window_start) & (st[BEGIN] < window_end)].copy()
    st[BEGIN] = (st[BEGIN] - window_start).clip(lower=0.0)
    st[END] = (st[END] - window_start).clip(upper=duration)
    return st


def clip_selection_table(st: pd.DataFrame, duration: float) -> pd.DataFrame:
    """Drop events that begin past the end of the decoded audio.

    The guard WABAD and DCLDE2026 apply to unwindowed reads, kept here so every
    dataset in this family spells it the same way.

    Parameters
    ----------
    st : pd.DataFrame
        Selection table with recording-absolute times. A table missing
        `Begin Time (s)` is returned unchanged.
    duration : float
        Duration of the decoded audio, in seconds.

    Returns
    -------
    pd.DataFrame
        The clipped selection table.
    """
    if BEGIN not in st.columns:
        return st
    return st[st[BEGIN] < duration].copy()
