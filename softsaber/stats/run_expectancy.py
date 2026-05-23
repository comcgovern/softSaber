"""Run-expectancy matrix (RE24) construction.

Given a PA table with ``state_before`` (one of 24 base-out states) and
``runs_on_play``, plus the runs scored *for the remainder of the half-inning*,
we estimate the expected runs scored from each starting state.

Formula:
    For each row r in a half-inning:
        future_runs(r) = sum(runs_on_play[r:]) for all subsequent rows in
                         the same half-inning.
    RE[state] = mean(future_runs) over all rows with state_before == state.

Linear weights then fall out as::

    run_value(outcome) = mean(
        RE[state_after] - RE[state_before] + runs_on_play
    )

This module is the dependency for :mod:`softsaber.stats.linear_weights`.

Status: STUB. The math below is correct and concise; it just needs a
populated PA table to run against.
"""

from __future__ import annotations

import pandas as pd

ALL_STATES = [
    f"{b1}{b2}{b3}:{o}"
    for b1 in ("_", "1")
    for b2 in ("_", "2")
    for b3 in ("_", "3")
    for o in (0, 1, 2)
]


def compute_re_matrix(pa: pd.DataFrame) -> pd.Series:
    """Return RE per base-out state, indexed by ``state_before``.

    Restricts to fully-resolved half-innings (``half_inning_ok``) so that
    badly-parsed halves don't bias the matrix.
    """
    if pa.empty or "state_before" not in pa.columns:
        return pd.Series(dtype=float)

    pa = pa.copy()
    if "half_inning_ok" in pa.columns:
        pa = pa[pa["half_inning_ok"].astype(bool)]
    if pa.empty:
        return pd.Series(dtype=float)

    keys = ["game_id", "inning", "top_bottom"]
    pa = pa.sort_values([*keys, "play_id"]).reset_index(drop=True)

    # Reverse-cumulative-sum within each half-inning: ``future_runs[i]`` =
    # runs scored from play i to the end of the half-inning, inclusive. That
    # is exactly the value of being in ``state_before[i]``.
    pa["future_runs"] = 0.0
    for _, g in pa.groupby(keys, sort=False):
        idx = g.index
        rev_cum = g["runs_on_play"][::-1].cumsum()[::-1]
        pa.loc[idx, "future_runs"] = rev_cum.astype(float).values
    return pa.groupby("state_before")["future_runs"].mean()


def compute_re24(pa: pd.DataFrame, re: pd.Series) -> pd.DataFrame:
    """Add per-play RE24 (run value relative to state) to a PA table."""
    out = pa.copy()
    out["re_before"] = out["state_before"].map(re).fillna(0.0)
    out["re_after"] = out["state_after"].map(re).fillna(0.0)
    # End-of-inning states (3 outs) have RE = 0 by definition.
    out.loc[out["state_after"].str.endswith(":3"), "re_after"] = 0.0
    out["re24"] = out["re_after"] - out["re_before"] + out["runs_on_play"]
    return out
