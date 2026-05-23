"""Reconstruct base-out state from PBP text, half-inning by half-inning.

This is the layer that lets us compute RE24 (and therefore principled linear
weights) without explicit baserunner tracking. The idea:

  * Walk a half-inning's events in order.
  * Maintain a tuple ``(runner_1B, runner_2B, runner_3B, outs)`` of runner
    names + outs.
  * For each event line, parse:
      - the batter's own outcome (uses :mod:`softsaber.parse.events`)
      - any embedded runner movement clauses, separated by ``;``, e.g.::
            "Smith singled to right; Jones to second; Brown scored."
  * Update bases/outs. Record the ``(bases_before, outs_before)`` state and
    ``runs_scored_on_play`` on each PA row — those are the inputs for
    run-expectancy estimation.

We do not have to get this perfect to be useful. As long as we resolve
≥95% of half-innings to a consistent end state (3 outs OR walk-off), the
remainder can be flagged and dropped from the RE24 training set without
biasing the model.

Status: STUB. The full implementation lives in :func:`reconstruct_half_inning`
below; right now it's a skeleton showing the data shape and clause-parsing
hooks. To finish:

1. Decide canonical base/outs representation. Suggest:
   ``BasesOuts = (occ_1B: bool, occ_2B: bool, occ_3B: bool, outs: int)``
   — runner identity isn't needed for RE24, just occupancy.
2. Implement ``_apply_event`` for each outcome code in
   :data:`softsaber.parse.events.OUTCOMES`.
3. Implement ``_apply_runner_clause`` for the ``; Name to second/scored``
   movement fragments.
4. Validate by replaying a season's PBP and checking that
   ``runs_scored_in_half_inning`` matches the score-diff between the first
   and last row of each half (cheap consistency check we already have data
   for from the score column).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from .events import classify

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasesOuts:
    """Occupancy-only base-out state. 24 possible values (8 base × 3 outs)."""

    occ_1B: bool = False
    occ_2B: bool = False
    occ_3B: bool = False
    outs: int = 0

    def as_state_id(self) -> str:
        """Stringified state for use as a dict key, e.g. ``'_2_:1'``."""
        bases = (
            ("1" if self.occ_1B else "_")
            + ("2" if self.occ_2B else "_")
            + ("3" if self.occ_3B else "_")
        )
        return f"{bases}:{self.outs}"


def reconstruct_half_inning(rows: pd.DataFrame) -> pd.DataFrame:
    """Augment one half-inning's PBP rows with base-out state and runs scored.

    ``rows`` must be sorted in play order. Adds columns:
        state_before, state_after, runs_on_play, batter, outcome

    STUB: returns the rows unchanged with placeholder columns. Implementation
    plan is documented in the module docstring.
    """
    out = rows.copy()
    out["batter"] = None
    out["outcome"] = None
    out["state_before"] = None
    out["state_after"] = None
    out["runs_on_play"] = 0

    # Pass 1: classify each row's outcome (cheap, deterministic).
    for i in out.index:
        ev = classify(str(out.at[i, "events"]))
        out.at[i, "batter"] = ev.batter
        out.at[i, "outcome"] = ev.outcome

    # Pass 2: simulate base-out state. TODO — see module docstring.
    return out


def attach_base_out_state(pbp_raw: pd.DataFrame) -> pd.DataFrame:
    """Apply :func:`reconstruct_half_inning` to every half-inning in a PBP frame."""
    if pbp_raw.empty:
        return pbp_raw
    keys = ["game_id", "inning", "top_bottom"]
    sorted_pbp = pbp_raw.sort_values([*keys, "row_idx"])
    out = (
        sorted_pbp.groupby(keys, group_keys=False)
        .apply(reconstruct_half_inning)
        .reset_index(drop=True)
    )
    return out
