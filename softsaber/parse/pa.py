"""Collapse raw PBP into a clean PA-level table for downstream stats.

Output schema (one row per plate appearance):

    season, game_id, inning, top_bottom, batting_team, fielding_team,
    batter, outcome, rbi,
    state_before, state_after, runs_on_play,
    play_id

Rows where ``outcome`` is None are dropped (and logged) so the linear-weights
math never sees an unclassified event.
"""

from __future__ import annotations

import logging

import pandas as pd

from .baserunners import attach_base_out_state

log = logging.getLogger(__name__)


def build_pa_table(pbp_raw: pd.DataFrame) -> pd.DataFrame:
    if pbp_raw.empty:
        return pbp_raw

    enriched = attach_base_out_state(pbp_raw)

    cols = [
        "season",
        "game_id",
        "inning",
        "top_bottom",
        "batting_team",
        "fielding_team",
        "batter",
        "outcome",
        "rbi",
        "state_before",
        "state_after",
        "runs_on_play",
        "play_id",
    ]
    # ``rbi`` isn't parsed onto the row yet — events.classify carries it.
    # When the parse layer is finished, plumb it through baserunners.py.
    enriched["rbi"] = enriched.get("rbi", 0)

    pa = enriched.loc[enriched["outcome"].notna(), [c for c in cols if c in enriched.columns]]
    dropped = len(enriched) - len(pa)
    if dropped:
        log.info("PA build: dropped %d unclassified rows", dropped)
    return pa.reset_index(drop=True)
