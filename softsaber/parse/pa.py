"""Collapse raw PBP into a clean PA-level table for downstream stats.

Output schema (one row per plate appearance):

    season, game_id, inning, top_bottom, batting_team, fielding_team,
    batter, outcome, rbi,
    state_before, state_after, runs_on_play,
    half_inning_ok, play_id
"""

from __future__ import annotations

import logging

import pandas as pd

from .baserunners import attach_base_out_state

log = logging.getLogger(__name__)

CANONICAL_COLUMNS = [
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
    "half_inning_ok",
    "play_id",
]


def build_pa_table(pbp_raw: pd.DataFrame) -> pd.DataFrame:
    if pbp_raw.empty:
        return pbp_raw

    enriched = attach_base_out_state(pbp_raw)
    pa = enriched.loc[enriched["outcome"].notna(), :].copy()
    dropped = len(enriched) - len(pa)
    if dropped:
        log.info("PA build: dropped %d unclassified rows", dropped)

    # Ensure every canonical column exists (some come from PBP, some from
    # reconstruction). Keep extras for debugging.
    for col in CANONICAL_COLUMNS:
        if col not in pa.columns:
            pa[col] = None
    return pa.reset_index(drop=True)
