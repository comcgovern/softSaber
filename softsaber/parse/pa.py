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


def resolve_batter_names(pa: pd.DataFrame, game_players: pd.DataFrame) -> pd.DataFrame:
    """Replace raw PBP batter strings with proper-case player names.

    Uses nameutil.match_player to join each PA row's raw batter token
    (e.g. ``"KNIGHT, S"``) to the matching row in ``game_players``,
    replacing ``batter`` with the ``player_name`` value (e.g. ``"Shelby Knight"``).
    Rows that cannot be resolved keep their original batter string.

    Adds a boolean column ``batter_resolved`` flagging successful lookups.

    ``game_players`` must have: ``game_id``, ``team_id``, ``first_name``,
    ``last_name``, ``player_name``, ``starter``.
    """
    from .nameutil import match_player

    if pa.empty:
        return pa
    if game_players.empty:
        out = pa.copy()
        out["batter_resolved"] = False
        return out

    # Pre-index by (game_id, team_id) for batting-team-scoped lookup and by
    # game_id alone as a fallback (Shape-A PBP has no batting_team_id).
    gp_by_game_team: dict[tuple[str, str], pd.DataFrame] = {}
    gp_by_game: dict[str, pd.DataFrame] = {}
    for gid, grp in game_players.groupby("game_id"):
        gp_by_game[str(gid)] = grp.reset_index(drop=True)
        for tid, tgrp in grp.groupby("team_id"):
            gp_by_game_team[(str(gid), str(tid))] = tgrp.reset_index(drop=True)

    out = pa.copy()
    names: list[str | None] = []
    flags: list[bool] = []
    has_tid = "batting_team_id" in out.columns

    for r in out.itertuples(index=False):
        raw: str | None = getattr(r, "batter", None)
        if not raw:
            names.append(raw)
            flags.append(False)
            continue

        gid = str(r.game_id)
        tid = str(getattr(r, "batting_team_id", "") or "") if has_tid else ""

        team_players = gp_by_game_team.get((gid, tid)) if tid else None
        players = team_players if team_players is not None else gp_by_game.get(gid)

        if players is None:
            names.append(raw)
            flags.append(False)
            continue

        hit = match_player(raw, players)
        if hit is not None:
            names.append(str(hit["player_name"]))
            flags.append(True)
        else:
            names.append(raw)
            flags.append(False)

    out["batter"] = names
    out["batter_resolved"] = flags
    n = sum(flags)
    log.info(
        "resolve_batter_names: resolved %d/%d batters (%.0f%%)",
        n, len(out), 100 * n / len(out) if len(out) else 0,
    )
    return out


__all__ = ["build_pa_table", "resolve_batter_names", "CANONICAL_COLUMNS"]
