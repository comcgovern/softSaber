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


def resolve_batter_names(
    pa: pd.DataFrame,
    game_players: pd.DataFrame,
    rosters: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Replace raw PBP batter strings with proper-case player names.

    Uses nameutil.match_player to join each PA row's raw batter token
    (e.g. ``"KNIGHT, S"``) to the matching row in ``game_players``,
    replacing ``batter`` with the ``player_name`` value (e.g. ``"Shelby Knight"``).
    Rows that cannot be resolved keep their original batter string.

    Adds a boolean column ``batter_resolved`` flagging successful lookups.

    ``game_players`` must have: ``game_id``, ``team_id``, ``first_name``,
    ``last_name``, ``player_name``, ``starter``.

    ``rosters`` is the optional season-level roster table (richer than
    ``game_players``: it has GameCenter-upgraded first names for the
    ~12% of boxscore entries that arrive as a first-initial or empty).
    When provided, each batter is looked up there first via
    ``(team_id, last_name, jersey)``; we fall back to ``game_players``
    only if the rosters table can't resolve the name.
    """
    from .nameutil import match_player

    if pa.empty:
        return pa
    if game_players.empty and (rosters is None or rosters.empty):
        out = pa.copy()
        out["batter_resolved"] = False
        return out

    # Prepare an optional roster lookup scoped by (team_id) — rosters are
    # season-level so we don't filter by game_id.  We give roster rows a
    # ``starter`` column so match_player's tie-breaker still works.
    roster_by_team: dict[str, pd.DataFrame] = {}
    if rosters is not None and not rosters.empty and "team_id" in rosters.columns:
        prepped = rosters.copy()
        if "starter" not in prepped.columns:
            prepped["starter"] = False
        # NCAA roster pages render names as "Last, First"; split into the
        # first_name/last_name columns match_player expects.  Fall back to
        # "First Last" via rsplit when no comma is present.
        if "last_name" not in prepped.columns or "first_name" not in prepped.columns:
            split = prepped["player_name"].fillna("").str.split(",", n=1, expand=True)
            last = split[0].fillna("").str.strip()
            if split.shape[1] > 1:
                first = split[1].fillna("").str.strip()
            else:
                first = pd.Series("", index=prepped.index, dtype=str)
            no_comma = first == ""
            if no_comma.any():
                alt = (
                    prepped.loc[no_comma, "player_name"]
                    .fillna("")
                    .str.rsplit(" ", n=1, expand=True)
                )
                if alt.shape[1] == 2:
                    first.loc[no_comma] = alt[0].fillna("").str.strip()
                    last.loc[no_comma] = alt[1].fillna("").str.strip()
            prepped["first_name"] = first
            prepped["last_name"] = last
        if "player_name" not in prepped.columns:
            prepped["player_name"] = (
                prepped["first_name"].fillna("") + " " + prepped["last_name"].fillna("")
            ).str.strip()
        for tid, grp in prepped.groupby("team_id"):
            roster_by_team[str(tid)] = grp.reset_index(drop=True)

    # Pre-index by (game_id, team_id) for batting-team-scoped lookup and by
    # game_id alone as a fallback (Shape-A PBP has no batting_team_id).
    gp_by_game_team: dict[tuple[str, str], pd.DataFrame] = {}
    gp_by_game: dict[str, pd.DataFrame] = {}
    # Bridge: PA's batting_team_id is the numeric henrygd teamId, but
    # rosters.team_id stores the seoname slug ("elon", "valparaiso") it
    # inherited from the casablanca scoreboard.  game_players carries both
    # columns, so use it to translate numeric → seoname when looking up
    # rosters by team.
    team_seo_by_id: dict[str, str] = {}
    if not game_players.empty:
        for gid, grp in game_players.groupby("game_id"):
            gp_by_game[str(gid)] = grp.reset_index(drop=True)
            for tid, tgrp in grp.groupby("team_id"):
                gp_by_game_team[(str(gid), str(tid))] = tgrp.reset_index(drop=True)
        if "team_seoname" in game_players.columns:
            for tid, seo in zip(
                game_players["team_id"].astype(str),
                game_players["team_seoname"].astype(str),
            ):
                if tid and seo and tid not in team_seo_by_id:
                    team_seo_by_id[tid] = seo

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

        # Try the richer season-roster table first when available.
        # Rosters are keyed by seoname; translate from numeric tid.
        hit = None
        roster_key = team_seo_by_id.get(tid, tid) if tid else ""
        if roster_key and roster_key in roster_by_team:
            hit = match_player(raw, roster_by_team[roster_key])

        if hit is None:
            team_players = gp_by_game_team.get((gid, tid)) if tid else None
            players = team_players if team_players is not None else gp_by_game.get(gid)
            if players is not None:
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
