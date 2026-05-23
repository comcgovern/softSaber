"""wOBA and wRC+ per player-season.

Formula (Fangraphs-style, adapted to the no-pitchers-batting environment of
NCAA softball, where the "non-pitcher" denominator collapses to the plain
league wRC/PA):

    wOBA       = sum_c weight_c * count_c / wOBA_denom
                 where weight_c is the scaled run value per event,
                       wOBA_denom = AB + BB - IBB + SF + HBP.
    wOBAscale  = lgOBP / lgwOBA_raw                  (forces lg wOBA == lg OBP)
    wRAA       = ((wOBA - lgwOBA) / wOBAscale) * PA
    wRC+       = (((wOBA - lgwOBA) / wOBAscale + lgR/PA)
                  / (park_factor * lgR/PA)) * 100

Inputs:
    pa            — PA-level table (one row per PA, with batter, outcome,
                    batting_team, season).
    weights       — output of :func:`linear_weights.compute_linear_weights`.
    park_factors  — output of :func:`park_factors.regress_park_factors` or
                    ``multi_year_park_factors``. One row per home_team_id
                    with column ``pf``. The PA table's ``batting_team``
                    field is used to look up each player's park factor —
                    callers should make sure that maps to the same id used
                    in the park-factors table (in v0.1, both are ``team``
                    strings from the scoreboard).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .linear_weights import WOBA_OUTCOMES

log = logging.getLogger(__name__)


OUTCOME_COLS = (
    "1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "K",
    "GO", "FO", "LO", "PO", "FC", "ROE", "SF", "SH", "DP",
)


def _woba_denom_count(outcome_counts: pd.Series) -> float:
    """wOBA denominator = AB + BB - IBB + SF + HBP = PA - IBB - SH."""
    pa = sum(outcome_counts.get(c, 0) for c in OUTCOME_COLS)
    ibb = outcome_counts.get("IBB", 0)
    sh = outcome_counts.get("SH", 0)
    return float(pa - ibb - sh)


def compute_player_seasons(pa: pd.DataFrame) -> pd.DataFrame:
    """Pivot PA-level rows to one row per (season, batter, team) with
    counts for each outcome plus PA / AB / H aggregates.
    """
    pa = pa.dropna(subset=["batter", "outcome"]).copy()
    counts = (
        pa.groupby(["season", "batter", "batting_team", "outcome"])
        .size()
        .unstack("outcome", fill_value=0)
        .reset_index()
    )
    # Make sure every canonical outcome column exists, even if zero.
    for c in ("1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "K",
              "GO", "FO", "LO", "PO", "FC", "ROE", "SF", "SH", "DP"):
        if c not in counts.columns:
            counts[c] = 0

    counts["PA"] = counts[[c for c in counts.columns if c not in (
        "season", "batter", "batting_team")]].sum(axis=1)
    counts["H"] = counts["1B"] + counts["2B"] + counts["3B"] + counts["HR"]
    counts["AB"] = counts["PA"] - counts["BB"] - counts["IBB"] - counts["HBP"] - counts["SH"] - counts["SF"]
    return counts


def _league_woba_scale(weights: pd.DataFrame, lg_counts: pd.Series) -> tuple[float, float]:
    """Return (lg_wOBA, wOBA_scale) so that lg_wOBA == lg_OBP."""
    # Raw weighted sum across the league (BB, HBP, 1B, 2B, 3B, HR).
    w = weights.set_index("outcome")["run_value_above_out"]
    raw_num = 0.0
    for code in WOBA_OUTCOMES:
        raw_num += w.get(code, 0.0) * lg_counts.get(code, 0)
    denom = _woba_denom_count(lg_counts)
    if denom <= 0:
        return 0.0, 1.0
    raw_woba = raw_num / denom

    # League OBP = (H + BB + IBB + HBP) / (AB + BB + IBB + HBP + SF).
    H = lg_counts.get("1B", 0) + lg_counts.get("2B", 0) + lg_counts.get("3B", 0) + lg_counts.get("HR", 0)
    BB = lg_counts.get("BB", 0) + lg_counts.get("IBB", 0)
    HBP = lg_counts.get("HBP", 0)
    SF = lg_counts.get("SF", 0)
    SH = lg_counts.get("SH", 0)
    PA = sum(lg_counts.get(c, 0) for c in OUTCOME_COLS)
    AB = PA - BB - HBP - SH - SF
    obp_denom = AB + BB + HBP + SF
    lg_obp = (H + BB + HBP) / obp_denom if obp_denom > 0 else 0.0

    scale = lg_obp / raw_woba if raw_woba > 0 else 1.0
    return lg_obp, scale


def player_wrc_plus(
    pa: pd.DataFrame,
    weights: pd.DataFrame,
    park_factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute wOBA, wRAA, and wRC+ per player-season.

    If ``park_factors`` is None or missing a team, that team gets pf=1.0.
    """
    if pa.empty:
        return pd.DataFrame()

    player = compute_player_seasons(pa)
    lg_counts = (
        player.drop(columns=["season", "batter", "batting_team"]).sum(axis=0)
    )

    lg_obp, woba_scale = _league_woba_scale(weights, lg_counts)
    w = weights.set_index("outcome")["run_value_above_out"] * woba_scale

    # League runs per PA from actual scored runs in the PA table. Falls back
    # to a wOBA-weighted estimate if ``runs_on_play`` is missing.
    lg_PA = int(sum(lg_counts.get(c, 0) for c in OUTCOME_COLS))
    if "runs_on_play" in pa.columns and lg_PA:
        lg_runs_per_pa = float(pa["runs_on_play"].sum()) / lg_PA
    elif lg_PA:
        lg_runs_per_pa = float(
            (lg_counts[list(WOBA_OUTCOMES)] * w.loc[list(WOBA_OUTCOMES)]).sum() / lg_PA
        )
    else:
        lg_runs_per_pa = 0.0

    # Player wOBA.
    woba_num = sum(player[c] * w.get(c, 0.0) for c in WOBA_OUTCOMES)
    woba_denom = player["PA"] - player["IBB"] - player["SH"]
    player["wOBA"] = np.where(woba_denom > 0, woba_num / woba_denom, 0.0)

    player["wRAA"] = ((player["wOBA"] - lg_obp) / woba_scale) * player["PA"] if woba_scale else 0.0

    # Park factor lookup. We accept either ``home_team_id`` or ``team_name``
    # as the join key; in v0.1 the scoreboard scrape gives us team-name
    # strings on both sides.
    if park_factors is not None and not park_factors.empty:
        pf_col = "home_team_id" if "home_team_id" in park_factors.columns else park_factors.columns[0]
        pf_map = park_factors.set_index(pf_col)["pf"]
        player["park_factor"] = player["batting_team"].map(pf_map).fillna(1.0)
    else:
        player["park_factor"] = 1.0

    denom = player["park_factor"] * lg_runs_per_pa
    player["wRC+"] = np.where(
        denom > 0,
        (((player["wOBA"] - lg_obp) / woba_scale + lg_runs_per_pa) / denom) * 100.0,
        np.nan,
    )

    return player.loc[:, [
        "season", "batter", "batting_team", "PA", "AB", "H",
        "1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "K", "SF", "SH",
        "wOBA", "wRAA", "park_factor", "wRC+",
    ]]
