"""Batter and pitcher rate statistics from the PA table.

These are the "counting → rate" derivations that sit alongside wRC+:
classic slash lines, plate-discipline rates, and batted-ball mix.

Two important data caveats baked into the column naming:

* **Batted-ball mix is OUTS-ONLY.**  NCAA PBP records the trajectory of
  *outs* (ground/fly/line/pop) but not of hits — a single "to right
  field" doesn't say whether it was a liner or a grounder.  So GB/FB/LD
  percentages here are over balls-in-play that became outs, and are
  suffixed ``_bbo`` (batted-ball-outs) to flag that.  They're useful for
  *relative* comparison within this dataset, not directly comparable to
  Fangraphs batted-ball rates.

* **HR/FB uses fly-ball-OUTS as the denominator** for the same reason,
  named ``hr_per_fbo``.  It runs high versus true HR/FB because fly-ball
  hits (non-HR) aren't counted in the denominator.

Pitcher innings-based stats (ERA, WHIP, K/9, BB/9) come from the
boxscore ``game_players`` partition, which carries the official IP/ER
line.  Everything else is derived from PBP outcomes attributed to each
pitcher by :mod:`softsaber.parse.pitcher`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# All canonical outcomes that count as a plate appearance.
PA_OUTCOMES = (
    "1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "K",
    "GO", "FO", "LO", "PO", "FC", "ROE", "SF", "SH", "DP",
)

# Batted-ball-out trajectory groupings (outs only — see module docstring).
_GB_CODES = ("GO",)
_FB_CODES = ("FO", "PO", "SF")
_LD_CODES = ("LO",)


def _pivot_outcome_counts(pa: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Group PA rows by ``group_cols`` and pivot ``outcome`` into one
    column per canonical outcome (zero-filled)."""
    counts = (
        pa.groupby([*group_cols, "outcome"])
        .size()
        .unstack("outcome", fill_value=0)
        .reset_index()
    )
    for c in PA_OUTCOMES:
        if c not in counts.columns:
            counts[c] = 0
    return counts


def _add_slash_and_discipline(df: pd.DataFrame) -> pd.DataFrame:
    """Add PA/AB/H aggregates plus slash line, plate-discipline, and
    batted-ball-out columns.  Operates in place and returns ``df``."""
    df["PA"] = df[list(PA_OUTCOMES)].sum(axis=1)
    df["H"] = df["1B"] + df["2B"] + df["3B"] + df["HR"]
    df["BB_total"] = df["BB"] + df["IBB"]
    df["AB"] = (
        df["PA"] - df["BB_total"] - df["HBP"] - df["SH"] - df["SF"]
    )
    df["TB"] = df["1B"] + 2 * df["2B"] + 3 * df["3B"] + 4 * df["HR"]
    df["XBH"] = df["2B"] + df["3B"] + df["HR"]

    ab = df["AB"].replace(0, np.nan)
    pa = df["PA"].replace(0, np.nan)

    df["AVG"] = (df["H"] / ab).fillna(0.0)
    obp_denom = (df["AB"] + df["BB_total"] + df["HBP"] + df["SF"]).replace(0, np.nan)
    df["OBP"] = ((df["H"] + df["BB_total"] + df["HBP"]) / obp_denom).fillna(0.0)
    df["SLG"] = (df["TB"] / ab).fillna(0.0)
    df["OPS"] = df["OBP"] + df["SLG"]
    df["ISO"] = df["SLG"] - df["AVG"]

    babip_denom = (df["AB"] - df["K"] - df["HR"] + df["SF"]).replace(0, np.nan)
    df["BABIP"] = ((df["H"] - df["HR"]) / babip_denom).fillna(0.0)

    df["K_pct"] = (df["K"] / pa).fillna(0.0)
    df["BB_pct"] = (df["BB_total"] / pa).fillna(0.0)
    df["HBP_pct"] = (df["HBP"] / pa).fillna(0.0)
    df["BB_K"] = (df["BB_total"] / df["K"].replace(0, np.nan)).fillna(0.0)
    df["XBH_pct"] = (df["XBH"] / pa).fillna(0.0)

    # Batted-ball-out mix (see module docstring for the outs-only caveat).
    gb = df[list(_GB_CODES)].sum(axis=1)
    fb = df[list(_FB_CODES)].sum(axis=1)
    ld = df[list(_LD_CODES)].sum(axis=1)
    bbo = (gb + fb + ld).replace(0, np.nan)
    df["GB_bbo"] = gb
    df["FB_bbo"] = fb
    df["LD_bbo"] = ld
    df["GB_pct_bbo"] = (gb / bbo).fillna(0.0)
    df["FB_pct_bbo"] = (fb / bbo).fillna(0.0)
    df["LD_pct_bbo"] = (ld / bbo).fillna(0.0)
    df["GB_FB_bbo"] = (gb / fb.replace(0, np.nan)).fillna(0.0)
    df["HR_per_fbo"] = (df["HR"] / (df["HR"] + fb).replace(0, np.nan)).fillna(0.0)
    return df


def batter_rates(pa: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, batter, batting_team) with counting stats,
    slash line, plate-discipline rates, and batted-ball-out mix."""
    if pa.empty:
        return pd.DataFrame()
    clean = pa.dropna(subset=["batter", "outcome"]).copy()
    if clean.empty:
        return pd.DataFrame()

    df = _pivot_outcome_counts(clean, ["season", "batter", "batting_team"])
    df = _add_slash_and_discipline(df)
    df = df.rename(columns={"batter": "player", "batting_team": "team"})
    df = df.sort_values(["season", "PA"], ascending=[True, False]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Pitcher rates
# ---------------------------------------------------------------------------

def _parse_innings_pitched(ip: object) -> float:
    """Convert NCAA IP notation ('6.2' = 6 ⅔ innings) to a float in innings.

    The decimal digit is *outs* (0, 1, or 2), not tenths.  Returns 0.0 for
    unparseable input.
    """
    if ip is None or (isinstance(ip, float) and np.isnan(ip)):
        return 0.0
    s = str(ip).strip()
    if not s:
        return 0.0
    try:
        if "." in s:
            whole_str, frac_str = s.split(".", 1)
            whole = int(whole_str) if whole_str else 0
            outs = int(frac_str[0]) if frac_str else 0
            if outs > 2:  # malformed (treat as decimal innings)
                return float(s)
            return whole + outs / 3.0
        return float(int(s))
    except (ValueError, TypeError):
        return 0.0


def _pitcher_boxscore_totals(game_players: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-pitcher IP/ER/H/BB/K from the boxscore game_players.

    Keyed by (season, player_name, team_id-derived team).  Only rows with
    a non-null ``ip`` (i.e. actually pitched) are counted.
    """
    if game_players.empty or "ip" not in game_players.columns:
        return pd.DataFrame()

    gp = game_players.copy()
    gp = gp[gp["ip"].notna()]
    if gp.empty:
        return pd.DataFrame()

    gp["IP"] = gp["ip"].apply(_parse_innings_pitched)
    gp = gp[gp["IP"] > 0]
    if gp.empty:
        return pd.DataFrame()

    # season may not be on game_players; callers join on player+team and we
    # backfill season from the pa-derived frame instead.
    group_cols = ["player_name", "team_id"]
    agg = gp.groupby(group_cols).agg(
        IP=("IP", "sum"),
        ER=("er", "sum"),
        H_allowed=("hits_allowed", "sum"),
        BB_allowed=("bb_pit", "sum"),
        K_box=("k_pit", "sum"),
        appearances=("IP", "size"),
    ).reset_index()
    return agg


def pitcher_rates(
    pa: pd.DataFrame,
    game_players: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per (season, pitcher, fielding_team) with PBP-derived rates,
    plus IP-based ERA/WHIP/K9/BB9 joined from the boxscore when available.

    PBP-derived (from outcomes attributed to the pitcher):
        TBF, K%, BB%, K-BB%, BAA, OBP-against, SLG-against, BABIP-against,
        batted-ball-out mix, HR/FBO.

    Boxscore-derived (need official IP):
        IP, ERA, WHIP, K/9, BB/9.
    """
    if pa.empty or "pitcher" not in pa.columns:
        return pd.DataFrame()

    clean = pa.dropna(subset=["pitcher", "outcome"]).copy()
    clean = clean[clean["pitcher"].astype(str) != ""]
    if clean.empty:
        return pd.DataFrame()

    df = _pivot_outcome_counts(clean, ["season", "pitcher", "fielding_team"])
    df = _add_slash_and_discipline(df)
    # Reframe slash-line column names from the pitcher's perspective.
    df = df.rename(columns={
        "pitcher": "player",
        "fielding_team": "team",
        "PA": "TBF",
        "AVG": "BAA",
        "OBP": "OBP_against",
        "SLG": "SLG_against",
        "OPS": "OPS_against",
        "BABIP": "BABIP_against",
    })

    # Join boxscore IP-based stats.  game_players carries team_id (numeric
    # henrygd id), which is what fielding-team resolution in pbp also uses,
    # but the pa "fielding_team" is a NAME string — so we join on player
    # name + season only, accepting that same-named pitchers on different
    # teams in one season (vanishingly rare) would merge.
    if game_players is not None and not game_players.empty:
        box = _pitcher_boxscore_totals(game_players)
        if not box.empty:
            box_by_name = (
                box.groupby("player_name")
                .agg(
                    IP=("IP", "sum"),
                    ER=("ER", "sum"),
                    H_allowed=("H_allowed", "sum"),
                    BB_allowed=("BB_allowed", "sum"),
                    K_box=("K_box", "sum"),
                    appearances=("appearances", "sum"),
                )
                .reset_index()
                .rename(columns={"player_name": "player"})
            )
            df = df.merge(box_by_name, on="player", how="left")
            ip = df["IP"].replace(0, np.nan)
            # Softball is a 7-inning game, so rate stats normalize to 7 IP.
            df["ERA"] = (7.0 * df["ER"] / ip).round(3)
            df["WHIP"] = ((df["BB_allowed"] + df["H_allowed"]) / ip).round(3)
            df["K7"] = (7.0 * df["K_box"] / ip).round(2)
            df["BB7"] = (7.0 * df["BB_allowed"] / ip).round(2)

    df = df.sort_values(["season", "TBF"], ascending=[True, False]).reset_index(drop=True)
    return df


__all__ = ["batter_rates", "pitcher_rates"]
