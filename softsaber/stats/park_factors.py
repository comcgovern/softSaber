"""Multi-year park factors with regression to the mean.

Park factor for venue v in year y, basic form:

    raw_pf(v, y) = (R_home_for + R_home_against) / G_home
                 / ((R_away_for + R_away_against) / G_away)

We then:

  1. Aggregate across N years with weights (most recent year highest), e.g.
     weights = [5, 4, 3] for a 3-year window.
  2. Regress toward 1.0 based on home-game sample size (Fangraphs uses
     ~0.5 regression weight for ~150-game samples; NCAA softball seasons
     are ~25-30 home games, so regression weight should be much higher).
  3. Optionally split into ``pf_runs``, ``pf_hr``, ``pf_2b``, etc. for use
     in component-wRC+ extensions later.

Status: STUB. The single-year raw computation below is correct; multi-year
weighting and regression are documented but not yet implemented.
"""

from __future__ import annotations

import pandas as pd

# Per-season home-game count is small (~25-30), so regress hard.
# Choose regression_pa so that a team with that many home games gets equal
# weight from raw_pf and league_avg=1.0.
DEFAULT_REGRESSION_GAMES = 100


def compute_raw_park_factors(games: pd.DataFrame) -> pd.DataFrame:
    """Single-season raw park factors keyed by home team's venue (proxied by
    ``home_team_id``).

    ``games`` schema: season, home_team_id, away_team_id, home_team_runs,
    away_team_runs.
    """
    if games.empty:
        return pd.DataFrame()

    # Total runs scored in each game.
    games = games.copy()
    games["total_runs"] = games["home_team_runs"].fillna(0) + games["away_team_runs"].fillna(0)

    # Home half: how many runs per game does this team see at home?
    home = (
        games.groupby(["season", "home_team_id"])["total_runs"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "home_rpg", "count": "home_g"})
    )
    # Road half: same team as the AWAY side — runs per game while travelling.
    away = (
        games.groupby(["season", "away_team_id"])["total_runs"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "away_rpg", "count": "away_g"})
    )
    away.index = away.index.set_names(["season", "home_team_id"])

    pf = home.join(away, how="inner").reset_index()
    pf["raw_pf"] = pf["home_rpg"] / pf["away_rpg"]
    return pf


def regress_park_factors(pf: pd.DataFrame, regression_games: int = DEFAULT_REGRESSION_GAMES) -> pd.DataFrame:
    """Shrink raw PF toward 1.0 based on home-game sample size."""
    out = pf.copy()
    w = out["home_g"] / (out["home_g"] + regression_games)
    out["pf"] = w * out["raw_pf"] + (1 - w) * 1.0
    return out


def multi_year_park_factors(
    games_by_year: dict[int, pd.DataFrame],
    weights: dict[int, float] | None = None,
    regression_games: int = DEFAULT_REGRESSION_GAMES,
) -> pd.DataFrame:
    """Weighted multi-year park factors with regression.

    STUB: implement once :func:`compute_raw_park_factors` is verified on
    real data. The structure should be:

        raw_pfs = {y: compute_raw_park_factors(games_by_year[y]) for y in ...}
        combined = weighted mean of raw_pf across years per home_team_id
        return regress_park_factors(combined)
    """
    raise NotImplementedError("multi_year_park_factors not yet implemented")
