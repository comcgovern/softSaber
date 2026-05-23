"""Season roster ingest, bridging the teams table to stats.ncaa.org roster pages.

The pipeline has two steps:

1. **ID discovery** — call :func:`ncaa_stats.discover_team_season_ids` with the
   game ``contestId`` list from the games table.  This fetches per-game stats
   pages on stats.ncaa.org and parses ``/teams/{year_specific_id}`` links.
   The resulting ``{team_name: stats_ncaa_team_id}`` map is joined to the teams
   table and written back to disk.

2. **Roster fetch** — for each team with a known ``stats_ncaa_team_id``, fetch
   ``stats.ncaa.org/teams/{id}/roster`` and parse player rows.  Each player row
   carries a ``ncaa_player_id`` from the ``/player/{id}`` link, giving us a
   global, stable player identifier we can join across seasons and back to PBP.

Output parquet schema (``rosters/{season}``):

    season, team_name, stats_ncaa_team_id,
    ncaa_player_id, player_name, jersey, position, class_year
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .. import storage
from ..config import REQUEST_WORKERS, WSB_D1_RANKING_PERIOD, WSB_D1_RANKING_STAT_SEQ
from . import ncaa_stats

log = logging.getLogger(__name__)


def discover_and_update_teams(
    teams: pd.DataFrame,
    games: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Discover ``stats_ncaa_team_id`` for each team and write an updated teams table.

    Uses ``contestId`` values from ``games`` as the primary discovery path
    (no config needed) and falls back to the national ranking page when
    ``WSB_D1_RANKING_STAT_SEQ[year]`` is configured.

    Returns the updated teams DataFrame with a ``stats_ncaa_team_id`` column.
    """
    contest_ids = games["game_id"].astype(str).tolist()
    ranking_period = WSB_D1_RANKING_PERIOD.get(year)

    division_id = 1  # D1 hardcoded; extend via Season.division_code if needed
    id_map = ncaa_stats.discover_team_season_ids(
        year,
        division_id=division_id,
        contest_ids=contest_ids,
        stat_seq=WSB_D1_RANKING_STAT_SEQ,
        ranking_period=ranking_period,
    )

    if not id_map:
        log.warning(
            "year %s: no stats_ncaa_team_id found — roster fetch will be skipped. "
            "If the contest pages returned 404s, check that CONTEST_STATS_URL is "
            "correct for the current NCAA site layout.",
            year,
        )
        teams = teams.copy()
        if "stats_ncaa_team_id" not in teams.columns:
            teams["stats_ncaa_team_id"] = None
        return teams

    teams = teams.copy()
    teams["stats_ncaa_team_id"] = teams["team_name"].map(id_map)
    matched = teams["stats_ncaa_team_id"].notna().sum()
    log.info(
        "year %s: matched stats_ncaa_team_id for %d/%d teams",
        year,
        matched,
        len(teams),
    )

    storage.write_partition("teams", str(year), teams)
    return teams


def ingest_season_rosters(
    teams: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Fetch rosters for all teams that have a known ``stats_ncaa_team_id``.

    ``teams`` must have ``team_name`` and ``stats_ncaa_team_id`` columns
    (populated by :func:`discover_and_update_teams`).

    Writes ``rosters/{year}.parquet`` and returns the combined DataFrame.
    """
    if "stats_ncaa_team_id" not in teams.columns:
        log.warning("teams table has no stats_ncaa_team_id — run discover first")
        return pd.DataFrame()

    eligible = teams[teams["stats_ncaa_team_id"].notna()].copy()
    if eligible.empty:
        log.warning("no teams with stats_ncaa_team_id for year %s", year)
        return pd.DataFrame()

    team_records = list(eligible.itertuples(index=False))
    log.info("rosters: fetching %d teams with %d workers", len(team_records), REQUEST_WORKERS)

    def _fetch_roster(row) -> pd.DataFrame:  # type: ignore[type-arg]
        tid = str(row.stats_ncaa_team_id)
        df = ncaa_stats.fetch_team_roster(tid, year)
        if df.empty:
            return df
        df["team_name"] = row.team_name
        df["stats_ncaa_team_id"] = tid
        df["season"] = year
        return df

    with ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as exe:
        results = list(exe.map(_fetch_roster, team_records))
    frames = [df for df in results if not df.empty]

    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    if not combined.empty:
        # Normalise columns to the canonical schema — extras are kept as-is.
        for col in ("ncaa_player_id", "player_name", "jersey", "position", "class_year"):
            if col not in combined.columns:
                combined[col] = None
        storage.write_partition("rosters", str(year), combined)
        log.info("rosters year=%s: wrote %d player rows", year, len(combined))

    return combined


__all__ = [
    "discover_and_update_teams",
    "ingest_season_rosters",
]
