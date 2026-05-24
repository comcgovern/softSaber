"""Season roster ingest — maps stats.ncaa.org ncaa_player_id to player names.

The pipeline has two steps:

1. **Team ID discovery** — :func:`discover_and_update_teams` fetches per-game
   stats pages on stats.ncaa.org and parses ``/teams/{year_specific_id}``
   links, writing ``stats_ncaa_team_id`` back to the teams partition.

2. **Player ID extraction** — stats.ncaa.org/teams/{id}/roster pages are
   JavaScript-rendered (2 KB empty shell).  Instead, the ``individual_stats``
   page for each contest is fetched; these are server-rendered and embed
   ``/player/{ncaa_player_id}`` links for every player in the game.

Output parquet schema (``rosters/{season}``):

    season, ncaa_player_id, player_name

Team context is intentionally omitted: the ``individual_stats`` pages do not
carry ``/teams/`` links so team association must be resolved downstream via
the games and game_players partitions.
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from .. import storage
from ..config import WSB_D1_RANKING_PERIOD, WSB_D1_RANKING_STAT_SEQ
from . import ncaa_stats

log = logging.getLogger(__name__)


# NCAA.com (henrygd) and stats.ncaa.org disagree on team-name formatting:
# the former uses abbreviations like "Penn St." and "N.C. State"; the latter
# spells things out as "Penn State" and "North Carolina State".  Normalising
# both sides through the same set of substitutions lets us join cleanly.
_ABBREV_REPLACEMENTS = [
    (re.compile(r"\bst\.?\b"), "state"),
    (re.compile(r"\buniv\.?\b"), "university"),
    (re.compile(r"\bcoll\.?\b"), "college"),
    (re.compile(r"\bn\.?\s*c\.?\b"), "north carolina"),
    (re.compile(r"\bs\.?\s*c\.?\b"), "south carolina"),
    (re.compile(r"\btex\.?\b"), "texas"),
    (re.compile(r"\bcal\.?\b"), "california"),
    (re.compile(r"\bmiss\.?\b"), "mississippi"),
    (re.compile(r"\bfla\.?\b"), "florida"),
    (re.compile(r"\bla\.?\b"), "louisiana"),
    (re.compile(r"\bga\.?\b"), "georgia"),
    (re.compile(r"\bva\.?\b"), "virginia"),
    (re.compile(r"\bky\.?\b"), "kentucky"),
    (re.compile(r"\bark\.?\b"), "arkansas"),
    (re.compile(r"\bmich\.?\b"), "michigan"),
    (re.compile(r"\bwash\.?\b"), "washington"),
    (re.compile(r"\bind\.?\b"), "indiana"),
    (re.compile(r"\b&\b"), "and"),
]


def _normalize_team_name(name: str) -> str:
    """Lower-case, expand common abbreviations, strip punctuation and whitespace."""
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    for pat, repl in _ABBREV_REPLACEMENTS:
        s = pat.sub(repl, s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


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

    # Match in two passes: first exact, then normalised (handles "St."/"State"
    # and similar abbreviation drift between ncaa.com and stats.ncaa.org).
    normalised_map: dict[str, str] = {}
    for raw_name, tid in id_map.items():
        normalised_map[_normalize_team_name(raw_name)] = tid

    def _lookup(name: str) -> str | None:
        if name in id_map:
            return id_map[name]
        return normalised_map.get(_normalize_team_name(name))

    teams["stats_ncaa_team_id"] = teams["team_name"].apply(_lookup)
    matched = teams["stats_ncaa_team_id"].notna().sum()
    log.info(
        "year %s: matched stats_ncaa_team_id for %d/%d teams",
        year,
        matched,
        len(teams),
    )
    unmatched = teams[teams["stats_ncaa_team_id"].isna()]["team_name"].tolist()
    if unmatched:
        sample_ncaa = list(id_map.keys())[:10]
        log.warning(
            "year %s: %d teams without stats_ncaa_team_id — "
            "unmatched henrygd names: %s | sample ncaa names: %s",
            year,
            len(unmatched),
            unmatched,
            sample_ncaa,
        )

    storage.write_partition("teams", str(year), teams)
    return teams


def ingest_season_rosters(
    teams: pd.DataFrame,
    games: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Extract player ncaa_player_ids from contest box_score pages.

    stats.ncaa.org roster pages are JavaScript-rendered and return empty HTML
    to plain HTTP clients.  This function instead re-uses the contest box_score
    pages already cached during :func:`discover_and_update_teams` to pull
    ``/player/{id}`` links, capturing every player who appeared in any game in
    ``games``.

    ``teams`` must have ``team_name`` and ``stats_ncaa_team_id`` columns
    (populated by :func:`discover_and_update_teams`).

    Writes ``rosters/{year}.parquet`` and returns the combined DataFrame.
    """
    if "stats_ncaa_team_id" not in teams.columns:
        log.warning("teams table has no stats_ncaa_team_id — run discover first")
        return pd.DataFrame()

    contest_ids = games["game_id"].astype(str).tolist()
    players = ncaa_stats.build_ncaa_player_map(contest_ids, year)

    if players.empty:
        log.warning("rosters year=%s: no players found from individual_stats pages", year)
        return pd.DataFrame()

    players["season"] = year

    storage.write_partition("rosters", str(year), players)
    log.info("rosters year=%s: wrote %d player rows", year, len(players))
    return players


__all__ = [
    "discover_and_update_teams",
    "ingest_season_rosters",
]
