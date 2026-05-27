"""Season roster ingest — fetches per-team rosters from stats.ncaa.org.

Rosters are stable within a season, so we fetch each team's roster page
exactly once and cache the cleared HTML on disk.  Re-runs that hit a
cached page skip the browser entirely.

stats.ncaa.org is fronted by an Akamai WAF that serves a JavaScript
challenge (``bm-verify``) to ``curl_cffi``.  Real Chrome via Playwright
clears it; bundled Chromium gets static-blocked.  We open one
``BrowserSession`` per ingest call and drive it serially across all
teams — the JS challenge is only paid on the first navigation.

Output parquet schema (``rosters/{season}``):

    season, team_name, team_seoname, team_id, stats_ncaa_team_id,
    first_name, last_name, player_name, jersey, position,
    class_year, ncaa_player_id
"""

from __future__ import annotations

import logging
import re

import pandas as pd
from tqdm import tqdm

from .. import storage
from . import ncaa_stats

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team-name normalisation (for the stats_ncaa_team_id discovery step)
# ---------------------------------------------------------------------------

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
    *,
    browser_session: object = None,
) -> pd.DataFrame:
    """Discover ``stats_ncaa_team_id`` for each team and write an updated teams table.

    Uses ``contestId`` values from ``games`` as the primary discovery path,
    with the national-ranking page as a fallback when ranking config is
    available for the year.  Returns the updated teams DataFrame.
    """
    from ..config import WSB_D1_RANKING_PERIOD, WSB_D1_RANKING_STAT_SEQ

    contest_ids = games["game_id"].astype(str).tolist()
    ranking_period = WSB_D1_RANKING_PERIOD.get(year)

    id_map = ncaa_stats.discover_team_season_ids(
        year,
        division_id=1,
        contest_ids=contest_ids,
        stat_seq=WSB_D1_RANKING_STAT_SEQ,
        ranking_period=ranking_period,
        browser_session=browser_session,
    )

    teams = teams.copy()
    if not id_map:
        log.warning("year %s: no stats_ncaa_team_id found via contest or ranking pages", year)
        if "stats_ncaa_team_id" not in teams.columns:
            teams["stats_ncaa_team_id"] = None
        return teams

    normalised_map: dict[str, str] = {
        _normalize_team_name(n): tid for n, tid in id_map.items()
    }

    def _lookup(name: str) -> str | None:
        if name in id_map:
            return id_map[name]
        return normalised_map.get(_normalize_team_name(name))

    teams["stats_ncaa_team_id"] = teams["team_name"].apply(_lookup)
    matched = teams["stats_ncaa_team_id"].notna().sum()
    log.info("year %s: matched stats_ncaa_team_id for %d/%d teams", year, matched, len(teams))

    storage.write_partition("teams", str(year), teams)
    return teams


# ---------------------------------------------------------------------------
# Roster fetch via headless Chrome
# ---------------------------------------------------------------------------

def ingest_season_rosters(
    teams: pd.DataFrame,
    games: pd.DataFrame,
    year: int,
    *,
    browser_session: object = None,
) -> pd.DataFrame:
    """Fetch one roster page per team from stats.ncaa.org via Playwright.

    Cached HTML from a prior run is reused (per-team-per-season), so a
    second run with the same teams costs nothing and a partial run only
    fetches the teams missing from the cache.

    The ``games`` parameter is accepted for symmetry with the rest of the
    ingest API but isn't used here — rosters are per-team-season, not
    per-game.

    Requires the ``akamai`` extra (Playwright + real Chrome).  Without it,
    returns an empty DataFrame and logs an actionable error.
    """
    del games  # signature consistency only

    eligible = teams[teams["stats_ncaa_team_id"].notna()].copy()
    if eligible.empty:
        log.warning("rosters year=%s: no teams with stats_ncaa_team_id; "
                    "run `ingest teams` and `ingest rosters` discovery first", year)
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []
    rows = list(eligible.itertuples(index=False))
    has_seo = "team_seoname" in eligible.columns

    def _drive(bs) -> None:
        for row in tqdm(rows, desc=f"rosters {year}", unit="team"):
            tid = str(row.stats_ncaa_team_id)
            df = ncaa_stats.fetch_team_roster(tid, year, browser_session=bs)
            if df.empty:
                continue
            df["team_name"] = row.team_name
            df["stats_ncaa_team_id"] = tid
            if has_seo:
                df["team_seoname"] = getattr(row, "team_seoname", "")
            df["season"] = year
            frames.append(df)

    if browser_session is not None:
        _drive(browser_session)
    else:
        # BrowserSession defers the Playwright import to __enter__, so a
        # missing-Playwright environment can surface as either ImportError
        # (no akamai_session module) or RuntimeError (playwright not
        # installed) — catch both and bail with an actionable message.
        try:
            from .akamai_session import BrowserSession
            with BrowserSession() as bs:
                _drive(bs)
        except (ImportError, RuntimeError) as e:
            log.error(
                "rosters year=%s: browser fallback unavailable (%s). Run:\n"
                "    pip install -e .[akamai]\n"
                "    playwright install chromium",
                year, e,
            )
            return pd.DataFrame()

    if not frames:
        log.warning("rosters year=%s: no rosters fetched", year)
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True).reset_index(drop=True)
    storage.write_partition("rosters", str(year), combined)
    sample = combined["player_name"].head(5).tolist() if "player_name" in combined.columns else []
    log.info(
        "rosters year=%s: wrote %d player rows from %d teams, sample=%s",
        year, len(combined), len(frames), sample,
    )
    return combined


__all__ = [
    "discover_and_update_teams",
    "ingest_season_rosters",
]
