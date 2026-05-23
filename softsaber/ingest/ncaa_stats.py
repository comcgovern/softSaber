"""Scraper for stats.ncaa.org team and roster pages.

stats.ncaa.org uses a year-specific numeric team ID that differs from both the
henrygd API ``teamId`` and the stable ``team_id`` from ``/game_upload/team_codes``.
For example, Oklahoma softball is ID 571946 for 2024 and 613592 for 2026.

These IDs are discovered by parsing ``/teams/{id}`` links from pages that list
teams, then stored as ``stats_ncaa_team_id`` in the teams table so the roster
fetcher can hit the right URL each season.

Discovery paths (both parse ``/teams/{numeric_id}`` links from HTML):

1. **Via game individual-stats pages** — we already have ``contestId`` values
   from the henrygd scoreboard; stats.ncaa.org game pages link out to both
   teams.  No extra config needed, covers all teams that played in our game
   cache.  URL: ``CONTEST_STATS_URL``.

2. **Via the national rankings page** — lists every ranked player in the
   division with a team link, so it covers all active teams in one fetch.
   Requires knowing the sport-specific ``stat_seq`` value; see ``config.py``.
   URL: ``NATIONAL_RANKING_URL``.

Roster page (once the team-season ID is known):

    https://stats.ncaa.org/teams/{team_season_id}/roster

Player rows on the roster page carry links of the form ``/player/{ncaa_player_id}/...``,
giving us the global, stable NCAA player ID we can use to join across seasons.
"""

from __future__ import annotations

import io
import logging
import re
from typing import Any

import pandas as pd
from lxml import html as lxml_html

from ..http_cache import FetchError, fetch

log = logging.getLogger(__name__)

# --- URL templates -----------------------------------------------------------

ROSTER_URL = "https://stats.ncaa.org/teams/{team_season_id}/roster"

# Game individual-stats page: links to both team pages using year-specific IDs.
CONTEST_STATS_URL = "https://stats.ncaa.org/contests/{contest_id}/individual_stats"

# National ranking page: one per sport-division-year, lists all teams.
# ranking_period is a year-specific ID (e.g. 113 for WSB 2026 end-of-season).
# stat_seq=271 is batting average for WSB (stable across years).
# See WSB_D1_RANKING_STAT_SEQ and WSB_D1_RANKING_PERIOD in config.py.
NATIONAL_RANKING_URL = (
    "https://stats.ncaa.org/rankings/national_ranking"
    "?academic_year={year}.0&division={division_id}.0"
    "&ranking_period={ranking_period}.0&sport_code=WSB&stat_seq={stat_seq}.0"
)


# --- Link parsing ------------------------------------------------------------

_TEAM_LINK_RE = re.compile(r"/teams/(\d+)")
_PLAYER_LINK_RE = re.compile(r"/player/(\d+)")


def _parse_team_links(html: str) -> dict[str, str]:
    """Extract ``{team_name: stats_ncaa_team_id}`` from any HTML page with
    ``<a href="/teams/{id}">`` links.

    Ignores entries whose link text is empty or purely numeric (nav artifacts).
    """
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return {}

    result: dict[str, str] = {}
    for a in tree.xpath('//a[contains(@href, "/teams/")]'):
        href = str(a.get("href") or "")
        m = _TEAM_LINK_RE.search(href)
        if not m:
            continue
        name = a.text_content().strip()
        if name and not name.isdigit():
            result[name] = m.group(1)
    return result


# --- Discovery ---------------------------------------------------------------

def _discover_via_contest(contest_id: str) -> dict[str, str]:
    """Fetch one game's stats page and extract team links.

    Returns an empty dict (not an error) when the page is unavailable — the
    caller should continue with the next contest_id.
    """
    url = CONTEST_STATS_URL.format(contest_id=contest_id)
    try:
        html = fetch(url, namespace="ncaa_stats/contests")
        found = _parse_team_links(html)
        if found:
            log.debug("contest %s: found %d team links", contest_id, len(found))
        return found
    except FetchError as e:
        log.debug("contest %s: skipped (%s)", contest_id, e)
        return {}
    except Exception as e:  # noqa: BLE001
        log.warning("contest %s: unexpected error: %s", contest_id, e)
        return {}


def _discover_via_ranking(
    year: int, division_id: int, stat_seq: int, ranking_period: int
) -> dict[str, str]:
    """Fetch the national ranking page and extract all team links in one shot."""
    url = NATIONAL_RANKING_URL.format(
        year=year,
        division_id=division_id,
        stat_seq=stat_seq,
        ranking_period=ranking_period,
    )
    try:
        html = fetch(url, namespace=f"ncaa_stats/rankings/{year}")
        found = _parse_team_links(html)
        log.info("ranking page year=%s: found %d team links", year, len(found))
        return found
    except FetchError as e:
        log.warning("ranking page year=%s: %s", year, e)
        return {}


def discover_team_season_ids(
    year: int,
    division_id: int = 1,
    *,
    contest_ids: list[str] | None = None,
    stat_seq: int | None = None,
    ranking_period: int | None = None,
    max_contests: int = 30,
) -> dict[str, str]:
    """Build a ``{team_name: stats_ncaa_team_id}`` mapping for one season.

    Tries both discovery paths and merges the results.

    Parameters
    ----------
    contest_ids:
        List of NCAA contest IDs (from the games table) to probe via
        ``CONTEST_STATS_URL``.  Covers two teams per request; ~30 games is
        enough for a full D1 field.
    stat_seq:
        When provided alongside ``ranking_period``, the national rankings page
        is also fetched — covers the full division in one request.
        ``stat_seq=271`` is batting average for WSB (stable across years).
    ranking_period:
        Year-specific period ID for the rankings page (e.g. 113 for WSB 2026).
        See ``WSB_D1_RANKING_PERIOD`` in config.py.
    max_contests:
        Upper bound on contest pages to fetch (default 30).
    """
    results: dict[str, str] = {}

    # Path 1: per-game pages (no extra config needed).
    if contest_ids:
        for cid in contest_ids[:max_contests]:
            found = _discover_via_contest(str(cid))
            results.update(found)

    # Path 2: national rankings page (covers 100 % of teams in one fetch).
    if stat_seq is not None and ranking_period is not None:
        results.update(_discover_via_ranking(year, division_id, stat_seq, ranking_period))

    log.info(
        "discover_team_season_ids year=%s: found %d teams total", year, len(results)
    )
    return results


# --- Roster scraping ---------------------------------------------------------

def _parse_roster_html(html: str, team_season_id: str) -> pd.DataFrame:
    """Parse the roster HTML into a per-player DataFrame.

    NCAA roster pages vary year-to-year in column ordering, so we take two
    passes:
    1. Use pandas ``read_html`` to get the tabular data (handles varied column
       headers gracefully).
    2. Re-scan with lxml to pull ``ncaa_player_id`` from ``/player/{id}``
       links, which ``read_html`` discards.
    """
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []

    if not tables:
        log.debug("team %s: no tables found on roster page", team_season_id)
        return pd.DataFrame()

    # Take the largest table — the roster is usually the biggest one.
    df = max(tables, key=len).copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "#": "jersey",
        "no.": "jersey",
        "no": "jersey",
        "name": "player_name",
        "player": "player_name",
        "pos": "position",
        "position": "position",
        "yr": "class_year",
        "cl": "class_year",
        "class": "class_year",
        "ht": "height",
        "hometown": "hometown",
        "high_school": "high_school",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["team_season_id"] = team_season_id

    # Second pass: pull ncaa_player_id from href="/player/{id}" links.
    player_ids: list[str | None] = []
    try:
        tree = lxml_html.fromstring(html)
        for a in tree.xpath('//a[contains(@href, "/player/")]'):
            m = _PLAYER_LINK_RE.search(str(a.get("href") or ""))
            if m:
                player_ids.append(m.group(1))
    except Exception:
        pass

    # Align player ID list to DataFrame rows (best-effort; drop if mismatched).
    if len(player_ids) == len(df):
        df["ncaa_player_id"] = player_ids
    elif player_ids:
        log.debug(
            "team %s: %d player links vs %d table rows — skipping ID join",
            team_season_id,
            len(player_ids),
            len(df),
        )

    return df.reset_index(drop=True)


def fetch_team_roster(
    team_season_id: str,
    year: int,
    *,
    force: bool = False,
) -> pd.DataFrame:
    """Fetch and parse the roster for one team-season.

    Returns an empty DataFrame if the page is unavailable or unparseable.
    """
    url = ROSTER_URL.format(team_season_id=team_season_id)
    try:
        html = fetch(url, namespace=f"ncaa_stats/rosters/{year}", force=force)
    except FetchError as e:
        log.warning("roster team_season_id=%s: %s", team_season_id, e)
        return pd.DataFrame()
    except Exception as e:  # noqa: BLE001
        log.warning("roster team_season_id=%s: unexpected error: %s", team_season_id, e)
        return pd.DataFrame()

    df = _parse_roster_html(html, team_season_id)
    log.info("roster team_season_id=%s: %d players", team_season_id, len(df))
    return df


__all__ = [
    "CONTEST_STATS_URL",
    "NATIONAL_RANKING_URL",
    "ROSTER_URL",
    "discover_team_season_ids",
    "fetch_team_roster",
]
