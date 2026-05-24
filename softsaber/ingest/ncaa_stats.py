"""Scraper for stats.ncaa.org team and player pages.

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

Player ID extraction (no separate roster fetch needed):

    stats.ncaa.org/teams/{id}/roster pages are JavaScript-rendered and return
    an empty HTML shell to non-browser clients.  Instead, player IDs are
    extracted from the contest box_score pages already fetched for team
    discovery — each box score page embeds ``/player/{ncaa_player_id}`` links
    for every player who appeared in that game.
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd
from lxml import html as lxml_html

from ..config import REQUEST_WORKERS
from ..http_cache import FetchError, fetch

log = logging.getLogger(__name__)

# --- URL templates -----------------------------------------------------------

ROSTER_URL = "https://stats.ncaa.org/teams/{team_season_id}/roster"

# Game boxscore page: server-rendered HTML with /teams/{id} links for both teams.
CONTEST_STATS_URL = "https://stats.ncaa.org/contests/{contest_id}/box_score"

# Individual stats page: server-rendered HTML with /player/{id} links (no team links).
INDIVIDUAL_STATS_URL = "https://stats.ncaa.org/contests/{contest_id}/individual_stats"

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

    Prefers the direct text node of the anchor (``a.text``) over
    ``text_content()`` to avoid picking up child-element text such as
    conference abbreviations rendered inside a ``<span>``.  Falls back to
    ``text_content()`` when the direct text node is blank.

    Ignores entries whose extracted text is empty or purely numeric.
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
        # Prefer the direct text node; child elements (spans, etc.) often
        # carry conference names or decorators that pollute text_content().
        name = (a.text or "").strip() or a.text_content().strip()
        # stats.ncaa.org appends conference in parens: "Oklahoma (SEC)" → strip it.
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip()
        if name and not name.isdigit():
            result[name] = m.group(1)

    if result:
        sample = list(result.items())[:5]
        log.debug("_parse_team_links: %d entries, sample=%s", len(result), sample)
    return result


def _parse_ranking_page_players(html: str) -> list[dict[str, str]]:
    """Extract player records from an individual-player national ranking page.

    Each table row on stat_seq=271 (individual batting average) has one
    ``/player/{id}`` link (the player) and one ``/teams/{id}`` link (their
    team).  We scan per-row so the two links are reliably paired, giving us
    ``ncaa_player_id``, ``player_name``, and ``stats_ncaa_team_id`` in one
    pass.

    This page is server-rendered (same infrastructure as the team-batting
    ranking page that successfully returns 300+ team links).
    """
    try:
        tree = lxml_html.fromstring(html)
    except Exception:
        return []

    players: list[dict[str, str]] = []
    seen: set[str] = set()

    for row in tree.xpath("//tr"):
        player_id = player_name = team_id = None

        for a in row.xpath('.//a'):
            href = str(a.get("href") or "")

            pm = _PLAYER_LINK_RE.search(href)
            if pm and player_id is None:
                name = (a.text or "").strip() or a.text_content().strip()
                if name and not name.isdigit():
                    player_id = pm.group(1)
                    player_name = name

            tm = _TEAM_LINK_RE.search(href)
            if tm and team_id is None:
                team_id = tm.group(1)

        if player_id and player_name and player_id not in seen:
            seen.add(player_id)
            players.append(
                {
                    "ncaa_player_id": player_id,
                    "player_name": player_name,
                    "stats_ncaa_team_id": team_id or "",
                }
            )

    return players


def _discover_players_via_ranking(
    year: int,
    division_id: int,
    stat_seq: int,
    ranking_period: int,
) -> list[dict[str, str]]:
    """Fetch the individual-player national ranking page and extract player records.

    Uses ``stat_seq=271`` (individual batting average) by default, which lists
    every qualifying batter in the division with both a player link and a team
    link on each row.  The page is server-rendered, the same infrastructure as
    the team-batting ranking page (stat_seq=281).
    """
    url = NATIONAL_RANKING_URL.format(
        year=year,
        division_id=division_id,
        stat_seq=stat_seq,
        ranking_period=ranking_period,
    )
    try:
        html = fetch(url, namespace=f"ncaa_stats/player_rankings/{year}")
        players = _parse_ranking_page_players(html)
        sample = [p["player_name"] for p in players[:5]]
        log.info(
            "player ranking page year=%s: found %d players, sample=%s",
            year,
            len(players),
            sample,
        )
        return players
    except FetchError as e:
        log.warning("player ranking page year=%s: %s", year, e)
        return []


def build_ncaa_player_map(
    year: int,
    division_id: int = 1,
    *,
    stat_seq: int = 271,
    ranking_period: int | None = None,
) -> pd.DataFrame:
    """Build an ``{ncaa_player_id, player_name, stats_ncaa_team_id}`` table.

    Uses the national individual-player ranking page (``stat_seq=271``,
    individual batting average) which is server-rendered and lists every
    qualifying batter in the division with both player and team links per row.

    Requires ``ranking_period`` — the year-specific period ID from
    ``WSB_D1_RANKING_PERIOD`` in config.py.  Returns an empty DataFrame if
    ``ranking_period`` is None or the page is unavailable.
    """
    if ranking_period is None:
        log.warning("player map: ranking_period not set for year %s — skipping", year)
        return pd.DataFrame()

    players = _discover_players_via_ranking(year, division_id, stat_seq, ranking_period)
    if not players:
        return pd.DataFrame()

    df = pd.DataFrame(players)
    log.info("player map year=%s: %d unique players", year, len(df))
    return df.reset_index(drop=True)


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
        sample = list(found.keys())[:10]
        log.info("ranking page year=%s: found %d team links, sample=%s", year, len(found), sample)
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
        batch = [str(c) for c in contest_ids[:max_contests]]
        with ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as exe:
            for found in exe.map(_discover_via_contest, batch):
                results.update(found)

    # Path 2: national rankings page (covers 100 % of teams in one fetch).
    if stat_seq is not None and ranking_period is not None:
        results.update(_discover_via_ranking(year, division_id, stat_seq, ranking_period))

    log.info(
        "discover_team_season_ids year=%s: found %d teams total", year, len(results)
    )
    return results


__all__ = [
    "CONTEST_STATS_URL",
    "INDIVIDUAL_STATS_URL",
    "NATIONAL_RANKING_URL",
    "ROSTER_URL",
    "build_ncaa_player_map",
    "discover_team_season_ids",
]
