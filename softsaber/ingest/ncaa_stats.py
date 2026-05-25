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

from ..config import NCAA_STATS_WORKERS
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
# stats.ncaa.org uses /players/{id} (plural) on roster and team pages,
# and /player/{id} (singular) on some older pages — match both.
_PLAYER_LINK_RE = re.compile(r"/players?/(\d+)")


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


# --- Roster scraping ---------------------------------------------------------
#
# stats.ncaa.org/teams/{id}/roster is server-rendered HTML (the data is in the
# page, not loaded by JS).  However, Akamai bot-detection serves a ~2 KB
# challenge page to fresh curl_cffi sessions.  The challenge sets cookies in
# the session; a second request with those cookies typically receives the real
# page.  We detect the challenge by checking for a small response with no
# <table> elements, then retry once with force=True to bypass the disk cache.

_ROSTER_CHALLENGE_BYTE_LIMIT = 8_000  # real roster pages are 15–25 KB


def _parse_roster_html(html: str, team_season_id: str) -> pd.DataFrame:
    """Parse the /teams/{id}/roster HTML into a per-player DataFrame.

    Two-pass approach:
    1. ``pd.read_html`` for the tabular data.
    2. lxml scan over the *roster table's rows specifically* for
       ``/players/{id}`` hrefs.  Scoping to the table avoids picking up
       links from sidebars (team leaders, season bests) that would make
       the link count exceed the row count.
    """
    import io

    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception:
        tables = []

    if not tables:
        return pd.DataFrame()

    df = max(tables, key=len).copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename = {
        "#": "jersey", "no.": "jersey", "no": "jersey",
        "name": "player_name", "player": "player_name",
        "pos": "position", "position": "position",
        "yr": "class_year", "cl": "class_year", "class": "class_year",
        "ht": "height", "hometown": "hometown", "high_school": "high_school",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["team_season_id"] = team_season_id

    # Find the roster table in the DOM (largest <table> by row count, same
    # heuristic pd.read_html effectively uses) and pull one player link per
    # row.  A row with no link gets None — typically a coach/staff entry
    # or a player whose page doesn't exist yet.
    player_ids: list[str | None] = []
    try:
        tree = lxml_html.fromstring(html)
        tables_dom = tree.xpath("//table")
        if tables_dom:
            roster_table = max(tables_dom, key=lambda t: len(t.xpath(".//tr")))
            for tr in roster_table.xpath(".//tr"):
                # Skip header rows (they have <th> not <td>).
                if not tr.xpath(".//td"):
                    continue
                pid: str | None = None
                for a in tr.xpath('.//a[contains(@href, "/players/")]'):
                    m = _PLAYER_LINK_RE.search(str(a.get("href") or ""))
                    if m:
                        pid = m.group(1)
                        break
                player_ids.append(pid)
    except Exception as e:  # noqa: BLE001
        log.debug("team %s: link extraction failed: %s", team_season_id, e)

    if len(player_ids) == len(df):
        df["ncaa_player_id"] = player_ids
        unlinked = sum(1 for p in player_ids if p is None)
        if unlinked:
            log.debug(
                "team %s: %d/%d players have no /players/ link "
                "(coach, redshirt, or page not yet published)",
                team_season_id, unlinked, len(df),
            )
    elif player_ids:
        log.warning(
            "team %s: %d player links vs %d table rows — skipping ID join",
            team_season_id, len(player_ids), len(df),
        )

    return df.reset_index(drop=True)


def _looks_like_challenge(html: str | None) -> bool:
    if not html:
        return True
    if len(html) < _ROSTER_CHALLENGE_BYTE_LIMIT:
        return True
    return 'meta http-equiv="refresh"' in html.lower()


def fetch_team_roster(
    team_season_id: str,
    year: int,
    *,
    force: bool = False,
    browser_session: object = None,
) -> pd.DataFrame:
    """Fetch and parse the roster for one team-season.

    Akamai bot-detection serves a JS challenge page to ``curl_cffi``; when
    a ``browser_session`` is provided, ``fetch_or_browser`` falls back to
    real Chrome and caches the cleared HTML for future runs.
    """
    from .akamai_session import fetch_or_browser

    url = ROSTER_URL.format(team_season_id=team_season_id)
    html = fetch_or_browser(
        url,
        namespace=f"ncaa_stats/rosters/{year}",
        browser_session=browser_session,
        force=force,
    )
    if html is None or _looks_like_challenge(html):
        return pd.DataFrame()

    df = _parse_roster_html(html, team_season_id)
    log.info("roster team_season_id=%s: %d players", team_season_id, len(df))
    return df


# --- Discovery ---------------------------------------------------------------

def _discover_via_contest(contest_id: str, browser_session=None) -> dict[str, str]:
    """Fetch one game's stats page and extract team links.

    Returns an empty dict (not an error) when the page is unavailable — the
    caller should continue with the next contest_id.
    """
    from .akamai_session import fetch_or_browser

    url = CONTEST_STATS_URL.format(contest_id=contest_id)
    html = fetch_or_browser(
        url, namespace="ncaa_stats/contests", browser_session=browser_session
    )
    if not html:
        return {}
    found = _parse_team_links(html)
    if found:
        log.debug("contest %s: found %d team links", contest_id, len(found))
    return found


def _discover_via_ranking(
    year: int, division_id: int, stat_seq: int, ranking_period: int,
    *, browser_session=None,
) -> dict[str, str]:
    """Fetch the national ranking page and extract all team links in one shot."""
    from .akamai_session import fetch_or_browser

    url = NATIONAL_RANKING_URL.format(
        year=year,
        division_id=division_id,
        stat_seq=stat_seq,
        ranking_period=ranking_period,
    )
    html = fetch_or_browser(
        url, namespace=f"ncaa_stats/rankings/{year}", browser_session=browser_session
    )
    if not html:
        log.warning("ranking page year=%s: unreachable", year)
        return {}
    found = _parse_team_links(html)
    sample = list(found.keys())[:10]
    log.info("ranking page year=%s: found %d team links, sample=%s", year, len(found), sample)
    return found


def discover_team_season_ids(
    year: int,
    division_id: int = 1,
    *,
    contest_ids: list[str] | None = None,
    stat_seq: int | None = None,
    ranking_period: int | None = None,
    max_contests: int = 30,
    browser_session=None,
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

    # Path 1: per-game pages.  Browser fetches must be serial (one
    # context, one tab), but most contest pages are cached from prior
    # runs; the threadpool stays useful for cache hits.
    if contest_ids:
        batch = [str(c) for c in contest_ids[:max_contests]]
        if browser_session is None:
            with ThreadPoolExecutor(max_workers=NCAA_STATS_WORKERS) as exe:
                for found in exe.map(_discover_via_contest, batch):
                    results.update(found)
        else:
            for contest_id in batch:
                results.update(_discover_via_contest(contest_id, browser_session))

    # Path 2: national rankings page (covers 100 % of teams in one fetch).
    if stat_seq is not None and ranking_period is not None:
        results.update(_discover_via_ranking(
            year, division_id, stat_seq, ranking_period,
            browser_session=browser_session,
        ))

    log.info(
        "discover_team_season_ids year=%s: found %d teams total", year, len(results)
    )
    return results


__all__ = [
    "CONTEST_STATS_URL",
    "INDIVIDUAL_STATS_URL",
    "NATIONAL_RANKING_URL",
    "ROSTER_URL",
    "discover_team_season_ids",
    "fetch_team_roster",
]
