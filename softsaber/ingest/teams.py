"""Pull the master team-code table and join to softball team IDs.

stats.ncaa.org has two ID systems we have to bridge:

* ``team_id`` — the NCAA-wide institution code, listed at
  ``/game_upload/team_codes``. Stable across sports and years.
* ``softball_id`` — the per-team URL slug used on /teams/<id> pages, which is
  what every other softball endpoint (PBP, box score) ultimately resolves
  against. Discovered by walking the scoreboard for a season.

This module fetches the first; the scoreboard module produces the second.
The join lives in :func:`build_teams_table`.
"""

from __future__ import annotations

import io
import logging

import pandas as pd
from lxml import html as lxml_html

from .. import storage
from ..http_cache import fetch

log = logging.getLogger(__name__)

TEAM_CODES_URL = "https://stats.ncaa.org/game_upload/team_codes"


def fetch_team_codes(force: bool = False, browser_session: object = None) -> pd.DataFrame:
    """Return a DataFrame of (team_id, team_name) for every NCAA institution."""
    from .akamai_session import fetch_or_browser

    raw = fetch_or_browser(
        TEAM_CODES_URL,
        namespace="team_codes",
        browser_session=browser_session,
        force=force,
    )
    if not raw:
        raise RuntimeError(f"team_codes unreachable ({TEAM_CODES_URL})")
    # First HTML table on the page has the codes.
    tables = pd.read_html(io.StringIO(raw))
    if not tables:
        raise RuntimeError("no tables found at team_codes URL")
    df = tables[0]
    # Page uses repeated header rows; strip them and rename.
    df.columns = ["team_id", "team_name"]
    df = df[~df["team_id"].isin(["NCAA Codes", "ID"])].copy()
    df["team_id"] = df["team_id"].astype(str).str.strip()
    df["team_name"] = df["team_name"].astype(str).str.strip()
    return df.drop_duplicates(subset=["team_id"]).reset_index(drop=True)


def build_teams_table(
    season_softball_ids: pd.DataFrame,
    season: int,
    *,
    browser_session: object = None,
) -> pd.DataFrame:
    """Join team-code list to softball-side IDs gathered from the scoreboard.

    ``season_softball_ids`` must have columns ``team_name`` and ``softball_id``;
    produced by :func:`softsaber.ingest.scoreboard.discover_team_softball_ids`.
    """
    codes = fetch_team_codes(browser_session=browser_session)
    merged = (
        season_softball_ids.merge(codes, on="team_name", how="left")
        .assign(season=season)
        .loc[:, ["season", "team_name", "team_id", "softball_id"]]
    )
    missing = merged["team_id"].isna().sum()
    if missing:
        log.warning("season %s: %d teams missing NCAA team_id (name mismatch)", season, missing)
    storage.write_partition("teams", str(season), merged)
    return merged


# Helper kept here because it's HTML-shaped (lxml import already paid for).
def parse_links(html: str, xpath: str) -> list[str]:
    tree = lxml_html.fromstring(html)
    return [el for el in tree.xpath(xpath) if el]
