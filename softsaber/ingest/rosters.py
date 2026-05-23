"""Roster ingest — needed to map batter-name strings in PBP back to player IDs.

stats.ncaa.org exposes a roster page per team-season at
``/teams/{softball_id}/roster`` (the exact path is what softballR's
``load_ncaa_softball_rosters`` resolves under the hood). The roster gives us
jersey number, full name, position, class, and a player-level ID we'll need
for season aggregation later.

This module is left as a small functional stub: the canonical URL pattern is
recorded here, and we wire up a parquet writer with the expected schema so
the stats layer can be developed against fixtures while we firm up the
scraping details against a real page.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from .. import storage
from ..http_cache import fetch

log = logging.getLogger(__name__)

ROSTER_URL_FMT = "https://stats.ncaa.org/teams/{softball_id}/roster"


def fetch_team_roster(softball_id: str, season: int) -> pd.DataFrame:
    """Fetch a single team's roster table.

    NOTE: The exact column set varies year to year as NCAA updates the page.
    We normalize to a minimal schema here and pass through any extras as
    columns prefixed with ``raw_``.
    """
    url = ROSTER_URL_FMT.format(softball_id=softball_id)
    html = fetch(url, namespace=f"roster/{season}")
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        log.warning("roster %s: no tables found", softball_id)
        return pd.DataFrame()

    df = tables[0].copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {
        "jersey": "jersey",
        "player": "player_name",
        "name": "player_name",
        "pos": "position",
        "position": "position",
        "yr": "class_year",
        "class": "class_year",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["softball_id"] = softball_id
    df["season"] = season
    return df


def ingest_season_rosters(team_ids: pd.DataFrame, season: int) -> pd.DataFrame:
    """``team_ids`` must have columns ``softball_id`` and ``team_name``."""
    frames = []
    for _, row in team_ids.iterrows():
        try:
            r = fetch_team_roster(str(row["softball_id"]), season)
            if not r.empty:
                r["team_name"] = row["team_name"]
                frames.append(r)
        except Exception as e:  # noqa: BLE001
            log.warning("roster team %s failed: %s", row["softball_id"], e)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        storage.write_partition("rosters", str(season), df)
    return df
