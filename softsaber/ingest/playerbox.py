"""Season-level player box scores (hitting + pitching).

These are computed by NCAA from the same underlying games, so for analytic
purposes they're a sanity check rather than a primary source — if our PBP
aggregation for AVG / OBP / SLG doesn't match the playerbox, something is
broken in the parse layer.

We keep them in the ingest pipeline because:
  * they cover players we may want for wRC+ leaderboards even when their
    team's PBP scrape is partial,
  * the season-level columns (AB, PA, H, 2B, 3B, HR, BB, HBP, SF, SH, K)
    are the exact denominators wRC+ wants per player-season.
"""

from __future__ import annotations

import io
import logging

import pandas as pd

from .. import storage
from ..http_cache import fetch

log = logging.getLogger(__name__)

# Stat-category IDs on stats.ncaa.org. These are stable across years for the
# softball ranking page. Confirmed pattern from softballR's playerbox source.
HITTING_STAT_IDS = {2024: None, 2025: None, 2026: None}  # TODO discover/confirm
PITCHING_STAT_IDS = {2024: None, 2025: None, 2026: None}

PLAYERBOX_URL_FMT = (
    "https://stats.ncaa.org/rankings/national_ranking?academic_year={year}.0"
    "&division={division_id}.0&ranking_period=0&sport_code=WSB&stat_seq={stat_seq}"
)


def fetch_playerbox(year: int, division_id: int, stat_seq: int) -> pd.DataFrame:
    url = PLAYERBOX_URL_FMT.format(year=year, division_id=division_id, stat_seq=stat_seq)
    html = fetch(url, namespace=f"playerbox/{year}")
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        return pd.DataFrame()
    # The data table is usually the largest one on the page.
    df = max(tables, key=len).copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df["season"] = year
    return df


def ingest_season_playerbox(year: int, division_id: int) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for label, ids in (("hitting", HITTING_STAT_IDS), ("pitching", PITCHING_STAT_IDS)):
        stat_seq = ids.get(year)
        if stat_seq is None:
            log.info("playerbox %s %s: stat_seq not configured, skipping", year, label)
            out[label] = pd.DataFrame()
            continue
        df = fetch_playerbox(year, division_id, stat_seq)
        if not df.empty:
            storage.write_partition(f"playerbox_{label}", str(year), df)
        out[label] = df
    return out
