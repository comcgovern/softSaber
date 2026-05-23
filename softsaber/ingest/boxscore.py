"""Boxscore ingest from the ncaa-api.henrygd.me REST wrapper.

One call per gameId returns a payload with team and player line totals. We
don't have a parser/flattener for boxscore yet, so this module's job is to
warm the on-disk cache (``data/raw/ncaa_api/boxscore/...``) for a set of
games. Once a parser is added, fan-out and partition writes belong here.
"""

from __future__ import annotations

import logging

import pandas as pd

from . import ncaa_api

log = logging.getLogger(__name__)


def fetch_game_boxscore(game_id: str) -> dict | None:
    """Return the raw boxscore payload for one game, or None on failure."""
    try:
        return ncaa_api.fetch_boxscore(game_id)
    except Exception as e:  # noqa: BLE001
        log.warning("game %s: boxscore fetch failed: %s", game_id, e)
        return None


def ingest_boxscores_for_games(games: pd.DataFrame) -> int:
    """Fetch boxscores for every game in ``games``. Returns the count fetched.

    Each response is cached on disk by ``http_cache``, so re-runs are cheap.
    """
    game_ids = games["game_id"].astype(str).tolist()
    total = len(game_ids)
    fetched = 0
    for i, gid in enumerate(game_ids, 1):
        log.debug("boxscore game %s (%d/%d)", gid, i, total)
        if fetch_game_boxscore(gid) is not None:
            fetched += 1
    return fetched


__all__ = ["fetch_game_boxscore", "ingest_boxscores_for_games"]
