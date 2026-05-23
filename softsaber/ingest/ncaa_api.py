"""Thin client for NCAA data, via the ncaa-api.henrygd.me REST wrapper.

henrygd's wrapper (MIT-licensed, ``github.com/henrygd/ncaa-api``) scrapes
the live ncaa.com pages and exposes them as REST JSON. We use it for all
three endpoints we care about:

* ``/scoreboard/<sport>/<division>[/YYYY/MM/DD]`` — list of contests.
* ``/game/<gameId>/play-by-play`` — per-period play arrays.
* ``/game/<gameId>/boxscore`` — team and player line totals.

NCAA's own backends (sdataprod GraphQL, data.ncaa.com casablanca) were
tried first but return empty / NoSuchKey for prior-season softball dates.
"""

from __future__ import annotations

import logging
from typing import Any

from ..http_cache import FetchError, fetch

log = logging.getLogger(__name__)

SCOREBOARD_HOST = "https://ncaa-api.henrygd.me"

# Sport-path slug used by the REST scoreboard.
SPORT_PATH_SOFTBALL = "softball"


class NcaaApiError(RuntimeError):
    pass


def _get_json(url: str, namespace: str, force: bool = False) -> dict[str, Any]:
    import json

    text = fetch(url, namespace=namespace, force=force, ext="json")
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise NcaaApiError(f"non-JSON response from {url}: {e}") from e


def fetch_scoreboard(
    sport_path: str,
    division: str,
    contest_date: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Return the REST scoreboard payload for one date.

    ``sport_path`` is the URL slug (``softball``); ``division`` is ``d1``/``d2``/``d3``;
    ``contest_date`` is ``YYYY/MM/DD``. Caller reads ``payload["games"]``.

    Days with no scheduled contests return a 404, which is normalized to an
    empty ``{"games": []}`` payload so callers can iterate calendars cleanly.
    """
    url = f"{SCOREBOARD_HOST}/scoreboard/{sport_path}/{division.lower()}/{contest_date}"
    ns = f"ncaa_api/scoreboard/{sport_path}_{division.lower()}"
    try:
        return _get_json(url, ns, force=force)
    except FetchError as e:
        if "status 404" in str(e):
            log.debug("no scoreboard for %s (%s)", contest_date, e)
            return {"games": []}
        raise


def fetch_play_by_play(contest_id: str, *, force: bool = False) -> dict[str, Any]:
    """Return the REST play-by-play payload for one game."""
    url = f"{SCOREBOARD_HOST}/game/{contest_id}/play-by-play"
    return _get_json(url, namespace="ncaa_api/pbp", force=force)


def fetch_boxscore(contest_id: str, *, force: bool = False) -> dict[str, Any]:
    """Return the REST boxscore payload for one game."""
    url = f"{SCOREBOARD_HOST}/game/{contest_id}/boxscore"
    return _get_json(url, namespace="ncaa_api/boxscore", force=force)


__all__ = [
    "NcaaApiError",
    "SCOREBOARD_HOST",
    "SPORT_PATH_SOFTBALL",
    "fetch_boxscore",
    "fetch_play_by_play",
    "fetch_scoreboard",
]
