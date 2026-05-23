"""Thin client for NCAA's data endpoints.

Two backends are in play:

* **ncaa-api.henrygd.me** — third-party REST wrapper around the ncaa.com
  site (MIT-licensed, ``github.com/henrygd/ncaa-api``). Date-keyed path:
  ``/scoreboard/<sport>/<division>/<YYYY>/<MM>/<DD>``. Used for scoreboard
  ingest because sdataprod's GraphQL scoreboard returns empty payloads for
  prior-season dates and ``data.ncaa.com``'s casablanca bucket doesn't have
  per-date keys for softball.
* **sdataprod.ncaa.com GraphQL** — Apollo backend speaking persisted queries
  (sha256 hash + ``variables`` blob). Used for boxscore and play-by-play,
  which the REST endpoints don't expose. Hashes were lifted from
  henrygd/ncaa-api (MIT); if NCAA rotates them, view source on any game
  center page and grep for ``sha256Hash``.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

from ..http_cache import FetchError, fetch

log = logging.getLogger(__name__)

GRAPHQL_HOST = "https://sdataprod.ncaa.com/"
SCOREBOARD_HOST = "https://ncaa-api.henrygd.me"

# Sport-path slug used by the REST scoreboard (distinct from the GraphQL
# ``sportCode``: REST uses "softball", GraphQL uses "WSB").
SPORT_PATH_SOFTBALL = "softball"

# Persisted-query hashes for the GraphQL endpoints. Update if NCAA rotates them.
HASH_PBP_GENERIC = "57f922d56d60d88326b62202b3d88e8cd3cfb6687931bc0b5b3dfab089b84faa"
HASH_BOXSCORE_SOFTBALL = "8fcd4071199071483be215ff66a2e3676f98563e26a7ff1ba113d56ce28a398d"


class NcaaApiError(RuntimeError):
    pass


def _build_url(hash_: str, variables: dict[str, Any]) -> str:
    extensions = {"persistedQuery": {"version": 1, "sha256Hash": hash_}}
    return (
        f"{GRAPHQL_HOST}?extensions={quote(json.dumps(extensions, separators=(',', ':')))}"
        f"&variables={quote(json.dumps(variables, separators=(',', ':')))}"
    )


def _get_json(url: str, namespace: str, force: bool = False) -> dict[str, Any]:
    text = fetch(url, namespace=namespace, force=force, ext="json")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as e:
        raise NcaaApiError(f"non-JSON response from {url}: {e}") from e
    if "errors" in payload and payload["errors"]:
        raise NcaaApiError(f"GraphQL errors from {url}: {payload['errors']}")
    return payload


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
    url = (
        f"{SCOREBOARD_HOST}/scoreboard/"
        f"{sport_path}/{division.lower()}/{contest_date}"
    )
    ns = f"ncaa_api/scoreboard/{sport_path}_{division.lower()}"
    try:
        return _get_json(url, ns, force=force)
    except FetchError as e:
        if "status 404" in str(e):
            log.debug("no scoreboard for %s (%s)", contest_date, e)
            return {"games": []}
        raise


def fetch_play_by_play(contest_id: str, *, force: bool = False) -> dict[str, Any]:
    """Return the GraphQL payload for one contest's play-by-play."""
    variables = {"contestId": str(contest_id), "staticTestEnv": None}
    url = _build_url(HASH_PBP_GENERIC, variables)
    payload = _get_json(url, namespace="ncaa_api/pbp", force=force)
    if "data" not in payload:
        raise NcaaApiError(f"no `data` field in pbp response: keys={list(payload)}")
    return payload


def fetch_boxscore(contest_id: str, *, force: bool = False) -> dict[str, Any]:
    """Return the GraphQL payload for one contest's softball boxscore."""
    variables = {"contestId": str(contest_id), "staticTestEnv": None}
    url = _build_url(HASH_BOXSCORE_SOFTBALL, variables)
    payload = _get_json(url, namespace="ncaa_api/boxscore", force=force)
    if "data" not in payload:
        raise NcaaApiError(f"no `data` field in boxscore response: keys={list(payload)}")
    return payload


__all__ = [
    "HASH_BOXSCORE_SOFTBALL",
    "HASH_PBP_GENERIC",
    "NcaaApiError",
    "SCOREBOARD_HOST",
    "SPORT_PATH_SOFTBALL",
    "fetch_boxscore",
    "fetch_play_by_play",
    "fetch_scoreboard",
]
