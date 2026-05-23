"""Thin client for NCAA's Apollo/GraphQL backend at sdataprod.ncaa.com.

The endpoint speaks persisted queries: the client sends only a sha256 hash
identifying a server-stored query plus a small ``variables`` blob. The hashes
below were lifted from henrygd/ncaa-api (MIT) which keeps them current. If
NCAA rotates them, view source on any game center page (e.g.
``https://www.ncaa.com/game/<contestId>``) and search for ``sha256Hash``.

Three endpoints we need:

* **Scoreboard** for one date / sport / division → list of contests with IDs.
* **Boxscore** for one contestId → team and player line totals (softball-specific
  shape returned by the ``TeamStatsSoftball`` hash).
* **Play-by-play** for one contestId → per-period play arrays, generic across
  most NCAA sports (``PlayByPlayGenericSport``).
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

from ..http_cache import fetch

log = logging.getLogger(__name__)

GRAPHQL_HOST = "https://sdataprod.ncaa.com/"

# Persisted-query hashes. Update if NCAA rotates them.
HASH_SCOREBOARD = "7287cda610a9326931931080cb3a604828febe6fe3c9016a7e4a36db99efdb7c"
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
    if "data" not in payload:
        raise NcaaApiError(f"no `data` field in response from {url}: keys={list(payload)}")
    return payload


def fetch_scoreboard(
    sport_code: str,
    division: int,
    season_year: int,
    contest_date: str,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Return the GraphQL payload for one day's scoreboard.

    ``contest_date`` is ``YYYY/MM/DD``. Caller reads ``payload["data"]["contests"]``.
    """
    variables = {
        "sportCode": sport_code,
        "division": division,
        "seasonYear": season_year,
        "contestDate": contest_date,
    }
    url = _build_url(HASH_SCOREBOARD, variables)
    ns = f"ncaa_api/scoreboard/{sport_code}_d{division}_{season_year}"
    return _get_json(url, ns, force=force)


def fetch_play_by_play(contest_id: str, *, force: bool = False) -> dict[str, Any]:
    """Return the GraphQL payload for one contest's play-by-play."""
    variables = {"contestId": str(contest_id), "staticTestEnv": None}
    url = _build_url(HASH_PBP_GENERIC, variables)
    return _get_json(url, namespace="ncaa_api/pbp", force=force)


def fetch_boxscore(contest_id: str, *, force: bool = False) -> dict[str, Any]:
    """Return the GraphQL payload for one contest's softball boxscore."""
    variables = {"contestId": str(contest_id), "staticTestEnv": None}
    url = _build_url(HASH_BOXSCORE_SOFTBALL, variables)
    return _get_json(url, namespace="ncaa_api/boxscore", force=force)


__all__ = [
    "HASH_BOXSCORE_SOFTBALL",
    "HASH_PBP_GENERIC",
    "HASH_SCOREBOARD",
    "NcaaApiError",
    "fetch_boxscore",
    "fetch_play_by_play",
    "fetch_scoreboard",
]
