"""Thin client for NCAA's sdataprod GraphQL backend.

The henrygd REST wrapper proxies NCAA's GraphQL backend at
``sdataprod.ncaa.com`` using persisted-query hashes.  We hit one query
directly here because henrygd doesn't expose it as a REST route and it
carries fuller team-and-player metadata than the box stat-line:

* ``GetGamecenterGameById_web`` — game-center payload.  Useful for
  upgrading degraded player names in the boxscore (~12% of softball
  entries arrive as a first-initial or empty firstName).

If the backend is unreachable (Akamai 403 / network error) we return
``None`` and callers fall back to whatever they had.
"""

from __future__ import annotations

import json
import logging
from typing import Any
from urllib.parse import quote

from ..http_cache import FetchError, fetch

log = logging.getLogger(__name__)

SDATAPROD_HOST = "https://sdataprod.ncaa.com"

# Persisted-query hash for GetGamecenterGameById_web.  If NCAA rotates
# this, refresh by inspecting the network traffic on ncaa.com/game/<id>.
GAMECENTER_HASH = (
    "93a02c7193c89d85bcdda8c1784925d9b64657f73ef584382e2297af555acd4b"
)


def _build_url(operation: str, sha256: str, variables: dict[str, Any]) -> str:
    extensions = {
        "persistedQuery": {"version": 1, "sha256Hash": sha256},
    }
    return (
        f"{SDATAPROD_HOST}/?meta=GetGamecenterGameById_web_{operation}"
        f"&extensions={quote(json.dumps(extensions, separators=(',', ':')))}"
        f"&queryName={operation}"
        f"&variables={quote(json.dumps(variables, separators=(',', ':')))}"
    )


def fetch_gamecenter(contest_id: str, *, force: bool = False) -> dict[str, Any] | None:
    """Return the ``data`` payload of GetGamecenterGameById_web, or None.

    ``None`` indicates the backend was unreachable (e.g. Akamai 403) or
    returned a non-JSON / errored response.  Callers should treat that as
    "no enrichment available" and proceed with the boxscore data.
    """
    variables = {"contestId": str(contest_id), "staticTestEnv": None}
    url = _build_url("GetGamecenterGameById_web", GAMECENTER_HASH, variables)
    try:
        text = fetch(url, namespace="sdataprod/gamecenter", force=force, ext="json")
    except FetchError as e:
        log.debug("gamecenter %s unreachable: %s", contest_id, e)
        return None

    try:
        body = json.loads(text)
    except json.JSONDecodeError as e:
        log.debug("gamecenter %s non-JSON: %s", contest_id, e)
        return None

    if not isinstance(body, dict):
        return None
    if body.get("errors"):
        log.debug("gamecenter %s GraphQL errors: %s", contest_id, body["errors"])
        return None
    data = body.get("data")
    return data if isinstance(data, dict) else None


__all__ = ["GAMECENTER_HASH", "SDATAPROD_HOST", "fetch_gamecenter"]
