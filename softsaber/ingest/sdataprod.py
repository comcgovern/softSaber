"""Thin client for NCAA's sdataprod GraphQL backend.

The henrygd REST wrapper proxies NCAA's GraphQL backend at
``sdataprod.ncaa.com`` using persisted-query hashes.  We hit one query
directly here because henrygd doesn't expose it as a REST route and it
carries fuller team-and-player metadata than the box stat-line:

* ``GetGamecenterGameById_web`` — game-center payload.  Useful for
  upgrading degraded player names in the boxscore (~12% of softball
  entries arrive as a first-initial or empty firstName).

We send a standard Apollo Persisted Query POST::

    POST https://sdataprod.ncaa.com/
    {
      "operationName": "GetGamecenterGameById_web",
      "variables":     {"contestId": "...", "staticTestEnv": null},
      "extensions":    {"persistedQuery": {"version": 1, "sha256Hash": "..."}}
    }

If the backend is unreachable (Akamai 403 / network error) or the hash
has been rotated server-side, we return ``None`` and callers fall back
to whatever they had.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from ..http_cache import FetchError, post_json

log = logging.getLogger(__name__)

SDATAPROD_URL = "https://sdataprod.ncaa.com/"

# Persisted-query hash for GetGamecenterGameById_web.  If NCAA rotates
# this you'll see ``PersistedQueryNotFound`` GraphQL errors; refresh by
# inspecting the network traffic on ncaa.com/game/<id>.
GAMECENTER_HASH = (
    "93a02c7193c89d85bcdda8c1784925d9b64657f73ef584382e2297af555acd4b"
)
GAMECENTER_OP = "GetGamecenterGameById_web"


def _apollo_body(operation: str, sha256: str, variables: dict[str, Any]) -> dict[str, Any]:
    return {
        "operationName": operation,
        "variables": variables,
        "extensions": {
            "persistedQuery": {"version": 1, "sha256Hash": sha256},
        },
    }


def fetch_gamecenter(contest_id: str, *, force: bool = False) -> dict[str, Any] | None:
    """Return the ``data`` payload of GetGamecenterGameById_web, or None.

    ``None`` indicates the backend was unreachable, returned non-JSON,
    or replied with GraphQL errors (e.g. ``PersistedQueryNotFound`` if
    NCAA rotated the hash).  Callers should treat that as "no enrichment
    available" and proceed with the boxscore data.
    """
    body = _apollo_body(
        GAMECENTER_OP,
        GAMECENTER_HASH,
        {"contestId": str(contest_id), "staticTestEnv": None},
    )
    try:
        text = post_json(
            SDATAPROD_URL, body, namespace="sdataprod/gamecenter", force=force
        )
    except FetchError as e:
        log.debug("gamecenter %s unreachable: %s", contest_id, e)
        return None

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        log.debug("gamecenter %s non-JSON: %s", contest_id, e)
        return None

    if not isinstance(parsed, dict):
        return None
    if parsed.get("errors"):
        log.warning(
            "gamecenter %s GraphQL errors (hash may be stale): %s",
            contest_id, parsed["errors"],
        )
        return None
    data = parsed.get("data")
    if not isinstance(data, dict):
        return None

    # Sanity check: a real GameCenter payload will not contain a top-level
    # ``contests`` list (that's the scoreboard query). If we see one,
    # the server ignored our persisted-query hash and fell back to a
    # default — treat as a miss so we don't pollute the upgrade index.
    if "contests" in data and "boxscore" not in data and "teamBoxscore" not in data:
        log.warning(
            "gamecenter %s: server returned scoreboard shape — hash %s likely stale",
            contest_id, GAMECENTER_HASH[:12],
        )
        return None

    return data


__all__ = ["GAMECENTER_HASH", "GAMECENTER_OP", "SDATAPROD_URL", "fetch_gamecenter"]
