"""Boxscore ingest from the ncaa-api.henrygd.me REST wrapper.

One call per gameId returns a payload with team and player line totals.

``parse_boxscore`` flattens the payload into a per-player DataFrame with the
schema below.  This is written to the ``game_players`` partition so the parse
layer can resolve batter name strings (LASTNAME, FI format) from PBP back to
actual players.

Output schema (one row per player-game appearance):

    game_id, team_id, team_seoname, is_home,
    first_name, last_name, player_name,
    jersey, position, starter, participated,
    ab, r, h, rbi, bb, k,
    ip, er, bb_pit, k_pit, hits_allowed, er_pit

``team_id`` is the numeric ID from the API (e.g. ``"41974"``).  It is NOT the
same as the ``team_id`` from ``stats.ncaa.org/game_upload/team_codes``; for
that join use ``team_seoname`` ↔ ``softball_id`` in the teams table.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pandas as pd

from .. import storage
from ..config import HENRYGD_WORKERS
from . import ncaa_api

log = logging.getLogger(__name__)


def fetch_game_boxscore(game_id: str) -> dict | None:
    """Return the raw boxscore payload for one game, or None on failure."""
    try:
        return ncaa_api.fetch_boxscore(game_id)
    except Exception as e:  # noqa: BLE001
        log.warning("game %s: boxscore fetch failed: %s", game_id, e)
        return None


def _safe_str(v: Any) -> str:
    return "" if v is None else str(v).strip()


def _safe_int(v: Any, default: int = 0) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def parse_boxscore(payload: dict[str, Any], game_id: str) -> pd.DataFrame:
    """Flatten one boxscore payload into a per-player DataFrame.

    Returns an empty DataFrame if the payload is unrecognised or has no player data.
    """
    if not payload:
        return pd.DataFrame()

    # Build a teamId → {seoname, is_home} lookup from the top-level teams array.
    team_meta: dict[str, dict[str, Any]] = {}
    for t in payload.get("teams") or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("teamId", ""))
        team_meta[tid] = {
            "team_seoname": _safe_str(t.get("seoname")),
            "is_home": bool(t.get("isHome")),
        }

    rows: list[dict[str, Any]] = []
    for team_box in payload.get("teamBoxscore") or []:
        if not isinstance(team_box, dict):
            continue
        tid = str(team_box.get("teamId", ""))
        meta = team_meta.get(tid, {"team_seoname": "", "is_home": False})

        for p in team_box.get("playerStats") or []:
            if not isinstance(p, dict):
                continue
            if not p.get("participated"):
                continue

            first = _safe_str(p.get("firstName"))
            last = _safe_str(p.get("lastName"))
            # Repair upstream bug: firstName="" + full name in lastName.
            if not first and " " in last:
                parts = last.split()
                if len(parts) >= 2:
                    first, last = parts[0], " ".join(parts[1:])

            bat = p.get("batterStats") or {}
            pit = p.get("pitcherStats") or {}

            row: dict[str, Any] = {
                "game_id": game_id,
                "team_id": tid,
                "team_seoname": meta["team_seoname"],
                "is_home": meta["is_home"],
                "first_name": first,
                "last_name": last,
                "player_name": f"{first} {last}".strip(),
                "jersey": _safe_int(p.get("number"), -1),
                "position": _safe_str(p.get("position")),
                "starter": bool(p.get("starter")),
                "participated": True,
                # Batting stats
                "ab": _safe_int(bat.get("atBats")),
                "r": _safe_int(bat.get("runsScored")),
                "h": _safe_int(bat.get("hits")),
                "rbi": _safe_int(bat.get("runsBattedIn")),
                "bb": _safe_int(bat.get("walks")),
                "k": _safe_int(bat.get("strikeouts")),
                # Pitching stats (None for non-pitchers)
                "ip": _safe_str(pit.get("inningsPitched")) if pit else None,
                "er": _safe_int(pit.get("earnedRuns")) if pit else None,
                "bb_pit": _safe_int(pit.get("walks")) if pit else None,
                "k_pit": _safe_int(pit.get("strikeouts")) if pit else None,
                "hits_allowed": _safe_int(pit.get("hitsAllowed")) if pit else None,
            }
            rows.append(row)

    return pd.DataFrame(rows)


def ingest_boxscores_for_games(games: pd.DataFrame, partition: str | None = None) -> int:
    """Fetch and parse boxscores for every game in ``games``.

    Writes a ``game_players`` parquet partition when ``partition`` is given
    (use the same partition key as the matching PBP/games write, e.g.
    ``"2024-05-04"``).  Returns the count of games successfully fetched.
    """
    game_ids = games["game_id"].astype(str).tolist()
    log.info("boxscore: fetching %d games with %d workers", len(game_ids), HENRYGD_WORKERS)

    def _fetch_and_parse(gid: str) -> pd.DataFrame:
        payload = fetch_game_boxscore(gid)
        if payload is None:
            return pd.DataFrame()
        return parse_boxscore(payload, gid)

    with ThreadPoolExecutor(max_workers=HENRYGD_WORKERS) as exe:
        results = list(exe.map(_fetch_and_parse, game_ids))
    frames = [df for df in results if not df.empty]
    fetched = sum(1 for df in results if not df.empty)

    if frames and partition is not None:
        combined = pd.concat(frames, ignore_index=True)
        storage.write_partition("game_players", partition, combined)

    return fetched


__all__ = ["fetch_game_boxscore", "ingest_boxscores_for_games", "parse_boxscore"]
