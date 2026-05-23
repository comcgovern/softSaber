"""Play-by-play ingest from the ncaa-api.henrygd.me REST wrapper.

One call per gameId returns a payload with ``periods[]`` where each period
is an inning containing a list of play objects (``playbyplayStats`` from
the GraphQL backend, or ``plays`` from henrygd's scrape — both shapes are
handled). We flatten that into the row shape the parse layer consumes:

    game_id, inning, top_bottom, batting_team, fielding_team,
    away_team_runs, home_team_runs, events, play_id, row_idx

**Field-name caveat:** NCAA does not publish a schema for this data and
the per-play field names drift between the upstream sources. The
candidate names below are reasonable guesses; the flattener logs the keys
it actually sees the first time it can't find what it expects so we can
tune from real data without another scrape.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

import pandas as pd

from .. import storage
from . import ncaa_api

log = logging.getLogger(__name__)

# Candidate keys for the per-play text field, in priority order. Add to this
# list as you discover variants in real responses; the first present non-empty
# string wins.
_TEXT_KEYS = ("text", "playText", "description", "summary", "narrative")

# Candidate keys for which side is batting (boolean: True = home batting).
_HOME_BATTING_KEYS = ("isHomeBatting", "homeBatting", "battingHome", "isHomeTeamBatting")

# Candidate keys for the running score columns.
_AWAY_SCORE_KEYS = ("awayScore", "scoreAway", "visitingScore")
_HOME_SCORE_KEYS = ("homeScore", "scoreHome")

# Candidate keys for period (inning) number on the period object.
_PERIOD_NUM_KEYS = ("periodNumber", "period", "inning", "number", "ordinal")

_warned_keys: set[str] = set()


def _first(d: dict[str, Any], keys: Iterable[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _warn_once(tag: str, keys: list[str]) -> None:
    if tag in _warned_keys:
        return
    _warned_keys.add(tag)
    log.warning("PBP shape: could not find %s; saw keys=%s", tag, keys)


def _half_inning(period: dict[str, Any]) -> tuple[str, str]:
    """Some payloads split top/bottom inside one period as two sub-lists
    (e.g. ``home`` and ``away`` arrays), others interleave plays with a
    per-play flag. We handle the per-play flag here and return ('top'/'bottom',
    field_name) once we've inspected the first play.
    """
    return ("top", "top")  # caller resolves per-play; helper kept for symmetry


def _find_periods(payload: dict[str, Any]) -> list[Any]:
    """Pull the ``periods`` list out of a PBP payload, tolerating shape drift.

    henrygd's REST wrapper puts ``periods`` at the top level; the older
    GraphQL backend nested it under ``data.playbyplay``. Check both, plus a
    couple of related variants.
    """
    if not payload:
        return []
    if isinstance(payload.get("periods"), list):
        return payload["periods"]
    data = payload.get("data") or {}
    if isinstance(data, dict):
        pbp = data.get("playbyplay") or data.get("playByPlay") or {}
        if isinstance(pbp, dict) and isinstance(pbp.get("periods"), list):
            return pbp["periods"]
        if isinstance(data.get("periods"), list):
            return data["periods"]
    return []


def flatten_pbp(payload: dict[str, Any], game_id: str) -> pd.DataFrame:
    """Turn one PBP payload into a flat DataFrame.

    Returns an empty frame (no rows, no columns) if the response shape is
    unrecognized or contains no plays — same contract as the previous HTML
    implementation, so the ``ingest_season_pbp`` driver can treat it uniformly.
    """
    periods = _find_periods(payload)
    if not periods:
        log.debug("game %s: no periods in payload", game_id)
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for period in periods:
        if not isinstance(period, dict):
            continue
        inning = _first(period, _PERIOD_NUM_KEYS)
        if inning is None:
            _warn_once("period_number", list(period.keys()))
        try:
            inning_int = int(inning) if inning is not None else 0
        except (TypeError, ValueError):
            inning_int = 0

        # Two shapes seen in NCAA generic-PBP: (a) flat list of plays with a
        # per-play "home batting" flag, (b) split into half-inning sub-arrays.
        # henrygd's REST wrapper labels the array ``plays``; the GraphQL
        # backend used ``playbyplayStats``.
        plays = period.get("plays") or period.get("playbyplayStats")
        if isinstance(plays, dict):
            # Shape (b): {"top": [...], "bottom": [...]} or {"away": [...], "home": [...]}.
            buckets = [
                ("top", plays.get("top") or plays.get("away") or []),
                ("bottom", plays.get("bottom") or plays.get("home") or []),
            ]
        elif isinstance(plays, list):
            buckets = [("auto", plays)]
        else:
            if plays is not None:
                _warn_once("playbyplayStats_shape", [type(plays).__name__])
            continue

        for bucket_side, bucket_plays in buckets:
            row_idx = 0
            for play in bucket_plays:
                if not isinstance(play, dict):
                    continue
                text = _first(play, _TEXT_KEYS)
                if not isinstance(text, str) or not text.strip():
                    # Skip non-play rows (inning headers, etc.) silently.
                    continue

                if bucket_side == "auto":
                    home_bat = _first(play, _HOME_BATTING_KEYS)
                    if home_bat is None:
                        # Fall back to a heuristic: NCAA tends to alternate,
                        # but without a flag we just tag everything 'top' and
                        # let downstream catch the imbalance. Log once.
                        _warn_once("home_batting_flag", list(play.keys()))
                        side = "top"
                    else:
                        side = "bottom" if bool(home_bat) else "top"
                else:
                    side = bucket_side

                rows.append(
                    {
                        "game_id": game_id,
                        "inning": inning_int,
                        "top_bottom": side,
                        # batting/fielding team names are filled in by the
                        # caller from the games table; we don't have team
                        # names on every play here.
                        "batting_team": play.get("battingTeam") or "",
                        "fielding_team": play.get("fieldingTeam") or "",
                        "away_team_runs": _first(play, _AWAY_SCORE_KEYS),
                        "home_team_runs": _first(play, _HOME_SCORE_KEYS),
                        "events": text.strip(),
                        "play_id": (
                            f"{game_id}_{inning_int}_"
                            f"{0 if side == 'top' else 1}_{row_idx + 1}"
                        ),
                        "row_idx": row_idx,
                    }
                )
                row_idx += 1

    return pd.DataFrame(rows)


def fetch_game_pbp(game_id: str) -> pd.DataFrame:
    """Return the raw PBP rows for one game, or empty DataFrame if unavailable."""
    try:
        payload = ncaa_api.fetch_play_by_play(game_id)
    except Exception as e:  # noqa: BLE001
        log.warning("game %s: pbp fetch failed: %s", game_id, e)
        return pd.DataFrame()
    return flatten_pbp(payload, game_id)


def _attach_team_names(pbp: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """Backfill batting_team / fielding_team from the games table when PBP
    rows didn't carry them (the common case).
    """
    if pbp.empty:
        return pbp
    names = games.set_index("game_id")[["away_team", "home_team"]].to_dict("index")
    bt: list[str] = []
    ft: list[str] = []
    for r in pbp.itertuples(index=False):
        rec = names.get(str(r.game_id), {})
        away, home = rec.get("away_team", ""), rec.get("home_team", "")
        if r.top_bottom == "top":
            bt.append(away)
            ft.append(home)
        else:
            bt.append(home)
            ft.append(away)
    pbp = pbp.copy()
    pbp["batting_team"] = bt
    pbp["fielding_team"] = ft
    return pbp


def ingest_pbp_for_games(
    games: pd.DataFrame, season: int, partition: str
) -> pd.DataFrame:
    """Pull PBP for every game in ``games`` and write a parquet partition.

    ``partition`` controls the on-disk file name under ``pbp_raw/`` (e.g.
    ``"2024"`` for a full season, ``"2024-05-04"`` for a single day).
    """
    frames: list[pd.DataFrame] = []
    game_ids = games["game_id"].astype(str).tolist()
    total = len(game_ids)
    for i, gid in enumerate(game_ids, 1):
        log.debug("pbp game %s (%d/%d)", gid, i, total)
        df = fetch_game_pbp(gid)
        if not df.empty:
            frames.append(df)
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df = _attach_team_names(df, games)
        df["season"] = season
        storage.write_partition("pbp_raw", partition, df)
    return df


def ingest_season_pbp(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Pull PBP for every finalized game in ``games`` (full-season partition)."""
    return ingest_pbp_for_games(games, season, str(season))


__all__ = [
    "fetch_game_pbp",
    "flatten_pbp",
    "ingest_pbp_for_games",
    "ingest_season_pbp",
]
