"""Play-by-play ingest from the ncaa-api.henrygd.me REST wrapper.

One call per gameId returns a payload with ``periods[]`` where each period
is an inning.  The play list inside each period comes in one of three shapes:

Shape A — flat list of GenericPlay dicts with a per-play batting-team flag::

    period["plays"] = [{playText, homeScore, visitorScore, isHomeBatting?}, ...]

Shape B — dict split by half-inning::

    period["plays"] = {"top": [...], "bottom": [...]}  # or "away"/"home"

Shape C — list of at-bat groups, each grouping one half-inning's plays under
a ``teamId`` key (the actual format returned by henrygd as of 2024–2025)::

    period["playbyplayStats"] = [
        {"teamId": 41890, "plays": [{playText, homeScore, visitorScore}, ...]},
        {"teamId": 41974, "plays": [...]},
        ...
    ]

All three shapes are handled.  Shape C uses the top-level ``teams`` array
(``isHome`` flag) to map ``teamId`` → top/bottom inning half.

Output row schema::

    game_id, inning, top_bottom, batting_team_id, batting_team, fielding_team,
    away_team_runs, home_team_runs, events, play_id, row_idx
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Iterable

import pandas as pd

from .. import storage
from ..config import REQUEST_WORKERS
from . import ncaa_api

log = logging.getLogger(__name__)

# Candidate keys for the per-play text field, in priority order.
_TEXT_KEYS = ("text", "playText", "description", "summary", "narrative")

# Candidate keys for which side is batting (boolean: True = home batting).
_HOME_BATTING_KEYS = ("isHomeBatting", "homeBatting", "battingHome", "isHomeTeamBatting")

# Candidate keys for the running score columns.
# "visitorScore" is the key used in henrygd's Shape C inner plays.
_AWAY_SCORE_KEYS = ("awayScore", "scoreAway", "visitingScore", "visitorScore")
_HOME_SCORE_KEYS = ("homeScore", "scoreHome")

# Candidate keys for period (inning) number on the period object.
_PERIOD_NUM_KEYS = ("periodNumber", "period", "inning", "number", "ordinal")

# Boilerplate lines emitted by the NCAA feed that are not plate appearances.
_BOILERPLATE_RE = re.compile(
    r"^\s*(?:Batting (?:starts|ends)|Pitching change|Defensive sub|"
    r"Inning (?:starts|ends)|End of inning)",
    re.I,
)

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


def _home_team_id(payload: dict[str, Any]) -> str | None:
    """Return the numeric teamId string of the home team from the payload's teams array."""
    for t in payload.get("teams") or []:
        if isinstance(t, dict) and t.get("isHome"):
            tid = t.get("teamId")
            return str(tid) if tid is not None else None
    return None


def _buckets_for_period(
    period: dict[str, Any],
    home_tid: str | None,
) -> list[tuple[str, str | None, list[Any]]]:
    """Return (side, batting_team_id, plays) triples for one inning period.

    ``side`` is 'top', 'bottom', or 'auto'.  ``batting_team_id`` is the
    numeric teamId string from the group when available (Shape C), else None.
    """
    plays_raw = period.get("plays") or period.get("playbyplayStats")

    if isinstance(plays_raw, dict):
        # Shape B: {"top": [...], "bottom": [...]} or {"away": [...], "home": [...]}
        return [
            ("top", None, plays_raw.get("top") or plays_raw.get("away") or []),
            ("bottom", None, plays_raw.get("bottom") or plays_raw.get("home") or []),
        ]

    if not isinstance(plays_raw, list):
        if plays_raw is not None:
            _warn_once("playbyplayStats_shape", [type(plays_raw).__name__])
        return []

    # Distinguish Shape A (flat play list) from Shape C (list of at-bat groups).
    # Shape C groups have a "plays" key and a "teamId" key; Shape A plays have text keys.
    first_non_empty = next((x for x in plays_raw if isinstance(x, dict)), None)
    if first_non_empty is not None and "plays" in first_non_empty and "teamId" in first_non_empty:
        # Shape C: [{teamId, plays: [...]}, ...]
        buckets: list[tuple[str, str | None, list[Any]]] = []
        for group in plays_raw:
            if not isinstance(group, dict):
                continue
            inner = group.get("plays")
            if not isinstance(inner, list):
                continue
            tid = str(group.get("teamId", "")) or None
            if tid and home_tid:
                side = "bottom" if tid == home_tid else "top"
            else:
                side = "auto"
            buckets.append((side, tid, inner))
        return buckets

    # Shape A: flat list of plays.
    return [("auto", None, plays_raw)]


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

    home_tid = _home_team_id(payload)

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

        buckets = _buckets_for_period(period, home_tid)

        for bucket_side, batting_tid, bucket_plays in buckets:
            row_idx = 0
            for play in bucket_plays:
                if not isinstance(play, dict):
                    continue
                text = _first(play, _TEXT_KEYS)
                if not isinstance(text, str) or not text.strip():
                    continue
                if _BOILERPLATE_RE.match(text):
                    continue

                if bucket_side == "auto":
                    home_bat = _first(play, _HOME_BATTING_KEYS)
                    if home_bat is None:
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
                        # batting_team_id carries the numeric teamId when known
                        # (Shape C); used downstream for name resolution before
                        # the games-table join populates batting_team.
                        "batting_team_id": batting_tid or "",
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
    """Backfill batting_team / fielding_team from the games table.

    Uses ``top_bottom`` (away bats top, home bats bottom) to assign names.
    Rows that already have a non-empty ``batting_team`` are left unchanged.
    """
    if pbp.empty:
        return pbp
    cols = ["away_team", "home_team", "away_team_id", "home_team_id"]
    available = [c for c in cols if c in games.columns]
    names = games.set_index("game_id")[available].to_dict("index")
    bt: list[str] = []
    ft: list[str] = []
    for r in pbp.itertuples(index=False):
        if getattr(r, "batting_team", ""):
            bt.append(r.batting_team)
            ft.append(getattr(r, "fielding_team", ""))
            continue
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
    game_ids = games["game_id"].astype(str).tolist()
    log.info("pbp: fetching %d games with %d workers", len(game_ids), REQUEST_WORKERS)
    with ThreadPoolExecutor(max_workers=REQUEST_WORKERS) as exe:
        results = list(exe.map(fetch_game_pbp, game_ids))
    frames = [df for df in results if not df.empty]
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
