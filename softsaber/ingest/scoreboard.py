"""Season scoreboard ingest, via the ncaa-api.henrygd.me REST wrapper.

For each calendar day in a season we pull the list of contests, then keep
the ones that are final. The output is the join key for everything else:
a ``games`` table keyed by ``game_id`` (NCAA's ``gameID``) that points at
the boxscore and play-by-play endpoints.

The sdataprod GraphQL scoreboard was the original source here, but returns
empty payloads for prior-season dates; ``data.ncaa.com``'s casablanca
bucket doesn't have per-date keys for softball either. henrygd's wrapper
scrapes the live ncaa.com scoreboard page and is what works today.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from .. import storage
from ..config import Season
from . import ncaa_api

log = logging.getLogger(__name__)

SEASON_WINDOWS: dict[int, tuple[date, date]] = {
    # Inclusive start/end. Tune per-year if regionals/WCWS dates shift.
    2024: (date(2024, 2, 8), date(2024, 6, 7)),
    2025: (date(2025, 2, 6), date(2025, 6, 6)),
    2026: (date(2026, 2, 5), date(2026, 6, 5)),
}


@dataclass
class GameRow:
    game_id: str
    game_date: str
    away_team: str
    away_team_id: str
    away_team_runs: int | None
    home_team: str
    home_team_id: str
    home_team_runs: int | None
    status: str


def _iter_dates(start: date, end: date) -> Iterable[date]:
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def _to_int(v: Any) -> int | None:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _team_id(side: dict[str, Any]) -> str:
    """Stable team identifier from a casablanca side block.

    Casablanca reports the team URL slug in ``names.seo`` and sometimes also a
    numeric ``teamId``. Prefer numeric when present so the team-codes join
    round-trips cleanly.
    """
    if side.get("teamId"):
        return str(side["teamId"])
    names = side.get("names") or {}
    return str(names.get("seo") or names.get("short") or "")


def _team_name(side: dict[str, Any]) -> str:
    names = side.get("names") or {}
    return str(names.get("short") or names.get("full") or names.get("seo") or "")


def parse_scoreboard(payload: dict[str, Any], *, only_final: bool = True) -> list[GameRow]:
    """Flatten one casablanca scoreboard payload into ``GameRow``s.

    Reads ``payload["games"]``, where each entry is ``{"game": {...}}`` with
    ``home``/``away`` side blocks and a ``gameState`` of ``final``,
    ``live``, or ``pre``.
    """
    entries = (payload or {}).get("games") or []
    rows: list[GameRow] = []
    for entry in entries:
        g = entry.get("game") if isinstance(entry, dict) else None
        if not g:
            continue
        state = str(g.get("gameState") or "").lower()
        if only_final and state != "final":
            continue
        home = g.get("home") or {}
        away = g.get("away") or {}
        if not home or not away:
            log.debug("contest %s missing home/away split, skipping", g.get("gameID"))
            continue
        rows.append(
            GameRow(
                game_id=str(g.get("gameID") or ""),
                game_date=str(g.get("startDate") or ""),
                away_team=_team_name(away),
                away_team_id=_team_id(away),
                away_team_runs=_to_int(away.get("score")),
                home_team=_team_name(home),
                home_team_id=_team_id(home),
                home_team_runs=_to_int(home.get("score")),
                status="Final",
            )
        )
    return rows


def fetch_day(season: Season, d: date) -> list[GameRow]:
    payload = ncaa_api.fetch_scoreboard(
        sport_path=ncaa_api.SPORT_PATH_SOFTBALL,
        division=season.division,
        contest_date=f"{d.year:04d}/{d.month:02d}/{d.day:02d}",
    )
    return parse_scoreboard(payload)


def _write_games(rows: list[GameRow], season: Season, partition: str) -> pd.DataFrame:
    df = pd.DataFrame([g.__dict__ for g in rows])
    df["season"] = season.year
    df["division"] = season.division
    storage.write_partition("games", partition, df)
    return df


def ingest_date(season: Season, d: date) -> pd.DataFrame:
    """Pull and persist one day's scoreboard. Useful for smoke-testing ingest.

    Writes a dated partition (``games/<year>-<MM>-<DD>.parquet``) so it
    doesn't clobber the full-season file.
    """
    try:
        rows = fetch_day(season, d)
    except Exception as e:  # noqa: BLE001
        log.warning("scoreboard %s failed: %s", d, e, exc_info=log.isEnabledFor(logging.DEBUG))
        rows = []
    log.info("scoreboard %s: %d games", d, len(rows))
    return _write_games(rows, season, f"{season.year}-{d.month:02d}-{d.day:02d}")


def ingest_season(season: Season) -> pd.DataFrame:
    """Walk a season's calendar and write a ``games`` parquet partition."""
    if season.year not in SEASON_WINDOWS:
        raise KeyError(f"no scoreboard window configured for {season.year}")
    start, end = SEASON_WINDOWS[season.year]

    from tqdm import tqdm
    dates = list(_iter_dates(start, end))
    all_games: list[GameRow] = []
    for d in tqdm(dates, desc=f"scoreboard {season.year}", unit="day"):
        try:
            day_games = fetch_day(season, d)
        except Exception as e:  # noqa: BLE001 — keep ingesting other days
            log.warning("scoreboard %s failed: %s", d, e, exc_info=log.isEnabledFor(logging.DEBUG))
            continue
        all_games.extend(day_games)

    return _write_games(all_games, season, str(season.year))


def discover_team_softball_ids(games: pd.DataFrame) -> pd.DataFrame:
    """Collapse a games table into the unique (team_name, softball_id) map."""
    away = games[["away_team", "away_team_id"]].rename(
        columns={"away_team": "team_name", "away_team_id": "softball_id"}
    )
    home = games[["home_team", "home_team_id"]].rename(
        columns={"home_team": "team_name", "home_team_id": "softball_id"}
    )
    return pd.concat([away, home]).drop_duplicates().reset_index(drop=True)


__all__ = [
    "GameRow",
    "SEASON_WINDOWS",
    "discover_team_softball_ids",
    "fetch_day",
    "ingest_date",
    "ingest_season",
    "parse_scoreboard",
]
