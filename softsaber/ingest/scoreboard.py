"""Season scoreboard ingest, via the NCAA GraphQL API.

For each calendar day in a season we hit ``sdataprod.ncaa.com`` and pull the
list of contests, then keep the ones that are Final. The output is the join
key for everything else: a ``games`` table keyed by ``game_id`` (NCAA's
``contestId``) that points at the boxscore and play-by-play endpoints.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

import pandas as pd

from .. import storage
from ..config import SPORT_CODE_SOFTBALL, Season
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


def parse_scoreboard(payload: dict[str, Any], *, only_final: bool = True) -> list[GameRow]:
    """Flatten one GraphQL scoreboard payload into ``GameRow``s.

    Reads ``payload["data"]["contests"]``. Each contest has a ``teams`` array
    with ``isHome`` discriminating sides; the ``gameState`` field is "F" for
    finalized, "I" in-progress, "P" pre-game.
    """
    contests = ((payload or {}).get("data") or {}).get("contests") or []
    games: list[GameRow] = []
    for c in contests:
        state = c.get("gameState") or ""
        if only_final and state != "F":
            continue
        teams = c.get("teams") or []
        home = next((t for t in teams if t.get("isHome")), None)
        away = next((t for t in teams if not t.get("isHome")), None)
        if not home or not away:
            log.debug("contest %s missing home/away split, skipping", c.get("contestId"))
            continue
        games.append(
            GameRow(
                game_id=str(c.get("contestId") or ""),
                game_date=str(c.get("startDate") or ""),
                away_team=str(away.get("nameShort") or ""),
                away_team_id=str(away.get("teamId") or away.get("seoname") or ""),
                away_team_runs=_to_int(away.get("score")),
                home_team=str(home.get("nameShort") or ""),
                home_team_id=str(home.get("teamId") or home.get("seoname") or ""),
                home_team_runs=_to_int(home.get("score")),
                status="Final",
            )
        )
    return games


def fetch_day(season: Season, d: date) -> list[GameRow]:
    payload = ncaa_api.fetch_scoreboard(
        sport_code=SPORT_CODE_SOFTBALL,
        division=season.division_code,
        season_year=season.year,
        contest_date=f"{d.year:04d}/{d.month:02d}/{d.day:02d}",
    )
    return parse_scoreboard(payload)


def ingest_season(season: Season) -> pd.DataFrame:
    """Walk a season's calendar and write a ``games`` parquet partition."""
    if season.year not in SEASON_WINDOWS:
        raise KeyError(f"no scoreboard window configured for {season.year}")
    start, end = SEASON_WINDOWS[season.year]

    all_games: list[GameRow] = []
    for d in _iter_dates(start, end):
        try:
            day_games = fetch_day(season, d)
        except Exception as e:  # noqa: BLE001 — keep ingesting other days
            log.warning("scoreboard %s failed: %s", d, e, exc_info=log.isEnabledFor(logging.DEBUG))
            continue
        all_games.extend(day_games)
        log.info("scoreboard %s: %d games", d, len(day_games))

    df = pd.DataFrame([g.__dict__ for g in all_games])
    df["season"] = season.year
    df["division"] = season.division
    storage.write_partition("games", str(season.year), df)
    return df


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
    "ingest_season",
    "parse_scoreboard",
]
