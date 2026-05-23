"""Season scoreboard ingest.

For each calendar day in a season, hit the livestream scoreboard endpoint and
extract one row per Final game. The output is the join key for everything
else: a ``games`` table keyed by ``game_id`` that points at the box-score and
play-by-play endpoints.

URL pattern (from softballR):
    https://stats.ncaa.org/season_divisions/{division_id}/livestream_scoreboards
    ?utf8=%E2%9C%93&season_division_id=&game_date={M}%2F{D}%2F{YYYY}
    &conference_id=0&tournament_id=&commit=Submit

NCAA softball seasons run early-February through early-June, with the WCWS
ending in the first week of June.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
from lxml import html as lxml_html

from .. import storage
from ..config import Season
from ..http_cache import fetch

log = logging.getLogger(__name__)

SCOREBOARD_URL_FMT = (
    "https://stats.ncaa.org/season_divisions/{div_id}/livestream_scoreboards"
    "?utf8=%E2%9C%93&season_division_id=&game_date={m}%2F{d}%2F{y}"
    "&conference_id=0&tournament_id=&commit=Submit"
)

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


_RE_CONTEST_ID = re.compile(r'<tr id="contest_(\d+)"')
_RE_LOGO_ALT = re.compile(
    r'<img height="20px" width="30px" alt="([^"]+)" src="([^"]+)"'
)
_RE_TEAM_HREF = re.compile(
    r'<a target="TEAMS_WIN" class="skipMask" href="/teams/(\d+)">'
)


def parse_scoreboard(html: str) -> list[GameRow]:
    """Parse the scoreboard HTML for one day into a list of finalized games.

    The page is awkward — softballR uses line-grep against raw HTML rather
    than CSS because the markup has empty <td> placeholders for non-final
    games. We do the same here.
    """
    # Each game block sits between successive ``<tr id="contest_...">`` rows.
    lines = html.splitlines()
    contest_lines: list[int] = [i for i, ln in enumerate(lines) if _RE_CONTEST_ID.search(ln)]
    if not contest_lines:
        return []

    games: list[GameRow] = []
    # NCAA renders two ``<tr id="contest_X">`` per game (away row + home row)
    # so we step by 2 and look ahead up to ~70 lines for the block.
    for idx in range(0, len(contest_lines), 2):
        start = contest_lines[idx]
        end = (
            contest_lines[idx + 2]
            if idx + 2 < len(contest_lines)
            else min(start + 70, len(lines))
        )
        block = "\n".join(lines[start:end])

        if "Canceled" in block or "Ppd" in block:
            continue

        m_id = _RE_CONTEST_ID.search(block)
        if not m_id:
            continue
        game_id = m_id.group(1)

        team_alts = _RE_LOGO_ALT.findall(block)
        team_hrefs = _RE_TEAM_HREF.findall(block)
        if len(team_alts) < 2 or len(team_hrefs) < 2:
            continue

        # Score divs are emitted in away/home order. The score sits on the
        # line *after* the ``<div id="score_..."``.
        score_idxs = [i for i, ln in enumerate(lines[start:end]) if 'id="score_' in ln]
        if len(score_idxs) < 2:
            continue
        away_runs = lines[start + score_idxs[0] + 1].strip()
        home_runs = lines[start + score_idxs[1] + 1].strip()

        status_idxs = [i for i, ln in enumerate(lines[start:end]) if 'class="livestream' in ln]
        status = lines[start + status_idxs[0] + 1].strip() if status_idxs else ""
        if status != "Final":
            continue

        # Date sits on the line after the rowspan=2 td.
        date_idxs = [
            i
            for i, ln in enumerate(lines[start:end])
            if '<td rowspan="2" valign="middle">' in ln
        ]
        game_date = lines[start + date_idxs[0] + 1].strip() if date_idxs else ""
        game_date = re.sub(r"\s*\(\d\)|\s*<br/>\*If necessary", "", game_date)

        games.append(
            GameRow(
                game_id=game_id,
                game_date=game_date,
                away_team=team_alts[0][0],
                away_team_id=team_hrefs[0],
                away_team_runs=int(away_runs) if away_runs.isdigit() else None,
                home_team=team_alts[1][0],
                home_team_id=team_hrefs[1],
                home_team_runs=int(home_runs) if home_runs.isdigit() else None,
                status=status,
            )
        )
    return games


def fetch_day(season: Season, d: date) -> list[GameRow]:
    url = SCOREBOARD_URL_FMT.format(
        div_id=season.division_id, m=d.month, d=d.day, y=d.year
    )
    html = fetch(url, namespace=f"scoreboard/{season.year}")
    return parse_scoreboard(html)


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
            log.warning("scoreboard %s failed: %s", d, e)
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


# Re-exported for unit tests / notebook use.
__all__ = [
    "GameRow",
    "SEASON_WINDOWS",
    "discover_team_softball_ids",
    "fetch_day",
    "ingest_season",
    "parse_scoreboard",
]
