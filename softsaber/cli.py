"""Command-line entry point.

Examples::

    softsaber ingest scoreboard --season 2024
    softsaber ingest scoreboard --date 2024-05-04
    softsaber ingest pbp --date 2024-05-04
    softsaber ingest boxscore --date 2024-05-04
    softsaber ingest all --seasons 2024 2025 2026
    softsaber ingest all --date 2024-05-04
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Annotated

import pandas as pd
import typer

from .config import Season, TARGET_DIVISION, TARGET_SEASONS
from .ingest import boxscore as boxscore_mod
from .ingest import pbp as pbp_mod
from .ingest import rosters as rosters_mod
from .ingest import scoreboard as scoreboard_mod
from .ingest import teams as teams_mod

app = typer.Typer(help="Softball analytics ingest + stats CLI.")
ingest_app = typer.Typer(help="Pull data from ncaa-api.henrygd.me.")
stats_app = typer.Typer(help="Compute advanced stats from processed PBP.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(stats_app, name="stats")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _parse_date(s: str) -> date:
    try:
        return datetime.strptime(s, "%Y-%m-%d").date()
    except ValueError as e:
        raise typer.BadParameter(f"--date must be YYYY-MM-DD: {e}") from None


def _resolve_season(season: int | None, day: date | None, ctx: str) -> int:
    """Pick the season year from ``--season`` or fall back to ``--date.year``.

    ``--season`` and ``--date`` are mutually informative: if a date is given,
    its year IS the season, so requiring both would be redundant. Raises if
    neither is provided.
    """
    if season is not None:
        return season
    if day is not None:
        return day.year
    raise typer.BadParameter(f"{ctx}: pass --season or --date")


def _date_partition(season: int, day: date) -> str:
    return f"{season}-{day.month:02d}-{day.day:02d}"


def _games_for(season: int, day: date | None) -> tuple[pd.DataFrame, str]:
    """Read the games partition (single-day or full-season) for downstream ingests."""
    from . import storage

    partition = _date_partition(season, day) if day else str(season)
    games = storage.read_table("games", partitions=[partition])
    if games.empty:
        hint = f"--date {day.isoformat()}" if day else f"--season {season}"
        raise SystemExit(
            f"no games partition for {partition} — run `ingest scoreboard {hint}` first"
        )
    return games, partition


@ingest_app.command("scoreboard")
def ingest_scoreboard(
    season: Annotated[
        int | None,
        typer.Option(help="Season year, e.g. 2024. Required unless --date is given."),
    ] = None,
    division: Annotated[str, typer.Option(help="D1/D2/D3")] = TARGET_DIVISION,
    day: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="Single date (YYYY-MM-DD) to ingest. Season is inferred from "
            "the year if --season isn't passed.",
        ),
    ] = None,
    verbose: bool = False,
) -> None:
    """Walk a season's scoreboard (or one day) and write a ``games`` partition."""
    _setup_logging(verbose)
    d = _parse_date(day) if day else None
    year = _resolve_season(season, d, "ingest scoreboard")
    sn = Season(year, division)
    df = scoreboard_mod.ingest_date(sn, d) if d else scoreboard_mod.ingest_season(sn)
    typer.echo(f"games written: {len(df)}")


@ingest_app.command("teams")
def ingest_teams(
    season: int = 2024,
    division: str = TARGET_DIVISION,
    verbose: bool = False,
) -> None:
    """Join NCAA team codes to softball-side IDs for the given season."""
    _setup_logging(verbose)
    from . import storage

    games = storage.read_table("games", partitions=[str(season)])
    if games.empty:
        raise SystemExit(f"no games partition for {season} — run `ingest scoreboard` first")
    softball_ids = scoreboard_mod.discover_team_softball_ids(games)
    df = teams_mod.build_teams_table(softball_ids, season)
    typer.echo(f"teams written: {len(df)}")


@ingest_app.command("pbp")
def ingest_pbp(
    season: Annotated[int | None, typer.Option(help="Season year. Required unless --date is given.")] = None,
    day: Annotated[
        str | None,
        typer.Option("--date", help="Single date (YYYY-MM-DD). Season inferred from year."),
    ] = None,
    verbose: bool = False,
) -> None:
    """Pull play-by-play for every game in the matching games partition."""
    _setup_logging(verbose)
    d = _parse_date(day) if day else None
    year = _resolve_season(season, d, "ingest pbp")
    games, partition = _games_for(year, d)
    df = pbp_mod.ingest_pbp_for_games(games, year, partition)
    typer.echo(f"pbp rows written: {len(df)}")


@ingest_app.command("boxscore")
def ingest_boxscore(
    season: Annotated[int | None, typer.Option(help="Season year. Required unless --date is given.")] = None,
    day: Annotated[
        str | None,
        typer.Option("--date", help="Single date (YYYY-MM-DD). Season inferred from year."),
    ] = None,
    verbose: bool = False,
) -> None:
    """Warm the boxscore cache for every game in the matching games partition.

    Boxscores aren't parsed into parquet yet — this just populates the raw
    JSON cache (``data/raw/ncaa_api/boxscore/``) for downstream use.
    """
    _setup_logging(verbose)
    d = _parse_date(day) if day else None
    year = _resolve_season(season, d, "ingest boxscore")
    games, partition = _games_for(year, d)
    n = boxscore_mod.ingest_boxscores_for_games(games, partition=partition)
    typer.echo(f"boxscores cached: {n}")


@ingest_app.command("all")
def ingest_all(
    seasons: Annotated[
        list[int] | None,
        typer.Option("--seasons", "-s", help="Season years. Required unless --date is given."),
    ] = None,
    division: str = TARGET_DIVISION,
    day: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="Single date (YYYY-MM-DD) to run end-to-end. Season inferred from year; "
            "--seasons is ignored if set.",
        ),
    ] = None,
    verbose: bool = False,
) -> None:
    """End-to-end ingest (scoreboard → teams → pbp → boxscore) for seasons or one day."""
    _setup_logging(verbose)
    from . import storage

    if day:
        d = _parse_date(day)
        sn = Season(d.year, division)
        partition = _date_partition(d.year, d)
        scoreboard_mod.ingest_date(sn, d)
        games = storage.read_table("games", partitions=[partition])
        if games.empty:
            typer.echo(f"no games found for {d.isoformat()}; stopping")
            return
        softball_ids = scoreboard_mod.discover_team_softball_ids(games)
        teams_mod.build_teams_table(softball_ids, d.year)
        pbp_mod.ingest_pbp_for_games(games, d.year, partition)
        boxscore_mod.ingest_boxscores_for_games(games, partition=partition)
        return

    years = seasons if seasons else list(TARGET_SEASONS)
    for year in years:
        sn = Season(year, division)
        scoreboard_mod.ingest_season(sn)
        games = storage.read_table("games", partitions=[str(year)])
        softball_ids = scoreboard_mod.discover_team_softball_ids(games)
        teams_mod.build_teams_table(softball_ids, year)
        pbp_mod.ingest_season_pbp(games, year)
        boxscore_mod.ingest_boxscores_for_games(games, partition=str(year))


@ingest_app.command("rosters")
def ingest_rosters(
    season: Annotated[int | None, typer.Option(help="Season year.")] = None,
    day: Annotated[
        str | None,
        typer.Option("--date", help="Infer season year from this date (YYYY-MM-DD)."),
    ] = None,
    verbose: bool = False,
) -> None:
    """Discover stats.ncaa.org team IDs and fetch per-player rosters.

    Reads the games and teams partitions for the season, probes the
    stats.ncaa.org individual-stats pages for each game to discover
    year-specific team IDs, then fetches each team's roster page to get
    player names and NCAA player IDs.

    Writes an updated ``teams/{season}.parquet`` (with ``stats_ncaa_team_id``)
    and a ``rosters/{season}.parquet`` (one row per player).

    Tip: fill in ``WSB_D1_RANKING_STAT_SEQ`` in ``config.py`` to enable
    full-division discovery in a single request instead of per-game pages.
    """
    from . import storage

    _setup_logging(verbose)
    d = _parse_date(day) if day else None
    year = _resolve_season(season, d, "ingest rosters")

    games, _ = _games_for(year, d)
    teams = storage.read_table("teams", partitions=[str(year)])
    if teams.empty:
        raise SystemExit(f"no teams partition for {year} — run `ingest teams` first")

    teams = rosters_mod.discover_and_update_teams(teams, games, year)
    df = rosters_mod.ingest_season_rosters(teams, year)
    typer.echo(f"roster rows written: {len(df)}")


@stats_app.command("wrc")
def stats_wrc(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
    verbose: bool = False,
) -> None:
    """End-to-end stats run: PBP → PA → RE24 → wOBA wts → wRC+ leaderboard.

    Reads cached ``pbp_raw`` and ``games`` parquet partitions and writes
    ``data/processed/wrc_plus/<seasons>.parquet``.
    """
    _setup_logging(verbose)
    from . import storage
    from .parse.pa import build_pa_table
    from .stats.linear_weights import compute_linear_weights
    from .stats.park_factors import multi_year_park_factors
    from .stats.run_expectancy import compute_re24, compute_re_matrix
    from .stats.wrc_plus import player_wrc_plus

    pbp = storage.read_table("pbp_raw", partitions=[str(s) for s in seasons])
    if pbp.empty:
        raise SystemExit("no pbp_raw partitions found — run `ingest pbp` first")

    pa = build_pa_table(pbp)
    re = compute_re_matrix(pa)
    pa_re24 = compute_re24(pa, re)
    weights = compute_linear_weights(pa_re24)

    games_by_year = {
        y: storage.read_table("games", partitions=[str(y)]) for y in seasons
    }
    games_by_year = {y: g for y, g in games_by_year.items() if not g.empty}
    pf = multi_year_park_factors(games_by_year) if games_by_year else None

    wrc = player_wrc_plus(pa, weights, pf)
    storage.write_partition("wrc_plus", "_".join(str(s) for s in seasons), wrc)
    storage.write_partition("linear_weights", "_".join(str(s) for s in seasons), weights)
    if pf is not None and not pf.empty:
        storage.write_partition("park_factors", "_".join(str(s) for s in seasons), pf)

    leader = wrc.sort_values("wRC+", ascending=False).head(15)
    typer.echo(leader[["season", "batter", "batting_team", "PA", "wOBA", "wRC+"]].to_string(index=False))


if __name__ == "__main__":
    app()
