"""Command-line entry point.

Examples::

    softsaber ingest scoreboard --season 2024
    softsaber ingest pbp --season 2024
    softsaber ingest all --seasons 2024 2025 2026
"""

from __future__ import annotations

import logging
from typing import Annotated

import typer

from .config import Season, TARGET_DIVISION, TARGET_SEASONS
from .ingest import pbp as pbp_mod
from .ingest import scoreboard as scoreboard_mod
from .ingest import teams as teams_mod

app = typer.Typer(help="Softball analytics ingest + stats CLI.")
ingest_app = typer.Typer(help="Pull data from stats.ncaa.org.")
app.add_typer(ingest_app, name="ingest")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@ingest_app.command("scoreboard")
def ingest_scoreboard(
    season: Annotated[int, typer.Option(help="Season year, e.g. 2024.")] = 2024,
    division: Annotated[str, typer.Option(help="D1/D2/D3")] = TARGET_DIVISION,
    verbose: bool = False,
) -> None:
    """Walk a season's scoreboard and write ``data/processed/games/<year>.parquet``."""
    _setup_logging(verbose)
    df = scoreboard_mod.ingest_season(Season(season, division))
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
    season: int = 2024,
    verbose: bool = False,
) -> None:
    """Pull play-by-play for every game in the season partition."""
    _setup_logging(verbose)
    from . import storage

    games = storage.read_table("games", partitions=[str(season)])
    if games.empty:
        raise SystemExit(f"no games partition for {season} — run `ingest scoreboard` first")
    df = pbp_mod.ingest_season_pbp(games, season)
    typer.echo(f"pbp rows written: {len(df)}")


@ingest_app.command("all")
def ingest_all(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
    division: str = TARGET_DIVISION,
    verbose: bool = False,
) -> None:
    """End-to-end ingest for one or more seasons (scoreboard → teams → pbp)."""
    _setup_logging(verbose)
    from . import storage

    for year in seasons:
        sn = Season(year, division)
        scoreboard_mod.ingest_season(sn)
        games = storage.read_table("games", partitions=[str(year)])
        softball_ids = scoreboard_mod.discover_team_softball_ids(games)
        teams_mod.build_teams_table(softball_ids, year)
        pbp_mod.ingest_season_pbp(games, year)


if __name__ == "__main__":
    app()
