"""Command-line entry point.

Examples::

    softsaber ingest scoreboard --season 2024
    softsaber ingest pbp --season 2024
    softsaber ingest all --seasons 2024 2025 2026
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated

import typer

from .config import Season, TARGET_DIVISION, TARGET_SEASONS
from .ingest import pbp as pbp_mod
from .ingest import scoreboard as scoreboard_mod
from .ingest import teams as teams_mod

app = typer.Typer(help="Softball analytics ingest + stats CLI.")
ingest_app = typer.Typer(help="Pull data from NCAA's GraphQL API (sdataprod.ncaa.com).")
stats_app = typer.Typer(help="Compute advanced stats from processed PBP.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(stats_app, name="stats")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


@ingest_app.command("scoreboard")
def ingest_scoreboard(
    season: Annotated[int, typer.Option(help="Season year, e.g. 2024.")] = 2024,
    division: Annotated[str, typer.Option(help="D1/D2/D3")] = TARGET_DIVISION,
    day: Annotated[
        str | None,
        typer.Option(
            "--date",
            help="Single date (YYYY-MM-DD) to ingest instead of the full season. "
            "Useful for smoke-testing without a full-season scrape.",
        ),
    ] = None,
    verbose: bool = False,
) -> None:
    """Walk a season's scoreboard (or one day) and write a ``games`` partition."""
    _setup_logging(verbose)
    sn = Season(season, division)
    if day:
        try:
            d = datetime.strptime(day, "%Y-%m-%d").date()
        except ValueError as e:
            raise typer.BadParameter(f"--date must be YYYY-MM-DD: {e}") from None
        df = scoreboard_mod.ingest_date(sn, d)
    else:
        df = scoreboard_mod.ingest_season(sn)
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
