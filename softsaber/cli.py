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


@ingest_app.command("discover-team-ids")
def ingest_discover_team_ids(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
    verbose: bool = False,
) -> None:
    """Re-run only the stats.ncaa.org team-ID discovery step (no roster scrape).

    Cheap when contest pages are already cached — lets you iterate on the
    name-matching logic without re-fetching ~350 roster pages.  Updates
    ``teams/{season}.parquet`` with a fresh ``stats_ncaa_team_id`` column.
    """
    from . import storage

    _setup_logging(verbose)
    for year in seasons:
        games = storage.read_table("games", partitions=[str(year)])
        teams = storage.read_table("teams", partitions=[str(year)])
        if teams.empty:
            typer.echo(f"season {year}: no teams partition — run `ingest teams` first")
            continue
        rosters_mod.discover_and_update_teams(teams, games, year)


@ingest_app.command("inspect-rosters")
def ingest_inspect_rosters(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
) -> None:
    """Print rosters parquet shape, columns, and a sample to debug join keys."""
    from . import storage

    for season in seasons:
        df = storage.read_table("rosters", partitions=[str(season)])
        typer.echo(f"\n=== season {season} ===")
        typer.echo(f"rows={len(df)}, columns={list(df.columns)}")
        if df.empty:
            continue
        if "team_id" in df.columns:
            tid = df["team_id"].astype(str)
            typer.echo(f"team_id unique values: {tid.nunique()} "
                       f"(empty/nan: {(tid.isin(['', 'nan', 'None'])).sum()})")
            typer.echo(f"team_id sample: {tid.head(5).tolist()}")
        else:
            typer.echo("MISSING team_id column — re-run `ingest rosters --season N`")
        cols = [c for c in ["team_id", "team_name", "stats_ncaa_team_id",
                             "first_name", "last_name", "jersey"] if c in df.columns]
        typer.echo(df[cols].head(5).to_string(index=False))


@ingest_app.command("unmatched-teams")
def ingest_unmatched_teams(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
) -> None:
    """List teams missing ``stats_ncaa_team_id`` (no rosters will be scraped).

    These are the henrygd team names that didn't resolve to a stats.ncaa.org
    team ID during ``ingest rosters`` discovery — usually a name-format
    mismatch ("Lamar University" vs "Lamar") rather than a defunct school.
    """
    from . import storage

    for season in seasons:
        teams = storage.read_table("teams", partitions=[str(season)])
        if teams.empty:
            typer.echo(f"season {season}: no teams partition")
            continue
        if "stats_ncaa_team_id" not in teams.columns:
            typer.echo(f"season {season}: no stats_ncaa_team_id column — run `ingest discover-team-ids` first")
            continue
        miss = teams[teams["stats_ncaa_team_id"].isna()]
        typer.echo(f"season {season}: {len(miss)}/{len(teams)} unmatched")
        if not miss.empty:
            for n in sorted(miss["team_name"].dropna().unique().tolist()):
                typer.echo(f"  {n}")


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
    """Fetch and parse boxscores for every game in the matching games partition.

    Populates the raw JSON cache (``data/raw/ncaa_api/boxscore/``) and writes
    a ``game_players`` parquet partition with per-player batting/pitching lines.
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
    """End-to-end ingest: scoreboard → teams → rosters → boxscore → pbp.

    Rosters come before boxscore/pbp so name resolution can use the
    authoritative stats.ncaa.org player list rather than the degraded
    boxscore-derived names.
    """
    _setup_logging(verbose)
    from . import storage

    # One BrowserSession across all stats.ncaa.org calls in this run:
    # team-codes, contest discovery, ranking discovery, and roster fetches.
    # The JS challenge is paid once at the first navigation; everything
    # else reuses the cleared Akamai state.  BrowserSession defers the
    # Playwright import to __enter__, so a missing-Playwright environment
    # surfaces as RuntimeError there rather than ImportError on the import
    # of akamai_session — catch both, degrade to curl_cffi-only.
    bs_cm = None
    bs = None
    try:
        from .ingest.akamai_session import BrowserSession
        bs_cm = BrowserSession()
        bs = bs_cm.__enter__()
    except (ImportError, RuntimeError) as e:
        if bs_cm is not None:
            bs_cm = None
        typer.echo(
            f"warning: browser fallback unavailable ({e}); "
            "stats.ncaa.org endpoints behind Akamai will be skipped.",
            err=True,
        )

    def _run_year(year: int, games: pd.DataFrame, partition: str, bs) -> None:
        softball_ids = scoreboard_mod.discover_team_softball_ids(games)
        teams = teams_mod.build_teams_table(softball_ids, year, browser_session=bs)
        teams = rosters_mod.discover_and_update_teams(teams, games, year, browser_session=bs)
        rosters_mod.ingest_season_rosters(teams, games, year, browser_session=bs)
        boxscore_mod.ingest_boxscores_for_games(games, partition=partition)
        pbp_mod.ingest_pbp_for_games(games, year, partition)

    try:
        if day:
            d = _parse_date(day)
            sn = Season(d.year, division)
            partition = _date_partition(d.year, d)
            scoreboard_mod.ingest_date(sn, d)
            games = storage.read_table("games", partitions=[partition])
            if games.empty:
                typer.echo(f"no games found for {d.isoformat()}; stopping")
                return
            _run_year(d.year, games, partition, bs)
            return

        years = seasons if seasons else list(TARGET_SEASONS)
        for year in years:
            sn = Season(year, division)
            scoreboard_mod.ingest_season(sn)
            games = storage.read_table("games", partitions=[str(year)])
            _run_year(year, games, str(year), bs)
    finally:
        if bs_cm is not None:
            bs_cm.__exit__(None, None, None)


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
    df = rosters_mod.ingest_season_rosters(teams, games, year)
    typer.echo(f"roster rows written: {len(df)}")


def _partition_key(seasons: list[int]) -> str:
    return "_".join(str(s) for s in seasons)


def _build_pa(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shared front half of every stats command.

    Loads cached ``pbp_raw``, attributes pitchers, builds the PA table,
    resolves batter names, and computes linear weights.  Returns
    ``(pa, weights, game_players)``.  Raises ``SystemExit`` if no PBP
    data is cached.
    """
    from . import storage
    from .parse.pa import build_pa_table, resolve_batter_names
    from .parse.pitcher import attribute_pitchers
    from .stats.linear_weights import compute_linear_weights
    from .stats.run_expectancy import compute_re24, compute_re_matrix

    parts = [str(s) for s in seasons]
    pbp = storage.read_table("pbp_raw", partitions=parts)
    if pbp.empty:
        raise SystemExit("no pbp_raw partitions found — run `ingest pbp` first")

    game_players = storage.read_table("game_players", partitions=parts)
    if not game_players.empty:
        pbp = attribute_pitchers(pbp, game_players)
    else:
        typer.echo("warning: no game_players data — pitcher attribution skipped.", err=True)

    pa = build_pa_table(pbp)

    rosters = storage.read_table("rosters", partitions=parts)
    if game_players.empty and rosters.empty:
        typer.echo(
            "warning: no game_players or rosters data — batter names will be raw PBP tokens. "
            "Run `ingest boxscore` and `ingest rosters` to enable name resolution.",
            err=True,
        )
    else:
        pa = resolve_batter_names(
            pa,
            game_players if not game_players.empty else pd.DataFrame(),
            rosters=rosters if not rosters.empty else None,
        )

    pa_re24 = compute_re24(pa, compute_re_matrix(pa))
    weights = compute_linear_weights(pa_re24)
    return pa, weights, game_players


def _emit_table(df: pd.DataFrame, cols: list[str], csv: str | None) -> None:
    """Print selected columns to stdout, or write the full frame to CSV."""
    if csv:
        df.to_csv(csv, index=False)
        typer.echo(f"wrote {len(df)} rows to {csv}")
        return
    present = [c for c in cols if c in df.columns]
    typer.echo(df[present].to_string(index=False))


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
    from .stats.park_factors import multi_year_park_factors
    from .stats.wrc_plus import player_wrc_plus

    pa, weights, _ = _build_pa(seasons)

    games_by_year = {
        y: storage.read_table("games", partitions=[str(y)]) for y in seasons
    }
    games_by_year = {y: g for y, g in games_by_year.items() if not g.empty}
    pf = multi_year_park_factors(games_by_year) if games_by_year else None

    key = _partition_key(seasons)
    wrc = player_wrc_plus(pa, weights, pf)
    storage.write_partition("wrc_plus", key, wrc)
    storage.write_partition("linear_weights", key, weights)
    if pf is not None and not pf.empty:
        storage.write_partition("park_factors", key, pf)

    leader = wrc.sort_values("wRC+", ascending=False).head(15)
    typer.echo(leader[["season", "batter", "batting_team", "PA", "wOBA", "wRC+"]].to_string(index=False))


@stats_app.command("batters")
def stats_batters(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
    min_pa: Annotated[int, typer.Option(help="Minimum PA to display.")] = 50,
    sort: Annotated[str, typer.Option(help="Column to sort by (descending).")] = "OPS",
    top: Annotated[int, typer.Option(help="Rows to show (0 = all).")] = 25,
    csv: Annotated[str | None, typer.Option(help="Write full table to this CSV path.")] = None,
    verbose: bool = False,
) -> None:
    """Batter rate stats: slash line, plate discipline, batted-ball-out mix.

    Writes ``data/processed/batter_rates/<seasons>.parquet`` and prints a
    leaderboard (or dumps to ``--csv``).
    """
    _setup_logging(verbose)
    from . import storage
    from .stats.rates import batter_rates

    pa, _, _ = _build_pa(seasons)
    df = batter_rates(pa)
    if df.empty:
        raise SystemExit("no batter rows produced")

    storage.write_partition("batter_rates", _partition_key(seasons), df)

    shown = df[df["PA"] >= min_pa].copy()
    if sort in shown.columns:
        shown = shown.sort_values(sort, ascending=False)
    if top > 0:
        shown = shown.head(top)
    _emit_table(
        shown,
        ["season", "player", "team", "PA", "AB", "H", "HR", "AVG", "OBP",
         "SLG", "OPS", "ISO", "BABIP", "K_pct", "BB_pct", "GB_pct_bbo"],
        csv,
    )


@stats_app.command("pitchers")
def stats_pitchers(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
    min_tbf: Annotated[int, typer.Option(help="Minimum batters faced to display.")] = 100,
    sort: Annotated[str, typer.Option(help="Column to sort by (ascending for ERA-like).")] = "softSIERA",
    top: Annotated[int, typer.Option(help="Rows to show (0 = all).")] = 25,
    csv: Annotated[str | None, typer.Option(help="Write full table to this CSV path.")] = None,
    verbose: bool = False,
) -> None:
    """Pitcher rate stats: K%/BB%, BAA, ERA/WHIP, xFIP, and softSIERA.

    Writes ``data/processed/pitcher_rates/<seasons>.parquet`` and prints a
    leaderboard sorted by ``--sort`` (ascending, since the headline metrics
    are ERA-like where lower is better).
    """
    _setup_logging(verbose)
    from . import storage
    from .stats.fielding_independent import add_soft_siera, add_xfip
    from .stats.rates import pitcher_rates

    pa, weights, game_players = _build_pa(seasons)
    df = pitcher_rates(pa, game_players)
    if df.empty:
        raise SystemExit("no pitcher rows produced")

    df = add_xfip(df, weights=weights)
    df = add_soft_siera(df, min_tbf=min_tbf)

    storage.write_partition("pitcher_rates", _partition_key(seasons), df)

    shown = df[df["TBF"] >= min_tbf].copy()
    # ERA-like metrics sort ascending (lower is better); rate metrics descending.
    ascending = sort in {"ERA", "WHIP", "BB7", "softSIERA", "xFIP", "BAA"}
    if sort in shown.columns:
        shown = shown.sort_values(sort, ascending=ascending)
    if top > 0:
        shown = shown.head(top)
    _emit_table(
        shown,
        ["season", "player", "team", "TBF", "IP", "ERA", "WHIP", "K_pct",
         "BB_pct", "K7", "BB7", "BAA", "xFIP", "softSIERA"],
        csv,
    )


@stats_app.command("unresolved-batters")
def stats_unresolved_batters(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
    top: Annotated[int, typer.Option(help="How many distinct tokens to show.")] = 40,
    summary: Annotated[bool, typer.Option(help="Bucket ALL unresolved by cause.")] = False,
    verbose: bool = False,
) -> None:
    """List the most-frequent PBP batter strings that ``resolve_batter_names``
    couldn't match against a roster row.  For each token, also reports the
    team's roster / game_players size and how many of those rows share the
    trailing surname — which separates a missing-roster bug (team has 0 rows)
    from a genuine ambiguity (team has 2+ same-surname rows).

    With ``--summary``, also buckets every unresolved batter (not just the
    top-N) by cause so we can see whether the residual is mostly structural
    floor or fixable matcher gaps.
    """
    _setup_logging(verbose)
    from . import storage
    from .parse.nameutil import _normalize

    pa, _, game_players = _build_pa(seasons)
    if "batter_resolved" not in pa.columns:
        raise SystemExit("PA table has no batter_resolved flag — rerun stats wrc / batters")
    unresolved = pa[~pa["batter_resolved"].fillna(False)]
    total = len(pa)
    n_unres = len(unresolved)
    typer.echo(f"{n_unres}/{total} unresolved ({100*n_unres/total:.1f}%)")

    rosters = storage.read_table("rosters", partitions=[str(s) for s in seasons])

    # Pre-index rosters and game_players by team_id with a normalized surname
    # column so we can probe (#roster rows, #surname matches) per (token, team).
    # rosters carries only player_name (NCAA "Last, First"); derive last_name
    # the same way resolve_batter_names does so the diagnostic mirrors it.
    def _index_by_team(df: pd.DataFrame, derive_last: bool = False) -> dict[str, pd.Series]:
        if df.empty or "team_id" not in df.columns:
            return {}
        if "last_name" in df.columns:
            ln_raw = df["last_name"].fillna("")
        elif derive_last and "player_name" in df.columns:
            split = df["player_name"].fillna("").str.split(",", n=1, expand=True)
            ln_raw = split[0].fillna("")
            no_comma = ln_raw.str.strip() == df["player_name"].fillna("").str.strip()
            if no_comma.any():
                alt = df.loc[no_comma, "player_name"].fillna("").str.rsplit(" ", n=1, expand=True)
                if alt.shape[1] == 2:
                    ln_raw.loc[no_comma] = alt[1].fillna("")
        else:
            return {}
        ln = ln_raw.map(_normalize)
        out: dict[str, pd.Series] = {}
        for tid, idx in df.groupby(df["team_id"].astype(str)).groups.items():
            out[str(tid)] = ln.loc[idx]
        return out

    roster_ln = _index_by_team(rosters, derive_last=True)
    gp_ln = _index_by_team(game_players)

    # rosters.team_id is the seoname slug; pa.batting_team_id is numeric.
    # Build the numeric→seoname bridge from game_players (which has both).
    team_seo_by_id: dict[str, str] = {}
    if not game_players.empty and "team_seoname" in game_players.columns:
        for tid, seo in zip(
            game_players["team_id"].astype(str),
            game_players["team_seoname"].astype(str),
        ):
            if tid and seo and tid not in team_seo_by_id:
                team_seo_by_id[tid] = seo

    has_tid = "batting_team_id" in unresolved.columns
    grouped = (
        unresolved.assign(
            batting_team_id=unresolved["batting_team_id"].astype(str)
            if has_tid else ""
        )
        .groupby(["batter", "batting_team", "batting_team_id"])
        .size().reset_index(name="n")
        .sort_values("n", ascending=False)
        .head(top)
    )

    rows = []
    for r in grouped.itertuples(index=False):
        tail = r.batter.split(",")[0].split()[-1] if r.batter else ""
        surname = _normalize(tail)
        tid = str(r.batting_team_id)
        seo = team_seo_by_id.get(tid, tid)
        ros = roster_ln.get(seo, pd.Series(dtype=str))
        gp = gp_ln.get(tid, pd.Series(dtype=str))
        rows.append({
            "n": r.n,
            "batter": r.batter,
            "team": r.batting_team,
            "ros_n": len(ros),
            "ros_hits": int((ros == surname).sum()) if len(ros) else 0,
            "gp_n": len(gp),
            "gp_hits": int((gp == surname).sum()) if len(gp) else 0,
        })
    diag = pd.DataFrame(rows)
    typer.echo(diag.to_string(index=False))

    if not summary:
        return

    # Bucket EVERY unresolved PA row (not just the top-N tokens) by failure
    # cause.  We iterate the unresolved frame directly — earlier attempts via
    # groupby silently dropped rows with NaN keys, undercounting by ~99%.
    buckets = {
        "no batting_team_id (Shape-A PBP)": 0,
        "non-D1 team (no roster)": 0,
        "roster gap — 0 surname matches (fixable: format)": 0,
        "matcher bug — 1 surname match (should resolve)": 0,
        "genuine ambiguity — 2+ surname matches (floor)": 0,
        "no roster AND no game_players surname match": 0,
        "missing batter token": 0,
    }
    seen = 0
    for r in unresolved.itertuples(index=False):
        seen += 1
        batter = getattr(r, "batter", None)
        if not isinstance(batter, str) or not batter:
            buckets["missing batter token"] += 1
            continue
        tid_val = getattr(r, "batting_team_id", "") if has_tid else ""
        tid = str(tid_val) if tid_val is not None else ""
        if not tid or tid == "nan":
            buckets["no batting_team_id (Shape-A PBP)"] += 1
            continue
        seo = team_seo_by_id.get(tid, tid)
        ros = roster_ln.get(seo, pd.Series(dtype=str))
        gp = gp_ln.get(tid, pd.Series(dtype=str))
        tail = batter.split(",")[0].split()
        surname = _normalize(tail[-1]) if tail else ""
        ros_hits = int((ros == surname).sum()) if len(ros) else 0
        gp_hits = int((gp == surname).sum()) if len(gp) else 0
        if len(ros) == 0:
            if gp_hits == 0:
                buckets["no roster AND no game_players surname match"] += 1
            else:
                buckets["non-D1 team (no roster)"] += 1
        elif ros_hits == 0:
            buckets["roster gap — 0 surname matches (fixable: format)"] += 1
        elif ros_hits == 1:
            buckets["matcher bug — 1 surname match (should resolve)"] += 1
        else:
            buckets["genuine ambiguity — 2+ surname matches (floor)"] += 1

    typer.echo("\n=== unresolved-batter buckets (across ALL %d unresolved, %d scanned) ==="
               % (n_unres, seen))
    width = max(len(k) for k in buckets)
    for k, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        pct = 100 * v / n_unres if n_unres else 0
        typer.echo(f"  {k.ljust(width)}  {v:>7d}  ({pct:5.1f}%)")


@stats_app.command("export")
def stats_export(
    seasons: Annotated[list[int], typer.Option("--seasons", "-s")] = list(TARGET_SEASONS),
    fmt: Annotated[str, typer.Option("--format", help="csv | json | both")] = "json",
    out: Annotated[str, typer.Option("--out", help="Output directory.")] = "exports",
    sharded: Annotated[bool, typer.Option(help="JSON: one file per player.")] = False,
    verbose: bool = False,
) -> None:
    """Export rate stats for downstream consumers (website, Firestore, spreadsheets).

    Reads cached rate tables (``batter_rates`` / ``pitcher_rates`` / ``wrc_plus``)
    plus the rosters table, joins on (season, team, player) to attach
    ``ncaa_player_id`` where available, and writes:

    * ``<out>/batters.csv``, ``pitchers.csv``, ``wrc.csv`` for CSV.
    * ``<out>/players.json`` (or sharded ``<out>/players/<id>.json``)
      for JSON — one Firestore-shaped document per player with identity
      at the top and per-season stats nested under ``seasons.<year>``.

    Run after ``stats batters``, ``stats pitchers``, and ``stats wrc`` so
    the rate tables are populated.
    """
    _setup_logging(verbose)
    from pathlib import Path

    from . import storage
    from .export import build_player_documents, write_csv, write_json

    key = _partition_key(seasons)
    parts = [str(s) for s in seasons]
    batters = storage.read_table("batter_rates", partitions=[key])
    pitchers = storage.read_table("pitcher_rates", partitions=[key])
    wrc = storage.read_table("wrc_plus", partitions=[key])
    rosters = storage.read_table("rosters", partitions=parts)

    if batters.empty and pitchers.empty and wrc.empty:
        raise SystemExit(
            "no rate tables found for these seasons — run `stats batters`, "
            "`stats pitchers`, `stats wrc` first"
        )

    out_dir = Path(out)
    fmt = fmt.lower()
    if fmt not in {"csv", "json", "both"}:
        raise typer.BadParameter("--format must be csv, json, or both")

    if fmt in {"csv", "both"}:
        csv_paths = write_csv(out_dir, batters, pitchers, wrc if not wrc.empty else None)
        for name, path in csv_paths.items():
            typer.echo(f"csv {name}: {path}")

    if fmt in {"json", "both"}:
        docs = build_player_documents(
            rosters, batters, pitchers, wrc if not wrc.empty else None,
        )
        result = write_json(out_dir, docs, sharded=sharded)
        with_id = sum(1 for d in docs if not d.get("id_synthesized"))
        typer.echo(
            f"json: {result.get('players') or result.get('players_dir')} "
            f"({result['count']} players, {with_id} with ncaa_player_id)"
        )


if __name__ == "__main__":
    app()
