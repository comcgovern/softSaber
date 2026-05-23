# softsaber

Python pipeline for computing advanced softball statistics — park factors,
linear weights, and wRC+ — from NCAA Division I play-by-play data. Inspired
by the [softballR](https://github.com/sportsdataverse/softballR) R package,
which this project mirrors at the scraping layer and extends with a
sabermetrics layer comparable to what Fangraphs publishes for baseball.

## Status

Skeleton + ingest layer is in place. Statistics modules are stubs.

| layer | module | status |
|---|---|---|
| HTTP cache + retries | `softsaber.http_cache` | done |
| Scoreboard scrape | `softsaber.ingest.scoreboard` | done |
| Play-by-play scrape | `softsaber.ingest.pbp` | done |
| Team/roster scrape | `softsaber.ingest.teams`, `softsaber.ingest.rosters` | done (rosters needs real-page tuning) |
| Player-box scrape | `softsaber.ingest.playerbox` | needs `stat_seq` discovery |
| Event classifier | `softsaber.parse.events` | basic patterns done, needs corpus validation |
| Base-out reconstruction (RE24 prep) | `softsaber.parse.baserunners` | **stub** |
| PA-level table | `softsaber.parse.pa` | scaffold |
| Run expectancy / RE24 | `softsaber.stats.run_expectancy` | math written, depends on RE24 prep |
| Linear weights / wOBA | `softsaber.stats.linear_weights` | math written, depends on RE24 |
| Park factors (multi-year) | `softsaber.stats.park_factors` | single-year done, multi-year stub |
| wRC+ | `softsaber.stats.wrc_plus` | **stub** |

## Architecture

```
                   stats.ncaa.org (HTML scrape)
                              |
                              v
                     softsaber.http_cache
                  (disk-cached, retried GETs)
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
   scoreboard               teams                 pbp
  (games table)        (id mapping)         (raw pbp rows)
                                                    |
                                                    v
                                       softsaber.parse.events
                                       softsaber.parse.baserunners
                                       softsaber.parse.pa
                                            (PA-level table)
                                                    |
                            +-----------------------+----------------+
                            v                       v                v
                  run_expectancy            linear_weights      park_factors
                       (RE24)                 (wOBA wts)       (multi-year)
                            \\____________________ | ____________/
                                                  v
                                              wrc_plus
                                       (player-season wRC+)
```

## Scope (v0.1)

* **Division:** NCAA D1.
* **Seasons:** 2024, 2025, 2026.
* **Base-out approach:** RE24 via text-parsed baserunner movement.

For the data-source constraints behind these choices, see the docstrings on
`softsaber.parse.baserunners` and `softsaber.stats.park_factors`.

## Install & run

```sh
pip install -e .[dev]
softsaber ingest scoreboard --season 2024
softsaber ingest teams --season 2024
softsaber ingest pbp --season 2024
# or, end to end:
softsaber ingest all --seasons 2024 --seasons 2025 --seasons 2026
```

Cached HTML lands in `data/raw/`; processed parquet tables in
`data/processed/`. Both directories are gitignored.

## Tests

```sh
pytest -q
```

The event-classifier tests use synthetic strings; they should be replaced
with verbatim fixtures from cached PBP pages once we've run the scrape
once.

## Open items before the stats layer can run

1. **Confirm `division_id` for 2025 and 2026.** `softsaber.config.DIVISION_IDS`
   has best-guess values from the observed +160/year pattern; verify by
   probing a known-good game date once the season opens.
2. **Cache a representative sample of PBP pages** (e.g. one team-season's
   worth) and use it to:
   - validate the events classifier against real text,
   - implement `softsaber.parse.baserunners.reconstruct_half_inning`,
   - sanity-check by comparing aggregated AVG/OBP to the playerbox.
3. **Discover `stat_seq` values** for the hitting and pitching national
   rankings pages (one number per category per year). Fill in
   `softsaber.ingest.playerbox.HITTING_STAT_IDS` and `PITCHING_STAT_IDS`.
4. **Build season-aggregate player table** from the finished PA table
   (group by `batter` × `season` × `batting_team`).
5. **Wire `park_factors.multi_year_park_factors`** once two seasons of
   `games` data are in hand.
6. **Implement `wrc_plus.player_wrc_plus`.** All upstream pieces have to be
   working first; the formula itself is ~10 lines.
