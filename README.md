# softsaber

Python pipeline for computing advanced softball statistics — park factors,
linear weights, and wRC+ — from NCAA Division I play-by-play data. Inspired
by the [softballR](https://github.com/sportsdataverse/softballR) R package,
which this project mirrors at the scraping layer and extends with a
sabermetrics layer comparable to what Fangraphs publishes for baseball.

## Status

Full pipeline implemented end-to-end. Tested against synthesized PBP that
exercises every code path; the integration test asserts that linear weights
come out in Tango order (HR > 3B > 2B > 1B > BB > 0 > outs) and that a known
power hitter shows up well above 100 wRC+ while the league average is 100.

| layer | module | status |
|---|---|---|
| HTTP cache + retries | `softsaber.http_cache` | done |
| Scoreboard scrape | `softsaber.ingest.scoreboard` | done |
| Play-by-play scrape | `softsaber.ingest.pbp` | done |
| Team scrape | `softsaber.ingest.teams` | done |
| Roster scrape | `softsaber.ingest.rosters` | done (real-page tuning when needed) |
| Player-box scrape | `softsaber.ingest.playerbox` | needs `stat_seq` discovery |
| Event classifier | `softsaber.parse.events` | done; expand corpus as real samples land |
| Base-out reconstruction | `softsaber.parse.baserunners` | done |
| PA-level table | `softsaber.parse.pa` | done |
| Run expectancy / RE24 | `softsaber.stats.run_expectancy` | done |
| Linear weights / wOBA | `softsaber.stats.linear_weights` | done |
| Park factors (multi-year) | `softsaber.stats.park_factors` | done |
| wRC+ | `softsaber.stats.wrc_plus` | done |

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
# or end-to-end ingest:
softsaber ingest all --seasons 2024 --seasons 2025 --seasons 2026
# then compute stats:
softsaber stats wrc --seasons 2024 --seasons 2025 --seasons 2026
```

The stats command writes:

* `data/processed/linear_weights/<seasons>.parquet` — wOBA weights per outcome,
* `data/processed/park_factors/<seasons>.parquet` — multi-year regressed PFs,
* `data/processed/wrc_plus/<seasons>.parquet` — per-player-season leaderboard.

It also prints the top 15 wRC+ to stdout.

Cached HTML lands in `data/raw/`; processed parquet tables in
`data/processed/`. Both directories are gitignored.

## Tests

```sh
pytest -q
```

The event-classifier tests use synthetic strings; they should be replaced
with verbatim fixtures from cached PBP pages once we've run the scrape
once.

## Open items / next steps

1. **Confirm `division_id` for 2025 and 2026.** `softsaber.config.DIVISION_IDS`
   has best-guess values from the observed +160/year pattern; verify by
   probing a known-good game date.
2. **Validate the events classifier against real PBP** once the scrape has
   run. The synthetic test corpus covers the canonical phrasings; NCAA
   stringers will surface phrasings we haven't seen. Failures appear as
   ``outcome=None`` rows in the PA build log — easy to grep.
3. **Discover `stat_seq` values** for the hitting and pitching national
   rankings pages and fill in `softsaber.ingest.playerbox.HITTING_STAT_IDS`
   and `PITCHING_STAT_IDS` so the playerbox sanity check is wired up.
4. **Sanity-check aggregation against the playerbox.** Once #3 is done,
   compare `compute_player_seasons` output to the NCAA-reported per-player
   AVG/OBP/SLG for the same season. Any discrepancy points at a parsing bug.
5. **Decide the venue key for park factors.** Right now we proxy venue by
   `home_team_id`, which is correct for ~95% of D1 teams but wrong for
   teams that share a stadium / play neutral-site tournaments. If you
   want true venue-level PFs we need to scrape the box score header
   (which carries the venue string).
