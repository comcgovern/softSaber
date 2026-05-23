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
| NCAA GraphQL client | `softsaber.ingest.ncaa_api` | done |
| Scoreboard ingest (GraphQL) | `softsaber.ingest.scoreboard` | done |
| Play-by-play ingest (GraphQL) | `softsaber.ingest.pbp` | done; per-play field names need confirmation on first real run |
| Team scrape | `softsaber.ingest.teams` | done (stats.ncaa.org HTML; redundant with `teamId` from GraphQL, can be retired) |
| Roster scrape | `softsaber.ingest.rosters` | done (real-page tuning when needed) |
| Player-box scrape | `softsaber.ingest.playerbox` | retire in favor of `ncaa_api.fetch_boxscore` |
| Event classifier | `softsaber.parse.events` | done; expand corpus as real samples land |
| Base-out reconstruction | `softsaber.parse.baserunners` | done |
| PA-level table | `softsaber.parse.pa` | done |
| Run expectancy / RE24 | `softsaber.stats.run_expectancy` | done |
| Linear weights / wOBA | `softsaber.stats.linear_weights` | done |
| Park factors (multi-year) | `softsaber.stats.park_factors` | done |
| wRC+ | `softsaber.stats.wrc_plus` | done |

## Architecture

```
                 sdataprod.ncaa.com (GraphQL persisted queries)
                              |
                              v
                     softsaber.ingest.ncaa_api
                  (scoreboard / pbp / boxscore)
                              |
                              v
                     softsaber.http_cache
                  (disk-cached, retried GETs)
                              |
        +---------------------+---------------------+
        |                     |                     |
        v                     v                     v
   scoreboard               teams                 pbp
  (games table)     (name → ncaa team_id)     (raw pbp rows)
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

1. **Confirm the GraphQL PBP shape on a real response.** The flatten layer in
   `softsaber.ingest.pbp` tries a small ordered list of candidate field names
   (`text`, `playText`, etc., plus `isHomeBatting` variants) and logs the
   actual keys it sees once when none match. Run `softsaber ingest pbp` for
   one season, grep the log for ``PBP shape:`` warnings, and add the real
   field names to the constants at the top of that module.
2. **Refresh persisted-query hashes if NCAA rotates them.** `ncaa_api.py`
   carries three hashes (`HASH_SCOREBOARD`, `HASH_PBP_GENERIC`,
   `HASH_BOXSCORE_SOFTBALL`). If a call ever returns
   ``PersistedQueryNotFound``, view source on a game center page and copy
   the new ``sha256Hash`` from the inlined ``__APOLLO_STATE__`` script.
3. **Validate the events classifier against real PBP.** The synthetic test
   corpus covers canonical phrasings; NCAA stringers will surface variants
   we haven't seen. Failures appear as ``outcome=None`` rows in the PA build
   log — easy to grep.
4. **Retire or rewrite `ingest.teams` / `ingest.playerbox`.** The GraphQL
   scoreboard already returns `teamId`, and `ncaa_api.fetch_boxscore` covers
   per-player line totals, so both legacy stats.ncaa.org scrapers can be
   replaced by a thin pass over cached payloads.
5. **Decide the venue key for park factors.** Right now we proxy venue by
   `home_team_id`, which is correct for ~95% of D1 teams but wrong for
   teams that share a stadium / play neutral-site tournaments. If you want
   true venue-level PFs we need the venue string from the boxscore payload
   (likely under `data.boxscore.gameInfo` or similar).
