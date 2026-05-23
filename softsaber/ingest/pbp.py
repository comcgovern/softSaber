"""Play-by-play ingest.

Two-step fetch per game, mirroring softballR's flow:

1. GET ``/contests/{game_id}/box_score`` to discover the internal
   ``pbp_id`` (the ID used by the standalone play-by-play page).
2. GET ``/game/play_by_play/{pbp_id}`` and parse the per-inning tables.

The PBP page renders one HTML table per *half-inning*, alternating away/home,
plus a few header tables we skip. Each table has three columns: away
action, score, home action. Empty cells indicate the other team is batting,
which is how we recover the top/bottom orientation.
"""

from __future__ import annotations

import io
import logging
import re

import pandas as pd

from .. import storage
from ..http_cache import fetch

log = logging.getLogger(__name__)

BOX_URL_FMT = "https://stats.ncaa.org/contests/{game_id}/box_score"
PBP_URL_FMT = "https://stats.ncaa.org/game/play_by_play/{pbp_id}"

_RE_PBP_LINK = re.compile(r'/game/play_by_play/(\d+)')

# Event-text fragments that signal a row is a true plate appearance (or
# baserunning result) and not a "pitching change" / "now batting" header.
# Kept here verbatim from softballR for parity; the parse layer turns these
# into outcome classes.
PLATE_EVENT_FRAGMENTS = (
    "struck out swinging",
    "struck out looking",
    "grounded out",
    "flied out",
    "infield fly",
    "hit into double play",
    "lined out",
    "out",
    "reached on a fielder's choice",
    "reached on an error",
    "reached on a throwing error",
    "reached on a fielding error",
    "lined into double play",
    "grounded into double play",
    "hit by pitch",
    "walked",
    "intentionally walked",
    "fouled out",
    "fouled into double play",
    "popped up",
    "singled",
    "doubled",
    "tripled",
    "homered",
)

_PLATE_EVENT_RE = re.compile("|".join(re.escape(p) for p in PLATE_EVENT_FRAGMENTS))


def _discover_pbp_id(game_id: str) -> str | None:
    html = fetch(BOX_URL_FMT.format(game_id=game_id), namespace="box_score")
    m = _RE_PBP_LINK.search(html)
    return m.group(1) if m else None


def fetch_game_pbp(game_id: str) -> pd.DataFrame:
    """Return the raw PBP rows for one game, or empty DataFrame if unavailable.

    Output columns:
        game_id, inning, top_bottom, batting_team, fielding_team,
        away_team_runs, home_team_runs, events, play_id, row_idx

    No event parsing happens here — see :mod:`softsaber.parse.events` for the
    outcome classifier, and :mod:`softsaber.parse.baserunners` for RE24 state
    reconstruction.
    """
    pbp_id = _discover_pbp_id(game_id)
    if pbp_id is None:
        log.warning("game %s: no pbp link on box score", game_id)
        return pd.DataFrame()
    log.debug("game %s: discovered pbp_id=%s", game_id, pbp_id)

    html = fetch(PBP_URL_FMT.format(pbp_id=pbp_id), namespace="pbp")
    tables = pd.read_html(io.StringIO(html))
    log.debug("game %s: %d tables on pbp page", game_id, len(tables))
    if len(tables) < 6:
        log.warning("game %s: pbp page returned %d tables (expected ≥6)", game_id, len(tables))
        return pd.DataFrame()

    # softballR drops tables[0:4] (game header / lineups) and then keeps every
    # other table from index 4 onwards. The "skipped" tables are the inning
    # summary scoreboards that interleave the PBP tables.
    half_inning_tables = tables[4:][1::2]

    rows: list[dict] = []
    away_team: str | None = None
    home_team: str | None = None

    inning_counter = 0
    last_top_bottom: str | None = None

    for tbl in half_inning_tables:
        if tbl.shape[1] < 3:
            continue
        # Row 0 in each table is the header [away_name, "Score", home_name].
        header = tbl.iloc[0].tolist()
        if away_team is None:
            away_team = str(header[0])
            home_team = str(header[2])

        body = tbl.iloc[1:].copy()
        body.columns = ["away_action", "score", "home_action"]
        body = body.fillna("")

        # Which side is batting? Whichever column is non-empty for the
        # first PA-shaped row.
        side = _infer_side(body)
        if side is None:
            continue

        if side != last_top_bottom:
            # Switching halves: starts a new half-inning. Top → new inning,
            # Bottom → same inning as the preceding top.
            if side == "top" or last_top_bottom is None:
                inning_counter += 1
        last_top_bottom = side

        batting_team = away_team if side == "top" else home_team
        fielding_team = home_team if side == "top" else away_team

        for ri, r in enumerate(body.itertuples(index=False)):
            away_act, score, home_act = r.away_action, r.score, r.home_action
            event_text = away_act if side == "top" else home_act
            if not isinstance(event_text, str) or not event_text.strip():
                continue
            if event_text.strip().lower() in {"score", str(away_team).lower(), str(home_team).lower()}:
                continue

            away_runs, home_runs = _split_score(score)

            rows.append(
                {
                    "game_id": game_id,
                    "inning": inning_counter,
                    "top_bottom": side,
                    "batting_team": batting_team,
                    "fielding_team": fielding_team,
                    "away_team_runs": away_runs,
                    "home_team_runs": home_runs,
                    "events": event_text.strip(),
                    "play_id": f"{game_id}_{inning_counter}_{0 if side == 'top' else 1}_{ri+1}",
                    "row_idx": ri,
                }
            )

    return pd.DataFrame(rows)


def _infer_side(body: pd.DataFrame) -> str | None:
    """Decide whether this half-inning is the top (away batting) or bottom."""
    for _, r in body.iterrows():
        a, h = str(r["away_action"]), str(r["home_action"])
        if _PLATE_EVENT_RE.search(a):
            return "top"
        if _PLATE_EVENT_RE.search(h):
            return "bottom"
    return None


def _split_score(cell: object) -> tuple[int | None, int | None]:
    if not isinstance(cell, str) or "-" not in cell:
        return None, None
    a, h = cell.split("-", 1)
    try:
        return int(a.strip()), int(h.strip())
    except ValueError:
        return None, None


def ingest_season_pbp(games: pd.DataFrame, season: int) -> pd.DataFrame:
    """Pull PBP for every finalized game in ``games`` and write a parquet partition."""
    frames = []
    game_ids = games["game_id"].astype(str).tolist()
    total = len(game_ids)
    for i, gid in enumerate(game_ids, 1):
        log.debug("pbp game %s (%d/%d)", gid, i, total)
        try:
            frames.append(fetch_game_pbp(gid))
        except Exception as e:  # noqa: BLE001 — never let one game halt the season
            log.warning("pbp game %s failed: %s", gid, e, exc_info=log.isEnabledFor(logging.DEBUG))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df["season"] = season
        storage.write_partition("pbp_raw", str(season), df)
    return df
