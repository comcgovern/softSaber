"""Pitcher attribution: stamp the active pitcher onto every PBP row.

NCAA PBP gives us per-PA outcomes and substitution boilerplate, but no
explicit "this PA was thrown by X" column.  We reconstruct it by:

1. Reading each game's starting pitchers from ``game_players`` (rows with
   ``position == 'P'`` and ``starter == True``).
2. Walking the PBP rows in order, watching for pitching-change boilerplate.
3. Stamping the currently-active pitcher of the fielding team onto every
   row, including PA rows so the pitcher column survives ``build_pa_table``.

Patterns vary by NCAA feed.  This module covers the common formats; any
substitution-shaped line that doesn't match a known pattern is logged at
debug so unrecognised formats can be added without re-scraping.
"""

from __future__ import annotations

import logging
import re
from typing import Iterable

import pandas as pd

log = logging.getLogger(__name__)


# Pitching-change patterns.  Each captures the *incoming* pitcher's name
# in a group named ``name``.  Order matters: more-specific patterns
# should come first so a longer match wins.
#
# Anchored against the *start* of the event text (after stripping
# leading whitespace) to avoid matching mid-narration fragments inside
# an at-bat description.
_PITCHER_CHANGE_PATTERNS: list[re.Pattern[str]] = [
    # Team-code prefix: "ND pitching change: Weiss,Brianne"
    # The team code is 2-5 uppercase letters (ND, GCU, FURM, DUKE, ARIZ, ...).
    re.compile(
        r"^\s*[A-Z]{2,5}\s+pitching\s+change\s*:\s*"
        r"(?P<name>[A-Z][\w'\-]+(?:,\s*[\w'\-]+(?:\s+[\w'\-]+)?)?)\b",
        re.I,
    ),
    # "Pitching: SMITH, J. for JONES, B." or "Pitching change: SMITH, J."
    re.compile(
        r"^\s*Pitching(?:\s+change)?\s*:\s*"
        r"(?P<name>[A-Z][\w'\-]+(?:,\s*[\w'\-]+(?:\s+[\w'\-]+)?)?)",
        re.I,
    ),
    # "Now pitching: SMITH, J."
    re.compile(
        r"^\s*Now\s+pitching\s*:\s*"
        r"(?P<name>[A-Z][\w'\-]+(?:,\s*[\w'\-]+(?:\s+[\w'\-]+)?)?)",
        re.I,
    ),
    # "P: SMITH, J."
    re.compile(r"^\s*P\s*:\s*(?P<name>[A-Z][\w'\-]+(?:,\s*[A-Z]\.?)?)", re.I),
    # "SMITH, J. to p[itcher] for JONES, B." or "SMITH, J. to p."
    # LASTNAME, FI form — anchored to start so at-bat text like
    # "X grounded out to p" never matches.
    re.compile(
        r"^\s*(?P<name>[A-Z][\w'\-]+,\s*[A-Z]\.?)\s+to\s+p(?:itcher)?\b",
        re.I,
    ),
    # "Firstname Lastname to p[itcher] for X" / "Firstname Lastname to p."
    # Title-case first + last, then " to p" — same start-anchor protects
    # us from picking up "Player Name grounded out to p" because
    # "grounded" isn't part of the name capture.
    re.compile(
        r"^\s*(?P<name>[A-Z][a-z]+(?:[\s'\-][A-Z][a-z]+)+)\s+to\s+p(?:itcher)?\b",
        re.I,
    ),
    # "SMITH, J. relieved JONES, B."
    re.compile(
        r"^\s*(?P<name>[A-Z][\w'\-]+(?:,\s*[A-Z]\.?)?)\s+relieved\b", re.I
    ),
    # "SMITH, J. in to pitch for JONES, B."
    re.compile(
        r"^\s*(?P<name>[A-Z][\w'\-]+(?:,\s*[A-Z]\.?)?)\s+in(?:to)?\s+to\s+pitch\b",
        re.I,
    ),
]

# Detector for lines that *specifically* look like pitching changes.
# Non-pitching subs (pinch-run, pinch-hit, defensive change) won't
# match.  At-bat narration is filtered out separately by checking
# events.classify before reaching this regex, so "X grounded out to p"
# and "X singled to pitcher" never reach the unmatched log.
_PITCHER_SUB_HINT_RE = re.compile(
    r"(?:^\s*(?:[A-Z]{2,5}\s+)?pitching\s+change\b)|"
    r"(?:^\s*now\s+pitching\b)|"
    r"(?:^\s*p\s*:)|"
    r"(?:\bto\s+p(?:itcher)?\s*(?:\.|for\b|$))",
    re.I,
)


def parse_pitcher_change(text: str) -> str | None:
    """Return the incoming pitcher's name token, or ``None`` if the line
    isn't a recognised pitching change.

    Examples::

        parse_pitcher_change("Pitching: SMITH, J. for JONES, B.")  → "SMITH, J."
        parse_pitcher_change("SMITH, J. to p for JONES, B.")        → "SMITH, J."
        parse_pitcher_change("SMITH, J. singled to right.")         → None
    """
    if not text:
        return None
    for pat in _PITCHER_CHANGE_PATTERNS:
        m = pat.match(text)
        if m:
            return m.group("name").strip()
    return None


def _is_starting_pitcher_position(pos: str) -> bool:
    """A player is a starting pitcher when the first slot in their
    position string is ``P``.

    NCAA boxscores list two-way players with slash-separated positions:

    * ``p``          → pure pitcher
    * ``p/dp``       → started at pitcher, also designated player
    * ``p/3b``       → started at pitcher, moved to 3B later
    * ``dp/p``       → started as DP, came in to pitch later  → NOT starter
    * ``1b/p``       → started at 1B, moved to pitch later     → NOT starter

    We accept the first form group and reject the rest.
    """
    if not pos:
        return False
    first = str(pos).strip().upper().split("/")[0].strip()
    return first == "P"


def _starting_pitchers(game_players: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """Return ``{(game_id, team_id): {"name": str, "player_id": str|None}}``
    for the starter at position P on each team in each game.

    If multiple starters at P are recorded (shouldn't happen but defends
    against bad data), the first is used.
    """
    if game_players.empty or "position" not in game_players.columns:
        return {}
    starters = game_players[
        game_players["position"].apply(_is_starting_pitcher_position)
        & game_players["starter"].fillna(False).astype(bool)
    ]
    out: dict[tuple[str, str], dict] = {}
    for (gid, tid), grp in starters.groupby(["game_id", "team_id"]):
        row = grp.iloc[0]
        out[(str(gid), str(tid))] = {
            "name": str(row.get("player_name") or "").strip(),
            "player_id": (
                str(row.get("ncaa_player_id"))
                if pd.notna(row.get("ncaa_player_id"))
                else None
            ),
        }
    return out


def _fielding_team_id(row, away_id: str, home_id: str) -> str:
    """Pick the fielding team_id for a PBP row using ``top_bottom``."""
    tb = (str(getattr(row, "top_bottom", "") or "")).lower()
    if tb == "top":
        return home_id  # away bats top → home fields
    if tb == "bottom":
        return away_id
    return ""


def attribute_pitchers(
    pbp_raw: pd.DataFrame,
    game_players: pd.DataFrame,
) -> pd.DataFrame:
    """Add ``pitcher`` and ``pitcher_id`` columns to ``pbp_raw``.

    ``pbp_raw`` is the raw event-level table written by the PBP ingest
    (one row per event, including substitution boilerplate).  Returns a
    copy with the new columns; rows where the pitcher can't be inferred
    (no starter recorded, no in-game change parsed) are left blank.

    ``game_players`` must carry ``game_id``, ``team_id``, ``position``,
    ``starter``, ``player_name``, and optionally ``ncaa_player_id``.
    """
    if pbp_raw.empty:
        out = pbp_raw.copy()
        out["pitcher"] = None
        out["pitcher_id"] = None
        return out

    starters = _starting_pitchers(game_players)
    log.info(
        "attribute_pitchers: %d game-team starting pitchers loaded from game_players",
        len(starters),
    )

    # Pull a stable lookup for resolving substitution name tokens
    # ("SMITH, J.") to a real player row, scoped by (game_id, team_id).
    from .events import classify
    from .nameutil import match_player
    gp_by_game_team: dict[tuple[str, str], pd.DataFrame] = {}
    if not game_players.empty:
        for (gid, tid), grp in game_players.groupby(["game_id", "team_id"]):
            gp_by_game_team[(str(gid), str(tid))] = grp.reset_index(drop=True)

    pbp = pbp_raw.copy().sort_values(["game_id", "row_idx"]).reset_index(drop=True)
    has_tb = "top_bottom" in pbp.columns
    has_ev = "events" in pbp.columns

    pitchers: list[str | None] = []
    pitcher_ids: list[str | None] = []

    # Per-game state.  Reset at every game_id change.
    cur_game: str | None = None
    away_id = home_id = ""
    state: dict[str, dict] = {}  # team_id -> {"name", "player_id"}
    unmatched_subs = 0

    for r in pbp.itertuples(index=False):
        gid = str(getattr(r, "game_id", "") or "")
        if gid != cur_game:
            cur_game = gid
            # Infer away/home team ids for this game.  PBP carries
            # batting_team_id; map it via top_bottom to get the pair.
            game_rows = pbp[pbp["game_id"].astype(str) == gid]
            tb_to_tid: dict[str, str] = {}
            if "batting_team_id" in game_rows.columns and has_tb:
                for tb in ("top", "bottom"):
                    sub = game_rows[
                        game_rows["top_bottom"].astype(str).str.lower() == tb
                    ]
                    tid_vals = sub["batting_team_id"].astype(str)
                    tid_vals = tid_vals[tid_vals != ""]
                    if not tid_vals.empty:
                        tb_to_tid[tb] = tid_vals.iloc[0]
            away_id = tb_to_tid.get("top", "")
            home_id = tb_to_tid.get("bottom", "")
            # Seed the pitcher state from the boxscore starters.
            state = {}
            if away_id:
                s = starters.get((gid, away_id))
                if s:
                    state[away_id] = s
            if home_id:
                s = starters.get((gid, home_id))
                if s:
                    state[home_id] = s

        f_tid = _fielding_team_id(r, away_id, home_id)
        events = str(getattr(r, "events", "") or "") if has_ev else ""

        # Pitching change handling — must run before stamping so the new
        # pitcher takes effect from this row forward.  Only consider
        # rows that aren't real PA events; classify() returns an outcome
        # for at-bat rows ("X grounded out to p" → GO), so we skip them
        # entirely and avoid both false-positive matches and noisy logs.
        incoming: str | None = None
        is_at_bat = False
        if events:
            is_at_bat = classify(events).outcome is not None
            if not is_at_bat:
                incoming = parse_pitcher_change(events)

        if incoming and f_tid:
            players = gp_by_game_team.get((gid, f_tid))
            if players is not None and not players.empty:
                hit = match_player(incoming, players)
                if hit is not None:
                    state[f_tid] = {
                        "name": str(hit.get("player_name") or incoming),
                        "player_id": (
                            str(hit.get("ncaa_player_id"))
                            if pd.notna(hit.get("ncaa_player_id"))
                            else None
                        ),
                    }
                else:
                    state[f_tid] = {"name": incoming, "player_id": None}
            else:
                state[f_tid] = {"name": incoming, "player_id": None}
        elif (
            events and not is_at_bat and _PITCHER_SUB_HINT_RE.search(events)
        ):
            # Looks pitching-specific but didn't match any pattern — log
            # so we can iterate on patterns later.
            unmatched_subs += 1
            if log.isEnabledFor(logging.DEBUG):
                log.debug("unmatched pitching-change line: %s", events[:120])

        active = state.get(f_tid, {}) if f_tid else {}
        pitchers.append(active.get("name") or None)
        pitcher_ids.append(active.get("player_id"))

    pbp["pitcher"] = pitchers
    pbp["pitcher_id"] = pitcher_ids

    resolved = sum(1 for p in pitchers if p)
    log.info(
        "attribute_pitchers: stamped pitcher on %d/%d rows (%.0f%%); "
        "unmatched substitution-shaped lines: %d",
        resolved, len(pbp),
        100 * resolved / len(pbp) if len(pbp) else 0,
        unmatched_subs,
    )
    return pbp


__all__ = ["attribute_pitchers", "parse_pitcher_change"]
