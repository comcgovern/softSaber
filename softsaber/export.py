"""Export rate stats in shapes suitable for downstream consumers.

Two output formats:

* **CSV** — one file per stat table (``batters.csv``, ``pitchers.csv``,
  ``wrc.csv``).  Faithful dumps of the rate-stats partition tables; easy
  to drop into a spreadsheet.

* **JSON** — Firestore-shaped per-player documents keyed on
  ``ncaa_player_id``.  Identity at the top level, per-season batting and
  pitching stats nested.  Players whose name + team + season triple
  doesn't resolve to a roster row get a synthesized key based on the
  slugged team and name, marked with ``id_synthesized: true`` so the
  loader can keep them out of the canonical players collection.

The JSON can be emitted as a single ``players.json`` array or as a
sharded ``players/<id>.json`` tree for incremental upload.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .parse.nameutil import _normalize

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Join key
# ---------------------------------------------------------------------------

def _norm_team(name: object) -> str:
    s = "" if name is None else str(name)
    return _normalize(s)


def _norm_player(name: object) -> str:
    s = "" if name is None else str(name)
    return _normalize(s)


def _roster_index(rosters: pd.DataFrame) -> dict[tuple[int, str, str], dict]:
    """Build ``{(season, norm_team, norm_player): roster_row_dict}``.

    When the rosters table has both ``player_name`` and ``first_name``/
    ``last_name``, we also index ``"first last"`` and ``"last, first"``
    so rate-stat names that come in slightly differently can still join.
    """
    if rosters.empty:
        return {}
    idx: dict[tuple[int, str, str], dict] = {}
    needed = {"season", "team_name", "player_name"}
    if not needed.issubset(rosters.columns):
        log.warning("rosters table missing columns %s; skipping ID join",
                    needed - set(rosters.columns))
        return {}

    for r in rosters.itertuples(index=False):
        d = r._asdict()
        season = int(d.get("season", 0) or 0)
        team_n = _norm_team(d.get("team_name"))
        if not season or not team_n:
            continue
        candidates = [d.get("player_name")]
        first = d.get("first_name") if "first_name" in rosters.columns else None
        last = d.get("last_name") if "last_name" in rosters.columns else None
        if first and last:
            candidates.append(f"{first} {last}")
            candidates.append(f"{last}, {first}")
        for cand in candidates:
            key = (season, team_n, _norm_player(cand))
            if key[2] and key not in idx:
                idx[key] = d
    return idx


# ---------------------------------------------------------------------------
# ID assignment
# ---------------------------------------------------------------------------

_SLUG_NONALPHA = re.compile(r"[^a-z0-9]+")


def _slug(s: object) -> str:
    norm = _normalize(str(s) if s is not None else "")
    return _SLUG_NONALPHA.sub("-", norm).strip("-")


def _synth_id(team: str, player: str) -> str:
    return f"x-{_slug(team)}-{_slug(player)}" or "x-unknown"


# ---------------------------------------------------------------------------
# Per-season payload extraction
# ---------------------------------------------------------------------------

_BATTING_COLS = (
    "PA", "AB", "H", "1B", "2B", "3B", "HR",
    "BB", "IBB", "HBP", "K", "SF", "SH",
    "TB", "XBH",
    "AVG", "OBP", "SLG", "OPS", "ISO", "BABIP",
    "K_pct", "BB_pct", "HBP_pct", "BB_K", "XBH_pct",
    "GB_bbo", "FB_bbo", "LD_bbo",
    "GB_pct_bbo", "FB_pct_bbo", "LD_pct_bbo", "GB_FB_bbo", "HR_per_fbo",
)

_PITCHING_COLS = (
    "TBF", "IP", "ER", "H_allowed", "BB_allowed", "K_box", "appearances",
    "ERA", "WHIP", "K7", "BB7",
    "BAA", "OBP_against", "SLG_against", "OPS_against", "BABIP_against",
    "K_pct", "BB_pct", "HBP_pct", "BB_K",
    "GB_bbo", "FB_bbo", "LD_bbo",
    "GB_pct_bbo", "FB_pct_bbo", "LD_pct_bbo", "HR_per_fbo",
    "xFIP", "softSIERA",
)

_WRC_COLS = ("PA", "wOBA", "wRAA", "park_factor", "wRC+")


def _row_payload(row: pd.Series, cols: Iterable[str]) -> dict[str, Any]:
    """Pull a JSON-friendly subset of a row.  Skips NaN and inf values
    so the resulting document doesn't carry junk that Firestore would
    reject.

    Normalises numpy scalars to Python natives via ``.item()`` first —
    on Windows ``numpy.int64`` is not a subclass of ``int`` and would
    otherwise fall through to the string branch.
    """
    out: dict[str, Any] = {}
    for c in cols:
        if c not in row:
            continue
        v = row[c]
        # numpy scalars: int64 → int, float64 → float, bool_ → bool.
        if hasattr(v, "item") and not isinstance(v, (str, bytes)):
            try:
                v = v.item()
            except (ValueError, AttributeError):
                pass
        if v is None:
            continue
        try:
            if pd.isna(v):
                continue
        except (TypeError, ValueError):
            pass
        if isinstance(v, bool):
            out[c] = v
        elif isinstance(v, float):
            if v != v or v in (float("inf"), float("-inf")):
                continue
            out[c] = v
        elif isinstance(v, int):
            out[c] = v
        else:
            out[c] = str(v)
    return out


# ---------------------------------------------------------------------------
# Document assembly
# ---------------------------------------------------------------------------

def build_player_documents(
    rosters: pd.DataFrame,
    batter_rates: pd.DataFrame,
    pitcher_rates: pd.DataFrame,
    wrc: pd.DataFrame | None = None,
) -> list[dict[str, Any]]:
    """Merge rate stats with rosters into per-player documents.

    Each document carries one ``seasons`` dict keyed by year, with
    ``batting``, ``pitching``, and ``wrc`` blocks where data exists.
    """
    roster_idx = _roster_index(rosters)

    # docs keyed by player_id during accumulation; we emit list at end.
    docs: dict[str, dict[str, Any]] = {}

    def _player_doc(player_id: str, identity: dict, season: int) -> dict:
        doc = docs.get(player_id)
        if doc is None:
            doc = {
                "ncaa_player_id": identity.get("ncaa_player_id"),
                "id_synthesized": identity.get("id_synthesized", False),
                "player_name": identity.get("player_name"),
                "first_name": identity.get("first_name"),
                "last_name": identity.get("last_name"),
                "team_name": identity.get("team_name"),
                "team_seoname": identity.get("team_seoname"),
                "stats_ncaa_team_id": identity.get("stats_ncaa_team_id"),
                "position": identity.get("position"),
                "class_year": identity.get("class_year"),
                "jersey": identity.get("jersey"),
                "seasons": {},
            }
            docs[player_id] = doc
        season_block = doc["seasons"].setdefault(str(season), {})
        season_block.setdefault("team_name", identity.get("team_name"))
        return season_block

    def _resolve(season: int, team: str, player: str) -> tuple[str, dict]:
        roster = roster_idx.get((season, _norm_team(team), _norm_player(player)))
        if roster and roster.get("ncaa_player_id"):
            return str(roster["ncaa_player_id"]), {
                "ncaa_player_id": str(roster["ncaa_player_id"]),
                "id_synthesized": False,
                "player_name": str(roster.get("player_name") or player),
                "first_name": str(roster.get("first_name") or ""),
                "last_name": str(roster.get("last_name") or ""),
                "team_name": str(roster.get("team_name") or team),
                "team_seoname": str(roster.get("team_seoname") or ""),
                "stats_ncaa_team_id": str(roster.get("stats_ncaa_team_id") or ""),
                "position": str(roster.get("position") or ""),
                "class_year": str(roster.get("class_year") or ""),
                "jersey": roster.get("jersey"),
            }
        return _synth_id(team, player), {
            "ncaa_player_id": None,
            "id_synthesized": True,
            "player_name": player,
            "first_name": "",
            "last_name": "",
            "team_name": team,
            "team_seoname": "",
            "stats_ncaa_team_id": "",
            "position": "",
            "class_year": "",
            "jersey": None,
        }

    if not batter_rates.empty:
        for _, row in batter_rates.iterrows():
            season = int(row.get("season", 0) or 0)
            player = str(row.get("player") or "")
            team = str(row.get("team") or "")
            if not season or not player:
                continue
            pid, identity = _resolve(season, team, player)
            block = _player_doc(pid, identity, season)
            block["batting"] = _row_payload(row, _BATTING_COLS)

    if not pitcher_rates.empty:
        for _, row in pitcher_rates.iterrows():
            season = int(row.get("season", 0) or 0)
            player = str(row.get("player") or "")
            team = str(row.get("team") or "")
            if not season or not player:
                continue
            pid, identity = _resolve(season, team, player)
            block = _player_doc(pid, identity, season)
            block["pitching"] = _row_payload(row, _PITCHING_COLS)

    if wrc is not None and not wrc.empty:
        for _, row in wrc.iterrows():
            season = int(row.get("season", 0) or 0)
            player = str(row.get("batter") or "")
            team = str(row.get("batting_team") or "")
            if not season or not player:
                continue
            pid, identity = _resolve(season, team, player)
            block = _player_doc(pid, identity, season)
            block["wrc"] = _row_payload(row, _WRC_COLS)

    return list(docs.values())


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_csv(
    out_dir: Path,
    batter_rates: pd.DataFrame,
    pitcher_rates: pd.DataFrame,
    wrc: pd.DataFrame | None,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    if not batter_rates.empty:
        p = out_dir / "batters.csv"
        batter_rates.to_csv(p, index=False)
        written["batters"] = p
    if not pitcher_rates.empty:
        p = out_dir / "pitchers.csv"
        pitcher_rates.to_csv(p, index=False)
        written["pitchers"] = p
    if wrc is not None and not wrc.empty:
        p = out_dir / "wrc.csv"
        wrc.to_csv(p, index=False)
        written["wrc"] = p
    return written


def write_json(
    out_dir: Path,
    documents: list[dict[str, Any]],
    *,
    sharded: bool = False,
) -> dict[str, Path | int]:
    """Write the player documents to JSON.

    When ``sharded=False`` (default), one ``players.json`` file containing
    the full array — easy for one-shot import.  When ``sharded=True``,
    one ``players/<id>.json`` file per player — easier for diff-based
    incremental upload to Firestore.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    if not sharded:
        path = out_dir / "players.json"
        path.write_text(json.dumps(documents, indent=2, sort_keys=True))
        return {"players": path, "count": len(documents)}

    players_dir = out_dir / "players"
    players_dir.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        pid = doc.get("ncaa_player_id") or _synth_id(
            doc.get("team_name", ""), doc.get("player_name", "")
        )
        path = players_dir / f"{pid}.json"
        path.write_text(json.dumps(doc, indent=2, sort_keys=True))
    return {"players_dir": players_dir, "count": len(documents)}


__all__ = [
    "build_player_documents",
    "write_csv",
    "write_json",
]
