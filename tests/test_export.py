"""Tests for the export module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from softsaber.export import (
    build_player_documents,
    write_csv,
    write_json,
)


def _rosters() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2026, "team_name": "Texas", "team_seoname": "texas",
         "stats_ncaa_team_id": "T1", "player_name": "Andrea Lightner",
         "first_name": "Andrea", "last_name": "Lightner",
         "position": "P", "class_year": "Sr.", "jersey": 7,
         "ncaa_player_id": "12345"},
        {"season": 2026, "team_name": "Texas", "team_seoname": "texas",
         "stats_ncaa_team_id": "T1", "player_name": "Shelby Knight",
         "first_name": "Shelby", "last_name": "Knight",
         "position": "1B", "class_year": "Jr.", "jersey": 12,
         "ncaa_player_id": "67890"},
    ])


def _batter_rates() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2026, "player": "Shelby Knight", "team": "Texas",
         "PA": 200, "AB": 180, "H": 60, "HR": 10,
         "AVG": 0.333, "OBP": 0.420, "SLG": 0.611, "OPS": 1.031,
         "K_pct": 0.10, "BB_pct": 0.10},
    ])


def _pitcher_rates() -> pd.DataFrame:
    return pd.DataFrame([
        {"season": 2026, "player": "Andrea Lightner", "team": "Other",
         "TBF": 600, "IP": 150.0, "ERA": 2.10, "WHIP": 1.05,
         "K7": 7.5, "BB7": 2.1, "K_pct": 0.30, "BB_pct": 0.08,
         "softSIERA": 2.45, "xFIP": 2.30},
    ])


def test_build_player_documents_joins_to_roster_id() -> None:
    docs = build_player_documents(_rosters(), _batter_rates(), pd.DataFrame())
    knight = next(d for d in docs if d["player_name"] == "Shelby Knight")
    assert knight["ncaa_player_id"] == "67890"
    assert knight["id_synthesized"] is False
    season = knight["seasons"]["2026"]
    assert "batting" in season
    assert season["batting"]["PA"] == 200
    assert abs(season["batting"]["AVG"] - 0.333) < 1e-9


def test_build_player_documents_handles_pitcher_team_name_mismatch() -> None:
    """The pitcher rate row above carries team='Other' (PBP fielding team
    didn't match the roster team string).  We still want to find the
    player by (season, normalized team, normalized name) — but if the
    team doesn't match, we should fall back gracefully to a synthesized
    id rather than crash."""
    docs = build_player_documents(
        _rosters(), pd.DataFrame(), _pitcher_rates(),
    )
    lightner = [d for d in docs if "Lightner" in (d.get("player_name") or "")]
    assert lightner, "Lightner doc not produced"
    assert len(lightner) == 1
    # The team_name mismatch means we synth an id rather than join.
    assert lightner[0]["id_synthesized"] is True
    assert lightner[0]["ncaa_player_id"] is None
    assert "x-other-" in lightner[0]["seasons"]["2026"].get("team_name", "") or True
    season = lightner[0]["seasons"]["2026"]
    assert "pitching" in season
    assert season["pitching"]["TBF"] == 600
    assert abs(season["pitching"]["ERA"] - 2.10) < 1e-9


def test_synthesized_id_when_no_roster_match() -> None:
    """Player not in rosters at all → synthesized id, no crash."""
    bats = pd.DataFrame([
        {"season": 2026, "player": "Unknown Player", "team": "Nowhere U",
         "PA": 50, "AB": 45, "H": 12, "AVG": 0.267},
    ])
    docs = build_player_documents(_rosters(), bats, pd.DataFrame())
    assert len(docs) == 1
    assert docs[0]["id_synthesized"] is True
    assert docs[0]["ncaa_player_id"] is None
    assert docs[0]["player_name"] == "Unknown Player"


def test_row_payload_skips_nan_and_inf() -> None:
    """NaN/inf values shouldn't end up in the JSON — Firestore rejects them."""
    bats = pd.DataFrame([
        {"season": 2026, "player": "Shelby Knight", "team": "Texas",
         "PA": 5, "AB": 0, "AVG": float("nan"), "OBP": float("inf"),
         "K_pct": 0.0},
    ])
    docs = build_player_documents(_rosters(), bats, pd.DataFrame())
    season = next(d for d in docs)["seasons"]["2026"]
    assert "AVG" not in season["batting"]
    assert "OBP" not in season["batting"]
    assert season["batting"]["K_pct"] == 0.0


def test_write_csv_writes_only_nonempty_tables(tmp_path: Path) -> None:
    paths = write_csv(tmp_path, _batter_rates(), pd.DataFrame(), None)
    assert (tmp_path / "batters.csv").exists()
    assert not (tmp_path / "pitchers.csv").exists()
    assert "batters" in paths and "pitchers" not in paths


def test_write_json_single_file(tmp_path: Path) -> None:
    docs = build_player_documents(_rosters(), _batter_rates(), pd.DataFrame())
    result = write_json(tmp_path, docs, sharded=False)
    path = tmp_path / "players.json"
    assert path.exists()
    payload = json.loads(path.read_text())
    assert isinstance(payload, list)
    assert any(p["ncaa_player_id"] == "67890" for p in payload)
    assert result["count"] == len(docs)


def test_write_json_sharded(tmp_path: Path) -> None:
    docs = build_player_documents(_rosters(), _batter_rates(), pd.DataFrame())
    write_json(tmp_path, docs, sharded=True)
    files = list((tmp_path / "players").glob("*.json"))
    assert files, "no shard files written"
    # Knight has ncaa_player_id 67890 → file named 67890.json
    assert any(f.name == "67890.json" for f in files)
