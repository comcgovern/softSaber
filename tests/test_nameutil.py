"""Tests for nameutil and resolve_batter_names."""

from __future__ import annotations

import pandas as pd
import pytest

from softsaber.parse.nameutil import match_player, normalize_pbp_name
from softsaber.parse.pa import build_pa_table, resolve_batter_names


# ---------------------------------------------------------------------------
# normalize_pbp_name
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw,expected", [
    ("KNIGHT, S", ("knight", "s")),
    ("KNIGHT,S", ("knight", "s")),
    ("KNIGHT, S.", ("knight", "s")),
    ("McANALLY, L", ("mcanally", "l")),
    ("SMITH-JONES, A", ("smith-jones", "a")),
    ("HORNBUCKLE,S", ("hornbuckle", "s")),
])
def test_normalize_pbp_name(raw: str, expected: tuple[str, str]) -> None:
    assert normalize_pbp_name(raw) == expected


def test_normalize_pbp_name_no_match() -> None:
    assert normalize_pbp_name("not a name") is None
    assert normalize_pbp_name("smith jones") is None  # no comma


# ---------------------------------------------------------------------------
# match_player
# ---------------------------------------------------------------------------

def _make_players(*rows: tuple[str, str, bool]) -> pd.DataFrame:
    """Helper: rows are (first_name, last_name, starter)."""
    return pd.DataFrame(
        [{"first_name": f, "last_name": l, "starter": s,
          "player_name": f"{f} {l}".strip()}
         for f, l, s in rows]
    )


def test_match_player_exact() -> None:
    players = _make_players(("Shelby", "Knight", True), ("Amy", "Adams", False))
    hit = match_player("KNIGHT, S", players)
    assert hit is not None
    assert hit["player_name"] == "Shelby Knight"


def test_match_player_no_space_after_comma() -> None:
    players = _make_players(("Shelby", "Knight", True))
    assert match_player("KNIGHT,S", players) is not None


def test_match_player_prefers_starter() -> None:
    players = _make_players(
        ("Sara", "Smith", False),
        ("Samantha", "Smith", True),
    )
    hit = match_player("SMITH, S", players)
    assert hit is not None
    assert hit["player_name"] == "Samantha Smith"


def test_match_player_last_name_only_unique() -> None:
    players = _make_players(("Shelby", "Knight", True), ("Amy", "Adams", False))
    hit = match_player("KNIGHT", players)
    assert hit is not None
    assert hit["player_name"] == "Shelby Knight"


def test_match_player_last_name_only_ambiguous_returns_none() -> None:
    players = _make_players(("Sara", "Smith", True), ("Sam", "Smith", False))
    assert match_player("SMITH", players) is None


def test_match_player_no_match() -> None:
    players = _make_players(("Shelby", "Knight", True))
    assert match_player("JONES, A", players) is None


def test_match_player_empty_df() -> None:
    assert match_player("KNIGHT, S", pd.DataFrame()) is None


# ---------------------------------------------------------------------------
# resolve_batter_names
# ---------------------------------------------------------------------------

def _make_pbp_row(**kwargs) -> dict:
    defaults = {
        "game_id": "g1", "inning": 1, "top_bottom": "top",
        "batting_team": "Away", "fielding_team": "Home",
        "away_team_runs": 0, "home_team_runs": 0,
        "row_idx": 0, "season": 2024,
        "batting_team_id": "11",
        "play_id": "g1_1_0_1",
    }
    defaults.update(kwargs)
    return defaults


def test_resolve_batter_names_basic() -> None:
    pbp = pd.DataFrame([
        _make_pbp_row(events="KNIGHT, S singled to right.", row_idx=0, play_id="g1_1_0_1"),
        _make_pbp_row(events="ADAMS, A walked.", row_idx=1, play_id="g1_1_0_2"),
        _make_pbp_row(events="JONES, B struck out swinging.", row_idx=2, play_id="g1_1_0_3"),
    ])
    game_players = pd.DataFrame([
        {"game_id": "g1", "team_id": "11", "first_name": "Shelby", "last_name": "Knight",
         "player_name": "Shelby Knight", "starter": True},
        {"game_id": "g1", "team_id": "11", "first_name": "Amy", "last_name": "Adams",
         "player_name": "Amy Adams", "starter": False},
        # Jones not in game_players — should keep raw name.
    ])

    pa = build_pa_table(pbp)
    resolved = resolve_batter_names(pa, game_players)

    assert "batter_resolved" in resolved.columns
    by_batter = resolved.set_index("batter")

    assert "Shelby Knight" in resolved["batter"].values
    assert "Amy Adams" in resolved["batter"].values
    # JONES not in game_players; raw token kept.
    assert "JONES, B" in resolved["batter"].values

    # Confirm flags.
    assert resolved.loc[resolved["batter"] == "Shelby Knight", "batter_resolved"].all()
    assert not resolved.loc[resolved["batter"] == "JONES, B", "batter_resolved"].any()


def test_resolve_batter_names_empty_game_players() -> None:
    pbp = pd.DataFrame([_make_pbp_row(events="KNIGHT, S singled to right.")])
    pa = build_pa_table(pbp)
    resolved = resolve_batter_names(pa, pd.DataFrame())
    assert "batter_resolved" in resolved.columns
    assert not resolved["batter_resolved"].any()
    assert resolved["batter"].iloc[0] == "KNIGHT, S"


def test_resolve_batter_names_team_scoped() -> None:
    """Players from game_players are filtered to batting_team_id first."""
    pbp = pd.DataFrame([
        _make_pbp_row(events="SMITH, J singled to left.", batting_team_id="11"),
    ])
    # Two Smiths in the same game but on different teams.
    game_players = pd.DataFrame([
        {"game_id": "g1", "team_id": "11", "first_name": "Jane", "last_name": "Smith",
         "player_name": "Jane Smith", "starter": True},
        {"game_id": "g1", "team_id": "22", "first_name": "Julie", "last_name": "Smith",
         "player_name": "Julie Smith", "starter": True},
    ])
    pa = build_pa_table(pbp)
    resolved = resolve_batter_names(pa, game_players)
    assert resolved["batter"].iloc[0] == "Jane Smith"
