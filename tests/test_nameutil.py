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


def test_match_player_full_first_with_comma() -> None:
    """Team-code pitching-change format: 'Lastname,Firstname'."""
    players = _make_players(("Brianne", "Weiss", True), ("Amy", "Adams", False))
    hit = match_player("Weiss,Brianne", players)
    assert hit is not None
    assert hit["player_name"] == "Brianne Weiss"


def test_match_player_firstname_lastname_order() -> None:
    """Free-text feed format: 'Firstname Lastname' with no comma."""
    players = _make_players(("Julia", "Pike", True), ("Sarah", "Wall", False))
    hit = match_player("Julia Pike", players)
    assert hit is not None
    assert hit["player_name"] == "Julia Pike"


def test_match_player_initial_leading_all_caps() -> None:
    """Initial-leads form ('T. THOMAS') common in some NCAA softball feeds."""
    players = _make_players(("Tabby", "Thomas", True), ("Amy", "Adams", False))
    hit = match_player("T. THOMAS", players)
    assert hit is not None
    assert hit["player_name"] == "Tabby Thomas"
    # No-space variant.
    hit = match_player("T.THOMAS", players)
    assert hit is not None
    assert hit["player_name"] == "Tabby Thomas"
    # No-dot variant.
    hit = match_player("T THOMAS", players)
    assert hit is not None
    assert hit["player_name"] == "Tabby Thomas"


def test_match_player_two_letter_initials() -> None:
    """Run-together initials: 'CJ Haney', 'AG Batson', 'MJ Nicholson'."""
    players = _make_players(("CJ", "Haney", True), ("Amy", "Adams", False))
    hit = match_player("CJ Haney", players)
    assert hit is not None
    assert hit["player_name"] == "CJ Haney"
    # The first token need only share its initial with the roster first name.
    players2 = _make_players(("Abigail", "Batson", True))
    hit = match_player("AG Batson", players2)
    assert hit is not None
    assert hit["player_name"] == "Abigail Batson"


def test_match_player_apostrophe_first_name() -> None:
    players = _make_players(("Z'Natria", "Evans", True), ("Bob", "Smith", False))
    hit = match_player("Z'Natria Evans", players)
    assert hit is not None
    assert hit["player_name"] == "Z'Natria Evans"


def test_match_player_truncated_surname() -> None:
    """Fixed-width feeds truncate the surname; match by prefix + first name."""
    players = _make_players(("Lydia", "VanderWoude", True), ("Amy", "Adams", False))
    hit = match_player("Lydia Vander", players)
    assert hit is not None
    assert hit["player_name"] == "Lydia VanderWoude"
    # Single-token truncated surname resolves when unique.
    players2 = _make_players(("Jane", "Brockenbrough", True), ("Amy", "Adams", False))
    hit = match_player("Brockenbroug", players2)
    assert hit is not None
    assert hit["player_name"] == "Jane Brockenbrough"


def test_match_player_surname_only_when_unique() -> None:
    players = _make_players(("Kim", "Jackson", True), ("Amy", "Adams", False))
    hit = match_player("Jackson", players)
    assert hit is not None
    assert hit["player_name"] == "Kim Jackson"


def test_match_player_ambiguous_surname_only_returns_none() -> None:
    players = _make_players(("Ana", "Flores", True), ("Ivy", "Flores", False))
    assert match_player("Flores", players) is None


def test_match_player_concatenated_surname_initial() -> None:
    """Some feeds (e.g. Lamar) concatenate surname+initial: 'FloresA', 'FloresI'."""
    players = _make_players(("Araceli", "Flores", True), ("Bob", "Smith", False))
    hit = match_player("FloresA", players)
    assert hit is not None
    assert hit["player_name"] == "Araceli Flores"
    # Two Flores but different initials — still disambiguates.
    players2 = _make_players(("Ana", "Flores", True), ("Ivy", "Flores", False))
    hit2 = match_player("FloresA", players2)
    assert hit2 is not None
    assert hit2["player_name"] == "Ana Flores"
    # Truly ambiguous: two players with same surname AND same initial.
    players3 = _make_players(("Ana", "Flores", True), ("Abby", "Flores", False))
    assert match_player("FloresA", players3) is None


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


def test_resolve_batter_names_degraded_boxscore_upgraded_by_rosters() -> None:
    """Degraded boxscore name resolves to full stats.ncaa.org roster name.

    Simulates the production scenario:
    - game_players has the degraded first name ``"A."`` from the boxscore.
    - rosters has the proper ``"Andrea"`` from stats.ncaa.org.
    resolve_batter_names should pick the richer rosters row.
    """
    pbp = pd.DataFrame([
        _make_pbp_row(events="LIGHTNER, A singled to right.", batting_team_id="11"),
    ])
    game_players = pd.DataFrame([
        {"game_id": "g1", "team_id": "11",
         "first_name": "A.", "last_name": "Lightner",
         "player_name": "A. Lightner", "starter": True},
    ])
    rosters = pd.DataFrame([
        {"team_id": "11",
         "first_name": "Andrea", "last_name": "Lightner",
         "player_name": "Andrea Lightner", "jersey": 7},
    ])

    pa = build_pa_table(pbp)
    resolved = resolve_batter_names(pa, game_players, rosters=rosters)

    assert resolved["batter"].iloc[0] == "Andrea Lightner"
    assert resolved["batter_resolved"].iloc[0]


def test_resolve_batter_names_falls_back_to_game_players_when_rosters_miss() -> None:
    pbp = pd.DataFrame([
        _make_pbp_row(events="KNIGHT, S singled to right.", batting_team_id="11"),
    ])
    game_players = pd.DataFrame([
        {"game_id": "g1", "team_id": "11", "first_name": "Shelby", "last_name": "Knight",
         "player_name": "Shelby Knight", "starter": True},
    ])
    # Rosters has a different team — should not match, fallback wins.
    rosters = pd.DataFrame([
        {"team_id": "99", "first_name": "Other", "last_name": "Knight",
         "player_name": "Other Knight", "jersey": 1},
    ])
    pa = build_pa_table(pbp)
    resolved = resolve_batter_names(pa, game_players, rosters=rosters)
    assert resolved["batter"].iloc[0] == "Shelby Knight"


def test_split_combined_name_repairs_empty_first() -> None:
    from softsaber.ingest.boxscore import split_combined_name
    assert split_combined_name("", "Libby Pippin") == ("Libby", "Pippin")
    assert split_combined_name("", "Andrea Mae Lightner") == ("Andrea", "Mae Lightner")
    # Untouched when firstName is already populated
    assert split_combined_name("Libby", "Pippin") == ("Libby", "Pippin")
    # Untouched when lastName has no whitespace
    assert split_combined_name("", "Pippin") == ("", "Pippin")


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
