"""Tests for pitcher attribution from PBP."""

from __future__ import annotations

import pandas as pd

from softsaber.parse.pitcher import attribute_pitchers, parse_pitcher_change


def test_parse_pitcher_change_patterns() -> None:
    assert parse_pitcher_change("Pitching: SMITH, J. for JONES, B.") == "SMITH, J."
    assert parse_pitcher_change("SMITH, J. to p for JONES, B.") == "SMITH, J."
    assert parse_pitcher_change("Now pitching: SMITH, J.") == "SMITH, J."
    assert parse_pitcher_change("SMITH, J. relieved JONES, B.") == "SMITH, J."
    assert parse_pitcher_change("P: SMITH, J.") == "SMITH, J."


def test_parse_pitcher_change_ignores_at_bat_text() -> None:
    assert parse_pitcher_change("SMITH, J. singled to right center.") is None
    assert parse_pitcher_change("ADAMS, A. struck out swinging.") is None
    assert parse_pitcher_change("") is None


def test_parse_pitcher_change_team_prefix() -> None:
    """Team-code prefix format: 'ND pitching change: Weiss,Brianne'."""
    assert parse_pitcher_change("ND pitching change: Weiss,Brianne") == "Weiss,Brianne"
    assert parse_pitcher_change("GCU pitching change: Jones,Abi") == "Jones,Abi"
    assert parse_pitcher_change("ARIZ pitching change: Holder,Rylie") == "Holder,Rylie"


def test_parse_pitcher_change_firstname_lastname() -> None:
    """Free-text format: 'Julia Pike to p.' or 'Cassie Brown to p for X'."""
    assert parse_pitcher_change("Julia Pike to p.") == "Julia Pike"
    assert parse_pitcher_change("Hailey Errichiello to p for Julia Smith.") == "Hailey Errichiello"
    assert parse_pitcher_change("Tori Grifone to p for Alyssa Twome.") == "Tori Grifone"


def test_parse_pitcher_change_doesnt_match_at_bat_outcomes() -> None:
    """At-bat lines containing 'to p' or 'pitcher' should NOT match."""
    # "X grounded out to p" — outcome verb between name and "to p"
    assert parse_pitcher_change("Julia Rowley grounded out to p") is None
    assert parse_pitcher_change("Ava Beal grounded out to p (0-1 K).") is None
    # "singled to pitcher" is a hit destination, not a sub
    assert parse_pitcher_change("Cate Lehner singled to pitcher, RBI.") is None
    # pinch-running / pinch-hitting are subs but not pitching
    assert parse_pitcher_change("Charlotte Constantine pinch ran for X.") is None
    assert parse_pitcher_change("Sami Levine pinch hit for Akira") is None


def test_attribute_pitchers_seeds_from_starters_and_follows_changes() -> None:
    # Game g1: away team 11 bats top, home team 22 bats bottom.
    # Away starter: ALPHA, A. (id A1)
    # Home starter: BRAVO, B. (id B1)
    # Mid-game, home brings in CHARLIE, C. (id C1).
    game_players = pd.DataFrame([
        {"game_id": "g1", "team_id": "11", "first_name": "Alice", "last_name": "Alpha",
         "player_name": "Alice Alpha", "position": "P", "starter": True,
         "ncaa_player_id": "A1"},
        {"game_id": "g1", "team_id": "22", "first_name": "Bob", "last_name": "Bravo",
         "player_name": "Bob Bravo", "position": "P", "starter": True,
         "ncaa_player_id": "B1"},
        {"game_id": "g1", "team_id": "22", "first_name": "Carol", "last_name": "Charlie",
         "player_name": "Carol Charlie", "position": "P", "starter": False,
         "ncaa_player_id": "C1"},
    ])
    pbp = pd.DataFrame([
        # Inning 1 top — home pitcher (Bravo) faces away batter
        {"game_id": "g1", "row_idx": 0, "inning": 1, "top_bottom": "top",
         "batting_team_id": "11", "events": "ALPHA, A. singled to right."},
        # Inning 1 bottom — away pitcher (Alpha) faces home batter
        {"game_id": "g1", "row_idx": 1, "inning": 1, "top_bottom": "bottom",
         "batting_team_id": "22", "events": "BRAVO, B. doubled to left."},
        # Inning 2 top — pitching change for home
        {"game_id": "g1", "row_idx": 2, "inning": 2, "top_bottom": "top",
         "batting_team_id": "11", "events": "Pitching: CHARLIE, C. for BRAVO, B."},
        # Inning 2 top — Charlie now pitching
        {"game_id": "g1", "row_idx": 3, "inning": 2, "top_bottom": "top",
         "batting_team_id": "11", "events": "ADAMS, A. struck out swinging."},
    ])

    result = attribute_pitchers(pbp, game_players)

    # Row 0 (top of 1st): home pitcher = Bravo
    assert result.iloc[0]["pitcher"] == "Bob Bravo"
    assert result.iloc[0]["pitcher_id"] == "B1"
    # Row 1 (bottom of 1st): away pitcher = Alpha
    assert result.iloc[1]["pitcher"] == "Alice Alpha"
    assert result.iloc[1]["pitcher_id"] == "A1"
    # Row 3 (top of 2nd, after change): home pitcher = Charlie
    assert result.iloc[3]["pitcher"] == "Carol Charlie"
    assert result.iloc[3]["pitcher_id"] == "C1"


def test_attribute_pitchers_handles_empty() -> None:
    pbp = pd.DataFrame()
    game_players = pd.DataFrame()
    out = attribute_pitchers(pbp, game_players)
    assert out.empty
    assert "pitcher" in out.columns
    assert "pitcher_id" in out.columns


def test_attribute_pitchers_no_starter_no_pitcher() -> None:
    """When game_players has no starter at P, the column is None until a
    pitching-change line is seen."""
    game_players = pd.DataFrame()  # no starters known
    pbp = pd.DataFrame([
        {"game_id": "g1", "row_idx": 0, "inning": 1, "top_bottom": "top",
         "batting_team_id": "11", "events": "ALPHA, A. singled to right."},
    ])
    result = attribute_pitchers(pbp, game_players)
    assert result.iloc[0]["pitcher"] is None
    assert result.iloc[0]["pitcher_id"] is None
