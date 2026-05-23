"""Round-trip test for the casablanca scoreboard parser.

Uses a minimal payload that matches the relevant subset of the
``data.ncaa.com/casablanca/scoreboard/.../scoreboard.json`` response. If NCAA
changes the shape, this test fails before the real ingest does.
"""

from __future__ import annotations

from softsaber.ingest.scoreboard import parse_scoreboard

FIXTURE = {
    "games": [
        {
            "game": {
                "gameID": "1234567",
                "startDate": "03/15/2024",
                "gameState": "final",
                "away": {
                    "score": "5",
                    "teamId": "30001",
                    "names": {"short": "Oklahoma", "seo": "oklahoma"},
                },
                "home": {
                    "score": "3",
                    "teamId": "30002",
                    "names": {"short": "Texas", "seo": "texas"},
                },
            }
        },
        {
            # Live game — should be excluded by default.
            "game": {
                "gameID": "1234568",
                "startDate": "03/15/2024",
                "gameState": "live",
                "away": {"score": "1", "names": {"short": "A", "seo": "a"}},
                "home": {"score": "0", "names": {"short": "B", "seo": "b"}},
            }
        },
    ]
}


def test_parse_scoreboard_minimal() -> None:
    games = parse_scoreboard(FIXTURE)
    assert len(games) == 1
    g = games[0]
    assert g.game_id == "1234567"
    assert g.away_team == "Oklahoma"
    assert g.home_team == "Texas"
    assert g.away_team_runs == 5
    assert g.home_team_runs == 3
    assert g.away_team_id == "30001"
    assert g.home_team_id == "30002"
    assert g.status == "Final"


def test_parse_scoreboard_team_id_falls_back_to_seo() -> None:
    payload = {
        "games": [
            {
                "game": {
                    "gameID": "9",
                    "startDate": "03/15/2024",
                    "gameState": "final",
                    "away": {"score": "1", "names": {"short": "A", "seo": "a-slug"}},
                    "home": {"score": "0", "names": {"short": "B", "seo": "b-slug"}},
                }
            }
        ]
    }
    [g] = parse_scoreboard(payload)
    assert g.away_team_id == "a-slug"
    assert g.home_team_id == "b-slug"


def test_parse_scoreboard_empty_payload() -> None:
    assert parse_scoreboard({}) == []
    assert parse_scoreboard({"games": []}) == []
