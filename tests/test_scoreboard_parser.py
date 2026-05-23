"""Round-trip test for the GraphQL scoreboard parser.

Uses a minimal payload that matches the relevant subset of the
sdataprod.ncaa.com scoreboard response. If NCAA changes the shape, this test
fails before the real ingest does.
"""

from __future__ import annotations

from softsaber.ingest.scoreboard import parse_scoreboard

FIXTURE = {
    "data": {
        "contests": [
            {
                "contestId": 1234567,
                "startDate": "2024-03-15",
                "gameState": "F",
                "teams": [
                    {
                        "isHome": False,
                        "teamId": 30001,
                        "nameShort": "Oklahoma",
                        "score": 5,
                    },
                    {
                        "isHome": True,
                        "teamId": 30002,
                        "nameShort": "Texas",
                        "score": 3,
                    },
                ],
            },
            {
                # Live game — should be excluded by default.
                "contestId": 1234568,
                "startDate": "2024-03-15",
                "gameState": "I",
                "teams": [
                    {"isHome": False, "teamId": 1, "nameShort": "A", "score": 1},
                    {"isHome": True, "teamId": 2, "nameShort": "B", "score": 0},
                ],
            },
        ]
    }
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


def test_parse_scoreboard_empty_payload() -> None:
    assert parse_scoreboard({}) == []
    assert parse_scoreboard({"data": {"contests": []}}) == []
