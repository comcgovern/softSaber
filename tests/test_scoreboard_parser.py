"""Round-trip test for the scoreboard HTML parser.

Uses a hand-crafted minimal HTML fixture that matches the exact markup
softballR's scraper keys off of. If stats.ncaa.org changes their template,
this test will fail loudly before the real ingest does.
"""

from __future__ import annotations

from softsaber.ingest.scoreboard import parse_scoreboard

FIXTURE = """
<html><body>
<tr id="contest_1234567">
  <td rowspan="2" valign="middle">
  03/15/2024
  </td>
  <td><a target="TEAMS_WIN" class="skipMask" href="/teams/30001">
    <img height="20px" width="30px" alt="Oklahoma" src="/logo1.png" />
  </a></td>
  <td><div id="score_away">
  5
  </div></td>
  <td><div class="livestream">
  Final
  </div></td>
</tr>
<tr id="contest_1234567_h">
  <td><a target="TEAMS_WIN" class="skipMask" href="/teams/30002">
    <img height="20px" width="30px" alt="Texas" src="/logo2.png" />
  </a></td>
  <td><div id="score_home">
  3
  </div></td>
</tr>
<tr id="contest_1234568">
  <td>...</td>
</tr>
</body></html>
""".strip()


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
