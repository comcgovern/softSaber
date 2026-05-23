"""Tests for the half-inning base-out simulator."""

from __future__ import annotations

import pandas as pd

from softsaber.parse.baserunners import (
    EMPTY,
    BasesOuts,
    _apply_outcome,
    _force_advance,
    reconstruct_half_inning,
)
from softsaber.parse.events import ParsedEvent


def _ev(code: str) -> ParsedEvent:
    return ParsedEvent(outcome=code, batter=None, raw="")


def test_force_advance_bases_loaded() -> None:
    loaded = BasesOuts(True, True, True, 0)
    new, runs = _force_advance(loaded)
    assert runs == 1
    assert new == BasesOuts(True, True, True, 0)


def test_apply_outcome_single_empty() -> None:
    new, r = _apply_outcome(EMPTY, _ev("1B"))
    assert r == 0
    assert new == BasesOuts(True, False, False, 0)


def test_apply_outcome_homer_runners_on() -> None:
    s = BasesOuts(True, True, False, 1)
    new, r = _apply_outcome(s, _ev("HR"))
    assert r == 3  # batter + 2 runners
    assert new == BasesOuts(False, False, False, 1)


def test_apply_outcome_sf_clears_third() -> None:
    s = BasesOuts(False, False, True, 0)
    new, r = _apply_outcome(s, _ev("SF"))
    assert r == 1
    assert new == BasesOuts(False, False, False, 1)


def test_apply_outcome_dp_two_outs() -> None:
    s = BasesOuts(True, False, False, 0)
    new, _ = _apply_outcome(s, _ev("DP"))
    assert new.outs == 2
    assert not new.occ_1B


def test_reconstruct_half_inning_simple() -> None:
    # Top of 1st, two singles + a homer scoring all three, then three outs.
    rows = pd.DataFrame(
        [
            {"events": "Smith singled to right.", "away_team_runs": 0, "home_team_runs": 0, "row_idx": 0,
             "game_id": "g1", "inning": 1, "top_bottom": "top", "play_id": "p1"},
            {"events": "Jones doubled to left; Smith to third.", "away_team_runs": 0, "home_team_runs": 0, "row_idx": 1,
             "game_id": "g1", "inning": 1, "top_bottom": "top", "play_id": "p2"},
            {"events": "Lopez homered to left, 3 RBI.", "away_team_runs": 3, "home_team_runs": 0, "row_idx": 2,
             "game_id": "g1", "inning": 1, "top_bottom": "top", "play_id": "p3"},
            {"events": "Brown struck out swinging.", "away_team_runs": 3, "home_team_runs": 0, "row_idx": 3,
             "game_id": "g1", "inning": 1, "top_bottom": "top", "play_id": "p4"},
            {"events": "Davis grounded out to ss.", "away_team_runs": 3, "home_team_runs": 0, "row_idx": 4,
             "game_id": "g1", "inning": 1, "top_bottom": "top", "play_id": "p5"},
            {"events": "Wilson flied out to cf.", "away_team_runs": 3, "home_team_runs": 0, "row_idx": 5,
             "game_id": "g1", "inning": 1, "top_bottom": "top", "play_id": "p6"},
        ]
    )
    out = reconstruct_half_inning(rows)

    assert list(out["outcome"]) == ["1B", "2B", "HR", "K", "GO", "FO"]
    assert out["runs_on_play"].sum() == 3
    # State after the HR should be empty bases, 0 outs.
    hr_row = out[out["outcome"] == "HR"].iloc[0]
    assert hr_row["state_after"] == "___:0"
    # State at end: bases empty, 3 outs.
    assert out.iloc[-1]["state_after"].endswith(":3")
    assert bool(out.iloc[-1]["half_inning_ok"])


def test_state_id_round_trip() -> None:
    s = BasesOuts(True, False, True, 2)
    assert s.as_state_id() == "1_3:2"
