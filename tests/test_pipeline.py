"""End-to-end smoke test of the analytics pipeline against synthetic PBP.

Generates a small but realistic season's worth of plays from a known
distribution, runs PBP → PA table → RE matrix → linear weights → wRC+, and
asserts the obvious sanity properties:

  * Run values are ordered: HR > 3B > 2B > 1B > BB > out.
  * The "league average" hitter sits near wRC+ = 100.
  * An obvious power hitter has wRC+ well above league average.
"""

from __future__ import annotations

import random

import pandas as pd
import pytest

from softsaber.parse.pa import build_pa_table
from softsaber.stats.linear_weights import compute_linear_weights
from softsaber.stats.run_expectancy import compute_re24, compute_re_matrix
from softsaber.stats.wrc_plus import player_wrc_plus

EVENT_TEMPLATES: dict[str, list[str]] = {
    "1B": ["{name} singled to right."],
    "2B": ["{name} doubled to left."],
    "3B": ["{name} tripled to right center."],
    "HR": ["{name} homered to left, 1 RBI."],
    "BB": ["{name} walked."],
    "HBP": ["{name} hit by pitch."],
    "K": ["{name} struck out swinging."],
    "GO": ["{name} grounded out to ss."],
    "FO": ["{name} flied out to cf."],
    "PO": ["{name} popped up to 3b."],
}

# Per-batter PA distribution: roster of N hitters with their own outcome
# probabilities. Average team is calibrated to a reasonable softball line.
LEAGUE_DIST = {
    "1B": 0.18, "2B": 0.06, "3B": 0.01, "HR": 0.05,
    "BB": 0.10, "HBP": 0.02,
    "K":  0.18, "GO": 0.20, "FO": 0.14, "PO": 0.06,
}

POWER_HITTER_DIST = {
    "1B": 0.18, "2B": 0.10, "3B": 0.02, "HR": 0.18,
    "BB": 0.15, "HBP": 0.02,
    "K":  0.15, "GO": 0.10, "FO": 0.08, "PO": 0.02,
}


def _outcome_to_runs_advance(outcome: str) -> int:
    return {"HR": 4, "3B": 3, "2B": 2, "1B": 1, "BB": 1, "HBP": 1, "ROE": 1}.get(outcome, 0)


def _synth_half_inning(rng: random.Random, batters: list[tuple[str, dict[str, float]]],
                        batter_idx: int, game_id: str, inning: int, top_bottom: str,
                        score_state: list[int], side_idx: int) -> tuple[list[dict], int]:
    """Walk one half-inning. ``score_state`` is [away, home] cumulative runs;
    we mutate it in place. ``side_idx`` is 0 for away, 1 for home.
    """
    rows: list[dict] = []
    bases = [False, False, False]
    outs = 0
    row_idx = 0
    while outs < 3:
        name, dist = batters[batter_idx % len(batters)]
        batter_idx += 1
        codes = list(dist.keys())
        weights = list(dist.values())
        oc = rng.choices(codes, weights=weights, k=1)[0]
        text = EVENT_TEMPLATES[oc][0].format(name=name)

        # Apply outcome to simulated bases (mirror of _apply_outcome, but
        # we just need to know how many runs scored).
        runs = 0
        if oc == "HR":
            runs = 1 + sum(bases)
            bases = [False, False, False]
        elif oc == "3B":
            runs = sum(bases)
            bases = [False, False, True]
        elif oc == "2B":
            runs = (1 if bases[2] else 0) + (1 if bases[1] else 0)
            bases = [False, True, bases[0]]
        elif oc == "1B":
            runs = 1 if bases[2] else 0
            bases = [True, bases[0], bases[1]]
        elif oc in {"BB", "HBP"}:
            # Force advance.
            if all(bases):
                runs = 1
            elif bases[0] and bases[1]:
                bases = [True, True, True]
            elif bases[0] and bases[2]:
                bases = [True, True, True]
            elif bases[0]:
                bases = [True, True, False]
            else:
                bases[0] = True
        else:
            outs += 1

        score_state[side_idx] += runs
        rows.append({
            "events": text,
            "away_team_runs": score_state[0],
            "home_team_runs": score_state[1],
            "row_idx": row_idx,
            "game_id": game_id,
            "inning": inning,
            "top_bottom": top_bottom,
            "play_id": f"{game_id}_{inning}_{0 if top_bottom=='top' else 1}_{row_idx+1}",
            "batting_team": "Home" if top_bottom == "bottom" else "Away",
            "fielding_team": "Away" if top_bottom == "bottom" else "Home",
            "season": 2024,
        })
        row_idx += 1
        if row_idx > 30:  # safety: shouldn't happen
            break
    return rows, batter_idx


def _synth_pbp(n_games: int = 30, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    league_batters = [(f"Batter_{i}", LEAGUE_DIST) for i in range(18)]
    # Power hitter slotted at lineup spot 4 for the Away team.
    away_batters = list(league_batters[:3]) + [("PowerHitter", POWER_HITTER_DIST)] + list(league_batters[4:9])
    home_batters = league_batters[9:18]

    all_rows: list[dict] = []
    for g in range(n_games):
        score = [0, 0]
        away_idx = home_idx = 0
        for inn in range(1, 8):
            top_rows, away_idx = _synth_half_inning(
                rng, away_batters, away_idx, f"g{g}", inn, "top", score, side_idx=0
            )
            all_rows.extend(top_rows)
            bot_rows, home_idx = _synth_half_inning(
                rng, home_batters, home_idx, f"g{g}", inn, "bottom", score, side_idx=1
            )
            all_rows.extend(bot_rows)
    return pd.DataFrame(all_rows)


@pytest.fixture(scope="module")
def synth_pa() -> pd.DataFrame:
    pbp = _synth_pbp()
    return build_pa_table(pbp)


def test_pa_table_has_expected_columns(synth_pa: pd.DataFrame) -> None:
    for col in ("batter", "outcome", "state_before", "state_after", "runs_on_play"):
        assert col in synth_pa.columns


def test_re_matrix_monotonic(synth_pa: pd.DataFrame) -> None:
    re = compute_re_matrix(synth_pa)
    # Empty bases with 0 outs should have higher RE than 2 outs.
    assert re.get("___:0", 0) > re.get("___:2", 0)
    # Bases loaded 0 outs should beat empty bases 0 outs.
    assert re.get("123:0", 0) > re.get("___:0", 0)


def test_linear_weights_ordered(synth_pa: pd.DataFrame) -> None:
    re = compute_re_matrix(synth_pa)
    pa_re = compute_re24(synth_pa, re)
    lw = compute_linear_weights(pa_re).set_index("outcome")
    # The Tango ordering: HR > 3B > 2B > 1B > BB > 0 > K/GO/FO.
    assert lw.loc["HR", "run_value"] > lw.loc["3B", "run_value"] > lw.loc["2B", "run_value"]
    assert lw.loc["2B", "run_value"] > lw.loc["1B", "run_value"] > lw.loc["BB", "run_value"]
    assert lw.loc["BB", "run_value"] > 0
    # An out is worth less than a walk by a healthy margin.
    assert lw.loc["K", "run_value"] < lw.loc["BB", "run_value"]


def test_wrc_plus_power_hitter(synth_pa: pd.DataFrame) -> None:
    re = compute_re_matrix(synth_pa)
    pa_re = compute_re24(synth_pa, re)
    lw = compute_linear_weights(pa_re)
    wrc = player_wrc_plus(synth_pa, lw)

    # League average wRC+ across PAs should be ~100.
    wpa = (wrc["wRC+"] * wrc["PA"]).sum() / wrc["PA"].sum()
    assert 95 < wpa < 105, f"league average wRC+ = {wpa:.1f}, expected ~100"

    # PowerHitter should be well above 100.
    ph = wrc[wrc["batter"] == "PowerHitter"]
    assert not ph.empty, "PowerHitter missing from wRC+ output"
    assert ph["wRC+"].iloc[0] > 130, f"PowerHitter wRC+ = {ph['wRC+'].iloc[0]:.1f}"
