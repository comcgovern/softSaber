"""Tests for batter/pitcher rate stats and fielding-independent metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from softsaber.stats.fielding_independent import (
    add_soft_siera,
    add_xfip,
    league_hr_per_fbo,
)
from softsaber.stats.rates import (
    _parse_innings_pitched,
    batter_rates,
    pitcher_rates,
)


def _pa_rows(rows: list[tuple]) -> pd.DataFrame:
    """rows: (season, batter, batting_team, fielding_team, pitcher, outcome)."""
    return pd.DataFrame(
        [
            {
                "season": s, "batter": b, "batting_team": bt,
                "fielding_team": ft, "pitcher": p, "outcome": o,
                "game_id": "g1",
            }
            for (s, b, bt, ft, p, o) in rows
        ]
    )


# ---------------------------------------------------------------------------
# Innings parsing
# ---------------------------------------------------------------------------

def test_parse_innings_pitched() -> None:
    assert _parse_innings_pitched("6.2") == 6 + 2 / 3
    assert _parse_innings_pitched("6.0") == 6.0
    assert _parse_innings_pitched("7") == 7.0
    assert _parse_innings_pitched("0.1") == 1 / 3
    assert _parse_innings_pitched("") == 0.0
    assert _parse_innings_pitched(None) == 0.0


# ---------------------------------------------------------------------------
# Batter rates
# ---------------------------------------------------------------------------

def test_batter_rates_slash_line() -> None:
    # Player A: 1 single, 1 HR, 1 BB, 1 K, 1 GO across 5 PA.
    pa = _pa_rows([
        (2026, "A", "T1", "T2", "P1", "1B"),
        (2026, "A", "T1", "T2", "P1", "HR"),
        (2026, "A", "T1", "T2", "P1", "BB"),
        (2026, "A", "T1", "T2", "P1", "K"),
        (2026, "A", "T1", "T2", "P1", "GO"),
    ])
    df = batter_rates(pa)
    row = df[df["player"] == "A"].iloc[0]
    assert row["PA"] == 5
    assert row["AB"] == 4  # PA - BB
    assert row["H"] == 2
    assert row["AVG"] == 0.5  # 2/4
    # OBP = (H + BB + HBP) / (AB + BB + HBP + SF) = (2+1)/(4+1) = 0.6
    assert abs(row["OBP"] - 0.6) < 1e-9
    # SLG = TB/AB = (1 + 4)/4 = 1.25
    assert abs(row["SLG"] - 1.25) < 1e-9
    assert abs(row["K_pct"] - 0.2) < 1e-9
    assert abs(row["BB_pct"] - 0.2) < 1e-9


def test_batter_rates_batted_ball_outs() -> None:
    pa = _pa_rows([
        (2026, "A", "T1", "T2", "P1", "GO"),
        (2026, "A", "T1", "T2", "P1", "GO"),
        (2026, "A", "T1", "T2", "P1", "FO"),
        (2026, "A", "T1", "T2", "P1", "LO"),
    ])
    row = batter_rates(pa).iloc[0]
    # GB=2, FB=1, LD=1, total bbo=4
    assert row["GB_bbo"] == 2
    assert row["FB_bbo"] == 1
    assert row["LD_bbo"] == 1
    assert abs(row["GB_pct_bbo"] - 0.5) < 1e-9


def test_batter_rates_empty() -> None:
    assert batter_rates(pd.DataFrame()).empty


# ---------------------------------------------------------------------------
# Pitcher rates
# ---------------------------------------------------------------------------

def test_pitcher_rates_basic() -> None:
    pa = _pa_rows([
        (2026, "B1", "T2", "T1", "Ace P", "K"),
        (2026, "B2", "T2", "T1", "Ace P", "K"),
        (2026, "B3", "T2", "T1", "Ace P", "BB"),
        (2026, "B4", "T2", "T1", "Ace P", "1B"),
    ])
    df = pitcher_rates(pa)
    row = df[df["player"] == "Ace P"].iloc[0]
    assert row["TBF"] == 4
    assert abs(row["K_pct"] - 0.5) < 1e-9
    assert abs(row["BB_pct"] - 0.25) < 1e-9
    # BAA = H/AB = 1/(4 - 1 BB) = 1/3
    assert abs(row["BAA"] - 1 / 3) < 1e-9


def test_pitcher_rates_joins_boxscore_ip() -> None:
    pa = _pa_rows([
        (2026, "B1", "T2", "T1", "Ace P", "K"),
        (2026, "B2", "T2", "T1", "Ace P", "1B"),
    ])
    game_players = pd.DataFrame([
        {"player_name": "Ace P", "team_id": "100", "ip": "7.0",
         "er": 2, "hits_allowed": 5, "bb_pit": 1, "k_pit": 8},
    ])
    df = pitcher_rates(pa, game_players)
    row = df[df["player"] == "Ace P"].iloc[0]
    assert row["IP"] == 7.0
    # ERA = 7 * ER / IP = 7 * 2 / 7 = 2.0
    assert abs(row["ERA"] - 2.0) < 1e-9
    # WHIP = (BB + H) / IP = (1 + 5)/7
    assert abs(row["WHIP"] - 6 / 7) < 1e-3


def test_pitcher_rates_no_pitcher_column() -> None:
    pa = pd.DataFrame([{"season": 2026, "outcome": "K", "batter": "x"}])
    assert pitcher_rates(pa).empty


# ---------------------------------------------------------------------------
# Fielding independent
# ---------------------------------------------------------------------------

def _pitcher_frame() -> pd.DataFrame:
    """Synthetic pitcher-rates frame with the columns xFIP/SIERA need."""
    rng = np.random.default_rng(0)
    n = 60
    k_pct = rng.uniform(0.05, 0.35, n)
    bb_pct = rng.uniform(0.02, 0.15, n)
    gb = rng.uniform(0.3, 0.7, n)
    tbf = rng.integers(120, 600, n)
    # ERA loosely (negatively) tied to K%, positively to BB%.
    era = 5.0 - 8 * k_pct + 10 * bb_pct + rng.normal(0, 0.4, n)
    return pd.DataFrame({
        "season": 2026,
        "player": [f"P{i}" for i in range(n)],
        "team": "T",
        "K_pct": k_pct, "BB_pct": bb_pct, "GB_pct_bbo": gb,
        "TBF": tbf,
        "HR": rng.integers(0, 8, n),
        "FB_bbo": rng.integers(10, 60, n),
        "BB_total": (bb_pct * tbf).astype(int),
        "HBP": rng.integers(0, 5, n),
        "K": (k_pct * tbf).astype(int),
        "IP": rng.uniform(20, 120, n),
        "ERA": np.clip(era, 0.5, 9.0),
    })


def test_league_hr_per_fbo() -> None:
    df = pd.DataFrame({"HR": [2, 3], "FB_bbo": [18, 27]})
    # 5 / (5 + 45) = 0.1
    assert abs(league_hr_per_fbo(df) - 0.1) < 1e-9


def test_add_xfip_aligns_mean_to_era() -> None:
    df = add_xfip(_pitcher_frame())
    assert "xFIP" in df.columns
    q = df[df["IP"] >= 1.0]
    # IP-weighted mean xFIP should equal IP-weighted mean ERA by construction.
    w = q["IP"]
    mean_xfip = (q["xFIP"] * w).sum() / w.sum()
    mean_era = (q["ERA"] * w).sum() / w.sum()
    assert abs(mean_xfip - mean_era) < 1e-6


def test_soft_siera_recovers_signal() -> None:
    df = add_soft_siera(_pitcher_frame(), min_tbf=100)
    assert "softSIERA" in df.columns
    assert df["softSIERA"].notna().all()
    # Predicted SIERA should correlate strongly with the noisy ERA it was
    # fit on, since ERA here is a smooth function of the features.
    corr = np.corrcoef(df["softSIERA"], df["ERA"])[0, 1]
    assert corr > 0.7


def test_soft_siera_too_few_pitchers() -> None:
    df = _pitcher_frame().head(5)
    out = add_soft_siera(df)
    assert out["softSIERA"].isna().all()
