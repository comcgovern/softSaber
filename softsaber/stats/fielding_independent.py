"""Fielding-independent pitching: xFIP and a softball-calibrated SIERA.

Both are ERA-scaled and calibrated on the supplied data rather than
imported from MLB, because NCAA softball's run environment, strikeout
rates, and 7-inning games make MLB coefficients wrong here.

**xFIP** — fielding-independent, with home runs normalized to a league
fly-ball rate.  We replace each pitcher's actual HR with an *expected*
HR = (their fly-ball-outs) × (league HR per fly-ball-out), then weight
the three true outcomes (xHR, BB+HBP, K) by their run values and put the
result on an ERA scale via a league constant.

Run-value coefficients come from the linear-weights table when provided
(genuinely our-data-calibrated); otherwise we fall back to the classic
FIP 13/3/2 weights scaled to the 7-inning game.

*Caveat:* the fly-ball denominator is fly-ball-OUTS only (PBP doesn't
give trajectory for hits), so the league HR/FBO rate runs high versus a
true HR/FB.  Because the same proxy is used for both the league rate and
each pitcher's fly balls, the bias largely divides out of the expected-HR
estimate.

**softSIERA** — a SIERA-style regression fit on *this* dataset: regress
each qualifying pitcher's ERA on strikeout rate, walk rate, ground-ball
rate, and their squares and interactions, weighted by batters faced.
The fitted model is then evaluated for every pitcher.  This is the
honest "we calibrated our own coefficients on D1 softball" statistic.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Innings in a regulation NCAA softball game.
SOFTBALL_GAME_INNINGS = 7.0

# Classic FIP run weights (HR, BB+HBP, K), per-9-innings ERA scale.
# Scaled to 7 innings for the fallback path.
_FIP_HR = 13.0
_FIP_BB = 3.0
_FIP_K = 2.0


def league_hr_per_fbo(pitcher_df: pd.DataFrame) -> float:
    """League HR / (HR + fly-ball-outs) across all pitchers in the frame."""
    hr = float(pitcher_df["HR"].sum())
    fbo = float(pitcher_df["FB_bbo"].sum())
    denom = hr + fbo
    return hr / denom if denom > 0 else 0.0


def add_xfip(
    pitcher_df: pd.DataFrame,
    *,
    weights: pd.DataFrame | None = None,
    min_ip: float = 1.0,
) -> pd.DataFrame:
    """Add an ``xFIP`` column to a pitcher-rates frame.

    Requires columns: ``HR``, ``FB_bbo``, ``BB_total``, ``HBP``, ``K``,
    ``IP``, ``ERA``.  Pitchers with ``IP < min_ip`` get ``xFIP = NaN`` and
    are excluded from the league-constant calibration.
    """
    df = pitcher_df.copy()
    needed = {"HR", "FB_bbo", "BB_total", "HBP", "K", "IP", "ERA"}
    missing = needed - set(df.columns)
    if missing:
        log.warning("add_xfip: missing columns %s; skipping", missing)
        df["xFIP"] = np.nan
        return df

    lg_rate = league_hr_per_fbo(df)
    x_hr = df["FB_bbo"] * lg_rate  # expected home runs

    if weights is not None and not weights.empty and "outcome" in weights.columns:
        w = weights.set_index("outcome")["run_value_above_out"]
        w_hr = float(w.get("HR", _FIP_HR / SOFTBALL_GAME_INNINGS))
        w_bb = float(w.get("BB", _FIP_BB / SOFTBALL_GAME_INNINGS))
        w_k = float(w.get("K", -_FIP_K / SOFTBALL_GAME_INNINGS))  # negative
        # Per-inning fielding-independent runs, then onto a 7-inning scale.
        raw = (w_hr * x_hr + w_bb * (df["BB_total"] + df["HBP"]) + w_k * df["K"])
    else:
        # Fallback: classic FIP weights scaled from 9- to 7-inning.
        s = SOFTBALL_GAME_INNINGS / 9.0
        raw = (
            _FIP_HR * s * x_hr
            + _FIP_BB * s * (df["BB_total"] + df["HBP"])
            - _FIP_K * s * df["K"]
        )

    ip = df["IP"].replace(0, np.nan)
    per_game = (raw / ip) * SOFTBALL_GAME_INNINGS

    qualified = df["IP"] >= min_ip
    if qualified.any():
        # IP-weighted league means → constant aligns mean xFIP to mean ERA.
        w_ip = df.loc[qualified, "IP"]
        lg_era = float((df.loc[qualified, "ERA"] * w_ip).sum() / w_ip.sum())
        lg_raw = float((per_game[qualified] * w_ip).sum() / w_ip.sum())
        constant = lg_era - lg_raw
    else:
        constant = 0.0

    df["xFIP"] = np.where(qualified, per_game + constant, np.nan)
    log.info("add_xfip: lg HR/FBO=%.3f, constant=%.3f", lg_rate, constant)
    return df


# ---------------------------------------------------------------------------
# softSIERA
# ---------------------------------------------------------------------------

_SIERA_FEATURES = ("K_pct", "BB_pct", "GB_pct_bbo")


def _design_matrix(df: pd.DataFrame) -> np.ndarray:
    """Build the SIERA design matrix: intercept, the three rates, their
    squares, and their pairwise interactions."""
    k = df["K_pct"].to_numpy(dtype=float)
    bb = df["BB_pct"].to_numpy(dtype=float)
    gb = df["GB_pct_bbo"].to_numpy(dtype=float)
    n = len(df)
    return np.column_stack([
        np.ones(n),
        k, bb, gb,
        k * k, bb * bb, gb * gb,
        k * bb, k * gb, bb * gb,
    ])


def fit_soft_siera(
    pitcher_df: pd.DataFrame,
    *,
    min_tbf: int = 100,
) -> tuple[np.ndarray | None, dict]:
    """Fit the softSIERA coefficients on qualifying pitchers.

    Target is ERA; features are K%, BB%, GB%(bbo) with squares and
    interactions.  Rows are weighted by batters faced (``TBF``).  Returns
    ``(coefficients, info)`` where ``coefficients`` is the 10-vector (or
    ``None`` if too few qualifiers) and ``info`` carries the R² and N.
    """
    needed = {"ERA", "TBF", *_SIERA_FEATURES}
    if needed - set(pitcher_df.columns):
        return None, {"reason": "missing columns", "n": 0}

    q = pitcher_df[
        (pitcher_df["TBF"] >= min_tbf)
        & pitcher_df["ERA"].notna()
        & np.isfinite(pitcher_df["ERA"])
    ].copy()
    if len(q) < 20:
        return None, {"reason": "too few qualifying pitchers", "n": len(q)}

    X = _design_matrix(q)
    y = q["ERA"].to_numpy(dtype=float)
    w = np.sqrt(q["TBF"].to_numpy(dtype=float))  # WLS via row scaling

    Xw = X * w[:, None]
    yw = y * w
    coef, *_ = np.linalg.lstsq(Xw, yw, rcond=None)

    pred = X @ coef
    ss_res = float((w * (y - pred) ** 2).sum())
    ss_tot = float((w * (y - np.average(y, weights=w)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coef, {"n": len(q), "r2": r2, "min_tbf": min_tbf}


def add_soft_siera(
    pitcher_df: pd.DataFrame,
    *,
    min_tbf: int = 100,
) -> pd.DataFrame:
    """Add a ``softSIERA`` column by fitting on qualifiers then predicting
    for every pitcher.  Pitchers below ``min_tbf`` still get a prediction
    (the model just wasn't fit on them)."""
    df = pitcher_df.copy()
    coef, info = fit_soft_siera(df, min_tbf=min_tbf)
    if coef is None:
        log.warning("add_soft_siera: not fit (%s)", info.get("reason"))
        df["softSIERA"] = np.nan
        return df

    pred = _design_matrix(df) @ coef
    # Clip to a sane softball ERA range to keep wild extrapolations honest.
    df["softSIERA"] = np.clip(pred, 0.0, 15.0).round(3)
    log.info(
        "add_soft_siera: fit on %d pitchers (min_tbf=%d), R²=%.3f",
        info["n"], info["min_tbf"], info["r2"],
    )
    return df


__all__ = [
    "add_soft_siera",
    "add_xfip",
    "fit_soft_siera",
    "league_hr_per_fbo",
]
