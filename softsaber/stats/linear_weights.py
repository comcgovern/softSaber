"""Per-outcome linear weights and wOBA coefficients, per season.

Standard derivation (Tango / The Book):

    For each outcome c, run_value(c) = mean RE24 over all PAs with outcome c.
    Then wOBA weights are run_value(c) normalized so that the league-average
    PA has wOBA == league OBP:

        scale = (lgOBP - lgBB_weight * (lgBB+lgHBP)/lgPA - ...) / lgwOBAraw

    In practice we just emit the per-outcome run values and let
    :mod:`softsaber.stats.wrc_plus` consume them.

Status: STUB. Implementation lives in :func:`compute_linear_weights`.
"""

from __future__ import annotations

import pandas as pd


# Outcomes that count as plate appearances (denominators for OBP / wOBA).
PA_OUTCOMES = {"1B", "2B", "3B", "HR", "BB", "IBB", "HBP", "K", "GO", "FO",
               "LO", "PO", "FC", "ROE", "SF", "SH", "DP"}

# Outcomes used in the wOBA numerator. SH excluded by convention; SF included
# in the numerator-denominator both, matching Fangraphs.
WOBA_OUTCOMES = ("BB", "HBP", "1B", "2B", "3B", "HR")


def compute_linear_weights(pa_with_re24: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame indexed by outcome with run-value columns.

    Columns:
        outcome, n, run_value, woba_weight
    """
    if pa_with_re24.empty or "re24" not in pa_with_re24.columns:
        return pd.DataFrame(columns=["outcome", "n", "run_value", "woba_weight"])

    grp = pa_with_re24.groupby("outcome")["re24"].agg(["count", "mean"])
    grp.columns = ["n", "run_value"]
    grp = grp.reset_index()

    # Shift so that the average out has run_value == 0. wOBA weights become
    # the run-value above an out, scaled by lgOBP / sum(weighted PAs).
    out_mask = grp["outcome"].isin({"K", "GO", "FO", "LO", "PO", "DP", "SH"})
    avg_out_rv = (
        (grp.loc[out_mask, "run_value"] * grp.loc[out_mask, "n"]).sum()
        / grp.loc[out_mask, "n"].sum()
    ) if out_mask.any() else 0.0

    grp["run_value_above_out"] = grp["run_value"] - avg_out_rv
    # woba_weight = run_value_above_out * scale, where scale forces lg wOBA
    # to equal lg OBP. Scale computation is deferred to wrc_plus.compute().
    grp["woba_weight"] = grp["run_value_above_out"]
    return grp
