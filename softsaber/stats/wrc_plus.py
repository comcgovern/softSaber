"""wOBA and wRC+ per player-season.

Reference formula (Fangraphs):

    wOBA = sum_c weight_c * count_c / (AB + BB - IBB + SF + HBP)
    wRAA = ((wOBA - lgwOBA) / wOBAscale) * PA
    wRC  = wRAA + (lgR/PA) * PA
    wRC+ = ((wRAA / PA + lgR/PA) +
            (lgR/PA - parkFactor * lgR/PA)) / lg(wRC/PA, excluding pitchers)
           * 100

In a college softball context where every batter is on a "position-player"
roster (DH excepted), the "excluding pitchers" denominator collapses to the
plain league wRC/PA, simplifying to::

    wRC+ = (((wOBA - lgwOBA)/wOBAscale + lgR/PA)
            / (lgR/PA * park_factor)) * 100

Status: STUB — signature and structure only. Fill in once linear_weights
and park_factors return real numbers.
"""

from __future__ import annotations

import pandas as pd


def player_wrc_plus(
    pa: pd.DataFrame,
    weights: pd.DataFrame,
    park_factors: pd.DataFrame,
) -> pd.DataFrame:
    """Compute wOBA and wRC+ per (player, season).

    Inputs:
        pa:           PA-level table with columns batter, season, outcome,
                      batting_team (used to resolve park factor).
        weights:      output of :func:`linear_weights.compute_linear_weights`.
        park_factors: output of :func:`park_factors.regress_park_factors`.

    Output schema:
        season, batter, team, PA, AB, H, BB, HBP, woba, wraa, wrc_plus
    """
    raise NotImplementedError("player_wrc_plus not yet implemented")
