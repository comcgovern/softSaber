"""Diagnose the unattributed-pitcher gap.

Usage:
    python scripts/diagnose_starters.py 2026
"""
from __future__ import annotations

import sys

import pandas as pd

from softsaber.parse.pitcher import _is_starting_pitcher_position


def main(year: str) -> int:
    gp = pd.read_parquet(f"data/processed/game_players/{year}.parquet")
    starters_p = gp[
        gp["position"].apply(_is_starting_pitcher_position)
        & gp["starter"].fillna(False)
    ]

    n_games = gp["game_id"].nunique()
    n_game_teams = gp.groupby(["game_id", "team_id"]).ngroups
    n_starter_game_teams = starters_p.groupby(["game_id", "team_id"]).ngroups

    print(f"games:                {n_games}")
    print(f"game-team pairs:      {n_game_teams}")
    print(
        f"with known P starter: {n_starter_game_teams} "
        f"({100 * n_starter_game_teams / n_game_teams:.0f}%)"
    )

    seeded_keys = set(starters_p.groupby(["game_id", "team_id"]).groups.keys())
    all_keys = set(gp.groupby(["game_id", "team_id"]).groups.keys())
    missing_keys = list(all_keys - seeded_keys)

    print()
    print(f"game-team pairs WITHOUT a starter row: {len(missing_keys)}")
    print()
    print("Sample of missing pairs and their actual position values:")
    print("-" * 70)
    for gid, tid in missing_keys[:15]:
        sub = gp[(gp["game_id"] == gid) & (gp["team_id"] == tid)]
        positions = sub["position"].tolist()
        starters_in_team = sub["starter"].fillna(False).sum()
        print(
            f"game={gid} team={tid}: {len(sub)} players, "
            f"{starters_in_team} starters, positions={positions}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "2026"))
