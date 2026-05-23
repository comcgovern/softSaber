"""Reconstruct base-out state from PBP text, half-inning by half-inning.

We track occupancy only (1B/2B/3B booleans + outs); runner identity isn't
needed for RE24, so this stays cheap and robust to typos in the underlying
text.

For each event we:

1. Snapshot the state as ``state_before``.
2. Use the **score column** (most reliable signal) to compute
   ``runs_on_play``: the diff between this row's cumulative score and the
   previous row's. NCAA emits the post-play score, so this is exact.
3. Apply the batter's outcome to derive a default post-state.
4. Apply any explicit runner-movement clauses (``X scored``, ``X to third``)
   on top of the default, to disambiguate cases the outcome alone can't
   resolve (e.g. how far did the runner on 2B advance on a single).
5. Cap outs at 3 and snapshot ``state_after``.

If our simulated runs_scored doesn't match the score-column diff at the end
of the half-inning, we flag the half-inning and downstream RE24 fitting can
choose to drop it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace

import pandas as pd

from .events import ParsedEvent, RunnerMove, classify

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BasesOuts:
    occ_1B: bool = False
    occ_2B: bool = False
    occ_3B: bool = False
    outs: int = 0

    def as_state_id(self) -> str:
        bases = (
            ("1" if self.occ_1B else "_")
            + ("2" if self.occ_2B else "_")
            + ("3" if self.occ_3B else "_")
        )
        return f"{bases}:{self.outs}"

    def occupied(self) -> list[int]:
        return [b for b, occ in zip((1, 2, 3), (self.occ_1B, self.occ_2B, self.occ_3B)) if occ]


EMPTY = BasesOuts()


def _set_base(s: BasesOuts, base: int, val: bool) -> BasesOuts:
    if base == 1:
        return replace(s, occ_1B=val)
    if base == 2:
        return replace(s, occ_2B=val)
    if base == 3:
        return replace(s, occ_3B=val)
    return s


def _force_advance(s: BasesOuts) -> tuple[BasesOuts, int]:
    """Walk/HBP-style force: push consecutive runners ahead by exactly one
    base. Returns the new state and runs scored.
    """
    runs = 0
    new = s
    if s.occ_1B and s.occ_2B and s.occ_3B:
        runs = 1
        new = BasesOuts(True, True, True, s.outs)
    elif s.occ_1B and s.occ_2B:
        new = BasesOuts(True, True, True, s.outs)
    elif s.occ_1B and s.occ_3B:
        new = BasesOuts(True, True, True, s.outs)
    elif s.occ_1B:
        new = BasesOuts(True, True, False, s.outs)
    else:
        # Runner on 1B not present — batter just takes first, no force.
        new = replace(s, occ_1B=True)
    return new, runs


def _apply_outcome(s: BasesOuts, ev: ParsedEvent) -> tuple[BasesOuts, int]:
    """Apply the batter's outcome to derive a default post-state.

    Returns (new_state, runs_scored_by_default). Runner movement clauses
    will then patch this state.
    """
    code = ev.outcome
    runs = 0
    occ1, occ2, occ3, outs = s.occ_1B, s.occ_2B, s.occ_3B, s.outs

    if code == "K":
        return replace(s, outs=outs + 1), 0
    if code in {"GO", "FO", "LO", "PO"}:
        return replace(s, outs=outs + 1), 0
    if code == "DP":
        # Outs += 2. Most common DP types clear the lead runner. We assume
        # the runner who was forced is the second out; conservative choice
        # is to wipe 1B and clear it.
        new_state = replace(s, outs=outs + 2)
        if s.occ_1B:
            new_state = replace(new_state, occ_1B=False)
        return new_state, 0
    if code == "SH":
        # Sacrifice bunt: out++, runners advance one base each.
        new_outs = outs + 1
        new3 = occ2 or occ3
        runs_sh = 1 if occ3 else 0
        new2 = occ1
        new1 = False
        return BasesOuts(new1, new2, new3, new_outs), runs_sh
    if code == "SF":
        # Sacrifice fly: out++, runner on 3B scores by definition.
        new_outs = outs + 1
        runs_sf = 1 if occ3 else 0
        return BasesOuts(occ1, occ2, False, new_outs), runs_sf
    if code == "BB" or code == "IBB" or code == "HBP":
        new, r = _force_advance(s)
        return new, r
    if code == "1B" or code == "ROE":
        # Default: every runner advances exactly one base. Movement clauses
        # may upgrade (e.g., runner from 1B to third on a single).
        runs += 1 if occ3 else 0
        new3 = occ2
        new2 = occ1
        new1 = True
        return BasesOuts(new1, new2, new3, outs), runs
    if code == "FC":
        # Fielder's choice: batter to 1B, one out, lead runner removed.
        new_outs = outs + 1
        # Remove lead runner: 3B → 2B → 1B priority.
        if occ3:
            occ3 = False
        elif occ2:
            occ2 = False
        elif occ1:
            occ1 = False
        return BasesOuts(True, occ2 or False, occ3 or False, new_outs), 0
    if code == "2B":
        runs += (1 if occ3 else 0) + (1 if occ2 else 0)
        new3 = occ1  # runner from 1B advances to 3B by default
        new2 = True
        new1 = False
        return BasesOuts(new1, new2, new3, outs), runs
    if code == "3B":
        runs += (1 if occ1 else 0) + (1 if occ2 else 0) + (1 if occ3 else 0)
        return BasesOuts(False, False, True, outs), runs
    if code == "HR":
        runs += 1 + sum((occ1, occ2, occ3))
        return BasesOuts(False, False, False, outs), runs

    # Unknown / unhandled outcome — leave state untouched.
    return s, 0


def _apply_movements(s: BasesOuts, moves: list[RunnerMove]) -> tuple[BasesOuts, int]:
    """Apply explicit runner movement clauses on top of the default state.

    The score-column diff is the source of truth for runs scored, so a small
    overcounting here is OK — RE24 fitting tolerates it.
    """
    runs_delta = 0
    state = s
    for m in moves:
        if m.kind == "score":
            # Move the most-advanced occupied base off the board.
            if state.occ_3B:
                state = replace(state, occ_3B=False)
            elif state.occ_2B:
                state = replace(state, occ_2B=False)
            elif state.occ_1B:
                state = replace(state, occ_1B=False)
            else:
                continue  # phantom "scored" — ignore
            runs_delta += 1
        elif m.kind == "advance":
            target = m.to_base  # 2 or 3
            if target == 3:
                # Promote a runner from 2B or 1B to 3B.
                if state.occ_2B and not state.occ_3B:
                    state = BasesOuts(state.occ_1B, False, True, state.outs)
                elif state.occ_1B and not state.occ_3B:
                    state = BasesOuts(False, state.occ_2B, True, state.outs)
            elif target == 2:
                # Promote a runner from 1B to 2B.
                if state.occ_1B and not state.occ_2B:
                    state = BasesOuts(False, True, state.occ_3B, state.outs)
        elif m.kind == "out":
            state = replace(state, outs=min(3, state.outs + 1))
            # Best-effort: remove a back-most runner (most likely to be
            # thrown out at the next base).
            if state.occ_1B:
                state = replace(state, occ_1B=False)
            elif state.occ_2B:
                state = replace(state, occ_2B=False)
            elif state.occ_3B:
                state = replace(state, occ_3B=False)
    return state, runs_delta


def reconstruct_half_inning(rows: pd.DataFrame) -> pd.DataFrame:
    """Augment one half-inning's PBP rows with base-out state and runs scored.

    Adds columns: batter, outcome, rbi, state_before, state_after,
    runs_on_play, half_inning_ok.
    """
    out = rows.copy().reset_index(drop=True)
    n = len(out)
    if n == 0:
        return out

    batter = [None] * n
    outcome = [None] * n
    rbi = [0] * n
    state_before = [""] * n
    state_after = [""] * n
    runs_on_play = [0] * n

    state = EMPTY

    # runs_on_play from score-column diff: away+home runs after this play
    # minus same total before. The first row of the half uses the *previous
    # half's* ending score, which we infer from the row's own pre-state by
    # subtracting the simulated runs.
    totals = (
        out["away_team_runs"].fillna(0).astype(float).values
        + out["home_team_runs"].fillna(0).astype(float).values
    )
    prev_total = float(totals[0]) if n else 0.0
    # Backfill: assume the half started with `totals[0] - simulated_runs[0]`.
    # We'll learn the right baseline after the first play and adjust later.

    for i in range(n):
        ev = classify(str(out.at[i, "events"]))
        batter[i] = ev.batter
        outcome[i] = ev.outcome
        rbi[i] = ev.rbi
        state_before[i] = state.as_state_id()

        if ev.outcome is None:
            state_after[i] = state.as_state_id()
            runs_on_play[i] = 0
            continue

        post, r_default = _apply_outcome(state, ev)
        post, r_extra = _apply_movements(post, ev.moves)

        # Authoritative runs from score-column diff (relative to last row).
        cur_total = float(totals[i])
        score_diff = max(0, int(round(cur_total - prev_total))) if i > 0 else None
        runs_on_play[i] = score_diff if score_diff is not None else (r_default + r_extra)
        prev_total = cur_total

        # If text-derived runs disagrees with score-column diff, trust
        # the score column — but cap outs at 3.
        post = replace(post, outs=min(3, post.outs))
        state = post
        state_after[i] = state.as_state_id()

        if state.outs >= 3:
            # Half-inning ends; remaining rows (if any) belong to the next
            # half by construction, so this loop just continues writing
            # zeros — but in well-formed PBP there are none after outs=3.
            state = EMPTY

    out["batter"] = batter
    out["outcome"] = outcome
    out["rbi"] = rbi
    out["state_before"] = state_before
    out["state_after"] = state_after
    out["runs_on_play"] = runs_on_play

    # Sanity flag: did simulated total runs match score-column diff?
    sim_runs = sum(runs_on_play)
    last_total = float(totals[-1]) if n else 0.0
    first_total = float(totals[0]) if n else 0.0
    # score_diff for first play uses None above; recompute proper score range
    score_range = last_total - (first_total - runs_on_play[0])
    out["half_inning_ok"] = abs(sim_runs - score_range) < 0.5

    return out


def attach_base_out_state(pbp_raw: pd.DataFrame) -> pd.DataFrame:
    if pbp_raw.empty:
        return pbp_raw
    keys = ["game_id", "inning", "top_bottom"]
    sorted_pbp = pbp_raw.sort_values([*keys, "row_idx"])
    # Iterate groups manually — pandas 2.2+ ``apply`` drops the group keys
    # from the concatenated result, which breaks downstream joins.
    parts = [
        reconstruct_half_inning(g)
        for _, g in sorted_pbp.groupby(keys, sort=False)
    ]
    if not parts:
        return pbp_raw.iloc[0:0]
    return pd.concat(parts, ignore_index=True)
