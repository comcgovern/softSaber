"""Utilities for matching PBP batter-name strings to boxscore player rows.

NCAA PBP uses ``"LASTNAME, FI"`` format (e.g. ``"KNIGHT, S"``), sometimes
without a space after the comma (``"HORNBUCKLE,S"``), and occasionally with
only a last name when it is unambiguous within the game.  Boxscore rows
carry proper-case ``first_name`` / ``last_name``.

The public surface is small:

* :func:`normalize_pbp_name` — parse a raw PBP name token into (last, first_initial).
* :func:`match_player`       — look up a PBP token in a game_players DataFrame.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

# Matches "LASTNAME, FI" or "LASTNAME,FI" with optional trailing period on the initial.
_NAME_RE = re.compile(r"^([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*),\s*([A-Z]\.?)\s*$")


def _normalize(s: str) -> str:
    """Lower-case, strip accents, collapse whitespace."""
    nfkd = unicodedata.normalize("NFKD", s)
    ascii_s = nfkd.encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", ascii_s).strip().lower()


def normalize_pbp_name(raw: str) -> tuple[str, str] | None:
    """Parse a PBP name token into ``(last_normalized, first_initial_lower)``.

    Returns ``None`` if the string doesn't match the expected pattern.

    Examples::

        normalize_pbp_name("KNIGHT, S")      → ("knight", "s")
        normalize_pbp_name("McANALLY, L")    → ("mcanally", "l")
        normalize_pbp_name("HORNBUCKLE,S")   → ("hornbuckle", "s")
        normalize_pbp_name("SMITH-JONES, A") → ("smith-jones", "a")
    """
    raw = raw.strip()
    m = _NAME_RE.match(raw)
    if not m:
        return None
    last = _normalize(m.group(1))
    initial = m.group(2).strip(".").lower()
    return (last, initial)


def match_player(
    pbp_name: str,
    players: pd.DataFrame,
) -> pd.Series | None:
    """Find the best-matching row in a game_players DataFrame for a PBP name string.

    ``players`` must have ``last_name`` and ``first_name`` columns (as produced
    by :func:`softsaber.ingest.boxscore.parse_boxscore`).

    Resolution strategy:
    1. Exact last-name + first-initial match → return the unique row.
    2. If multiple rows share the same last name + initial (rare), return the
       starter, then fall back to the first match.
    3. Last-name-only match (when no comma/initial found) with a unique result.

    Returns ``None`` if no match is found.
    """
    if players.empty:
        return None

    parsed = normalize_pbp_name(pbp_name)

    if parsed is not None:
        last_norm, initial = parsed
        mask = (
            players["last_name"].str.lower().apply(_normalize) == last_norm
        ) & (
            players["first_name"].str.lower().str[:1] == initial
        )
        hits = players[mask]
        if len(hits) == 1:
            return hits.iloc[0]
        if len(hits) > 1:
            starters = hits[hits["starter"]]
            return starters.iloc[0] if not starters.empty else hits.iloc[0]

    # Fallback: last-name only (useful when PBP omits the initial).
    last_only = _normalize(pbp_name.split(",")[0].strip())
    mask_last = players["last_name"].str.lower().apply(_normalize) == last_only
    hits_last = players[mask_last]
    if len(hits_last) == 1:
        return hits_last.iloc[0]

    return None


__all__ = ["match_player", "normalize_pbp_name"]
