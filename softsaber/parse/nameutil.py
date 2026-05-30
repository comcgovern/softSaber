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
# Matches "Lastname,Firstname" (or with a space): full first name, no period.
# Distinct from _NAME_RE because the first name is 2+ letters; matched as
# a fallback so "SMITH, J." (initial form) takes the original path.
_FULL_NAME_RE = re.compile(
    r"^([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*),\s*([A-Z][a-z][\w'\-]*)\s*$"
)
# Matches "F. LASTNAME" / "F.LASTNAME" / "F LASTNAME" — first initial leads.
# Distinct from the title-case _FIRST_LAST_RE because here the surname
# is often ALL CAPS in PBP.
_INITIAL_LAST_RE = re.compile(
    r"^([A-Z])\.?\s*([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*)\s*$"
)
# Matches "Firstname Lastname" (no comma): two title-case tokens.
_FIRST_LAST_RE = re.compile(
    r"^([A-Z][a-z][\w'\-]*)\s+([A-Z][\w'\-]+(?:\s+[A-Z][\w'\-]+)*)\s*$"
)


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


def _resolve(players: pd.DataFrame, last_norm: str, first_pred) -> pd.Series | None:
    """Return the best row whose last_name matches and first_name passes
    ``first_pred``.  Prefers starters when ambiguous."""
    last_match = players["last_name"].fillna("").str.lower().apply(_normalize) == last_norm
    first_norm = players["first_name"].fillna("").str.lower().apply(_normalize)
    mask = last_match & first_norm.apply(first_pred)
    hits = players[mask]
    if len(hits) == 1:
        return hits.iloc[0]
    if len(hits) > 1:
        if "starter" in hits.columns:
            starters = hits[hits["starter"].fillna(False)]
            if not starters.empty:
                return starters.iloc[0]
        return hits.iloc[0]
    return None


def match_player(
    pbp_name: str,
    players: pd.DataFrame,
) -> pd.Series | None:
    """Find the best-matching row in a player DataFrame for a PBP name string.

    Supports three PBP name shapes:

    1. ``"LASTNAME, F"`` / ``"LASTNAME,F."`` — last + first-initial (NCAA standard).
    2. ``"Lastname,Firstname"`` — last + full first (team-code pitching changes).
    3. ``"Firstname Lastname"`` — first then last with no comma (free-text feeds).

    Resolution prefers starters when multiple rows share the same surname.
    Falls back to last-name-only when no first-name signal is available.
    Returns ``None`` if nothing matches.
    """
    if players.empty:
        return None

    raw = pbp_name.strip()

    # 1. "LASTNAME, FI"
    parsed = normalize_pbp_name(raw)
    if parsed is not None:
        last_norm, initial = parsed
        hit = _resolve(players, last_norm, lambda f: f.startswith(initial))
        if hit is not None:
            return hit

    # 2. "Lastname,Firstname" (full first name)
    m = _FULL_NAME_RE.match(raw)
    if m:
        last_norm = _normalize(m.group(1))
        first_norm = _normalize(m.group(2))
        hit = _resolve(players, last_norm, lambda f: f == first_norm)
        if hit is not None:
            return hit
        # Fall through to initial-only match if full-name didn't hit.
        hit = _resolve(players, last_norm, lambda f: f.startswith(first_norm[:1]))
        if hit is not None:
            return hit

    # 3. "Firstname Lastname" (no comma)
    m = _FIRST_LAST_RE.match(raw)
    if m:
        first_norm = _normalize(m.group(1))
        last_norm = _normalize(m.group(2))
        hit = _resolve(players, last_norm, lambda f: f == first_norm)
        if hit is not None:
            return hit
        hit = _resolve(players, last_norm, lambda f: f.startswith(first_norm[:1]))
        if hit is not None:
            return hit

    # 4. "F. LASTNAME" — initial leads, surname trails (often ALL CAPS).
    m = _INITIAL_LAST_RE.match(raw)
    if m:
        initial = m.group(1).lower()
        last_norm = _normalize(m.group(2))
        hit = _resolve(players, last_norm, lambda f: f.startswith(initial))
        if hit is not None:
            return hit

    # 5. Last-name only fallback.
    last_only = _normalize(raw.split(",")[0].strip())
    mask_last = players["last_name"].fillna("").str.lower().apply(_normalize) == last_only
    hits_last = players[mask_last]
    if len(hits_last) == 1:
        return hits_last.iloc[0]

    return None


__all__ = ["match_player", "normalize_pbp_name"]
