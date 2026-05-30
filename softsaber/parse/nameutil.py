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
# No-comma shapes ("Firstname Lastname", "F. LASTNAME", "CJ Haney") are
# handled by tokenizing inside match_player rather than dedicated regexes.


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


def _resolve(
    players: pd.DataFrame,
    ln: pd.Series,
    fn: pd.Series,
    last_pred,
    first_pred,
) -> pd.Series | None:
    """Return the best row whose normalized last/first names satisfy the
    given predicates.  ``ln``/``fn`` are the pre-normalized last/first
    name series for ``players`` (computed once by the caller).  Prefers
    starters when the predicates match more than one row."""
    mask = ln.map(last_pred) & fn.map(first_pred)
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


# A leading token that's 1-3 capital letters (optionally dotted) is an
# initial cluster, not a first name: "T.", "CJ", "AG", "MJ".
_INITIAL_CLUSTER = re.compile(r"^[A-Z]{1,3}\.?$")


def match_player(
    pbp_name: str,
    players: pd.DataFrame,
) -> pd.Series | None:
    """Find the best-matching row in a player DataFrame for a PBP name string.

    Handles the name shapes NCAA softball feeds actually emit:

    1. ``"LASTNAME, F"`` / ``"LASTNAME,F."`` — last + first-initial (standard).
    2. ``"Lastname,Firstname"`` — last + full first (some feeds).
    3. ``"Firstname Lastname"`` — first then last, no comma.
    4. ``"F. LASTNAME"`` / ``"CJ Haney"`` — initial(s) lead, surname trails.
    5. Truncated surnames (``"Lydia Vander"`` for "Lydia VanderWoude") via
       surname-prefix matching, gated by an exact/initial first-name match.
    6. Surname-only (``"Jackson"``) when unique.

    Prefers starters on ties.  Returns ``None`` if nothing resolves.
    """
    if players.empty:
        return None

    raw = pbp_name.strip()
    ln = players["last_name"].fillna("").map(_normalize)
    fn = players["first_name"].fillna("").map(_normalize)

    def eq(x: str):
        return lambda v: v == x

    def starts(x: str):
        return (lambda v: v.startswith(x)) if x else (lambda v: True)

    # --- Comma shapes -----------------------------------------------------
    parsed = normalize_pbp_name(raw)  # "LASTNAME, FI"
    if parsed is not None:
        last_norm, initial = parsed
        hit = _resolve(players, ln, fn, eq(last_norm), starts(initial))
        if hit is not None:
            return hit

    m = _FULL_NAME_RE.match(raw)  # "Lastname,Firstname"
    if m:
        last_norm = _normalize(m.group(1))
        first_norm = _normalize(m.group(2))
        hit = _resolve(players, ln, fn, eq(last_norm), eq(first_norm))
        if hit is None:
            hit = _resolve(players, ln, fn, eq(last_norm), starts(first_norm[:1]))
        if hit is not None:
            return hit

    # --- No-comma shapes: tokenize ---------------------------------------
    # Normalize the no-space initial-leads form ("T.THOMAS" → "T. THOMAS")
    # so the tokenizer below sees two tokens.  Require the dot so a bare
    # ALL-CAPS surname like "THOMAS" doesn't get split into "THO MAS".
    raw_tokenized = re.sub(
        r"^([A-Z]{1,3}\.)(?=[A-Z])", r"\1 ", raw
    )
    if "," not in raw_tokenized:
        toks = raw_tokenized.split()
        if len(toks) >= 2:
            first_tok, *rest = toks
            last_norm = _normalize(" ".join(rest))
            if _INITIAL_CLUSTER.match(first_tok):
                # "T. THOMAS" / "CJ Haney": match surname + first initial.
                initial = first_tok[0].lower()
                hit = _resolve(players, ln, fn, eq(last_norm), starts(initial))
                if hit is None and len(last_norm) >= 4:
                    hit = _resolve(players, ln, fn, starts(last_norm), starts(initial))
                if hit is not None:
                    return hit
            else:
                # "Firstname Lastname" — exact, then surname-prefix
                # (truncation) gated by the first name.
                first_norm = _normalize(first_tok)
                hit = _resolve(players, ln, fn, eq(last_norm), eq(first_norm))
                if hit is None and len(last_norm) >= 4:
                    hit = _resolve(players, ln, fn, starts(last_norm), eq(first_norm))
                if hit is None and len(last_norm) >= 4:
                    hit = _resolve(
                        players, ln, fn, starts(last_norm), starts(first_norm[:1])
                    )
                if hit is not None:
                    return hit

    # --- Surname-only fallback -------------------------------------------
    # Try the comma-split surname and the trailing token; accept an exact
    # unique match, then a unique prefix match (for truncated surnames).
    candidates: list[str] = []
    if "," in raw:
        candidates.append(_normalize(raw.split(",")[0]))
    toks = raw.split()
    if toks:
        candidates.append(_normalize(toks[-1]))
    seen: set[str] = set()
    for last_only in candidates:
        if not last_only or last_only in seen:
            continue
        seen.add(last_only)
        exact = players[ln == last_only]
        if len(exact) == 1:
            return exact.iloc[0]
        if len(last_only) >= 5:
            pref = players[ln.str.startswith(last_only)]
            if len(pref) == 1:
                return pref.iloc[0]

    return None


__all__ = ["match_player", "normalize_pbp_name"]
