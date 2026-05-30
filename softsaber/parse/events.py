"""Classify PBP ``events`` text into structured outcomes.

We map free-text descriptions like::

    "Smith, J. singled to right center, 2 RBI; Jones scored; Brown to third."

into a structured row with the batter's outcome plus a list of explicit
runner movement clauses (``Jones scored``, ``Brown to third``) that the
base-out simulator in :mod:`softsaber.parse.baserunners` consumes.

Anything we can't classify (pitching changes, defensive substitutions, etc.)
returns ``outcome=None`` and is dropped from the PA-level table so it can't
silently corrupt linear weights.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canonical outcome codes used everywhere downstream (linear weights, RE24,
# wOBA, wRC+). Keep this list small and stable.
OUTCOMES = (
    "1B",   # single
    "2B",   # double
    "3B",   # triple
    "HR",   # home run
    "BB",   # unintentional walk
    "IBB",  # intentional walk
    "HBP",  # hit by pitch
    "K",    # strikeout
    "GO",   # ground out (incl. force outs)
    "FO",   # fly out
    "LO",   # line out
    "PO",   # pop out / fouled out / infield fly
    "FC",   # fielder's choice (no error)
    "ROE",  # reached on error
    "SF",   # sacrifice fly
    "SH",   # sacrifice bunt
    "DP",   # any double play (out-on-play=2)
)

# Each entry: ordered list of (regex, outcome). First match wins.
# Order matters — double-play patterns must come BEFORE generic ground out etc.
OUTCOME_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bhomered\b", re.I), "HR"),
    (re.compile(r"\btripled\b", re.I), "3B"),
    (re.compile(r"\bdoubled\b", re.I), "2B"),
    (re.compile(r"\bsingled\b", re.I), "1B"),
    (re.compile(r"\bintentionally walked\b", re.I), "IBB"),
    (re.compile(r"\bwalked\b", re.I), "BB"),
    (re.compile(r"\bhit by pitch\b", re.I), "HBP"),
    (re.compile(r"\bstruck out\b", re.I), "K"),
    (re.compile(r"hit into double play|grounded into double play|"
                r"lined into double play|fouled into double play", re.I), "DP"),
    (re.compile(r"reached on (?:an? )?(?:fielding |throwing )?error", re.I), "ROE"),
    (re.compile(r"reached on a fielder'?s choice", re.I), "FC"),
    (re.compile(r"sacrifice fly|sac fly", re.I), "SF"),
    (re.compile(r"sacrifice bunt|sac bunt", re.I), "SH"),
    (re.compile(r"infield fly", re.I), "PO"),
    (re.compile(r"\bpopped up\b|\bfouled out\b", re.I), "PO"),
    (re.compile(r"\blined out\b", re.I), "LO"),
    (re.compile(r"\bflied out\b", re.I), "FO"),
    (re.compile(r"\bgrounded out\b", re.I), "GO"),
]

# Heuristic batter-name extractor.  Three PBP shapes observed in NCAA softball:
#   * "LASTNAME, F. singled..." — standard last-comma-initial
#   * "Firstname Lastname singled..." — free-text feed
#   * "F. LASTNAME singled..." — initial-leads form (e.g. "T. THOMAS")
# Optional "F. " prefix at the start covers the third case; the rest is the
# original lastname-or-titlecase capture.
_BATTER_RE = re.compile(
    r"^\s*(?P<name>"
    r"(?:[A-Z]\.?\s+)?"               # optional "F. " or "F " prefix
    r"[A-Z][\w'\-]+"                  # surname or first title-case token
    r"(?:,\s*[A-Z]\.?)?"              # optional ", F."
    r"(?:\s+[A-Z][\w'\-]+)?"          # optional " Lastname"
    r")\s+"
    r"(?:singled|doubled|tripled|homered|walked|"
    r"intentionally walked|grounded|flied|lined|popped|fouled|"
    r"struck out|hit (?:by pitch|into double play)|reached)"
)

_RBI_RE = re.compile(r"(\d+)\s*RBI", re.I)

# Movement-clause patterns. Order matters — "out at X" before "to X".
_MOVE_SCORED_RE = re.compile(r"\bscored\b", re.I)
_MOVE_OUT_RE = re.compile(
    r"\b(?:out at (?:second|third|home)|caught stealing|picked off|"
    r"thrown out|doubled off)\b",
    re.I,
)
_MOVE_TO_RE = re.compile(r"\bto (second|third|home)\b", re.I)

# Sacrifice fly detection (modifies outcome from FO/LO/PO to SF).
_SAC_FLY_RE = re.compile(r"\b(?:sf|sacrifice fly|sac fly)\b", re.I)
_SAC_BUNT_RE = re.compile(r"\bsac\b|\bsacrifice bunt\b|\bsac bunt\b", re.I)


# Per-base destination as small int: 1=1B, 2=2B, 3=3B, 4=home.
_BASE_INT = {"second": 2, "third": 3, "home": 4}


@dataclass
class RunnerMove:
    """One runner movement clause inside an event."""

    # 'advance' (to base 2 or 3), 'score' (cross home), 'out' (caught on bases)
    kind: str
    to_base: int  # 2, 3, 4 (score), or 0 (out)


@dataclass
class ParsedEvent:
    outcome: str | None
    batter: str | None
    rbi: int = 0
    moves: list[RunnerMove] = field(default_factory=list)
    raw: str = ""


def classify(text: str) -> ParsedEvent:
    """Classify one event-text row.

    Returns ``outcome=None`` for unmatched (substitution/comment) rows.
    The first ``;``-delimited clause is treated as the batter's primary
    outcome; subsequent clauses are parsed as runner movement.
    """
    out = ParsedEvent(outcome=None, batter=None, raw=text)

    m_b = _BATTER_RE.match(text)
    if m_b:
        out.batter = m_b.group("name").strip()

    primary, *rest = [c.strip() for c in text.split(";")]

    for pat, code in OUTCOME_PATTERNS:
        if pat.search(primary):
            out.outcome = code
            break

    # SF/SH override: a fly out / bunt that drives in a run is sacrifice.
    if out.outcome in {"FO", "LO", "PO"} and _SAC_FLY_RE.search(text):
        out.outcome = "SF"
    if out.outcome == "GO" and _SAC_BUNT_RE.search(text):
        out.outcome = "SH"

    m_rbi = _RBI_RE.search(text)
    if m_rbi:
        out.rbi = int(m_rbi.group(1))

    # Pull movement clauses from the non-primary fragments. Also scan the
    # primary for "scored"/"to third" mentions that come AFTER a comma
    # (some NCAA boxes put first-runner movement on the same clause).
    for clause in [primary, *rest]:
        # Skip the batter's own action in the primary — but a primary
        # like "Smith homered, Jones scored" still needs the second half
        # scanned. Easiest: scan every clause for movement keywords.
        for m in _MOVE_OUT_RE.finditer(clause):
            out.moves.append(RunnerMove(kind="out", to_base=0))
        for m in _MOVE_SCORED_RE.finditer(clause):
            out.moves.append(RunnerMove(kind="score", to_base=4))
        for m in _MOVE_TO_RE.finditer(clause):
            base = _BASE_INT[m.group(1).lower()]
            out.moves.append(RunnerMove(kind="advance" if base < 4 else "score", to_base=base))

    return out
