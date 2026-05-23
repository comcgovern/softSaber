"""Classify PBP ``events`` text into structured outcomes.

We map free-text descriptions like::

    "Smith, J. singled to right center, 2 RBI; Jones scored; Brown to third."

into a row with::

    batter="Smith, J.", outcome="1B", rbi=2, fielders=["RF","CF"],
    out_on_play=0, sac=False, ...

This is the layer that gates everything downstream. Anything we can't parse
is logged and excluded from the PA-level table so it doesn't silently
corrupt linear weights.

Status: STUB. Wire up the classifier here once we have real PBP samples
cached locally; build a comprehensive test suite in tests/test_events.py
covering each ``OUTCOME_PATTERNS`` entry below.
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

# Heuristic batter-name extractor. NCAA PBP usually starts each PA with the
# batter's "Last, F." or "Last F." followed by a verb fragment.
_BATTER_RE = re.compile(
    r"^\s*(?P<name>[A-Z][\w'\-]+(?:,\s*[A-Z]\.?)?(?:\s+[A-Z][\w'\-]+)?)\s+"
    r"(?:singled|doubled|tripled|homered|walked|"
    r"intentionally walked|grounded|flied|lined|popped|fouled|"
    r"struck out|hit (?:by pitch|into double play)|reached)"
)

_RBI_RE = re.compile(r"(\d+)\s*RBI", re.I)


@dataclass
class ParsedEvent:
    outcome: str | None
    batter: str | None
    rbi: int = 0
    raw: str = ""
    notes: list[str] = field(default_factory=list)


def classify(text: str) -> ParsedEvent:
    """Single-event classifier. Returns ``outcome=None`` for unmatched text."""
    out = ParsedEvent(outcome=None, batter=None, raw=text)

    m_b = _BATTER_RE.match(text)
    if m_b:
        out.batter = m_b.group("name").strip()

    for pat, code in OUTCOME_PATTERNS:
        if pat.search(text):
            out.outcome = code
            break

    m_rbi = _RBI_RE.search(text)
    if m_rbi:
        out.rbi = int(m_rbi.group(1))

    return out
