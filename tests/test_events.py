"""Smoke tests for the event-text classifier.

These cover the obvious cases. Once we have real PBP samples cached, replace
the synthetic strings with verbatim fixtures from stats.ncaa.org.
"""

from __future__ import annotations

import pytest

from softsaber.parse.events import classify


@pytest.mark.parametrize(
    "text,expected_outcome,expected_rbi",
    [
        ("Smith singled to right center, 2 RBI; Jones scored.", "1B", 2),
        ("Jones, A. doubled down the lf line.", "2B", 0),
        ("Brown tripled to right center, 1 RBI; Davis scored.", "3B", 1),
        ("Lopez homered to left, 3 RBI.", "HR", 3),
        ("Wilson walked.", "BB", 0),
        ("Garcia intentionally walked.", "IBB", 0),
        ("Patel hit by pitch.", "HBP", 0),
        ("Kim struck out swinging.", "K", 0),
        ("Nguyen struck out looking.", "K", 0),
        ("Adams grounded out to ss.", "GO", 0),
        ("Martin flied out to cf.", "FO", 0),
        ("Reed lined out to 2b.", "LO", 0),
        ("Cole popped up to 3b.", "PO", 0),
        ("Diaz fouled out to c.", "PO", 0),
        ("Hall grounded into double play c to 2b to 1b.", "DP", 0),
        ("Owens reached on a fielding error by ss.", "ROE", 0),
        ("Hayes reached on a fielder's choice.", "FC", 0),
        ("Park grounded out to p, SAC, 1 RBI; runner scored.", "SH", 1),
    ],
)
def test_classify_outcome(text: str, expected_outcome: str, expected_rbi: int) -> None:
    ev = classify(text)
    assert ev.outcome == expected_outcome, f"got {ev.outcome} for {text!r}"
    assert ev.rbi == expected_rbi


def test_classify_unmatched_returns_none() -> None:
    assert classify("Pitching change: Roberts to p.").outcome is None
