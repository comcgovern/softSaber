"""Project-wide configuration: paths, season metadata, scraping constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# NCAA's GraphQL backend (sdataprod.ncaa.com) keys softball by sportCode/division
# rather than a per-season div id, so we don't need a year-by-year ID table.
SPORT_CODE_SOFTBALL = "WSB"
DIVISION_CODES: dict[str, int] = {"D1": 1, "D2": 2, "D3": 3}

TARGET_SEASONS: tuple[int, ...] = (2024, 2025, 2026)
TARGET_DIVISION: str = "D1"

# stats.ncaa.org national ranking page parameters for WSB Division I.
#
# stat_seq=281 is TEAM batting average for WSB — confirmed from:
#   https://stats.ncaa.org/rankings/national_ranking?academic_year=2026.0
#     &division=1.0&ranking_period=113.0&sport_code=WSB&stat_seq=281.0
# Team rankings pages carry /teams/{id} links; individual rankings (stat_seq=271)
# carry only /player/{id} links and are useless for team ID discovery.
WSB_D1_RANKING_STAT_SEQ: int = 281

WSB_D1_RANKING_PERIOD: dict[int, int | None] = {
    2024: 88,    # confirmed
    2025: 101,   # confirmed
    2026: 113,   # confirmed
}

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT_S = 30
REQUEST_RETRY_MAX = 5
REQUEST_RETRY_BASE_DELAY_S = 2.0

# Rate limiting and concurrency.
#
# Two separate limits because the two backends have very different tolerance:
#
#   henrygd (ncaa-api.henrygd.me) — public shared instance; it makes its own
#   requests to sdataprod.ncaa.com / data.ncaa.com / ncaa.com on our behalf.
#   Being too aggressive hurts other users and the NCAA upstream.  The server
#   has a 45-second cache so parallel requests to the *same* URL are wasted;
#   1–2 concurrent requests with a ~0.5 s gap is plenty.
#
#   stats.ncaa.org — Akamai-fronted; curl_cffi handles TLS fingerprinting.
#   The site itself is fast; the bottleneck is the WAF, not throughput.

# henrygd / ncaa-api: be a polite guest on a public API.
HENRYGD_RATE_PER_SEC: float = 2.0
HENRYGD_WORKERS: int = 2

# stats.ncaa.org scraping (team discovery, roster pages).
NCAA_STATS_RATE_PER_SEC: float = 3.0
NCAA_STATS_WORKERS: int = 3

# Shared fallback used by older code paths and http_cache's _RateLimiter.
# Set to the more conservative of the two so a single limiter is safe for both.
REQUEST_RATE_PER_SEC: float = HENRYGD_RATE_PER_SEC
REQUEST_WORKERS: int = HENRYGD_WORKERS


@dataclass(frozen=True)
class Season:
    year: int
    division: str = TARGET_DIVISION

    @property
    def division_code(self) -> int:
        try:
            return DIVISION_CODES[self.division]
        except KeyError as e:
            raise KeyError(
                f"No NCAA division code mapped for {self.division}. "
                f"Known: {sorted(DIVISION_CODES)}"
            ) from e


def ensure_dirs() -> None:
    for p in (RAW_DIR, PROCESSED_DIR):
        p.mkdir(parents=True, exist_ok=True)
