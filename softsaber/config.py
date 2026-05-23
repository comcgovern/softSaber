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
# stat_seq=271 is batting average for WSB — confirmed from:
#   https://stats.ncaa.org/rankings/national_ranking?academic_year=2026.0
#     &division=1.0&ranking_period=113.0&sport_code=WSB&stat_seq=271.0
# stat_seq appears stable across years for the same sport/stat category.
#
# ranking_period is year-specific (the end-of-season period ID for that year).
# 2026 confirmed = 113. Fill in 2024/2025 by visiting the ranking page for
# those years and noting the ranking_period value in the URL.
WSB_D1_RANKING_STAT_SEQ: int = 271

WSB_D1_RANKING_PERIOD: dict[int, int | None] = {
    2024: None,  # TODO: inspect ranking URL for 2024 and note ranking_period
    2025: None,  # TODO: inspect ranking URL for 2025
    2026: 113,   # confirmed
}

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT_S = 30
REQUEST_RETRY_MAX = 5
REQUEST_RETRY_BASE_DELAY_S = 2.0
INTER_REQUEST_DELAY_S = 0.75


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
