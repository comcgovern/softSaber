"""Project-wide configuration: paths, season metadata, scraping constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# stats.ncaa.org uses a season_division_id keyed by (division, year). Values
# through 2024 come from softballR; 2025/2026 need to be discovered by hitting
# the scoreboard page for a known date and reading the redirect / page source.
# Override via NCAA_DIVISION_IDS env if you confirm new values.
DIVISION_IDS: dict[tuple[str, int], int] = {
    ("D1", 2021): 17540,
    ("D1", 2022): 17840,
    ("D1", 2023): 18101,
    ("D1", 2024): 18261,
    # TODO(2025/2026): confirm by probing
    #   https://stats.ncaa.org/rankings/national_ranking?academic_year=2025...
    # or by inspecting a 2025 scoreboard URL once a known game date is loaded.
    # Placeholders below are guesses based on the ~+160/year increment pattern
    # observed for D1 (17540 → 17840 → 18101 → 18261). Treat as unconfirmed.
    ("D1", 2025): 18420,
    ("D1", 2026): 18580,
}

# Default focus per the project plan.
TARGET_SEASONS: tuple[int, ...] = (2024, 2025, 2026)
TARGET_DIVISION: str = "D1"

# stats.ncaa.org blocks unrecognized clients; set a browser-like UA.
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

REQUEST_TIMEOUT_S = 30
REQUEST_RETRY_MAX = 5
REQUEST_RETRY_BASE_DELAY_S = 2.0
# stats.ncaa.org is slow; keep us friendly.
INTER_REQUEST_DELAY_S = 0.75


@dataclass(frozen=True)
class Season:
    year: int
    division: str = TARGET_DIVISION

    @property
    def division_id(self) -> int:
        try:
            return DIVISION_IDS[(self.division, self.year)]
        except KeyError as e:
            raise KeyError(
                f"No stats.ncaa.org division_id mapped for {self.division} {self.year}. "
                f"Add it to softsaber.config.DIVISION_IDS."
            ) from e


def ensure_dirs() -> None:
    for p in (RAW_DIR, PROCESSED_DIR):
        p.mkdir(parents=True, exist_ok=True)
