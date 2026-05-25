"""Season roster ingest — builds a per-season player table from boxscore data.

stats.ncaa.org is behind a WAF that blocks non-browser clients (roster and
ranking pages all return 403 or empty JS shells).  This module instead builds
rosters from the henrygd boxscore API, which we already call for every game.

The boxscore for each game carries first_name, last_name, jersey, position,
and team for every player who appeared.  Deduplicating across all games in the
season gives us a complete per-season roster for every team.

Output parquet schema (``rosters/{season}``):

    season, team_name, team_seoname, team_id,
    first_name, last_name, player_name, jersey, position
"""

from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .. import storage
from ..config import HENRYGD_WORKERS, NCAA_STATS_WORKERS
from . import ncaa_api, ncaa_stats, sdataprod

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Team-name normalisation (kept for discover_and_update_teams)
# ---------------------------------------------------------------------------

_ABBREV_REPLACEMENTS = [
    (re.compile(r"\bst\.?\b"), "state"),
    (re.compile(r"\buniv\.?\b"), "university"),
    (re.compile(r"\bcoll\.?\b"), "college"),
    (re.compile(r"\bn\.?\s*c\.?\b"), "north carolina"),
    (re.compile(r"\bs\.?\s*c\.?\b"), "south carolina"),
    (re.compile(r"\btex\.?\b"), "texas"),
    (re.compile(r"\bcal\.?\b"), "california"),
    (re.compile(r"\bmiss\.?\b"), "mississippi"),
    (re.compile(r"\bfla\.?\b"), "florida"),
    (re.compile(r"\bla\.?\b"), "louisiana"),
    (re.compile(r"\bga\.?\b"), "georgia"),
    (re.compile(r"\bva\.?\b"), "virginia"),
    (re.compile(r"\bky\.?\b"), "kentucky"),
    (re.compile(r"\bark\.?\b"), "arkansas"),
    (re.compile(r"\bmich\.?\b"), "michigan"),
    (re.compile(r"\bwash\.?\b"), "washington"),
    (re.compile(r"\bind\.?\b"), "indiana"),
    (re.compile(r"\b&\b"), "and"),
]


def _normalize_team_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    for pat, repl in _ABBREV_REPLACEMENTS:
        s = pat.sub(repl, s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------------------
# Team-ID discovery (unchanged — still useful for enriching teams table)
# ---------------------------------------------------------------------------

def discover_and_update_teams(
    teams: pd.DataFrame,
    games: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Discover ``stats_ncaa_team_id`` for each team and write an updated teams table.

    Uses ``contestId`` values from ``games`` as the primary discovery path.
    Falls back to the national ranking page when ranking config is available.

    Returns the updated teams DataFrame with a ``stats_ncaa_team_id`` column.
    """
    from ..config import WSB_D1_RANKING_PERIOD, WSB_D1_RANKING_STAT_SEQ

    contest_ids = games["game_id"].astype(str).tolist()
    ranking_period = WSB_D1_RANKING_PERIOD.get(year)

    id_map = ncaa_stats.discover_team_season_ids(
        year,
        division_id=1,
        contest_ids=contest_ids,
        stat_seq=WSB_D1_RANKING_STAT_SEQ,
        ranking_period=ranking_period,
    )

    teams = teams.copy()
    if not id_map:
        log.warning("year %s: no stats_ncaa_team_id found via contest or ranking pages", year)
        if "stats_ncaa_team_id" not in teams.columns:
            teams["stats_ncaa_team_id"] = None
        return teams

    normalised_map: dict[str, str] = {
        _normalize_team_name(n): tid for n, tid in id_map.items()
    }

    def _lookup(name: str) -> str | None:
        if name in id_map:
            return id_map[name]
        return normalised_map.get(_normalize_team_name(name))

    teams["stats_ncaa_team_id"] = teams["team_name"].apply(_lookup)
    matched = teams["stats_ncaa_team_id"].notna().sum()
    log.info("year %s: matched stats_ncaa_team_id for %d/%d teams", year, matched, len(teams))
    unmatched = teams[teams["stats_ncaa_team_id"].isna()]["team_name"].tolist()
    if unmatched:
        log.warning(
            "year %s: %d teams without stats_ncaa_team_id — "
            "unmatched henrygd names: %s | sample ncaa names: %s",
            year,
            len(unmatched),
            unmatched,
            list(id_map.keys())[:10],
        )

    storage.write_partition("teams", str(year), teams)
    return teams


# ---------------------------------------------------------------------------
# Roster extraction from boxscore data
# ---------------------------------------------------------------------------

def _is_degraded_first_name(first: str) -> bool:
    """A boxscore firstName is degraded if it's empty or just an initial.

    ~12% of softball entries arrive this way at the upstream source:
    ~8.8% first-initial (e.g. ``"A."``, ``"A"``) and ~3.1% empty.
    """
    if not first:
        return True
    stripped = first.replace(".", "").strip()
    return len(stripped) <= 1


def _split_combined_name(first: str, last: str) -> tuple[str, str]:
    """Repair the common upstream bug where the full name lands in lastName.

    Many softball boxscores arrive as ``firstName="" / lastName="Libby Pippin"``
    — the source didn't split the name at all.  When firstName is empty and
    lastName contains whitespace, treat the first whitespace-separated token
    as the given name and the remainder as the surname.

    Returns ``(first, last)`` unchanged if no repair applies.  Returns
    ``(first, last)`` with the split applied if firstName is empty and
    lastName has whitespace.

    Multi-token surnames (e.g. ``"De La Cruz"``) are still wrong after this
    repair — we choose the more common single-token-first-name case over
    perfect handling of compound surnames, since the GameCenter upgrade
    path is meant to fix the ambiguous cases.
    """
    if first or " " not in last.strip():
        return first, last
    parts = last.strip().split()
    if len(parts) < 2:
        return first, last
    return parts[0], " ".join(parts[1:])


def _gamecenter_name_index(payload: dict) -> dict[tuple[str, str, str], str]:
    """Walk a GameCenter payload and build a name-upgrade lookup.

    Returns a mapping ``(team_id, last_name_lower, jersey_str) → full_first_name``.
    Only entries with a usable (non-degraded) first name are included, so
    callers can blindly do ``index.get(key)`` to test for an upgrade.

    The GameCenter response shape isn't strictly contracted, so we
    recursively walk every dict, looking for ones that smell like a
    player record (carry both ``firstName`` and ``lastName``).  A nearby
    ``teamId`` and ``jerseyNumber``/``number`` are picked up from the
    same dict.
    """
    index: dict[tuple[str, str, str], str] = {}

    def _walk(node: object) -> None:
        if isinstance(node, dict):
            first = str(node.get("firstName") or "").strip()
            last = str(node.get("lastName") or "").strip()
            if first and last and not _is_degraded_first_name(first):
                tid = str(node.get("teamId") or node.get("team_id") or "").strip()
                jersey_raw = (
                    node.get("jerseyNumber")
                    if node.get("jerseyNumber") is not None
                    else node.get("number")
                )
                jersey = "" if jersey_raw is None else str(jersey_raw).strip()
                index[(tid, last.lower(), jersey)] = first
                # Also index without team_id and without jersey so the
                # lookup can fall back when boxscore lacks one of those.
                index.setdefault(("", last.lower(), jersey), first)
                index.setdefault((tid, last.lower(), ""), first)
                index.setdefault(("", last.lower(), ""), first)
            for v in node.values():
                _walk(v)
        elif isinstance(node, list):
            for v in node:
                _walk(v)

    _walk(payload)
    return index


def _upgrade_first_name(
    index: dict[tuple[str, str, str], str],
    team_id: str,
    last: str,
    jersey: object,
) -> str | None:
    """Look up a richer first name from a GameCenter index, or None."""
    if not index or not last:
        return None
    jstr = "" if jersey is None else str(jersey).strip()
    last_l = last.lower()
    for key in (
        (team_id, last_l, jstr),
        (team_id, last_l, ""),
        ("", last_l, jstr),
        ("", last_l, ""),
    ):
        hit = index.get(key)
        if hit:
            return hit
    return None


def _boxscore_to_player_rows(game_id: str) -> pd.DataFrame:
    """Fetch one game's boxscore and return a per-player DataFrame.

    Uses cached data when available; makes a live request otherwise.
    """
    import json

    from ..http_cache import FetchError, fetch
    from . import ncaa_api as _api

    url = f"{ncaa_api.SCOREBOARD_HOST}/game/{game_id}/boxscore"
    try:
        text = fetch(url, namespace="ncaa_api/boxscore", ext="json")
        payload = json.loads(text)
    except Exception as e:
        log.debug("game %s: boxscore unavailable for roster build: %s", game_id, e)
        return pd.DataFrame()

    rows = []
    team_meta: dict[str, dict] = {}
    for t in payload.get("teams") or []:
        if not isinstance(t, dict):
            continue
        tid = str(t.get("teamId", ""))
        team_meta[tid] = {
            "team_seoname": str(t.get("seoname") or ""),
            "team_name": str(t.get("name") or ""),
            "is_home": bool(t.get("isHome")),
        }

    gc_index: dict[tuple[str, str, str], str] | None = None

    for team_box in payload.get("teamBoxscore") or []:
        if not isinstance(team_box, dict):
            continue
        tid = str(team_box.get("teamId", ""))
        meta = team_meta.get(tid, {})
        for p in team_box.get("playerStats") or []:
            if not isinstance(p, dict) or not p.get("participated"):
                continue
            first = str(p.get("firstName") or "").strip()
            last = str(p.get("lastName") or "").strip()
            if not first and not last:
                continue
            jersey = p.get("number")

            # Repair the most common upstream degradation locally first:
            # firstName="" with the full name in lastName ("Libby Pippin").
            first, last = _split_combined_name(first, last)

            # If the firstName is still degraded ("", "A.") and the
            # GameCenter endpoint is reachable, upgrade via sdataprod.
            # (Akamai usually 403s this; the circuit breaker keeps us
            # from hammering once that's confirmed.)
            if last and _is_degraded_first_name(first):
                if gc_index is None:
                    gc_payload = sdataprod.fetch_gamecenter(game_id)
                    gc_index = _gamecenter_name_index(gc_payload) if gc_payload else {}
                upgrade = _upgrade_first_name(gc_index, tid, last, jersey)
                if upgrade:
                    first = upgrade

            rows.append(
                {
                    "team_id": tid,
                    "team_seoname": meta.get("team_seoname", ""),
                    "team_name": meta.get("team_name", ""),
                    "first_name": first,
                    "last_name": last,
                    "player_name": f"{first} {last}".strip(),
                    "jersey": jersey,
                    "position": str(p.get("position") or "").strip(),
                }
            )

    return pd.DataFrame(rows)


def _try_ncaa_rosters(teams: pd.DataFrame, year: int) -> pd.DataFrame:
    """Attempt to fetch rosters from stats.ncaa.org/teams/{id}/roster.

    Returns a combined DataFrame, or empty if the WAF blocks all requests.
    The Akamai challenge page is ~2 KB; fetch_team_roster retries once after
    the first request sets session cookies, which often bypasses the block.
    Requires ``stats_ncaa_team_id`` to be populated in ``teams``.
    """
    eligible = teams[teams["stats_ncaa_team_id"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame()

    log.info("rosters: trying stats.ncaa.org for %d teams", len(eligible))

    def _fetch(row) -> pd.DataFrame:  # type: ignore[type-arg]
        tid = str(row.stats_ncaa_team_id)
        df = ncaa_stats.fetch_team_roster(tid, year)
        if df.empty:
            return df
        df["team_name"] = row.team_name
        df["stats_ncaa_team_id"] = tid
        df["season"] = year
        return df

    with ThreadPoolExecutor(max_workers=NCAA_STATS_WORKERS) as exe:
        results = list(exe.map(_fetch, eligible.itertuples(index=False)))

    frames = [df for df in results if not df.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def ingest_season_rosters(
    teams: pd.DataFrame,
    games: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Build the season roster, preferring stats.ncaa.org, falling back to henrygd.

    **Primary**: fetches stats.ncaa.org/teams/{id}/roster for each team with a
    known ``stats_ncaa_team_id``.  The page is server-rendered; a one-retry
    pattern handles the Akamai bot-challenge that appears on the first request.
    Yields ``ncaa_player_id``, jersey, position, class_year, and full name.

    **Fallback**: if the NCAA site is fully blocked, we build the roster from
    the henrygd boxscore API (already cached).  This gives first_name,
    last_name, jersey, and position but no ``ncaa_player_id``.

    Writes ``rosters/{year}.parquet`` and returns the combined DataFrame.
    """
    combined = _try_ncaa_rosters(teams, year)

    if combined.empty:
        log.info("rosters: NCAA fetch yielded nothing — falling back to henrygd boxscores")
        game_ids = games["game_id"].astype(str).tolist()
        with ThreadPoolExecutor(max_workers=HENRYGD_WORKERS) as exe:
            results = list(exe.map(_boxscore_to_player_rows, game_ids))
        frames = [df for df in results if not df.empty]
        if not frames:
            log.warning("rosters year=%s: no player data from any source", year)
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)

        # A player appears in many games' boxscores across a season; some
        # appearances will have degraded names ("" or "A.") and others
        # will be full ("Libby Pippin"). Pick one row per player keeping
        # the best name we ever saw for them.
        #
        # Identity within a team is "same last_name + same jersey".  We
        # also fall back to last_name alone for rows missing jersey, so
        # a player who only ever appeared without a number doesn't get
        # dropped.
        def _name_quality(first: str) -> int:
            if not first:
                return 0
            stripped = first.replace(".", "").strip()
            if len(stripped) <= 1:
                return 1  # "A.", "A"
            return 2 + len(first)  # prefer longest full name

        combined["_q"] = combined["first_name"].fillna("").map(_name_quality)
        combined = combined.sort_values(
            ["team_seoname", "last_name", "jersey", "_q"],
            ascending=[True, True, True, False],
        )
        # Two-pass dedup: rows with a jersey collapse on (team, last, jersey);
        # rows without a jersey (or jersey == -1) collapse on (team, last).
        has_jersey = combined["jersey"].notna() & (combined["jersey"] != -1)
        with_j = combined[has_jersey].drop_duplicates(
            subset=["team_seoname", "last_name", "jersey"], keep="first"
        )
        without_j = combined[~has_jersey].drop_duplicates(
            subset=["team_seoname", "last_name"], keep="first"
        )
        # If a (team, last) pair shows up in both groups, prefer the jerseyed one.
        seen = set(zip(with_j["team_seoname"], with_j["last_name"]))
        without_j = without_j[
            ~without_j.apply(lambda r: (r["team_seoname"], r["last_name"]) in seen, axis=1)
        ]
        combined = (
            pd.concat([with_j, without_j], ignore_index=True)
            .drop(columns=["_q"])
            .reset_index(drop=True)
        )
        # Recompute player_name in case the surviving row had a stale value
        # (e.g., it came in before _split_combined_name updated the parts).
        combined["player_name"] = (
            combined["first_name"].fillna("") + " " + combined["last_name"].fillna("")
        ).str.strip()
        if "team_seoname" in teams.columns and "team_name" in teams.columns:
            seo_to_name = teams.set_index("team_seoname")["team_name"].to_dict()
            mask = combined["team_name"].eq("") | combined["team_name"].isna()
            combined.loc[mask, "team_name"] = combined.loc[mask, "team_seoname"].map(seo_to_name)

    combined["season"] = year
    storage.write_partition("rosters", str(year), combined)
    sample = combined.get("player_name", combined.get("last_name", pd.Series())).head(5).tolist()
    log.info("rosters year=%s: wrote %d player rows, sample=%s", year, len(combined), sample)
    return combined


__all__ = [
    "discover_and_update_teams",
    "ingest_season_rosters",
]
