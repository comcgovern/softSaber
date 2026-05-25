"""Quick probe: does following Akamai's bm-verify redirect get us a real roster page?

Usage:
    python scripts/probe_bm_verify.py 613592
"""
from __future__ import annotations

import re
import sys
import time

from curl_cffi import requests as r


def main(team_season_id: str) -> int:
    s = r.Session(impersonate="chrome124")
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Referer": "https://stats.ncaa.org/",
    })
    url = f"https://stats.ncaa.org/teams/{team_season_id}/roster"

    r1 = s.get(url, timeout=20)
    print(f"first  : {r1.status_code} {len(r1.content)}B")

    m = re.search(r"URL=['\"]([^'\"]+)['\"]", r1.text)
    if not m:
        print("no meta-refresh URL found in first response — printing head:")
        print(r1.text[:500])
        return 1

    next_path = m.group(1)
    next_url = (
        f"https://stats.ncaa.org{next_path}" if next_path.startswith("/") else next_path
    )
    print(f"follow : {next_url[:140]}{'...' if len(next_url) > 140 else ''}")

    time.sleep(5)
    r2 = s.get(next_url, timeout=20)
    print(f"second : {r2.status_code} {len(r2.content)}B")
    print()
    print("second body head (first 400 chars):")
    print(r2.text[:400])
    print()
    looks_real = any(
        marker in r2.text
        for marker in ("Jersey", "Player Name", "roster_table", "/players/")
    )
    print(f"contains player table? {looks_real}")

    if not looks_real and "meta http-equiv=\"refresh\"" in r2.text.lower():
        print()
        print("second response is ANOTHER challenge page — chained challenge.")
        m2 = re.search(r"URL=['\"]([^'\"]+)['\"]", r2.text)
        if m2:
            third_url = (
                f"https://stats.ncaa.org{m2.group(1)}"
                if m2.group(1).startswith("/") else m2.group(1)
            )
            print(f"follow2: {third_url[:140]}{'...' if len(third_url) > 140 else ''}")
            time.sleep(5)
            r3 = s.get(third_url, timeout=20)
            print(f"third  : {r3.status_code} {len(r3.content)}B")
            print()
            print("third body head (first 400 chars):")
            print(r3.text[:400])
            print()
            looks_real3 = any(
                marker in r3.text
                for marker in ("Jersey", "Player Name", "roster_table", "/players/")
            )
            print(f"contains player table? {looks_real3}")
            return 0 if looks_real3 else 2
    return 0 if looks_real else 2


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python scripts/probe_bm_verify.py <statsNcaaTeamId>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
