"""Akamai-cookie bootstrap for endpoints protected by NCAA's WAF.

The flow is:

1. Launch headless Chromium via Playwright.
2. Visit a real page on the target host so the Akamai JavaScript
   challenge runs and sets the ``_abck`` / ``bm_sz`` cookies.
3. Either pull the rendered HTML straight from the browser, or save
   the cookies to disk for ``curl_cffi`` to reuse.

This module is intentionally standalone — no other softsaber code
imports it yet.  Run the probe to verify Chromium can clear Akamai
from your network before wiring it into the ingest pipeline::

    python -m softsaber.ingest.akamai_session probe-roster <statsNcaaTeamId>
    python -m softsaber.ingest.akamai_session probe-gamecenter <contestId>

Install once::

    pip install -e .[akamai]
    playwright install chromium
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from ..config import RAW_DIR, ensure_dirs

log = logging.getLogger(__name__)

# How long to trust a cookie blob before re-running the browser.
COOKIE_TTL_S = 30 * 60

# Cookies Akamai sets that we actually need to forward.
AKAMAI_COOKIE_NAMES = {"_abck", "bm_sz", "bm_sv", "ak_bmsc", "bm_lso", "bm_so"}

# Hosts we mint cookies for.  Akamai cookies are per-domain so each
# protected host needs its own bootstrap visit.
STATS_HOST = "stats.ncaa.org"
NCAA_HOST = "www.ncaa.com"


def _cookie_path(host: str) -> Path:
    return RAW_DIR / "akamai_cookies" / f"{host}.json"


def _load_cached(host: str) -> dict[str, str] | None:
    path = _cookie_path(host)
    if not path.exists():
        return None
    age = time.time() - path.stat().st_mtime
    if age > COOKIE_TTL_S:
        log.debug("akamai cookies for %s expired (%.0fs old)", host, age)
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _save(host: str, cookies: dict[str, str]) -> None:
    ensure_dirs()
    path = _cookie_path(host)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cookies, indent=2))


def _looks_valid_abck(value: str) -> bool:
    """Akamai sets a placeholder ``_abck`` immediately, then replaces it
    once the JS challenge completes.  Treat anything over ~400 chars as
    challenge-cleared; below that we wait a bit more.
    """
    return len(value) > 400


def _new_browser(pw):  # type: ignore[no-untyped-def]
    browser = pw.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 800},
        locale="en-US",
    )
    return browser, context


def _wait_for_clearance(context, host: str, timeout_s: float = 20.0) -> str:
    """Block until a non-placeholder ``_abck`` cookie exists for ``host``."""
    deadline = time.time() + timeout_s
    last = ""
    while time.time() < deadline:
        jar = context.cookies(f"https://{host}")
        last = next((c["value"] for c in jar if c["name"] == "_abck"), "")
        if _looks_valid_abck(last):
            return last
        time.sleep(0.5)
    return last


def fetch_page_html(url: str) -> tuple[int, str, dict[str, str]]:
    """Load ``url`` in a headless browser and return (status, html, cookies).

    Cookies are scoped to the URL's host and saved to disk for reuse.
    """
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright is not installed.  Run:\n"
            "    pip install -e .[akamai]\n"
            "    playwright install chromium"
        ) from e

    from urllib.parse import urlparse
    host = urlparse(url).hostname or ""

    log.info("akamai: launching headless Chromium for %s", url)
    with sync_playwright() as pw:
        browser, context = _new_browser(pw)
        try:
            page = context.new_page()
            resp = page.goto(url, wait_until="networkidle", timeout=60_000)
            status = resp.status if resp else 0

            # Give the JS challenge a chance to complete before we read.
            _wait_for_clearance(context, host)

            html = page.content()
            cookies = {
                c["name"]: c["value"]
                for c in context.cookies()
                if c["name"] in AKAMAI_COOKIE_NAMES
            }
            if cookies:
                _save(host, cookies)
            log.info(
                "akamai: %s → status=%d, %d bytes, cookies=%s",
                url, status, len(html), sorted(cookies),
            )
            return status, html, cookies
        finally:
            browser.close()


def get_cookies(host: str, *, force_refresh: bool = False) -> dict[str, str]:
    """Return Akamai cookies for ``host``, minting via browser if needed.

    Bootstraps by visiting the host's root page (or a known-good path).
    """
    if not force_refresh:
        cached = _load_cached(host)
        if cached:
            log.debug("akamai: using cached cookies for %s", host)
            return cached
    bootstrap = {
        STATS_HOST: f"https://{STATS_HOST}/",
        NCAA_HOST: f"https://{NCAA_HOST}/scoreboard/softball/d1",
    }.get(host, f"https://{host}/")
    _, _, cookies = fetch_page_html(bootstrap)
    return cookies


def invalidate(host: str) -> None:
    path = _cookie_path(host)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------

def _probe_roster(team_season_id: str) -> dict[str, Any]:
    """Fetch a stats.ncaa.org roster page through Playwright and through
    curl_cffi-with-cookies, so we can see which path works."""
    url = f"https://{STATS_HOST}/teams/{team_season_id}/roster"

    status, html, cookies = fetch_page_html(url)
    snippet = html[:600]
    # Cheap "is this a real roster page" check: the table has these headers.
    looks_real = ("Jersey" in html and "Player" in html) or "roster" in html.lower()

    # Now try with curl_cffi using the captured cookies.
    curl_status: int | str
    curl_bytes = 0
    curl_snippet = ""
    try:
        from curl_cffi import requests as curl_requests
        sess = curl_requests.Session(impersonate="chrome124")
        sess.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Referer": f"https://{STATS_HOST}/",
        })
        resp = sess.get(url, cookies=cookies, timeout=20)
        curl_status = resp.status_code
        curl_bytes = len(resp.content)
        curl_snippet = resp.text[:300]
    except Exception as e:  # noqa: BLE001
        curl_status = f"err: {e}"

    return {
        "browser_status": status,
        "browser_bytes": len(html),
        "browser_looks_real": looks_real,
        "browser_snippet": snippet,
        "curl_status": curl_status,
        "curl_bytes": curl_bytes,
        "curl_snippet": curl_snippet,
    }


def _probe_gamecenter(contest_id: str) -> dict[str, Any]:
    """POST a GameCenter request to sdataprod with browser-minted cookies."""
    from curl_cffi import requests as curl_requests

    from .sdataprod import GAMECENTER_HASH, GAMECENTER_OP, SDATAPROD_URL

    # Bootstrap on the matching game page so cookies are minted in-context.
    _, _, cookies = fetch_page_html(f"https://{NCAA_HOST}/game/{contest_id}")

    body = {
        "operationName": GAMECENTER_OP,
        "variables": {"contestId": str(contest_id), "staticTestEnv": None},
        "extensions": {
            "persistedQuery": {"version": 1, "sha256Hash": GAMECENTER_HASH},
        },
    }
    sess = curl_requests.Session(impersonate="chrome124")
    sess.headers.update({
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Origin": f"https://{NCAA_HOST}",
        "Referer": f"https://{NCAA_HOST}/game/{contest_id}",
    })
    resp = sess.post(SDATAPROD_URL, json=body, cookies=cookies, timeout=20)
    return {
        "status": resp.status_code,
        "bytes": len(resp.content),
        "text_head": resp.text[:500],
    }


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) < 3:
        print(
            "usage:\n"
            "  python -m softsaber.ingest.akamai_session probe-roster <statsNcaaTeamId>\n"
            "  python -m softsaber.ingest.akamai_session probe-gamecenter <contestId>",
            file=sys.stderr,
        )
        return 2
    cmd, target = argv[1], argv[2]
    if cmd == "probe-roster":
        result = _probe_roster(target)
        print(f"browser status:     {result['browser_status']}")
        print(f"browser bytes:      {result['browser_bytes']}")
        print(f"browser looks real: {result['browser_looks_real']}")
        print(f"browser snippet (first 600 chars):")
        print(result["browser_snippet"])
        print()
        print(f"curl_cffi status:   {result['curl_status']}")
        print(f"curl_cffi bytes:    {result['curl_bytes']}")
        print(f"curl_cffi snippet:  {result['curl_snippet'][:200]}")
        return 0 if result["browser_looks_real"] else 1
    if cmd == "probe-gamecenter":
        result = _probe_gamecenter(target)
        print(f"sdataprod status: {result['status']}")
        print(f"sdataprod bytes:  {result['bytes']}")
        print(f"response head:\n{result['text_head']}")
        return 0 if result["status"] == 200 else 1
    print(f"unknown command: {cmd}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))


__all__ = ["fetch_page_html", "get_cookies", "invalidate"]
