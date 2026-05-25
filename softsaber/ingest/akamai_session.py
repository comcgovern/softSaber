"""Akamai-cookie bootstrap for endpoints protected by NCAA's WAF.

The flow is:

1. Launch headless Chromium via Playwright.
2. Visit a real ncaa.com page so the Akamai JavaScript challenge runs
   and sets the ``_abck`` / ``bm_sz`` cookies.
3. Save the cookies to disk with a TTL.
4. Hand them to ``curl_cffi`` for subsequent API calls.

This module is intentionally standalone — no other softsaber code
imports it yet.  Run the probe to verify Chromium can clear Akamai
from your network before wiring it into the ingest pipeline::

    python -m softsaber.ingest.akamai_session probe <contestId>

Install once::

    pip install playwright
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

# The page we visit to trigger the JS challenge.  Any real ncaa.com page
# works; the game-center page is convenient because it's exactly the
# kind of context that legitimately calls sdataprod afterwards.
BOOTSTRAP_URL_FMT = "https://www.ncaa.com/game/{contest_id}"
DEFAULT_BOOTSTRAP_URL = "https://www.ncaa.com/scoreboard/softball/d1"

# Cookies Akamai sets that we actually need to forward.
AKAMAI_COOKIE_NAMES = {"_abck", "bm_sz", "bm_sv", "ak_bmsc", "bm_lso", "bm_so"}


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
    once the JS challenge completes.  The challenged version is much
    longer and ends with ``~-1~-1~-1`` (or similar) — the placeholder
    ends with ``~-1~-1~-1~-1`` plus a fingerprint segment that's clearly
    incomplete.  Treat any value over ~400 chars as good-enough; below
    that we wait a bit more.
    """
    return len(value) > 400


def _fetch_cookies_via_browser(bootstrap_url: str) -> dict[str, str]:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as e:
        raise RuntimeError(
            "playwright is not installed.  Run:\n"
            "    pip install playwright\n"
            "    playwright install chromium"
        ) from e

    log.info("akamai: launching headless Chromium for %s", bootstrap_url)
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        try:
            context = browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport={"width": 1280, "height": 800},
                locale="en-US",
            )
            page = context.new_page()
            page.goto(bootstrap_url, wait_until="networkidle", timeout=45_000)

            # Wait for a real (non-placeholder) _abck.
            deadline = time.time() + 20
            while time.time() < deadline:
                jar = context.cookies("https://www.ncaa.com")
                abck = next((c["value"] for c in jar if c["name"] == "_abck"), "")
                if _looks_valid_abck(abck):
                    break
                page.wait_for_timeout(500)

            cookies = {
                c["name"]: c["value"]
                for c in context.cookies()
                if c["name"] in AKAMAI_COOKIE_NAMES
            }
            log.info(
                "akamai: captured %d cookies (%s)",
                len(cookies), ", ".join(sorted(cookies)),
            )
            return cookies
        finally:
            browser.close()


def get_cookies(
    *,
    host: str = "ncaa.com",
    bootstrap_url: str = DEFAULT_BOOTSTRAP_URL,
    force_refresh: bool = False,
) -> dict[str, str]:
    """Return Akamai cookies for ``host``, minting via browser if needed."""
    if not force_refresh:
        cached = _load_cached(host)
        if cached:
            log.debug("akamai: using cached cookies for %s", host)
            return cached
    cookies = _fetch_cookies_via_browser(bootstrap_url)
    if cookies:
        _save(host, cookies)
    return cookies


def invalidate(host: str = "ncaa.com") -> None:
    path = _cookie_path(host)
    if path.exists():
        path.unlink()


# ---------------------------------------------------------------------------
# Probe — verifies the cookies actually unlock sdataprod
# ---------------------------------------------------------------------------

def _probe_sdataprod(contest_id: str) -> dict[str, Any]:
    """POST a GameCenter request to sdataprod with browser-minted cookies."""
    from curl_cffi import requests as curl_requests

    from .sdataprod import GAMECENTER_HASH, GAMECENTER_OP, SDATAPROD_URL

    cookies = get_cookies(
        bootstrap_url=BOOTSTRAP_URL_FMT.format(contest_id=contest_id),
    )
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
        "Origin": "https://www.ncaa.com",
        "Referer": f"https://www.ncaa.com/game/{contest_id}",
    })
    resp = sess.post(SDATAPROD_URL, json=body, cookies=cookies, timeout=20)
    return {
        "status": resp.status_code,
        "bytes": len(resp.content),
        "text_head": resp.text[:500],
    }


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) < 2 or argv[1] != "probe":
        print(
            "usage: python -m softsaber.ingest.akamai_session probe <contestId>",
            file=sys.stderr,
        )
        return 2
    contest_id = argv[2] if len(argv) > 2 else ""
    if not contest_id:
        print("missing <contestId>", file=sys.stderr)
        return 2

    result = _probe_sdataprod(contest_id)
    print(f"sdataprod status:  {result['status']}")
    print(f"sdataprod bytes:   {result['bytes']}")
    print(f"response head:\n{result['text_head']}")
    return 0 if result["status"] == 200 else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))


__all__ = ["get_cookies", "invalidate"]
