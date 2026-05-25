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


def _new_browser(pw, *, use_real_chrome: bool = True):  # type: ignore[no-untyped-def]
    """Launch a browser.  ``use_real_chrome`` drives the user's installed
    Chrome instead of Playwright's bundled Chromium — Akamai often static-
    blocks bundled Chromium based on binary/fingerprint tells but lets
    real Chrome through.  Falls back to Chromium if Chrome isn't installed.
    """
    launch_kwargs: dict[str, Any] = {"headless": True}
    if use_real_chrome:
        launch_kwargs["channel"] = "chrome"
    try:
        browser = pw.chromium.launch(**launch_kwargs)
    except Exception as e:  # noqa: BLE001
        if use_real_chrome:
            log.warning(
                "real Chrome not available (%s); falling back to bundled Chromium",
                e,
            )
            browser = pw.chromium.launch(headless=True)
        else:
            raise
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
            # domcontentloaded fires as soon as the HTML parses — much
            # faster than networkidle, which never settles while the
            # Akamai sensor is pinging.  We do the real wait below by
            # polling for either the cleared _abck cookie or the URL
            # changing away from the challenge page.
            try:
                resp = page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                status = resp.status if resp else 0
            except Exception as e:  # noqa: BLE001
                log.warning("page.goto raised (%s); reading whatever we have", e)
                status = 0

            # Wait for the challenge to clear.  The challenge page is
            # small (~300B-3KB) and contains a meta-refresh; the real
            # page is much larger.  Poll until either the body grows
            # past a sane threshold or we time out.
            deadline = time.time() + 20
            while time.time() < deadline:
                body = page.content()
                if len(body) > 10_000 and "meta http-equiv=\"refresh\"" not in body.lower():
                    break
                page.wait_for_timeout(500)

            html = page.content()
            log.info("akamai: final URL = %s, body = %d bytes", page.url, len(html))
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


class BrowserSession:
    """Long-lived headless Chrome context for fetching multiple Akamai-
    protected pages without re-paying the challenge each time.

    Use as a context manager::

        with BrowserSession() as sess:
            html1 = sess.get('https://stats.ncaa.org/teams/613592/roster')
            html2 = sess.get('https://stats.ncaa.org/teams/613593/roster')

    The first ``get`` pays the JS-challenge cost (~3–5s).  Subsequent
    fetches in the same session reuse the warm Akamai state and finish
    in roughly the time of the navigation itself.
    """

    def __init__(self, *, use_real_chrome: bool = True) -> None:
        self._use_real_chrome = use_real_chrome
        self._pw = None
        self._browser = None
        self._context = None
        self._page = None

    def __enter__(self) -> "BrowserSession":
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as e:
            raise RuntimeError(
                "playwright is not installed. Run:\n"
                "    pip install -e .[akamai]\n"
                "    playwright install chromium"
            ) from e
        self._pw = sync_playwright().start()
        self._browser, self._context = _new_browser(
            self._pw, use_real_chrome=self._use_real_chrome
        )
        self._page = self._context.new_page()
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if self._browser is not None:
                self._browser.close()
        finally:
            if self._pw is not None:
                self._pw.stop()
            self._browser = self._context = self._page = self._pw = None

    def get(self, url: str, *, settle_min_bytes: int = 10_000) -> tuple[int, str]:
        """Navigate ``page`` to ``url`` and return (status, html) once the
        Akamai challenge has cleared.  Returns whatever HTML is loaded
        after a 20-second clearance timeout if the challenge stalls.
        """
        if self._page is None:
            raise RuntimeError("BrowserSession used outside its context manager")
        try:
            resp = self._page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            status = resp.status if resp else 0
        except Exception as e:  # noqa: BLE001
            log.warning("page.goto(%s) raised (%s); reading current state", url, e)
            status = 0
        deadline = time.time() + 20
        while time.time() < deadline:
            body = self._page.content()
            if (
                len(body) > settle_min_bytes
                and 'meta http-equiv="refresh"' not in body.lower()
            ):
                break
            self._page.wait_for_timeout(500)
        html = self._page.content()
        log.info("akamai: %s → %d bytes (final url %s)", url, len(html), self._page.url)
        return status, html


def fetch_or_browser(
    url: str,
    *,
    namespace: str,
    browser_session: "BrowserSession | None" = None,
    force: bool = False,
    challenge_threshold_bytes: int = 8_000,
) -> str | None:
    """Try ``curl_cffi`` first; fall back to ``browser_session`` on a
    challenge/error.  Returns the cleared HTML (and writes it to the
    standard http_cache path so subsequent runs hit the disk cache).

    Returns ``None`` when both paths fail.  This is the canonical helper
    for any stats.ncaa.org endpoint — discovery, rosters, team_codes.
    """
    from ..http_cache import FetchError, _cache_path, fetch  # type: ignore[attr-defined]
    from ..config import ensure_dirs

    def _is_challenge(html: str | None) -> bool:
        if not html:
            return True
        if len(html) < challenge_threshold_bytes:
            return True
        return 'meta http-equiv="refresh"' in html.lower()

    try:
        html = fetch(url, namespace=namespace, force=force)
    except FetchError as e:
        log.debug("fetch_or_browser %s: curl_cffi failed (%s)", url, e)
        html = None
    except Exception as e:  # noqa: BLE001
        log.warning("fetch_or_browser %s: unexpected curl_cffi error: %s", url, e)
        html = None

    if not _is_challenge(html):
        return html

    if browser_session is None:
        log.warning(
            "fetch_or_browser %s: challenge response and no browser_session available",
            url,
        )
        return html  # may be the challenge page; caller can decide

    try:
        _, html = browser_session.get(url)
    except Exception as e:  # noqa: BLE001
        log.warning("fetch_or_browser %s: browser fetch failed: %s", url, e)
        return None

    if html and not _is_challenge(html):
        ensure_dirs()
        path = _cache_path(url, namespace, ext="html")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"<!-- src: {url} -->\n{html}", encoding="utf-8")
        return html

    return html


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


def main(argv: list[str]) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(argv) < 3 or argv[1] != "probe-roster":
        print(
            "usage: python -m softsaber.ingest.akamai_session probe-roster "
            "<statsNcaaTeamId>",
            file=sys.stderr,
        )
        return 2
    result = _probe_roster(argv[2])
    print(f"browser status:     {result['browser_status']}")
    print(f"browser bytes:      {result['browser_bytes']}")
    print(f"browser looks real: {result['browser_looks_real']}")
    print("browser snippet (first 600 chars):")
    print(result["browser_snippet"])
    print()
    print(f"curl_cffi status:   {result['curl_status']}")
    print(f"curl_cffi bytes:    {result['curl_bytes']}")
    print(f"curl_cffi snippet:  {result['curl_snippet'][:200]}")
    return 0 if result["browser_looks_real"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))


__all__ = ["fetch_page_html", "get_cookies", "invalidate"]
