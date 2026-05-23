"""HTTP layer with disk cache and retry/backoff.

Every GET against stats.ncaa.org is expensive and the data is immutable once a
game is final, so we cache responses on disk keyed by URL. Re-running ingest
against the same season is then a local-only operation.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    INTER_REQUEST_DELAY_S,
    RAW_DIR,
    REQUEST_RETRY_BASE_DELAY_S,
    REQUEST_RETRY_MAX,
    REQUEST_TIMEOUT_S,
    USER_AGENT,
    ensure_dirs,
)

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    pass


def _cache_path(url: str, namespace: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return RAW_DIR / namespace / f"{digest}.html"


def _log_retry(retry_state) -> None:  # type: ignore[type-arg]
    log.debug(
        "retry %d/%d for %s after: %s",
        retry_state.attempt_number,
        REQUEST_RETRY_MAX,
        retry_state.args[1] if len(retry_state.args) > 1 else "?",
        retry_state.outcome.exception(),
    )


@retry(
    reraise=True,
    stop=stop_after_attempt(REQUEST_RETRY_MAX),
    wait=wait_exponential(multiplier=REQUEST_RETRY_BASE_DELAY_S, min=2, max=30),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, FetchError)),
    before_sleep=_log_retry,
)
def _do_get(session: requests.Session, url: str) -> str:
    resp = session.get(url, timeout=REQUEST_TIMEOUT_S)
    log.debug("GET %s → %d (%d bytes)", url, resp.status_code, len(resp.content))
    if resp.status_code == 429 or 500 <= resp.status_code < 600:
        raise FetchError(f"retryable status {resp.status_code} for {url}")
    if resp.status_code != 200:
        raise FetchError(f"status {resp.status_code} for {url}")
    return resp.text


_session: requests.Session | None = None


def session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )
        _session = s
    return _session


def fetch(url: str, *, namespace: str, force: bool = False) -> str:
    """GET ``url`` with disk cache. ``namespace`` groups cached files by source."""
    ensure_dirs()
    path = _cache_path(url, namespace)
    if path.exists() and not force:
        log.debug("cache hit %s", url)
        return path.read_text(encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    log.info("fetch %s", url)
    text = _do_get(session(), url)
    # Tag the URL into the file so cache contents are self-describing.
    path.write_text(f"<!-- src: {url} -->\n{text}", encoding="utf-8")
    time.sleep(INTER_REQUEST_DELAY_S)
    return text
