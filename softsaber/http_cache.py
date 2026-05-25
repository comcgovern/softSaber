"""HTTP layer with disk cache, token-bucket rate limiting, and retry/backoff.

Every GET against the NCAA APIs is expensive and the data is immutable once a
game is final, so we cache responses on disk keyed by URL. Re-running ingest
against the same season is then a local-only operation.

Concurrency notes
-----------------
``fetch`` is safe to call from multiple threads simultaneously.  Two things
make this work:

* **Thread-local sessions** — ``curl_cffi`` sessions wrap a libcurl handle
  which is not thread-safe.  Each OS thread gets its own session via
  ``threading.local``.

* **Token-bucket rate limiter** — ``_rate_limiter.acquire()`` serialises the
  *start* of each live HTTP request so we never exceed ``REQUEST_RATE_PER_SEC``
  requests per second across all threads.  Cache hits bypass the limiter
  entirely since they involve no network I/O.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path

import requests
from curl_cffi import requests as curl_requests
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import (
    RAW_DIR,
    REQUEST_RATE_PER_SEC,
    REQUEST_RETRY_BASE_DELAY_S,
    REQUEST_RETRY_MAX,
    REQUEST_TIMEOUT_S,
    USER_AGENT,
    ensure_dirs,
)

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """HTTP failure.  Retryable by default (timeout, 429, 5xx)."""


class PermanentFetchError(FetchError):
    """Non-retryable HTTP failure (4xx other than 429).

    Subclasses FetchError so existing ``except FetchError`` handlers
    still catch it; tenacity skips it via an explicit predicate.
    """


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Token-bucket rate limiter for concurrent callers.

    ``acquire()`` blocks the calling thread until a request slot is available,
    then immediately marks that slot as consumed.  With N threads each calling
    ``acquire()`` before making a network request, the aggregate throughput
    stays at ``rate`` requests per second regardless of N.
    """

    def __init__(self, rate: float) -> None:
        self._interval = 1.0 / rate
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                time.sleep(wait)
            self._next_allowed = time.monotonic() + self._interval


_rate_limiter = _RateLimiter(REQUEST_RATE_PER_SEC)


# ---------------------------------------------------------------------------
# Thread-local session
# ---------------------------------------------------------------------------

_local = threading.local()

_SESSION_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "Referer": "https://www.ncaa.com/",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "same-origin",
    "Sec-Fetch-User": "?1",
}


def session() -> curl_requests.Session:
    # stats.ncaa.org is fronted by a TLS-fingerprint-checking WAF (Akamai) that
    # 403s plain Python clients. curl_cffi impersonates Chrome's JA3 so we look
    # like a real browser at the TLS layer, not just in headers.
    if not hasattr(_local, "session"):
        s = curl_requests.Session(impersonate="chrome124")
        s.headers.update(_SESSION_HEADERS)
        _local.session = s
    return _local.session


# ---------------------------------------------------------------------------
# HTTP GET with retry
# ---------------------------------------------------------------------------

def _cache_path(url: str, namespace: str, ext: str = "html") -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:24]
    return RAW_DIR / namespace / f"{digest}.{ext}"


_RETRYABLE_STATUSES = {429, 503}  # 502/504 are usually upstream-down: retrying in seconds rarely helps.


def _classify_status(status: int, url: str) -> None:
    if status == 200:
        return
    if status in _RETRYABLE_STATUSES:
        raise FetchError(f"retryable status {status} for {url}")
    raise PermanentFetchError(f"status {status} for {url}")


_RETRYABLE_TYPES = (
    requests.ConnectionError,
    requests.Timeout,
    curl_requests.exceptions.RequestException,
)


def _is_retryable(exc: BaseException) -> bool:
    """tenacity predicate: retry on network errors and retryable FetchError,
    but NOT on PermanentFetchError (4xx other than 429)."""
    if isinstance(exc, PermanentFetchError):
        return False
    if isinstance(exc, FetchError):
        return True
    return isinstance(exc, _RETRYABLE_TYPES)


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
    retry=retry_if_exception(_is_retryable),
    before_sleep=_log_retry,
)
def _do_get(sess: curl_requests.Session, url: str) -> str:
    resp = sess.get(url, timeout=REQUEST_TIMEOUT_S)
    log.debug("GET %s → %d (%d bytes)", url, resp.status_code, len(resp.content))
    _classify_status(resp.status_code, url)
    return resp.text


@retry(
    reraise=True,
    stop=stop_after_attempt(REQUEST_RETRY_MAX),
    wait=wait_exponential(multiplier=REQUEST_RETRY_BASE_DELAY_S, min=2, max=30),
    retry=retry_if_exception(_is_retryable),
    before_sleep=_log_retry,
)
def _do_post_json(sess: curl_requests.Session, url: str, body: dict) -> str:
    resp = sess.post(
        url,
        json=body,
        timeout=REQUEST_TIMEOUT_S,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    log.debug("POST %s → %d (%d bytes)", url, resp.status_code, len(resp.content))
    _classify_status(resp.status_code, url)
    return resp.text


def fetch(url: str, *, namespace: str, force: bool = False, ext: str = "html") -> str:
    """GET ``url`` with disk cache. ``namespace`` groups cached files by source.

    Cache hits are returned immediately without touching the rate limiter.
    Live requests go through the token-bucket limiter so the aggregate
    request rate across all threads stays within ``REQUEST_RATE_PER_SEC``.

    ``ext`` controls the on-disk extension; use ``"json"`` for JSON responses
    so files round-trip through ``json.loads`` without a comment prefix.
    """
    ensure_dirs()
    path = _cache_path(url, namespace, ext=ext)
    if path.exists() and not force:
        log.debug("cache hit %s", url)
        return path.read_text(encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    _rate_limiter.acquire()
    log.info("fetch %s", url)
    text = _do_get(session(), url)
    if ext == "html":
        path.write_text(f"<!-- src: {url} -->\n{text}", encoding="utf-8")
    else:
        path.write_text(text, encoding="utf-8")
    return text


def post_json(
    url: str,
    body: dict,
    *,
    namespace: str,
    force: bool = False,
) -> str:
    """POST ``body`` as JSON to ``url`` with disk cache keyed on URL+body.

    Used for GraphQL persisted-query endpoints where the request body
    (operationName + variables + extensions) is the cache key.
    """
    import json as _json

    ensure_dirs()
    body_blob = _json.dumps(body, sort_keys=True, separators=(",", ":"))
    key = f"{url}\n{body_blob}"
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    path = RAW_DIR / namespace / f"{digest}.json"
    if path.exists() and not force:
        log.debug("cache hit POST %s", url)
        return path.read_text(encoding="utf-8")

    path.parent.mkdir(parents=True, exist_ok=True)
    _rate_limiter.acquire()
    log.info("POST %s body=%s", url, body_blob[:200])
    text = _do_post_json(session(), url, body)
    path.write_text(text, encoding="utf-8")
    return text
