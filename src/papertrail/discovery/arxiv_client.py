"""Bounded, polite HTTP adapter for the legacy arXiv search API."""

import re
import time
from collections.abc import Callable
from typing import Literal

import httpx

from papertrail.discovery.arxiv_parser import ArxivParseError, parse_arxiv_feed
from papertrail.models import PaperSearchResult

ARXIV_API = "https://export.arxiv.org/api/query"
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
SORT_BY = ("relevance", "submittedDate", "lastUpdatedDate")
SORT_ORDER = ("ascending", "descending")
MIN_INTERVAL_SECONDS = 3.0
MAX_RESPONSE_BYTES = 2_000_000  # Metadata only; never a PDF.


class ArxivRequestError(RuntimeError):
    """The upstream API could not complete a metadata request."""


def _search_expression(query: str) -> str:
    clean = " ".join(query.split())
    if not clean or len(clean) > 200:
        raise ValueError("Search query must contain 1–200 non-whitespace characters")
    # Raw arXiv field expressions are an explicit advanced mode.
    if re.match(r"^(all|ti|abs|au|cat|id):", clean, flags=re.IGNORECASE) or clean.startswith("("):
        return clean
    if '"' in clean or "\\" in clean:
        raise ValueError('Plain search query cannot include quotes or backslashes; use arXiv field syntax instead')
    return f'all:"{clean}"'


class ArxivClient:
    """One HTTP connection at a time. Rate limits apply to this instance/process.

    Callers running multiple processes must coordinate their request rate externally.
    """

    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        http_client: httpx.Client | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not (1 <= timeout_seconds <= 120):
            raise ValueError("Timeout must be between 1 and 120 seconds")
        if not (0 <= max_retries <= 5):
            raise ValueError("max_retries must be between 0 and 5")
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._clock = clock
        self._sleep = sleep
        self._next_allowed_at = 0.0
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            follow_redirects=False,
            headers={"User-Agent": "PaperTrail/0.2 (academic metadata research; read-only)"},
        )

    def __enter__(self) -> "ArxivClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def _wait_until_allowed(self) -> None:
        remaining = self._next_allowed_at - self._clock()
        if remaining > 0:
            self._sleep(remaining)
        # Leave a 3-second gap after every attempt, successful or not.
        self._next_allowed_at = self._clock() + MIN_INTERVAL_SECONDS

    def _backoff(self, attempt: int, retry_after: str | None = None) -> None:
        delay = min(30.0, 3.0 * (2 ** attempt))
        if retry_after:
            try:
                delay = max(delay, min(120.0, float(retry_after)))
            except ValueError:
                pass
        self._next_allowed_at = max(self._next_allowed_at, self._clock() + delay)

    def search(
        self,
        query: str,
        *,
        max_results: int = 5,
        start: int = 0,
        sort_by: Literal["relevance", "submittedDate", "lastUpdatedDate"] = "relevance",
        sort_order: Literal["ascending", "descending"] = "descending",
    ) -> PaperSearchResult:
        if not 1 <= max_results <= 20:
            raise ValueError("max_results must be between 1 and 20 for PaperTrail")
        if not 0 <= start <= 29999 or start + max_results > 30000:
            raise ValueError("start + max_results must be within the first 30,000 results")
        if sort_by not in SORT_BY or sort_order not in SORT_ORDER:
            raise ValueError("Unknown arXiv sorting option")
        expression = _search_expression(query)
        params = {
            "search_query": expression,
            "start": start,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": sort_order,
        }
        for attempt in range(self._max_retries + 1):
            self._wait_until_allowed()
            try:
                response = self._http.get(ARXIV_API, params=params, timeout=self._timeout)
            except httpx.RequestError as exc:
                if attempt == self._max_retries:
                    raise ArxivRequestError("arXiv API could not be reached after bounded retries") from exc
                self._backoff(attempt)
                continue
            if response.status_code in RETRY_STATUSES:
                if attempt == self._max_retries:
                    raise ArxivRequestError(f"arXiv API returned HTTP {response.status_code} after retries")
                self._backoff(attempt, response.headers.get("Retry-After"))
                continue
            if response.status_code != 200:
                raise ArxivRequestError(f"arXiv API returned HTTP {response.status_code}")
            if len(response.content) > MAX_RESPONSE_BYTES:
                raise ArxivParseError("arXiv metadata response exceeded the permitted size")
            return parse_arxiv_feed(response.content, query=query.strip(), start=start, requested=max_results)
        raise ArxivRequestError("arXiv API retry budget exceeded")  # Defensive safeguard.
