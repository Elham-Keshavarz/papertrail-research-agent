"""Bounded, arXiv-only PDF fetcher. Never accepts an arbitrary download URL."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from urllib.parse import quote, urljoin, urlsplit

import httpx

ARXIV_PDF_BASE = "https://arxiv.org/pdf/"
ARXIV_HOSTS = frozenset({"arxiv.org", "www.arxiv.org", "export.arxiv.org"})
RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})
MIN_INTERVAL_SECONDS = 3.0
MAX_REDIRECTS = 3
MAX_PDF_BYTES = 20 * 1024 * 1024
ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[A-Za-z][A-Za-z0-9.-]*/\d{7})(?:v[1-9]\d*)?\Z")


class PdfDownloadError(RuntimeError):
    """The PDF cannot be safely or reliably downloaded."""


def validate_paper_id(value: str) -> str:
    paper_id = value.strip()
    if not ARXIV_ID.fullmatch(paper_id):
        raise ValueError("Invalid arXiv paper ID. Example: 2510.22344 or hep-th/9901001")
    return paper_id


def _approved_pdf_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return (
            parsed.scheme == "https"
            and parsed.hostname in ARXIV_HOSTS
            and parsed.port in (None, 443)
            and parsed.username is None
            and parsed.password is None
            and parsed.path.startswith("/pdf/")
            and not parsed.fragment
        )
    except ValueError:
        return False


class ArxivPdfDownloader:
    """Download one paper at a time with retry, pacing, redirect checks, and size limits.

    Pacing is per instance; multiple processes must coordinate externally.
    """

    def __init__(
        self,
        *,
        http_client: httpx.Client | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
        max_bytes: int = MAX_PDF_BYTES,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not 1 <= timeout_seconds <= 120:
            raise ValueError("Timeout must be between 1 and 120 seconds")
        if not 0 <= max_retries <= 5:
            raise ValueError("max_retries must be between 0 and 5")
        if not 1024 <= max_bytes <= 100 * 1024 * 1024:
            raise ValueError("max_bytes must be between 1 KiB and 100 MiB")
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            headers={"User-Agent": "PaperTrail/0.3 (educational PDF research; single-file download)"},
            follow_redirects=False,
        )
        self._clock = clock
        self._sleep = sleep
        self._next_allowed_at = 0.0
        self._timeout = timeout_seconds
        self._max_retries = max_retries
        self._max_bytes = max_bytes

    def __enter__(self) -> ArxivPdfDownloader:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def _wait(self) -> None:
        remainder = self._next_allowed_at - self._clock()
        if remainder > 0:
            self._sleep(remainder)
        self._next_allowed_at = self._clock() + MIN_INTERVAL_SECONDS

    def _backoff(self, attempt: int, retry_after: str | None) -> None:
        delay = min(30.0, 3.0 * 2**attempt)
        if retry_after:
            try:
                delay = max(delay, min(120.0, max(0.0, float(retry_after))))
            except ValueError:
                pass
        self._next_allowed_at = max(self._next_allowed_at, self._clock() + delay)

    def fetch(self, paper_id: str) -> bytes:
        paper_id = validate_paper_id(paper_id)
        start_url = ARXIV_PDF_BASE + quote(paper_id, safe="/")
        for attempt in range(self._max_retries + 1):
            url = start_url
            retry_status: int | None = None
            retry_after: str | None = None
            try:
                for redirect in range(MAX_REDIRECTS + 1):
                    if not _approved_pdf_url(url):
                        raise PdfDownloadError("PDF redirect points outside the approved arXiv PDF hosts")
                    self._wait()
                    with self._http.stream("GET", url, timeout=self._timeout, follow_redirects=False) as response:
                        code = response.status_code
                        if code in (301, 302, 303, 307, 308):
                            if redirect == MAX_REDIRECTS:
                                raise PdfDownloadError("Too many arXiv PDF redirects")
                            location = response.headers.get("location")
                            if not location:
                                raise PdfDownloadError("arXiv PDF redirect has no Location header")
                            url = urljoin(str(response.request.url), location)
                            continue
                        if code in RETRY_STATUSES:
                            retry_status = code
                            retry_after = response.headers.get("retry-after")
                            break
                        if code != 200:
                            raise PdfDownloadError(f"arXiv PDF server returned HTTP {code}")
                        length = response.headers.get("content-length")
                        if length and length.isdecimal() and int(length) > self._max_bytes:
                            raise PdfDownloadError("PDF exceeds the maximum allowed download size")
                        payload = bytearray()
                        for chunk in response.iter_bytes(chunk_size=65536):
                            payload.extend(chunk)
                            if len(payload) > self._max_bytes:
                                raise PdfDownloadError("PDF exceeds the maximum allowed download size")
                        if not bytes(payload[:5]) == b"%PDF-":
                            raise PdfDownloadError("Downloaded content is not a PDF (invalid file signature)")
                        return bytes(payload)
            except httpx.RequestError as exc:
                if attempt == self._max_retries:
                    raise PdfDownloadError("arXiv PDF service could not be reached after retries") from exc
            else:
                if retry_status is None:
                    raise PdfDownloadError("Unable to follow arXiv PDF redirect")
                if attempt == self._max_retries:
                    raise PdfDownloadError(f"arXiv PDF server returned HTTP {retry_status} after retries")
            self._backoff(attempt, retry_after)
        raise PdfDownloadError("PDF download retry budget exceeded")
