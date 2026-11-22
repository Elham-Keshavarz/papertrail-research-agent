import httpx
import pytest

from papertrail.ingestion.pdf_downloader import ArxivPdfDownloader, PdfDownloadError, validate_paper_id


class FakeClock:
    def __init__(self):
        self.now = 100.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def make_client(handler, *, retries=0, max_bytes=1024 * 1024):
    clock = FakeClock()
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return ArxivPdfDownloader(http_client=http, max_retries=retries, max_bytes=max_bytes,
                              clock=clock, sleep=clock.sleep), clock


def test_downloads_valid_pdf_and_encodes_legacy_id():
    requested = []

    def handler(request):
        requested.append(str(request.url))
        return httpx.Response(200, content=b"%PDF-1.7\nhello")

    client, clock = make_client(handler)
    assert client.fetch("hep-th/9901001").startswith(b"%PDF-")
    assert requested == ["https://arxiv.org/pdf/hep-th/9901001"]
    assert client.fetch("2410.00123")
    assert clock.waits == [3.0]


def test_redirect_to_external_domain_is_blocked():
    client, _ = make_client(lambda request: httpx.Response(302, headers={"location": "https://example.org/pdf/evil"}))
    with pytest.raises(PdfDownloadError, match="outside"):
        client.fetch("2410.00123")


def test_redirect_with_credentials_or_wrong_path_is_blocked():
    client, _ = make_client(lambda request: httpx.Response(302, headers={"location": "https://user@arxiv.org/pdf/2410.00123"}))
    with pytest.raises(PdfDownloadError, match="outside"):
        client.fetch("2410.00123")


def test_same_host_redirect_supported():
    calls = []

    def handler(request):
        calls.append(str(request.url))
        if len(calls) == 1:
            return httpx.Response(301, headers={"location": "/pdf/2410.00123?download=1"})
        return httpx.Response(200, content=b"%PDF-1.7\nhello")

    client, clock = make_client(handler)
    assert client.fetch("2410.00123") == b"%PDF-1.7\nhello"
    assert len(calls) == 2
    assert clock.waits == [3.0]


@pytest.mark.parametrize("paper_id", ["", "abc", "../2410.00123", "2410.00123/../../etc", "https://evil/pdf/x", "2410.12"])
def test_bad_ids_are_rejected_without_network(paper_id):
    calls = []
    client, _ = make_client(lambda request: calls.append(1) or httpx.Response(200, content=b"%PDF-1.7"))
    with pytest.raises(ValueError):
        client.fetch(paper_id)
    assert not calls


def test_429_gets_retried_and_paced():
    counter = 0

    def handler(request):
        nonlocal counter
        counter += 1
        if counter == 1:
            return httpx.Response(429, headers={"retry-after": "5"})
        return httpx.Response(200, content=b"%PDF-1.7\ncontent")

    client, clock = make_client(handler, retries=1)
    assert client.fetch("2401.12345") == b"%PDF-1.7\ncontent"
    assert counter == 2
    assert clock.waits == [5.0]


def test_retry_limit_enforced():
    client, clock = make_client(lambda request: httpx.Response(503), retries=1)
    with pytest.raises(PdfDownloadError, match="503"):
        client.fetch("2401.12345")
    assert clock.waits == [3.0]


def test_bad_content_rejected():
    client, _ = make_client(lambda request: httpx.Response(200, content=b"<!DOCTYPE html>"))
    with pytest.raises(PdfDownloadError, match="not a PDF"):
        client.fetch("2401.12345")


def test_oversized_response_header_rejected():
    client, _ = make_client(lambda request: httpx.Response(200, headers={"content-length": "9000"}, content=b"%PDF-test"), max_bytes=1024)
    with pytest.raises(PdfDownloadError, match="maximum"):
        client.fetch("2401.12345")


def test_oversized_stream_rejected_even_without_length_header():
    client, _ = make_client(lambda request: httpx.Response(200, content=b"%PDF-" + b"x" * 3000), max_bytes=1024)
    with pytest.raises(PdfDownloadError, match="maximum"):
        client.fetch("2401.12345")


def test_transport_error_is_controlled():
    def handler(request):
        raise httpx.ConnectError("simulated offline", request=request)

    client, _ = make_client(handler)
    with pytest.raises(PdfDownloadError, match="could not be reached"):
        client.fetch("2401.12345")


def test_nonretryable_404():
    client, _ = make_client(lambda request: httpx.Response(404), retries=2)
    with pytest.raises(PdfDownloadError, match="404"):
        client.fetch("2401.12345")


def test_valid_ids():
    assert validate_paper_id(" 2410.00001v2 ") == "2410.00001v2"
    assert validate_paper_id("math.NT/0501001") == "math.NT/0501001"
