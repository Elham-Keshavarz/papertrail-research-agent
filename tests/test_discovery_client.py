import httpx
import pytest

from papertrail.discovery import ArxivClient, ArxivParseError, ArxivRequestError
from papertrail.discovery.arxiv_client import _search_expression
from test_discovery_parser import SAMPLE


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def client(handler, clock=None, **kwargs):
    timer = clock or FakeClock()
    return ArxivClient(
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        clock=timer, sleep=timer.sleep, **kwargs,
    )


def test_builds_correct_search_url_and_gets_results():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=SAMPLE)

    c = client(handler)
    result = c.search("retrieval augmented generation", max_results=5)
    assert len(result.papers) == 2
    assert str(calls[0].url).startswith("https://export.arxiv.org/api/query?")
    assert calls[0].url.params["search_query"] == 'all:"retrieval augmented generation"'
    assert calls[0].url.params["max_results"] == "5"
    assert calls[0].url.params["sortBy"] == "relevance"


def test_advanced_query_is_preserved():
    assert _search_expression("cat:cs.AI AND ti:retrieval") == "cat:cs.AI AND ti:retrieval"


@pytest.mark.parametrize("query", ["", "  ", '"bad"', "a" * 201])
def test_rejects_invalid_queries_before_network(query):
    c = client(lambda request: pytest.fail("Must not send a request"))
    with pytest.raises(ValueError):
        c.search(query)


@pytest.mark.parametrize("params", [
    {"max_results": 0}, {"max_results": 21}, {"start": -1},
    {"start": 30000}, {"start": 29999, "max_results": 5},
    {"sort_by": "unknown"}, {"sort_order": "wrong"},
])
def test_search_bounds_are_checked_before_network(params):
    c = client(lambda request: pytest.fail("Must not send a request"))
    with pytest.raises(ValueError):
        c.search("transformer", **params)


def test_rate_limiting_between_successful_calls():
    timer = FakeClock()
    c = client(lambda req: httpx.Response(200, content=SAMPLE), clock=timer)
    c.search("retrieval")
    c.search("retrieval")
    assert timer.sleeps == [3.0]


def test_retry_503_then_success_with_backoff():
    timer = FakeClock()
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(503) if count == 1 else httpx.Response(200, content=SAMPLE)

    c = client(handler, clock=timer)
    assert len(c.search("retrieval").papers) == 2
    assert count == 2
    assert timer.sleeps == [3.0]


def test_retry_after_header_is_honored():
    timer = FakeClock()
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(429, headers={"Retry-After": "8"}) if count == 1 else httpx.Response(200, content=SAMPLE)

    c = client(handler, clock=timer)
    c.search("retrieval")
    assert timer.sleeps == [8.0]


def test_retries_bounded_for_failure():
    timer = FakeClock()
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(429)

    c = client(handler, max_retries=2, clock=timer)
    with pytest.raises(ArxivRequestError, match="429"):
        c.search("retrieval")
    assert count == 3
    assert timer.sleeps == [3.0, 6.0]


def test_non_retryable_http_error_fails_immediately():
    count = 0

    def handler(request):
        nonlocal count
        count += 1
        return httpx.Response(404)

    c = client(handler)
    with pytest.raises(ArxivRequestError, match="404"):
        c.search("retrieval")
    assert count == 1


def test_transport_failure_retried_then_reported():
    def handler(request):
        raise httpx.ConnectError("blocked", request=request)

    c = client(handler, max_retries=1)
    with pytest.raises(ArxivRequestError, match="could not be reached"):
        c.search("retrieval")


def test_malformed_response_rejected_without_retry():
    c = client(lambda request: httpx.Response(200, text="Bad XML"))
    with pytest.raises(ArxivParseError):
        c.search("retrieval")


def test_oversized_metadata_rejected():
    c = client(lambda request: httpx.Response(200, content=b"A" * 2_000_001))
    with pytest.raises(ArxivParseError, match="size"):
        c.search("retrieval")
