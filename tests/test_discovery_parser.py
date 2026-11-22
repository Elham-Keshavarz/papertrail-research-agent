from pathlib import Path

import pytest

from papertrail.discovery import ArxivParseError, parse_arxiv_feed

SAMPLE = (Path(__file__).parent / "fixtures" / "arxiv_sample.xml").read_bytes()


def parse(payload=SAMPLE, **kwargs):
    return parse_arxiv_feed(payload, query="retrieval", start=0, requested=5, **kwargs)


def test_parses_two_papers_and_total():
    result = parse()
    assert result.total_results == 134
    assert result.start == 0
    assert len(result.papers) == 2
    paper = result.papers[0]
    assert paper.paper_id == "2401.12345"
    assert paper.title == "Retrieval Augmented Generation: Evaluation"
    assert paper.authors == ["Ada Researcher", "Bob Scientist"]
    assert paper.abstract == "A study of retrieval and evaluation methods."
    assert str(paper.url) == "https://arxiv.org/abs/2401.12345"
    assert str(paper.pdf_url) == "https://arxiv.org/pdf/2401.12345"
    assert paper.categories == ["cs.IR", "cs.AI"]
    assert paper.published_at.isoformat() == "2024-01-20T12:00:00+00:00"
    assert paper.updated_at.isoformat() == "2024-01-22T19:30:00+00:00"
    assert paper.journal_reference is not None
    assert paper.is_peer_reviewed is None


def test_legacy_identifiers_are_supported():
    assert parse().papers[1].paper_id == "hep-th/9901001"


def test_empty_feed():
    result = parse(b'<feed xmlns="http://www.w3.org/2005/Atom"/>')
    assert result.papers == []
    assert result.total_results is None


@pytest.mark.parametrize("payload", [
    b"not XML",
    b"<bad/>",
    b'<!DOCTYPE feed [<!ENTITY x "attack">]><feed xmlns="http://www.w3.org/2005/Atom">&x;</feed>',
    b'<feed xmlns="http://www.w3.org/2005/Atom"><title>Error</title><entry/></feed>',
])
def test_malformed_unsafe_and_error_feeds_rejected(payload):
    with pytest.raises(ArxivParseError):
        parse(payload)


def test_rejects_missing_author():
    with pytest.raises(ArxivParseError):
        parse(SAMPLE.replace(b"<author><name> Ada Researcher </name></author>", b"").replace(
            b"<author><name>  Bob Scientist </name></author>", b""))


def test_rejects_forged_source():
    with pytest.raises(ArxivParseError):
        parse(SAMPLE.replace(b"http://arxiv.org/abs/2401.12345v2", b"http://malicious.example/abs/2401.12345v2", 1))


def test_rejects_too_many_entries():
    with pytest.raises(ArxivParseError):
        parse_arxiv_feed(SAMPLE, query="retrieval", start=0, requested=1)


def test_invalid_pub_date_rejected():
    with pytest.raises(ArxivParseError):
        parse(SAMPLE.replace(b"2024-01-20T12:00:00Z", b"2024-01-20"))
