"""Offline, deterministic tests for page-level chunking and exact text attribution."""

from __future__ import annotations

import hashlib

import pytest

from papertrail.chunking.chunker import ChunkingError, chunk_document, validate_chunk_settings
from papertrail.ingestion.pdf_extractor import ExtractedDocument, ExtractedPage


def make_doc(texts: list[str], *, paper_id: str = "2510.22344", sha: str | None = None) -> ExtractedDocument:
    return ExtractedDocument(
        paper_id=paper_id,
        source_url=f"https://arxiv.org/abs/{paper_id}",
        sha256=sha or "a" * 64,
        total_pages=len(texts),
        extracted_pages=len(texts),
        truncated=False,
        empty_text_pages=[index for index, text in enumerate(texts, 1) if not text],
        pages=[ExtractedPage(page_number=index, text=text, character_count=len(text), has_text=bool(text)) for index, text in enumerate(texts, 1)],
    )


def assert_coverage(source_text: str, chunks) -> None:
    covered = set()
    for chunk in chunks:
        assert source_text[chunk.start_char:chunk.end_char] == chunk.text
        assert not chunk.text[0].isspace()
        assert not chunk.text[-1].isspace()
        covered.update(range(chunk.start_char, chunk.end_char))
    assert all(index in covered for index, char in enumerate(source_text) if not char.isspace())


def test_single_short_page_produces_one_chunk():
    doc = make_doc(["A short research paragraph."])
    output = chunk_document(doc)
    assert output.total_chunks == 1
    assert output.chunks[0].page_number == 1
    assert output.chunks[0].start_char == 0
    assert output.chunks[0].end_char == len(doc.pages[0].text)
    assert output.chunks[0].source_sha256 == "a" * 64


def test_multi_page_never_crosses_pages():
    doc = make_doc(["first " * 300, "second " * 250])
    result = chunk_document(doc, chunk_size=300, overlap=35)
    assert result.total_chunks > 2
    assert {chunk.page_number for chunk in result.chunks} == {1, 2}
    assert len({chunk.chunk_id for chunk in result.chunks}) == result.total_chunks
    for page in doc.pages:
        relevant = [chunk for chunk in result.chunks if chunk.page_number == page.page_number]
        assert_coverage(page.text, relevant)


def test_chunk_size_enforced_even_for_long_unbroken_token():
    text = "X" * 1400
    chunks = chunk_document(make_doc([text]), chunk_size=200, overlap=40).chunks
    assert len(chunks) > 3
    assert all(len(chunk.text) <= 200 for chunk in chunks)
    assert_coverage(text, chunks)


def test_text_with_multilingual_characters_and_newlines():
    text = ("هوش مصنوعی و پژوهش\nThe research study.\n\n" * 30).strip()
    result = chunk_document(make_doc([text]), chunk_size=140, overlap=20)
    assert_coverage(text, result.chunks)
    assert all(len(chunk.text) <= 140 for chunk in result.chunks)


def test_paragraph_boundary_preferred_to_hard_cut():
    text = "First paragraph is meaningful. " * 3 + "\n\n" + "Second paragraph is also meaningful. " * 10
    chunks = chunk_document(make_doc([text]), chunk_size=140, overlap=25).chunks
    assert chunks[0].text.endswith("meaningful.")
    assert chunks[0].end_char == text.index("\n\n") - 1


def test_whitespace_page_is_skipped():
    doc = make_doc(["  \n  ", "Important paper text."])
    out = chunk_document(doc)
    assert out.total_chunks == 1
    assert out.empty_text_pages == [1]
    assert out.chunks[0].page_number == 2


def test_all_empty_pages_produce_empty_index_not_error():
    result = chunk_document(make_doc(["", "\n "]))
    assert result.total_chunks == 0
    assert result.chunks == []
    assert result.empty_text_pages == [1, 2]


def test_leading_and_trailing_whitespace_have_exact_offsets():
    text = "  \n Actual evidence.  \n"
    c = chunk_document(make_doc([text])).chunks[0]
    assert c.text == "Actual evidence."
    assert text[c.start_char:c.end_char] == c.text
    assert c.start_char == text.index("Actual")


def test_repeated_runs_have_identical_json_and_ids():
    doc = make_doc(["Repeated reproducible evidence. " * 45])
    first = chunk_document(doc, chunk_size=160, overlap=35)
    second = chunk_document(doc, chunk_size=160, overlap=35)
    assert first.model_dump_json() == second.model_dump_json()


def test_changed_content_changes_chunk_id():
    a = chunk_document(make_doc(["This is a research result."])).chunks[0]
    b = chunk_document(make_doc(["This is a research result!"])).chunks[0]
    assert a.chunk_id != b.chunk_id


def test_old_arxiv_id_supported():
    res = chunk_document(make_doc(["This is evidence."], paper_id="hep-th/9901001"))
    assert res.paper_id == "hep-th/9901001"
    assert res.chunks[0].source_url.endswith("hep-th/9901001")


@pytest.mark.parametrize("size,overlap", [(99, 0), (8001, 0), (120, -1), (120, 60), (120, 120), (False, 0), (120, True)])
def test_invalid_settings_rejected(size, overlap):
    with pytest.raises(ChunkingError):
        validate_chunk_settings(size, overlap)


def test_no_overlap_produces_complete_nonwhitespace_coverage():
    text = "  ".join(["alpha beta gamma delta."] * 100)
    chunks = chunk_document(make_doc([text]), chunk_size=110, overlap=0).chunks
    assert_coverage(text, chunks)
    for first, second in zip(chunks, chunks[1:]):
        assert first.end_char <= second.start_char


def test_overlap_is_real_and_bounded():
    text = "the evidence of this research is important. " * 80
    chunks = chunk_document(make_doc([text]), chunk_size=150, overlap=30).chunks
    assert_coverage(text, chunks)
    for first, second in zip(chunks, chunks[1:]):
        assert second.start_char < first.end_char
        assert first.end_char - second.start_char <= 30


def test_bad_source_url_rejected():
    doc = make_doc(["Text"])
    doc.source_url = "https://example.com/other"
    with pytest.raises(ChunkingError, match="source URL"):
        chunk_document(doc)


def test_invalid_digest_rejected():
    doc = make_doc(["Text"], sha="bad")
    with pytest.raises(ChunkingError, match="SHA-256"):
        chunk_document(doc)


def test_bad_page_numbers_rejected():
    doc = make_doc(["One", "Two"])
    doc.pages[1].page_number = 1
    with pytest.raises(ChunkingError, match="numbered consecutively"):
        chunk_document(doc)


def test_bad_page_metadata_rejected():
    doc = make_doc(["Text"])
    doc.pages[0].character_count = 99
    with pytest.raises(ChunkingError, match="Inconsistent page"):
        chunk_document(doc)


def test_source_truncation_metadata_preserved():
    doc = make_doc(["Present."])
    doc.truncated = True
    doc.total_pages = 4
    result = chunk_document(doc)
    assert result.truncated is True
    assert result.total_pages == 4
    assert result.extracted_pages == 1
