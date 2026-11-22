import hashlib

import pymupdf
import pytest

from papertrail.ingestion.pdf_extractor import PdfExtractionError, extract_pdf


def make_pdf(*pages):
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_text((72, 72), text)
    output = doc.tobytes()
    doc.close()
    return output


def test_extracts_pages_with_correct_one_based_numbers():
    payload = make_pdf("PaperTrail abstract", "Limitations and future work")
    result = extract_pdf(payload, "2510.22344")
    assert result.paper_id == "2510.22344"
    assert result.total_pages == result.extracted_pages == 2
    assert [p.page_number for p in result.pages] == [1, 2]
    assert "PaperTrail" in result.pages[0].text
    assert "future work" in result.pages[1].text
    assert result.sha256 == hashlib.sha256(payload).hexdigest()
    assert result.empty_text_pages == []
    assert result.truncated is False


def test_blank_page_is_reported_not_hallucinated():
    result = extract_pdf(make_pdf("Start", None), "2401.12345")
    assert result.empty_text_pages == [2]
    assert result.pages[1].text == ""
    assert result.pages[1].has_text is False


def test_max_pages_is_explicitly_reported_as_truncation():
    result = extract_pdf(make_pdf("first", "second", "third"), "2401.12345", max_pages=2)
    assert result.total_pages == 3
    assert result.extracted_pages == 2
    assert result.truncated is True
    assert len(result.pages) == 2


@pytest.mark.parametrize("limit", [0, -1, 201])
def test_invalid_page_limit_is_rejected(limit):
    with pytest.raises(ValueError, match="max_pages"):
        extract_pdf(make_pdf("content"), "2401.12345", max_pages=limit)


@pytest.mark.parametrize("content", [b"not a pdf", b"%PDF- corrupt content"])
def test_non_pdf_or_corrupt_pdf_is_rejected(content):
    with pytest.raises(PdfExtractionError):
        extract_pdf(content, "2401.12345")


def test_encrypted_pdf_is_rejected():
    doc = pymupdf.open(stream=make_pdf("secret"), filetype="pdf")
    encrypted = doc.tobytes(encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner123", user_pw="reader123")
    doc.close()
    with pytest.raises(PdfExtractionError, match="Password-protected"):
        extract_pdf(encrypted, "2401.12345")


def test_invalid_paper_id_rejected():
    with pytest.raises(ValueError, match="Invalid arXiv"):
        extract_pdf(make_pdf("text"), "../../etc/passwd")
