"""Regression checks for column-interleaving observed on real research PDFs."""

import pymupdf
import pytest

from papertrail.ingestion.layout import extract_page_text
from papertrail.ingestion.pdf_extractor import extract_pdf


def _academic_pdf(*, two_columns=True):
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_text((60, 48), "FULL WIDTH PAPER TITLE ACROSS THE PAGE", fontsize=18)
    if two_columns:
        xpositions = (55, 325)
        texts = (
            [f"LEFT COLUMN evidence definition paragraph number {i}." for i in range(10)],
            [f"RIGHT COLUMN evaluation benchmarks paragraph number {i}." for i in range(10)],
        )
    else:
        xpositions = (55,)
        texts = ([f"SINGLE COLUMN scientific information paragraph number {i}." for i in range(20)],)
    for x, lines in zip(xpositions, texts):
        for i, line in enumerate(lines):
            page.insert_text((x, 115 + 17 * i), line, fontsize=9)
    output = doc.tobytes()
    doc.close()
    return output


def test_two_column_stays_in_column_order():
    pdf = _academic_pdf()
    result = extract_pdf(pdf, "2510.22344")
    page = result.pages[0]
    assert page.extraction_method == "two_column"
    assert page.text.startswith("FULL WIDTH PAPER TITLE")
    assert page.text.index("LEFT COLUMN evidence definition paragraph number 9") < page.text.index("RIGHT COLUMN evaluation benchmarks paragraph number 0")
    assert page.text.count("LEFT COLUMN") == 10
    assert page.text.count("RIGHT COLUMN") == 10
    assert page.character_count == len(page.text)
    assert not result.empty_text_pages


def test_simple_mode_is_available_for_audit_and_reproducibility():
    pdf = _academic_pdf()
    simple = extract_pdf(pdf, "2510.22344", layout="simple")
    assert simple.pages[0].extraction_method == "simple"
    assert simple.pages[0].text.count("LEFT COLUMN") == 10
    assert simple.pages[0].text.count("RIGHT COLUMN") == 10
    assert simple.pages[0].text != extract_pdf(pdf, "2510.22344").pages[0].text


def test_single_column_fallback():
    result = extract_pdf(_academic_pdf(two_columns=False), "2510.22344")
    assert result.pages[0].extraction_method == "simple"
    assert result.pages[0].text.count("SINGLE COLUMN") == 20


def test_invalid_layout_is_rejected():
    with pytest.raises(ValueError, match="layout"):
        extract_pdf(_academic_pdf(), "2510.22344", layout="magic")


def test_extract_page_api_exposes_method():
    doc = pymupdf.open(stream=_academic_pdf(), filetype="pdf")
    try:
        text, mode = extract_page_text(doc[0])
        assert mode == "two_column"
        assert "LEFT COLUMN" in text
    finally:
        doc.close()
