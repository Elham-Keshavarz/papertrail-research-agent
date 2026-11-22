"""One-paper ingestion: optional local cache -> download -> extract -> atomic save."""

from __future__ import annotations

from pathlib import Path

from papertrail.storage import atomic_write_bytes

from .pdf_downloader import ArxivPdfDownloader, MAX_PDF_BYTES, PdfDownloadError, validate_paper_id
from .pdf_extractor import ExtractedDocument, extract_pdf


def ingest_arxiv_pdf(
    paper_id: str,
    *,
    data_dir: Path,
    client: ArxivPdfDownloader,
    max_pages: int = 100,
    force: bool = False,
    layout: str = "auto",
) -> tuple[ExtractedDocument, Path, Path, bool]:
    """Return document, local PDF path, extracted JSON path, cache hit.

    Only verified arXiv identifiers determine file paths. Extract before persisting downloads.
    """
    paper_id = validate_paper_id(paper_id)
    if layout not in {"auto", "simple"}:
        raise ValueError("layout must be 'auto' or 'simple'")
    if not 1 <= max_pages <= 200:
        raise ValueError("max_pages must be between 1 and 200")
    safe_stem = paper_id.replace("/", "__")
    pdf_path = data_dir / "papers" / f"{safe_stem}.pdf"
    json_path = data_dir / "extracted" / f"{safe_stem}.json"
    cached = pdf_path.is_file() and not force
    if cached:
        if pdf_path.stat().st_size > MAX_PDF_BYTES:
            raise PdfDownloadError("Cached PDF exceeds allowed size; use --force to re-download")
        payload = pdf_path.read_bytes()
    else:
        payload = client.fetch(paper_id)
    if len(payload) > MAX_PDF_BYTES:
        raise PdfDownloadError("PDF exceeds maximum cache size")
    extracted = extract_pdf(payload, paper_id, max_pages=max_pages, layout=layout)
    if not cached:
        atomic_write_bytes(pdf_path, payload)
    atomic_write_bytes(json_path, extracted.model_dump_json(indent=2).encode("utf-8"))
    return extracted, pdf_path, json_path, cached
