"""Parse and validate arXiv Atom responses; never interpret source text as instructions."""

import re
from urllib.parse import unquote, urlsplit

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException
from pydantic import ValidationError
from xml.etree.ElementTree import ParseError

from papertrail.models import PaperMetadata, PaperSearchResult

ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV = "{http://arxiv.org/schemas/atom}"
OPENSEARCH = "{http://a9.com/-/spec/opensearch/1.1/}"
ARXIV_ID = re.compile(r"(?:\d{4}\.\d{4,5}|[a-z][\w.-]*/\d{7})(?:v\d+)?\Z", re.IGNORECASE)


class ArxivParseError(ValueError):
    """The response is not a valid arXiv search-result feed."""


def _text(parent, path: str, *, required: bool = False) -> str | None:
    node = parent.find(path)
    value = " ".join("".join(node.itertext()).split()) if node is not None else ""
    if required and not value:
        raise ArxivParseError(f"Missing required Atom metadata: {path}")
    return value or None


def _paper_id(raw_url: str) -> str:
    url = urlsplit(raw_url)
    if url.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"} or url.scheme not in {"https", "http"}:
        raise ArxivParseError("Paper ID URL does not belong to arxiv.org")
    if not url.path.startswith("/abs/"):
        raise ArxivParseError("Paper ID URL is not an /abs/ URL")
    identifier = unquote(url.path[len("/abs/"):])
    if not ARXIV_ID.fullmatch(identifier):
        raise ArxivParseError("Invalid arXiv identifier in Atom entry")
    return re.sub(r"v\d+$", "", identifier, flags=re.IGNORECASE)


def parse_arxiv_feed(payload: bytes, *, query: str, start: int, requested: int) -> PaperSearchResult:
    """Convert a response body into trustworthy *metadata contracts* (not trusted content)."""
    try:
        root = SafeET.fromstring(payload)
    except (ParseError, DefusedXmlException) as exc:
        raise ArxivParseError("Response is malformed or unsafe XML") from exc
    if root.tag != f"{ATOM}feed":
        raise ArxivParseError("Response does not contain an Atom feed")

    feed_title = _text(root, f"{ATOM}title")
    if feed_title and feed_title.casefold() == "error":
        raise ArxivParseError("arXiv returned an error feed; check your search expression")

    total_string = _text(root, f"{OPENSEARCH}totalResults")
    if total_string is not None:
        try:
            total = int(total_string)
        except ValueError as exc:
            raise ArxivParseError("Invalid OpenSearch totalResults") from exc
        if total < 0:
            raise ArxivParseError("OpenSearch totalResults cannot be negative")
    else:
        total = None

    papers = []
    for entry in root.findall(f"{ATOM}entry"):
        paper_id = _paper_id(_text(entry, f"{ATOM}id", required=True))
        authors = [name for author in entry.findall(f"{ATOM}author")
                   if (name := _text(author, f"{ATOM}name"))]
        categories = list(dict.fromkeys(
            category.attrib["term"]
            for category in entry.findall(f"{ATOM}category")
            if category.attrib.get("term")
        ))
        primary_category = entry.find(f"{ARXIV}primary_category")
        if primary_category is not None and primary_category.attrib.get("term"):
            primary = primary_category.attrib["term"]
            if primary not in categories:
                categories.insert(0, primary)
        try:
            papers.append(PaperMetadata(
                paper_id=paper_id,
                title=_text(entry, f"{ATOM}title", required=True),
                authors=authors,
                abstract=_text(entry, f"{ATOM}summary", required=True),
                published_at=_text(entry, f"{ATOM}published", required=True),
                updated_at=_text(entry, f"{ATOM}updated"),
                url=f"https://arxiv.org/abs/{paper_id}",
                pdf_url=f"https://arxiv.org/pdf/{paper_id}",
                categories=categories,
                journal_reference=_text(entry, f"{ARXIV}journal_ref"),
                is_peer_reviewed=None,  # Journal reference alone does not establish review status.
            ))
        except ValidationError as exc:
            raise ArxivParseError(f"Invalid metadata for arXiv entry {paper_id}") from exc
    if len(papers) > requested:
        raise ArxivParseError("arXiv returned more entries than requested")
    return PaperSearchResult(query=query, start=start, total_results=total, papers=papers)
