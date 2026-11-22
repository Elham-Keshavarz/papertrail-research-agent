"""Conservative, deterministic reading-order repair for two-column academic PDFs.

This is *not* a general document-layout engine. It only reorders detected,
non-overlapping left/right text blocks; ambiguous pages retain PyMuPDF's
ordinary sorted extraction and should be manually reviewed before citing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pymupdf

ExtractionMethod = Literal["simple", "two_column"]


@dataclass(frozen=True)
class _Block:
    x0: float
    y0: float
    x1: float
    y1: float
    text: str
    side: Literal["left", "right", "wide"]


def _text_blocks(page: pymupdf.Page) -> list[_Block]:
    """Get text blocks with page coordinates; skip image/non-text blocks."""
    midpoint = (page.rect.x0 + page.rect.x1) / 2
    tolerance = page.rect.width * 0.025
    blocks: list[_Block] = []
    for x0, y0, x1, y1, content, _number, block_type in page.get_text("blocks", sort=False):
        if block_type != 0 or not content.strip():
            continue
        if x1 <= midpoint + tolerance and x0 < midpoint - tolerance:
            side = "left"
        elif x0 >= midpoint - tolerance and x1 > midpoint + tolerance:
            side = "right"
        else:
            side = "wide"
        blocks.append(_Block(x0, y0, x1, y1, content.strip(), side))
    return blocks


def _is_two_column(page: pymupdf.Page, blocks: list[_Block]) -> bool:
    left = [b for b in blocks if b.side == "left" and len(b.text) >= 45]
    right = [b for b in blocks if b.side == "right" and len(b.text) >= 45]
    if not left or not right:
        return False
    if sum(len(b.text) for b in left) < 100 or sum(len(b.text) for b in right) < 100:
        return False
    left_top, left_bottom = min(b.y0 for b in left), max(b.y1 for b in left)
    right_top, right_bottom = min(b.y0 for b in right), max(b.y1 for b in right)
    overlap = min(left_bottom, right_bottom) - max(left_top, right_top)
    if overlap < min(70, 0.1 * page.rect.height):
        return False
    # Many wide blocks interspersed through the body indicate a complex page.
    # Avoid forcing a naive two-column ordering in that situation.
    wide = [b for b in blocks if b.side == "wide" and len(b.text) > 60]
    if len(wide) > len(left) + len(right):
        return False
    return True


def _column_order(blocks: list[_Block]) -> str:
    ordered: list[_Block] = []
    current: list[_Block] = []

    def flush() -> None:
        if not current:
            return
        ordered.extend(sorted((b for b in current if b.side == "left"), key=lambda b: (b.y0, b.x0)))
        ordered.extend(sorted((b for b in current if b.side == "right"), key=lambda b: (b.y0, b.x0)))
        current.clear()

    # A full-width heading separates vertical reading bands. Sort all blocks by
    # vertical position first, but read each band left-column then right-column.
    for block in sorted(blocks, key=lambda b: (b.y0, b.x0)):
        if block.side == "wide":
            flush()
            ordered.append(block)
        else:
            current.append(block)
    flush()
    return "\n\n".join(block.text for block in ordered).strip()


def extract_page_text(page: pymupdf.Page, *, layout: str = "auto") -> tuple[str, ExtractionMethod]:
    """Return text and selected reading order. ``simple`` keeps original behavior."""
    if layout not in {"auto", "simple"}:
        raise ValueError("layout must be 'auto' or 'simple'")
    if layout == "simple":
        return page.get_text("text", sort=True).strip(), "simple"
    blocks = _text_blocks(page)
    if _is_two_column(page, blocks):
        return _column_order(blocks), "two_column"
    return page.get_text("text", sort=True).strip(), "simple"
