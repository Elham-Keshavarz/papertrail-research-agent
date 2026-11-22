"""Page-aware, deterministic chunking for locally extracted research papers."""

from .chunker import ChunkedDocument, ChunkingError, TextChunk, chunk_document
from .pipeline import chunk_saved_document

__all__ = ["ChunkedDocument", "ChunkingError", "TextChunk", "chunk_document", "chunk_saved_document"]
