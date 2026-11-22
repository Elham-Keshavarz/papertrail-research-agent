"""Local sentence embeddings and hash-validated on-disk cache."""
from .pipeline import EmbeddingError, EmbeddingManifest, EmbeddingRun, embed_saved_document

__all__ = ["EmbeddingError", "EmbeddingManifest", "EmbeddingRun", "embed_saved_document"]
