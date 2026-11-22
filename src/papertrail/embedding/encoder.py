"""Local, pinned sentence embedding model. No cloud inference or API key."""
from __future__ import annotations

from typing import Protocol

import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# A public, immutable Hugging Face repository revision (not the moving 'main' branch).
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"


class Encoder(Protocol):
    dimension: int
    max_sequence_length: int

    def token_count(self, text: str) -> int: ...

    def encode_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


class SentenceTransformerEncoder:
    """Loads model lazily so importing core CLI never requires torch or HF packages."""

    def __init__(self, model_name: str = MODEL_NAME, revision: str = MODEL_REVISION):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError('Embedding dependencies missing. Install with: pip install -e ".[dev,embeddings]"') from exc
        try:
            self.model = SentenceTransformer(model_name, revision=revision, trust_remote_code=False)
        except Exception as exc:
            raise RuntimeError(f"Could not load local embedding model; check internet on first run or HF cache ({type(exc).__name__})") from exc
        dimension_getter = getattr(self.model, "get_embedding_dimension", None)
        if dimension_getter is None:
            dimension_getter = self.model.get_sentence_embedding_dimension
        self.dimension = int(dimension_getter() or 0)
        self.max_sequence_length = int(self.model.max_seq_length or 0)
        if self.dimension < 1 or self.max_sequence_length < 1:
            raise RuntimeError("Embedding model has invalid dimension or max sequence length")

    def token_count(self, text: str) -> int:
        # Explicitly disable truncation. The model's default may silently truncate >256 tokens.
        tokens = self.model.tokenizer(text, add_special_tokens=True, truncation=False)
        return len(tokens["input_ids"])

    def encode_documents(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        encode = getattr(self.model, "encode_document", None)
        if encode is None:
            encode = self.model.encode
        return np.asarray(encode(
            texts, batch_size=batch_size, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        ), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """Use retrieval query prompt if model supplies one; else conventional encoding."""
        encode = getattr(self.model, "encode_query", None)
        if encode is None:
            encode = self.model.encode
        return np.asarray(encode(
            text, convert_to_numpy=True, normalize_embeddings=True,
            show_progress_bar=False,
        ), dtype=np.float32)
