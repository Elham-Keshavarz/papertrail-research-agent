"""Deterministic, dependency-free BM25 retrieval for small paper corpora.

The index is built in memory from *verified* saved chunks. It is deliberately
not a persistent external search service. Scores are unbounded and must not
be interpreted as confidence or compared directly with cosine similarity.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from papertrail.chunking.chunker import TextChunk

# Avoid scoring ubiquitous words from question phrasing. Technical terms, numbers,
# model names and dataset names are retained, including their component tokens.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "do", "does",
    "for", "from", "how", "in", "into", "is", "it", "its", "of", "on", "or",
    "that", "the", "their", "these", "this", "those", "to", "was", "were",
    "what", "when", "where", "which", "who", "why", "with", "used", "using",
})
TOKEN_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Case-insensitive Unicode tokenization; preserve exact technical names.

    'FAIR-RAG' yields FAIR and RAG; '2WikiMultiHopQA' stays one token.
    Add a minimal plural normalization for common query-vs-heading variation.
    """
    tokens: list[str] = []
    for match in TOKEN_PATTERN.finditer(text.casefold()):
        token = match.group()
        if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        if len(token) > 1 and token not in STOPWORDS:
            tokens.append(token)
    return tokens


@dataclass(frozen=True)
class BM25Hit:
    position: int
    score: float
    rank: int


class BM25Index:
    """Okapi BM25; rows refer to the input chunk order, not PDF offsets."""

    def __init__(self, chunks: Sequence[TextChunk], *, k1: float = 1.2, b: float = 0.75):
        if not chunks:
            raise ValueError("BM25 needs at least one chunk")
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 parameters are invalid")
        self.k1 = k1
        self.b = b
        self.lengths: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.document_frequency: Counter[str] = Counter()
        for position, chunk in enumerate(chunks):
            counts = Counter(tokenize(chunk.text))
            self.lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((position, count))
                self.document_frequency[term] += 1
        self.count = len(chunks)
        self.avg_length = sum(self.lengths) / self.count if sum(self.lengths) else 1.0

    def search(self, query: str, *, limit: int = 60) -> list[BM25Hit]:
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("BM25 limit must be positive")
        query_tokens = Counter(tokenize(query))
        if not query_tokens:
            return []
        scores: dict[int, float] = defaultdict(float)
        for term, query_tf in query_tokens.items():
            df = self.document_frequency.get(term, 0)
            if not df:
                continue
            idf = math.log1p((self.count - df + 0.5) / (df + 0.5))
            for position, freq in self.postings[term]:
                length_norm = 1 - self.b + self.b * (self.lengths[position] / self.avg_length)
                denom = freq + self.k1 * length_norm
                scores[position] += idf * (freq * (self.k1 + 1) / denom) * min(query_tf, 2)
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:limit]
        return [BM25Hit(position=pos, score=round(score, 6), rank=i)
                for i, (pos, score) in enumerate(ranked, 1)]
