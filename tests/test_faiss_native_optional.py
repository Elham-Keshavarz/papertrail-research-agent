"""Native FAISS integration smoke test: runs in CI and after search extras install."""

import numpy as np
import pytest


def test_native_faiss_inner_product_search_and_serialization():
    faiss = pytest.importorskip("faiss")
    index = faiss.IndexFlatIP(3)
    matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
    index.add(matrix)
    blob = faiss.serialize_index(index)
    restored = faiss.deserialize_index(blob)
    query = np.array([[0, 1, 0]], dtype=np.float32)
    scores, ids = restored.search(query, 2)
    assert restored.ntotal == 3
    assert ids[0, 0] == 1
    assert scores[0, 0] == pytest.approx(1.0)
