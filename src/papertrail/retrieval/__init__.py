"""Page-attributed dense, lexical, and hybrid local retrieval."""

from .faiss_store import IndexBuild, RetrievalError, build_paper_index
from .search import SearchHit, SearchResult, search_paper
from .bm25 import BM25Index, BM25Hit, tokenize
from .hybrid import EvidencePassage, HybridHit, HybridResult, hybrid_search_paper
from .evaluation import Coverage, RetrievalComparison, compare_benchmark_retrieval
from .quality_eval import (RetrievalSuite, RetrievalEvaluationReport, load_retrieval_suite,
                           run_retrieval_evaluation, save_retrieval_report, score_at_k)

__all__ = [
    "IndexBuild", "RetrievalError", "build_paper_index", "SearchHit", "SearchResult", "search_paper",
    "BM25Index", "BM25Hit", "tokenize", "EvidencePassage", "HybridHit", "HybridResult",
    "hybrid_search_paper", "Coverage", "RetrievalComparison", "compare_benchmark_retrieval", "RetrievalSuite", "RetrievalEvaluationReport",
    "load_retrieval_suite", "run_retrieval_evaluation", "save_retrieval_report", "score_at_k",
]

from .fusion import fuse_candidates, validate_rrf_weights, FusedCandidate
from .optimization import (OptimizationReport, run_retrieval_optimization,
                           save_optimization_report)
