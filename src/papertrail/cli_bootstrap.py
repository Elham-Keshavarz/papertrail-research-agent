"""Lightweight CLI entry point: control CPU workers before native imports.

Do not import `papertrail.cli` or any NumPy/FAISS module above the function
below. Library consumers that only import `papertrail` remain unaffected.
"""
from __future__ import annotations

import os
from collections.abc import Sequence


def configure_numeric_threads() -> None:
    """Cap native math worker teams at launch, before NumPy/FAISS imports.

    PAPERTRAIL_MATH_THREADS accepts 1-8 (default 1), overruling potentially
    excessive inherited BLAS/OMP values. PAPERTRAIL_LIMIT_MATH_THREADS=0 opts
    out. No network, API calls or model loading happens here.
    """
    if os.environ.get("PAPERTRAIL_LIMIT_MATH_THREADS", "1") == "0":
        return
    requested = os.environ.get("PAPERTRAIL_MATH_THREADS", "1")
    if requested not in {str(i) for i in range(1, 9)}:
        requested = "1"
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = requested
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def main(argv: Sequence[str] | None = None) -> int:
    configure_numeric_threads()
    from .cli import main as cli_main
    return cli_main(argv)
