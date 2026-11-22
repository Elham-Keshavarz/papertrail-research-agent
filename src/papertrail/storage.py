"""Atomic local binary-file writes shared by document processing phases."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_bytes(destination: Path, content: bytes) -> None:
    """Write into the destination directory, then replace the final file atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(prefix=".papertrail_", suffix=".tmp", dir=destination.parent, delete=False) as temp:
            temp_path = Path(temp.name)
            temp.write(content)
        os.replace(temp_path, destination)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
