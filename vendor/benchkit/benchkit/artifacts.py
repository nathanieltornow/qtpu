"""Minimal filesystem-based artifact store."""

from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

ARTIFACT_ROOT = Path("benchkit/artifacts")
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)


def artifact(data: bytes, file_ext: str) -> str:
    """Store raw bytes as an artifact and return its filepath.

    Args:
        data (bytes): Bytes to store.
        file_ext (str): File extension.

    Returns:
        str: Path to the saved artifact (relative to ARTIFACT_ROOT).
    """
    today = dt.datetime.now(dt.timezone.utc).date().isoformat()
    uid = str(uuid.uuid4())[:8]

    folder = ARTIFACT_ROOT / today
    folder.mkdir(parents=True, exist_ok=True)

    path = folder / f"{uid}.{file_ext}"
    path.write_bytes(data)
    return str(path)


def load_artifact(path: str) -> bytes:
    """Load raw bytes from an artifact filepath.

    Args:
        path (str): Path to the artifact file.

    Returns:
        bytes: The loaded bytes.
    """
    return Path(path).read_bytes()
