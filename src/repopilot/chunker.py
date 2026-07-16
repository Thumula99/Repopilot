from __future__ import annotations

import hashlib
from typing import Iterable, List

from .models import SourceDocument


def stable_id(*parts: str) -> str:
    joined = "::".join(parts)
    return hashlib.sha1(joined.encode("utf-8", errors="ignore")).hexdigest()[:16]


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> list[str]:
    """Simple character chunker with overlap.

    Character chunking is intentionally dependency-free. It is not perfect, but
    it is reliable for README files, code snippets, and GitHub issue text.
    """
    text = text.replace("\r\n", "\n").strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        window = text[start:end]

        # Prefer ending near a newline or sentence boundary when possible.
        if end < len(text):
            split_at = max(window.rfind("\n\n"), window.rfind("\n"), window.rfind(". "))
            if split_at > chunk_size * 0.55:
                end = start + split_at + 1
                window = text[start:end]

        chunks.append(window.strip())
        if end >= len(text):
            break
        start = max(0, end - overlap)
    return [c for c in chunks if c]


def chunk_documents(
    documents: Iterable[SourceDocument],
    chunk_size: int = 1200,
    overlap: int = 180,
) -> List[SourceDocument]:
    chunked: list[SourceDocument] = []
    for doc in documents:
        pieces = chunk_text(doc.text, chunk_size=chunk_size, overlap=overlap)
        for index, piece in enumerate(pieces):
            metadata = dict(doc.metadata)
            metadata["chunk_index"] = index
            chunked.append(
                SourceDocument(
                    id=stable_id(doc.id, str(index), piece[:80]),
                    text=piece,
                    metadata=metadata,
                )
            )
    return chunked
