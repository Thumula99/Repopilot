from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SourceDocument:
    id: str
    text: str
    metadata: Dict[str, Any]


@dataclass
class SearchHit:
    text: str
    metadata: Dict[str, Any]
    score: float | None = None

    @property
    def label(self) -> str:
        source = self.metadata.get("source", "unknown")
        path = self.metadata.get("path") or self.metadata.get("title") or ""
        return f"{source}: {path}".strip()
