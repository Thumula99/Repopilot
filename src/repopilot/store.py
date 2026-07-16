from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from .models import SearchHit, SourceDocument


class FreeHashingEmbedder:
    """Offline embedding function for a zero-cost MVP.

    This uses scikit-learn's HashingVectorizer, so it does not need API calls,
    model downloads, or GPU access. It is weaker than modern embedding models,
    but good enough for a portfolio MVP and cheap demos.
    """

    def __init__(self, n_features: int = 512):
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm=None,
            ngram_range=(1, 2),
            lowercase=True,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self.vectorizer.transform(texts)
        matrix = normalize(matrix, norm="l2", axis=1)
        return matrix.astype(np.float32).toarray().tolist()


def safe_collection_name(name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", name).strip("-").lower()
    cleaned = cleaned[:60] or "repopilot"
    if len(cleaned) < 3:
        cleaned = f"repo-{cleaned}"
    return cleaned


class ChromaStore:
    def __init__(self, persist_dir: Path | str = "data/chroma", collection_name: str = "repopilot"):
        import chromadb

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = safe_collection_name(collection_name)
        self.embedder = FreeHashingEmbedder()
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, documents: Iterable[SourceDocument]) -> int:
        docs = list(documents)
        if not docs:
            return 0

        ids = [doc.id for doc in docs]
        texts = [doc.text for doc in docs]
        metadatas = [self._clean_metadata(doc.metadata) for doc in docs]
        embeddings = self.embedder.embed(texts)

        # Batch to keep memory small on free environments.
        batch_size = 100
        for start in range(0, len(docs), batch_size):
            end = start + batch_size
            self.collection.upsert(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metadatas[start:end],
                embeddings=embeddings[start:end],
            )
        return len(docs)

    def search(self, query: str, k: int = 6, where: dict | None = None) -> list[SearchHit]:
        query_embedding = self.embedder.embed([query])[0]
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[SearchHit] = []
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        for text, metadata, distance in zip(documents, metadatas, distances):
            score = None
            if distance is not None:
                score = max(0.0, 1.0 - float(distance))
            hits.append(SearchHit(text=text, metadata=metadata or {}, score=score))
        return hits

    def count(self) -> int:
        return int(self.collection.count())

    @staticmethod
    def _clean_metadata(metadata: dict) -> dict:
        cleaned = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            else:
                cleaned[key] = str(value)
        return cleaned
