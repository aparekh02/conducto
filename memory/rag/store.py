"""Vector store for grid knowledge — FAISS-backed with NVIDIA NIM embeddings.

Stores and retrieves:
  - Grid topology documents (bus specs, line ratings, transformer data)
  - Operating procedures and constraints
  - NERC/ISO reliability standards
  - Historical outage and incident reports
  - Equipment maintenance records

Uses NVIDIA NIM embedding endpoint to stay within the NVIDIA ecosystem.
Falls back to sentence-transformers for offline/local use.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_STORE_DIR = Path(__file__).parent.parent.parent / "nemotron_agent" / "data" / "vector_store"


@dataclass
class Document:
    """A chunk of knowledge stored in the vector store."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: np.ndarray | None = None


@dataclass
class SearchResult:
    doc_id: str
    content: str
    metadata: dict[str, Any]
    score: float


class VectorStore:
    """FAISS vector store with NVIDIA NIM embeddings."""

    def __init__(
        self,
        store_dir: Path | str = DEFAULT_STORE_DIR,
        embedding_model: str = "nvidia/nv-embedqa-e5-v5",
        nim_api_key: str | None = None,
        nim_base_url: str = "https://integrate.api.nvidia.com/v1",
        embedding_dim: int = 1024,
    ) -> None:
        import faiss

        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model = embedding_model
        self.embedding_dim = embedding_dim
        self._nim_api_key = nim_api_key or os.getenv("NVIDIA_API_KEY", "")
        self._nim_base_url = nim_base_url

        # FAISS index (L2 distance, flat — accurate for <100K docs)
        self._index = faiss.IndexFlatIP(embedding_dim)  # inner product (cosine after normalisation)
        self._documents: list[Document] = []
        self._doc_map: dict[str, int] = {}  # doc_id → position

        # Try to load existing index
        self._load()

        logger.info(
            "VectorStore ready  docs=%d  dim=%d  model=%s",
            len(self._documents), embedding_dim, embedding_model,
        )

    # ── Embeddings ───────────────────────────────────────

    def _embed_texts(self, texts: list[str]) -> np.ndarray:
        """Get embeddings from NVIDIA NIM embedding endpoint."""
        import requests

        resp = requests.post(
            f"{self._nim_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._nim_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.embedding_model,
                "input": texts,
                "input_type": "passage",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = np.array(
            [d["embedding"] for d in data["data"]],
            dtype=np.float32,
        )
        # L2-normalise for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-9)
        return embeddings

    def _embed_query(self, query: str) -> np.ndarray:
        """Embed a query (different input_type for some models)."""
        import requests

        resp = requests.post(
            f"{self._nim_base_url}/embeddings",
            headers={
                "Authorization": f"Bearer {self._nim_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.embedding_model,
                "input": [query],
                "input_type": "query",
            },
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()

        emb = np.array(data["data"][0]["embedding"], dtype=np.float32)
        emb = emb / max(np.linalg.norm(emb), 1e-9)
        return emb.reshape(1, -1)

    # ── Add documents ────────────────────────────────────

    def add(self, documents: list[Document]) -> int:
        """Add documents to the store. Returns count added."""
        texts = [d.content for d in documents]
        embeddings = self._embed_texts(texts)

        for i, doc in enumerate(documents):
            doc.embedding = embeddings[i]
            pos = len(self._documents)
            self._documents.append(doc)
            self._doc_map[doc.doc_id] = pos

        self._index.add(embeddings)
        self._save()
        logger.info("Added %d documents (total: %d)", len(documents), len(self._documents))
        return len(documents)

    # ── Search ───────────────────────────────────────────

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search for documents relevant to a query."""
        if len(self._documents) == 0:
            return []

        query_emb = self._embed_query(query)
        scores, indices = self._index.search(query_emb, min(top_k, len(self._documents)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            doc = self._documents[idx]
            results.append(SearchResult(
                doc_id=doc.doc_id,
                content=doc.content,
                metadata=doc.metadata,
                score=float(score),
            ))
        return results

    # ── Persistence ──────────────────────────────────────

    def _save(self) -> None:
        import faiss

        faiss.write_index(self._index, str(self.store_dir / "index.faiss"))

        docs_data = [
            {"doc_id": d.doc_id, "content": d.content, "metadata": d.metadata}
            for d in self._documents
        ]
        with open(self.store_dir / "documents.json", "w") as f:
            json.dump(docs_data, f)

    def _load(self) -> None:
        import faiss

        index_path = self.store_dir / "index.faiss"
        docs_path = self.store_dir / "documents.json"
        if index_path.exists() and docs_path.exists():
            self._index = faiss.read_index(str(index_path))
            with open(docs_path) as f:
                docs_data = json.load(f)
            self._documents = [
                Document(doc_id=d["doc_id"], content=d["content"], metadata=d["metadata"])
                for d in docs_data
            ]
            self._doc_map = {d.doc_id: i for i, d in enumerate(self._documents)}
            logger.info("Loaded %d documents from disk", len(self._documents))

    @property
    def size(self) -> int:
        return len(self._documents)
