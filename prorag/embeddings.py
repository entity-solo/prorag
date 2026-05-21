"""
Embedding store — wraps sentence-transformers for semantic node/edge matching.

Used in ProRAG's 2-phase vector retrieval:
  Phase 1: cosine similarity between question and entity nodes → seed selection
  Phase 2: cosine similarity between question and edge relations → BFS cost weighting
"""

from __future__ import annotations

import numpy as np


class EmbeddingStore:
    """
    Singleton embedding store backed by a local sentence-transformers model.

    The model is loaded lazily on first use. All embeddings are L2-normalized
    so cosine similarity reduces to a simple dot product.
    """

    _instance: "EmbeddingStore | None" = None

    def __new__(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingStore":
        if cls._instance is None:
            inst = super().__new__(cls)
            inst._model_name = model_name
            inst._model = None
            inst._cache: dict[str, np.ndarray] = {}
            cls._instance = inst
        return cls._instance

    # ── model loading ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError(
                    "sentence-transformers is required for vector retrieval. "
                    "Install it with: pip install sentence-transformers"
                ) from e
            self._model = SentenceTransformer(self._model_name)

    # ── public API ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized embedding vector for a text string."""
        if text not in self._cache:
            self._load()
            vec = self._model.encode(text, normalize_embeddings=True)
            self._cache[text] = vec.astype(np.float32)
        return self._cache[text]

    def similarity(self, a: str, b: str) -> float:
        """Cosine similarity between two texts (range: -1 to 1)."""
        return float(np.dot(self.embed(a), self.embed(b)))

    def top_k(
        self,
        query: str,
        candidates: list[str],
        k: int,
        threshold: float = 0.0,
    ) -> list[tuple[str, float]]:
        """
        Return the top-k most similar candidates to query.
        Sorted by similarity descending. Filters by threshold.
        """
        if not candidates:
            return []
        q_emb = self.embed(query)
        # Batch encode all candidates not yet cached
        uncached = [c for c in candidates if c not in self._cache]
        if uncached:
            self._load()
            vecs = self._model.encode(uncached, normalize_embeddings=True, batch_size=64)
            for text, vec in zip(uncached, vecs):
                self._cache[text] = vec.astype(np.float32)

        scores = [
            (c, float(np.dot(q_emb, self._cache[c])))
            for c in candidates
        ]
        scores.sort(key=lambda x: -x[1])
        return [(c, s) for c, s in scores[:k] if s >= threshold]
