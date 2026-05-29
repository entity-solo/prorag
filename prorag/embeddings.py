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
            inst._use_fallback = False
            inst._cache: dict[str, np.ndarray] = {}
            cls._instance = inst
        return cls._instance

    # ── model loading ─────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self._model is not None or self._use_fallback:
            return
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            torch.set_num_threads(1)
        except ImportError:
            self._use_fallback = True
            return

        import os
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        local_path = os.path.join(repo_root, "models", self._model_name)
        load_path = local_path if os.path.exists(local_path) else self._model_name

        try:
            self._model = SentenceTransformer(load_path)
        except Exception:
            self._use_fallback = True

    def _fallback_embed(self, text: str) -> np.ndarray:
        vector = np.zeros(512, dtype=np.float32)
        normalized = text.lower().strip()
        tokens = normalized.split()
        for token in tokens:
            vector[hash(("tok", token)) % vector.size] += 2.0
        condensed = normalized.replace(" ", "")
        for size in (3, 4):
            for idx in range(max(0, len(condensed) - size + 1)):
                gram = condensed[idx : idx + size]
                vector[hash(("gram", gram)) % vector.size] += 1.0
        norm = np.linalg.norm(vector)
        if norm == 0:
            return vector
        return vector / norm

    # ── public API ────────────────────────────────────────────────────────────

    def embed(self, text: str) -> np.ndarray:
        """Return a normalized embedding vector for a text string."""
        if text not in self._cache:
            self._load()
            if self._use_fallback or self._model is None:
                vec = self._fallback_embed(text)
            else:
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
            if self._use_fallback or self._model is None:
                for text in uncached:
                    self._cache[text] = self._fallback_embed(text).astype(np.float32)
            else:
                vecs = self._model.encode(uncached, normalize_embeddings=True, batch_size=64)
                for text, vec in zip(uncached, vecs):
                    self._cache[text] = vec.astype(np.float32)

        scores = [
            (c, float(np.dot(q_emb, self._cache[c])))
            for c in candidates
        ]
        scores.sort(key=lambda x: -x[1])
        return [(c, s) for c, s in scores[:k] if s >= threshold]
