"""
Public ProRAG API.
"""

from .extractor import ingest_file, ingest_text
from .graph import ProRAGGraph
from .pipeline import answer


class ProRAG:
    """High-level facade for ingest, ask, save, and load."""

    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.graph = ProRAGGraph()
        self.model = model

    def ingest(self, text: str, source: str = "") -> int:
        return ingest_text(text, self.graph, source=source, llm_model=self.model)

    def ingest_file(self, path: str, source: str | None = None) -> int:
        return ingest_file(path, self.graph, source=source, llm_model=self.model)

    def ask(self, question: str) -> dict:
        return answer(question, self.graph, llm_model=self.model)

    def save(self, path: str) -> None:
        self.graph.save(path)

    def load(self, path: str) -> None:
        self.graph.load(path)

    def stats(self) -> dict:
        return self.graph.stats()


__all__ = ["ProRAG", "ProRAGGraph", "ingest_text", "ingest_file", "answer"]
