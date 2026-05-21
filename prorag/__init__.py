"""
ProRAG — Proactive Knowledge Graph RAG

Fast start:
    from prorag import ProRAG

    rag = ProRAG()
    rag.ingest("Einstein developed the theory of relativity in 1905 in Bern.")
    result = rag.ask("Where did Einstein develop relativity?")
    print(result["answer"])
"""

from .graph import ProRAGGraph
from .extractor import ingest_text, ingest_file
from .pipeline import answer


class ProRAG:
    """
    High-level interface.

    Example:
        rag = ProRAG()
        rag.ingest_file("knowledge.txt")
        result = rag.ask("What did Einstein develop?")
    """

    def __init__(self, model: str = "llama3-70b-8192"):
        self.graph = ProRAGGraph()
        self.model = model

    def ingest(self, text: str, source: str = "", domains: list[str] | None = None) -> int:
        """Add raw text to the knowledge graph. Returns number of triples extracted."""
        return ingest_text(text, self.graph, source=source, llm_model=self.model, extra_domains=domains)

    def ingest_file(self, path: str, source: str | None = None) -> int:
        """Ingest a plain-text file. Returns number of triples extracted."""
        return ingest_file(path, self.graph, source=source, llm_model=self.model)

    def ask(self, question: str) -> dict:
        """Answer a question from the graph. Returns dict with answer + metadata."""
        return answer(question, self.graph, llm_model=self.model)

    def save(self, path: str) -> None:
        """Persist graph to disk."""
        self.graph.save(path)

    def load(self, path: str) -> None:
        """Load graph from disk."""
        self.graph.load(path)

    def stats(self) -> dict:
        return self.graph.stats()


__all__ = ["ProRAG", "ProRAGGraph", "ingest_text", "ingest_file", "answer"]
