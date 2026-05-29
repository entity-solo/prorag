"""
Public ProRAG API.
"""

from .extractor import ingest_file, ingest_text
from .graph import ProRAGGraph
from .modes import QualityMode
from .pipeline import answer


class ProRAG:
    """High-level facade for ingest, ask, save, and load."""

    def __init__(
        self,
        model: str = "llama-3.3-70b-versatile",
        quality_mode: QualityMode | str = "balanced",
    ):
        self.graph = ProRAGGraph()
        self.model = model
        self.quality_mode = quality_mode

    def ingest(
        self,
        text: str,
        source: str = "",
        *,
        quality_mode: QualityMode | str | None = None,
    ) -> int:
        count, _ = ingest_text(
            text,
            self.graph,
            source=source,
            llm_model=self.model,
            quality_mode=quality_mode or self.quality_mode,
        )
        return count

    def ingest_file(
        self,
        path: str,
        source: str | None = None,
        chunk_size: int = 1500,
        overlap_sentences: int = 1,
        quality_mode: QualityMode | str | None = None,
    ) -> int:
        return ingest_file(
            path,
            self.graph,
            source=source,
            llm_model=self.model,
            chunk_size=chunk_size,
            overlap_sentences=overlap_sentences,
            quality_mode=quality_mode or self.quality_mode,
        )

    def ask(
        self,
        question: str,
        *,
        include_source_text: bool = False,
        max_source_chars: int = 1200,
        quality_mode: QualityMode | str | None = None,
    ) -> dict:
        return answer(
            question,
            self.graph,
            llm_model=self.model,
            include_source_text=include_source_text,
            max_source_chars=max_source_chars,
            quality_mode=quality_mode or self.quality_mode,
        )

    def save(self, path: str) -> None:
        self.graph.save(path)

    def load(self, path: str) -> None:
        self.graph.load(path)

    def stats(self) -> dict:
        return self.graph.stats()


__all__ = ["ProRAG", "ProRAGGraph", "ingest_text", "ingest_file", "answer"]
