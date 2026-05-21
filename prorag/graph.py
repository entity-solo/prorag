"""
Knowledge graph storage and retrieval.
"""

from __future__ import annotations

import heapq
import json
import time
from dataclasses import asdict, dataclass, field

import networkx as nx
import numpy as np

from .entity_utils import is_unresolved_reference, normalize_entity_name


@dataclass
class NodeMeta:
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    updated_at: float = field(default_factory=time.time)


@dataclass
class EdgeMeta:
    condition: str = ""
    negated: bool = False
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    updated_at: float = field(default_factory=time.time)


class ProRAGGraph:
    """Entity graph backed by ``networkx.MultiDiGraph``."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        source: str = "",
        condition: str = "",
        negated: bool = False,
        confidence: float = 1.0,
    ) -> None:
        """Add a validated triple to the graph."""
        subject = self._coerce_text(subject)
        relation = self._coerce_text(relation).strip().lower()
        obj = self._coerce_text(obj)
        condition = str(condition or "").strip()

        if not subject or not relation or not obj:
            return
        if is_unresolved_reference(subject) or is_unresolved_reference(obj):
            return

        now = time.time()
        self._ensure_node(subject, source=source, confidence=confidence, now=now)
        self._ensure_node(obj, source=source, confidence=confidence, now=now)

        existing = self._find_existing_edge(subject, relation, obj, negated, condition)
        if existing is not None:
            meta = existing["meta"]
            if source and source not in meta.sources:
                meta.sources.append(source)
            meta.confidence = max(meta.confidence, confidence)
            meta.updated_at = now
            return

        contradiction = self._find_contradicting_edge(subject, relation, obj, negated, condition)
        self._store_edge(
            subject,
            obj,
            relation,
            EdgeMeta(
                condition=condition,
                negated=negated,
                sources=[source] if source else [],
                confidence=confidence,
                updated_at=now,
            ),
        )
        if contradiction is not None:
            self._add_contradiction_note(subject, relation, obj, contradiction, source)

    def _store_edge(self, subject: str, obj: str, relation: str, meta: EdgeMeta) -> None:
        self.g.add_edge(subject, obj, relation=relation, meta=meta)

    def _coerce_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        return normalize_entity_name(str(value))

    def _ensure_node(self, node: str, *, source: str, confidence: float, now: float) -> None:
        if node not in self.g:
            self.g.add_node(
                node,
                meta=NodeMeta(
                    sources=[source] if source else [],
                    confidence=confidence,
                    updated_at=now,
                ),
            )
            return

        meta: NodeMeta = self.g.nodes[node]["meta"]
        if source and source not in meta.sources:
            meta.sources.append(source)
        meta.confidence = max(meta.confidence, confidence)
        meta.updated_at = now

    def _find_existing_edge(
        self,
        subject: str,
        relation: str,
        obj: str,
        negated: bool,
        condition: str,
    ) -> dict | None:
        for _, tgt, data in self.g.out_edges(subject, data=True):
            meta: EdgeMeta = data["meta"]
            if (
                tgt == obj
                and data.get("relation") == relation
                and meta.negated == negated
                and meta.condition == condition
            ):
                return data
        return None

    def _find_contradicting_edge(
        self,
        subject: str,
        relation: str,
        obj: str,
        negated: bool,
        condition: str,
    ) -> dict | None:
        for _, tgt, data in self.g.out_edges(subject, data=True):
            meta: EdgeMeta = data["meta"]
            if (
                tgt == obj
                and data.get("relation") == relation
                and meta.negated != negated
                and meta.condition == condition
            ):
                return data
        return None

    def _add_contradiction_note(
        self,
        subject: str,
        relation: str,
        obj: str,
        existing_data: dict,
        source: str,
    ) -> None:
        existing_data["meta"].confidence *= 0.7
        self._store_edge(
            subject,
            obj,
            f"CONTRADICTS:{relation}",
            EdgeMeta(
                sources=[source] if source else [],
                confidence=0.5,
                updated_at=time.time(),
            ),
        )

    def query(
        self,
        keywords: list[str],
        max_hops: int = 2,
        top_k: int = 40,
    ) -> list[dict]:
        """Keyword-seeded graph traversal for fallback retrieval."""
        candidate_nodes: set[str] = set()
        for keyword in keywords:
            keyword = normalize_entity_name(keyword)
            if not keyword:
                continue
            for node in self.g.nodes:
                if keyword in node:
                    candidate_nodes.add(node)

        if not candidate_nodes:
            return []

        distances = {node: 0 for node in candidate_nodes}
        queue = [(0, node) for node in candidate_nodes]
        heapq.heapify(queue)

        while queue:
            dist, node = heapq.heappop(queue)
            if dist > distances[node] or dist >= max_hops:
                continue

            neighbors = set(self.g.successors(node)) | set(self.g.predecessors(node))
            for neighbor in neighbors:
                new_dist = dist + 1
                if new_dist <= max_hops and (
                    neighbor not in distances or new_dist < distances[neighbor]
                ):
                    distances[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))

        expanded = set(distances)
        triples: list[dict] = []
        for src in expanded:
            for _, tgt, data in self.g.out_edges(src, data=True):
                if tgt not in expanded:
                    continue
                meta: EdgeMeta = data["meta"]
                triples.append(
                    {
                        "subject": src,
                        "relation": data["relation"],
                        "object": tgt,
                        "negated": meta.negated,
                        "condition": meta.condition,
                        "confidence": meta.confidence,
                        "sources": meta.sources,
                        "distance": min(distances.get(src, max_hops), distances.get(tgt, max_hops)),
                    }
                )

        def relevance(triple: dict) -> int:
            haystack = f"{triple['subject']} {triple['relation']} {triple['object']}"
            return sum(1 for keyword in keywords if normalize_entity_name(keyword) in haystack)

        triples.sort(key=lambda item: (-relevance(item), item["distance"], -item["confidence"]))
        return triples[:top_k]

    def query_vector(
        self,
        question: str,
        max_cost: float = 2.0,
        top_k: int = 60,
        seed_k: int = 10,
        seed_threshold: float = 0.25,
        alias_threshold: float = 0.85,
    ) -> list[dict]:
        """Semantic graph retrieval with alias bridging."""
        from .embeddings import EmbeddingStore

        store = EmbeddingStore()
        all_nodes = list(self.g.nodes)
        if not all_nodes:
            return []

        question_embedding = store.embed(question)
        top_nodes = store.top_k(question, all_nodes, k=seed_k, threshold=seed_threshold)
        seeds = {node for node, _ in top_nodes} or set(all_nodes[:seed_k])

        node_embeddings = None
        if alias_threshold > 0.0 and len(all_nodes) > 1:
            embeddings = np.array([store.embed(node) for node in all_nodes])
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            node_embeddings = embeddings / np.maximum(norms, 1e-9)

        distances: dict[str, float] = {node: 0.0 for node in seeds}
        queue = [(0.0, node) for node in seeds]
        heapq.heapify(queue)

        while queue:
            dist, node = heapq.heappop(queue)
            if dist > distances[node] or dist >= max_cost:
                continue

            edges: list[tuple[str, dict]] = []
            for _, neighbor, data in self.g.out_edges(node, data=True):
                edges.append((neighbor, data))
            for pred, _, data in self.g.in_edges(node, data=True):
                edges.append((pred, data))

            for neighbor, data in edges:
                relation = data.get("relation", "")
                relation_similarity = float(np.dot(question_embedding, store.embed(relation)))
                entity_similarity = float(np.dot(question_embedding, store.embed(neighbor)))
                step_cost = 1.0 - max(0.0, relation_similarity, entity_similarity)
                new_dist = dist + step_cost
                if new_dist < max_cost and (
                    neighbor not in distances or new_dist < distances[neighbor]
                ):
                    distances[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))

            if alias_threshold > 0.0 and node_embeddings is not None:
                node_embedding = store.embed(node)
                normalized = node_embedding / max(np.linalg.norm(node_embedding), 1e-9)
                similarities = np.dot(node_embeddings, normalized)
                for idx, similarity in enumerate(similarities):
                    alias_node = all_nodes[idx]
                    if alias_node == node or similarity < alias_threshold:
                        continue
                    new_dist = dist + (1.0 - max(0.0, float(similarity)))
                    if new_dist < max_cost and (
                        alias_node not in distances or new_dist < distances[alias_node]
                    ):
                        distances[alias_node] = new_dist
                        heapq.heappush(queue, (new_dist, alias_node))

        expanded = set(distances)
        triples: list[dict] = []
        for src in expanded:
            for _, tgt, data in self.g.out_edges(src, data=True):
                if tgt not in expanded:
                    continue
                meta: EdgeMeta = data["meta"]
                triple_text = f"{src} {data['relation']} {tgt}"
                triples.append(
                    {
                        "subject": src,
                        "relation": data["relation"],
                        "object": tgt,
                        "negated": meta.negated,
                        "condition": meta.condition,
                        "confidence": meta.confidence,
                        "sources": meta.sources,
                        "distance": min(distances.get(src, max_cost), distances.get(tgt, max_cost)),
                        "similarity": float(np.dot(question_embedding, store.embed(triple_text))),
                    }
                )

        triples.sort(key=lambda item: (-item["similarity"], item["distance"], -item["confidence"]))
        return triples[:top_k]

    def stats(self) -> dict:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
        }

    def save(self, path: str) -> None:
        data = nx.node_link_data(self.g)
        edge_key = "links" if "links" in data else "edges"
        for node in data["nodes"]:
            if "meta" in node:
                node["meta"] = asdict(node["meta"])
        for edge in data[edge_key]:
            if "meta" in edge:
                edge["meta"] = asdict(edge["meta"])
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"graph": data}, handle, ensure_ascii=False)

    def load(self, path: str) -> None:
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        graph_data = raw["graph"] if isinstance(raw, dict) and "graph" in raw else raw
        edge_key = "links" if "links" in graph_data else "edges"
        for node in graph_data["nodes"]:
            if "meta" in node:
                meta = node["meta"]
                if "domains" in meta:
                    meta.pop("domains", None)
                node["meta"] = NodeMeta(**meta)
        for edge in graph_data[edge_key]:
            if "meta" in edge:
                edge["meta"] = EdgeMeta(**edge["meta"])
        self.g = nx.node_link_graph(graph_data)
