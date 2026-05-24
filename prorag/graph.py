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

from .entity_utils import normalize_entity_name


@dataclass
class NodeMeta:
    node_type: str = "entity"  # "entity" / "event" / "temporal"
    attributes: dict[str, str] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
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
    statement_time: str = ""
    temporal_aspect: str = "PRESENT"
    aspect: str = ""  # perfective / imperfective / prospective / habitual
    modality: str = ""  # certain / possible / necessary / counterfactual
    quantifier: str = ""  # all / some / most / over / less_than
    evidentiality: str = ""  # direct / reported / inferred
    speech_act: str = ""  # assertion / claim / question
    causal: str = ""  # event ID which is the cause


class ProRAGGraph:
    """Entity graph backed by ``networkx.MultiDiGraph``."""

    def __init__(self) -> None:
        self.g = nx.MultiDiGraph()
        self.chunks: dict[str, str] = {}

    def add_chunk(self, source: str, text: str) -> None:
        """Add a raw text chunk to the graph's storage linked to its source/chunk ID."""
        if source and text:
            self.chunks[source] = text.strip()

    def add_relation(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        source: str = "",
        condition: str = "",
        negated: bool = False,
        confidence: float = 1.0,
        statement_time: str = "",
        temporal_aspect: str = "PRESENT",
        aspect: str = "",
        modality: str = "",
        quantifier: str = "",
        evidentiality: str = "",
        speech_act: str = "",
        causal: str = "",
    ) -> None:
        """Add a standard relation edge between two entities."""
        subject = self._coerce_text(subject)
        relation = self._coerce_text(relation).strip().lower()
        obj = self._coerce_text(obj)
        condition = str(condition or "").strip()
        statement_time = str(statement_time or "").strip()
        temporal_aspect = str(temporal_aspect or "PRESENT").strip().upper()

        if not subject or not relation or not obj:
            return

        now = time.time()
        self._ensure_node(subject, source=source, confidence=confidence, now=now, node_type="entity")
        self._ensure_node(obj, source=source, confidence=confidence, now=now, node_type="entity")

        existing = self._find_existing_edge(subject, relation, obj, negated, condition, statement_time, temporal_aspect)
        if existing is not None:
            meta = existing["meta"]
            if source and source not in meta.sources:
                meta.sources.append(source)
            meta.confidence = max(meta.confidence, confidence)
            meta.updated_at = now
            return

        contradiction = self._find_contradicting_edge(subject, relation, obj, negated, condition, statement_time, temporal_aspect)
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
                statement_time=statement_time,
                temporal_aspect=temporal_aspect,
                aspect=aspect,
                modality=modality,
                quantifier=quantifier,
                evidentiality=evidentiality,
                speech_act=speech_act,
                causal=causal,
            ),
        )
        if contradiction is not None:
            self._add_contradiction_note(subject, relation, obj, contradiction, source)

    def add_attribute(
        self,
        subject: str,
        key: str,
        value: str,
        *,
        source: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Add or update an attribute of an entity."""
        subject = self._coerce_text(subject)
        key = normalize_entity_name(key)
        value = str(value).strip()
        if not subject or not key or not value:
            return

        now = time.time()
        self._ensure_node(subject, source=source, confidence=confidence, now=now, node_type="entity")
        meta: NodeMeta = self.g.nodes[subject]["meta"]
        meta.attributes[key] = value
        meta.updated_at = now

    def add_event(
        self,
        event_id: str,
        role: str,
        entity: str,
        *,
        source: str = "",
        condition: str = "",
        negated: bool = False,
        confidence: float = 1.0,
        statement_time: str = "",
        temporal_aspect: str = "PRESENT",
        aspect: str = "",
        modality: str = "",
        quantifier: str = "",
        evidentiality: str = "",
        speech_act: str = "",
        causal: str = "",
    ) -> None:
        """Add an event node and create participant role edges."""
        event_id = self._coerce_text(event_id)
        role = normalize_entity_name(role)
        entity = self._coerce_text(entity)
        if not event_id or not role or not entity:
            return

        now = time.time()
        node_type_target = "temporal" if role in ("time", "date") else "entity"

        self._ensure_node(event_id, source=source, confidence=confidence, now=now, node_type="event")
        self._ensure_node(entity, source=source, confidence=confidence, now=now, node_type=node_type_target)

        existing = self._find_existing_edge(event_id, role, entity, negated, condition, statement_time, temporal_aspect)
        if existing is not None:
            meta = existing["meta"]
            if source and source not in meta.sources:
                meta.sources.append(source)
            meta.confidence = max(meta.confidence, confidence)
            meta.updated_at = now
            return

        contradiction = self._find_contradicting_edge(event_id, role, entity, negated, condition, statement_time, temporal_aspect)
        self._store_edge(
            event_id,
            entity,
            role,
            EdgeMeta(
                condition=condition,
                negated=negated,
                sources=[source] if source else [],
                confidence=confidence,
                updated_at=now,
                statement_time=statement_time,
                temporal_aspect=temporal_aspect,
                aspect=aspect,
                modality=modality,
                quantifier=quantifier,
                evidentiality=evidentiality,
                speech_act=speech_act,
                causal=causal,
            ),
        )
        if contradiction is not None:
            self._add_contradiction_note(event_id, role, entity, contradiction, source)

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
        statement_time: str = "",
        temporal_aspect: str = "PRESENT",
    ) -> None:
        """Add a validated relation to the graph (for backward compatibility)."""
        self.add_relation(
            subject=subject,
            relation=relation,
            obj=obj,
            source=source,
            condition=condition,
            negated=negated,
            confidence=confidence,
            statement_time=statement_time,
            temporal_aspect=temporal_aspect,
        )

    def _store_edge(self, subject: str, obj: str, relation: str, meta: EdgeMeta) -> None:
        self.g.add_edge(subject, obj, relation=relation, meta=meta)

    def _coerce_text(self, value: object) -> str:
        if value is None:
            return ""
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        return normalize_entity_name(str(value))

    def _ensure_node(self, node: str, *, source: str, confidence: float, now: float, node_type: str = "entity") -> None:
        if node not in self.g:
            self.g.add_node(
                node,
                meta=NodeMeta(
                    node_type=node_type,
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
        statement_time: str,
        temporal_aspect: str,
    ) -> dict | None:
        for _, tgt, data in self.g.out_edges(subject, data=True):
            meta: EdgeMeta = data["meta"]
            if (
                tgt == obj
                and data.get("relation") == relation
                and meta.negated == negated
                and meta.condition == condition
                and meta.statement_time == statement_time
                and meta.temporal_aspect == temporal_aspect
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
        statement_time: str,
        temporal_aspect: str,
    ) -> dict | None:
        for _, tgt, data in self.g.out_edges(subject, data=True):
            meta: EdgeMeta = data["meta"]
            if (
                tgt == obj
                and data.get("relation") == relation
                and meta.negated != negated
                and meta.condition == condition
                and meta.statement_time == statement_time
                and meta.temporal_aspect == temporal_aspect
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
            for node, data in self.g.nodes(data=True):
                if keyword in node:
                    candidate_nodes.add(node)
                    continue
                meta = data.get("meta")
                if meta and hasattr(meta, "attributes") and meta.attributes:
                    for attr_key, attr_val in meta.attributes.items():
                        if keyword in attr_key.lower() or keyword in attr_val.lower():
                            candidate_nodes.add(node)
                            break

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
                        "statement_time": meta.statement_time,
                        "temporal_aspect": meta.temporal_aspect,
                    }
                )

        # Add virtual triples for attributes of expanded nodes
        for node in expanded:
            meta = self.g.nodes[node].get("meta")
            if meta and hasattr(meta, "attributes") and meta.attributes:
                for attr_key, attr_val in meta.attributes.items():
                    triples.append(
                        {
                            "subject": node,
                            "relation": attr_key,
                            "object": attr_val,
                            "negated": False,
                            "condition": "",
                            "confidence": meta.confidence,
                            "sources": meta.sources,
                            "distance": distances.get(node, max_hops),
                            "statement_time": "",
                            "temporal_aspect": "PRESENT",
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
                        "statement_time": meta.statement_time,
                        "temporal_aspect": meta.temporal_aspect,
                    }
                )

        # Add virtual triples for attributes of expanded nodes
        for node in expanded:
            meta = self.g.nodes[node].get("meta")
            if meta and hasattr(meta, "attributes") and meta.attributes:
                for attr_key, attr_val in meta.attributes.items():
                    triple_text = f"{node} {attr_key} {attr_val}"
                    triples.append(
                        {
                            "subject": node,
                            "relation": attr_key,
                            "object": attr_val,
                            "negated": False,
                            "condition": "",
                            "confidence": meta.confidence,
                            "sources": meta.sources,
                            "distance": distances.get(node, max_cost),
                            "similarity": float(np.dot(question_embedding, store.embed(triple_text))),
                            "statement_time": "",
                            "temporal_aspect": "PRESENT",
                        }
                    )

        triples.sort(key=lambda item: (-item["similarity"], item["distance"], -item["confidence"]))
        return triples[:top_k]

    def stats(self) -> dict:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
        }

    def merge_entities(
        self,
        name_similarity_threshold: float = 0.8,
        fingerprint_threshold: float = 0.4,
    ) -> None:
        """Merge similar entities based on name similarity and neighbor fingerprint overlap."""
        import difflib

        nodes = list(self.g.nodes)
        if len(nodes) < 2:
            return

        # Calculate fingerprint for each node
        # Fingerprint: set of (relation, neighbor, direction)
        fingerprints: dict[str, set[tuple[str, str, str]]] = {}
        for node in nodes:
            fp = set()
            # Outgoing edges
            for _, tgt, key, data in self.g.out_edges(node, keys=True, data=True):
                relation = data.get("relation") or key
                fp.add((relation, tgt, "out"))
            # Incoming edges
            for src, _, key, data in self.g.in_edges(node, keys=True, data=True):
                relation = data.get("relation") or key
                fp.add((relation, src, "in"))
            fingerprints[node] = fp

        # We will find pairs to merge
        # Disjoint-set (union-find) to track groups
        parent = {node: node for node in nodes}

        def find(n):
            path = []
            while parent[n] != n:
                path.append(n)
                n = parent[n]
            for node in path:
                parent[node] = n
            return n

        def union(n1, n2):
            r1 = find(n1)
            r2 = find(n2)
            if r1 != r2:
                # Make the one with longer name or more edges the root
                len1 = len(r1)
                len2 = len(r2)
                if len1 >= len2:
                    parent[r2] = r1
                else:
                    parent[r1] = r2

        for i in range(len(nodes)):
            node_a = nodes[i]
            fp_a = fingerprints[node_a]
            if not fp_a:
                continue
            for j in range(i + 1, len(nodes)):
                node_b = nodes[j]
                fp_b = fingerprints[node_b]
                if not fp_b:
                    continue

                # Check name similarity
                name_sim = difflib.SequenceMatcher(None, node_a, node_b).ratio()
                if name_sim < name_similarity_threshold:
                    continue

                # Calculate Jaccard similarity of fingerprints
                intersection = fp_a & fp_b
                union_set = fp_a | fp_b
                if not union_set:
                    continue
                jaccard = len(intersection) / len(union_set)

                if jaccard >= fingerprint_threshold:
                    union(node_a, node_b)

        # Now group nodes by their representative root
        groups: dict[str, list[str]] = {}
        for node in nodes:
            root = find(node)
            if root != node:
                groups.setdefault(root, []).append(node)

        # Execute merging for each group
        for root, aliases in groups.items():
            root_meta: NodeMeta = self.g.nodes[root]["meta"]
            for alias in aliases:
                alias_meta: NodeMeta = self.g.nodes[alias]["meta"]

                # Merge metadata
                root_meta.aliases.append(alias)
                if hasattr(alias_meta, "aliases"):
                    root_meta.aliases.extend(alias_meta.aliases)
                root_meta.aliases = list(set(root_meta.aliases))

                # Merge attributes
                if hasattr(alias_meta, "attributes") and alias_meta.attributes:
                    for k, v in alias_meta.attributes.items():
                        if k not in root_meta.attributes:
                            root_meta.attributes[k] = v

                # Merge sources
                for s in alias_meta.sources:
                    if s not in root_meta.sources:
                        root_meta.sources.append(s)

                root_meta.confidence = max(root_meta.confidence, alias_meta.confidence)
                root_meta.updated_at = max(root_meta.updated_at, alias_meta.updated_at)

                # Redirect incoming edges
                in_edges = list(self.g.in_edges(alias, data=True, keys=True))
                for src, _, key, data in in_edges:
                    if src == root:
                        continue
                    relation = data.get("relation") or key
                    meta = data.get("meta")
                    self.g.add_edge(src, root, relation=relation, meta=meta)

                # Redirect outgoing edges
                out_edges = list(self.g.out_edges(alias, data=True, keys=True))
                for _, tgt, key, data in out_edges:
                    if tgt == root:
                        continue
                    relation = data.get("relation") or key
                    meta = data.get("meta")
                    self.g.add_edge(root, tgt, relation=relation, meta=meta)

                # Remove the alias node
                self.g.remove_node(alias)

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
            json.dump({"graph": data, "chunks": self.chunks}, handle, ensure_ascii=False)

    def load(self, path: str) -> None:
        import inspect
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        graph_data = raw["graph"] if isinstance(raw, dict) and "graph" in raw else raw
        self.chunks = raw.get("chunks", {}) if isinstance(raw, dict) else {}
        edge_key = "links" if "links" in graph_data else "edges"

        node_fields = set(inspect.signature(NodeMeta).parameters.keys())
        edge_fields = set(inspect.signature(EdgeMeta).parameters.keys())

        for node in graph_data["nodes"]:
            if "meta" in node:
                meta = node["meta"]
                if "domains" in meta:
                    meta.pop("domains", None)
                filtered_meta = {k: v for k, v in meta.items() if k in node_fields}
                node["meta"] = NodeMeta(**filtered_meta)
        for edge in graph_data[edge_key]:
            if "meta" in edge:
                meta = edge["meta"]
                filtered_meta = {k: v for k, v in meta.items() if k in edge_fields}
                edge["meta"] = EdgeMeta(**filtered_meta)
        self.g = nx.node_link_graph(graph_data)
