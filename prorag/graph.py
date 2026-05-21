"""
Knowledge Graph engine — stores entities, relations, and metadata.
Partitioned by domain. Handles contradictions explicitly.
"""

import time
import json
import networkx as nx
from dataclasses import dataclass, field, asdict


@dataclass
class NodeMeta:
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    updated_at: float = field(default_factory=time.time)
    domains: list[str] = field(default_factory=list)


@dataclass
class EdgeMeta:
    condition: str = ""           # e.g. "at 1 atm pressure"
    negated: bool = False
    sources: list[str] = field(default_factory=list)
    confidence: float = 1.0
    updated_at: float = field(default_factory=time.time)


class ProRAGGraph:
    """
    Proactive, domain-partitioned knowledge graph.

    Key design decisions:
    - Nodes carry metadata: source, confidence, domain labels, timestamp
    - Edges carry conditions and negation flags (handles 'không', 'bị', 'được')
    - Contradictions are stored explicitly as CONTRADICTS edges — never silently overwritten
    - Each domain is a subgraph view; querying is scoped to relevant domains
    """

    def __init__(self):
        self.g = nx.MultiDiGraph()
        self._domain_index: dict[str, set[str]] = {}   # domain -> set of node ids

    # ── ingestion ──────────────────────────────────────────────────────────────

    def add_triple(
        self,
        subject: str,
        relation: str,
        obj: str,
        *,
        domains: list[str] | None = None,
        source: str = "",
        condition: str = "",
        negated: bool = False,
        confidence: float = 1.0,
    ) -> None:
        """Add (subject, relation, object) to the graph."""
        domains = domains or ["general"]
        now = time.time()

        for node, label in [(subject, subject), (obj, obj)]:
            if node not in self.g:
                self.g.add_node(node, meta=NodeMeta(
                    sources=[source] if source else [],
                    confidence=confidence,
                    updated_at=now,
                    domains=list(domains),
                ))
            else:
                m: NodeMeta = self.g.nodes[node]["meta"]
                if source and source not in m.sources:
                    m.sources.append(source)
                for d in domains:
                    if d not in m.domains:
                        m.domains.append(d)
                m.updated_at = now

        # Check for contradictions before adding
        existing = self._find_contradicting_edge(subject, relation, obj, negated)
        if existing:
            self._add_contradiction_note(subject, relation, obj, existing, source)
            return

        self.g.add_edge(
            subject, obj,
            relation=relation,
            meta=EdgeMeta(
                condition=condition,
                negated=negated,
                sources=[source] if source else [],
                confidence=confidence,
                updated_at=now,
            ),
        )

        # Update domain index
        for d in domains:
            self._domain_index.setdefault(d, set())
            self._domain_index[d].add(subject)
            self._domain_index[d].add(obj)

    def _find_contradicting_edge(self, subject, relation, obj, negated) -> dict | None:
        """Return existing edge data if it directly contradicts the new triple."""
        for _, tgt, data in self.g.out_edges(subject, data=True):
            if (
                tgt == obj
                and data.get("relation") == relation
                and data["meta"].negated != negated
            ):
                return data
        return None

    def _add_contradiction_note(self, subject, relation, obj, existing_data, new_source):
        """Mark contradiction — keep both claims, flag for LLM to resolve."""
        existing_data["meta"].confidence *= 0.7   # lower confidence on contested claim
        self.g.add_edge(
            subject, obj,
            relation=f"CONTRADICTS:{relation}",
            meta=EdgeMeta(
                sources=[new_source] if new_source else [],
                confidence=0.5,
                updated_at=time.time(),
            ),
        )

    # ── querying ───────────────────────────────────────────────────────────────

    def query(
        self,
        keywords: list[str],
        domains: list[str] | None = None,
        max_hops: int = 2,
        top_k: int = 40,
    ) -> list[dict]:
        """
        Return relevant triples for a set of keywords.
        Scoped to domain subgraph when domains provided.
        """
        candidate_nodes: set[str] = set()

        # Seed from keyword matches
        for kw in keywords:
            kw_lower = kw.lower()
            for node in self.g.nodes:
                if kw_lower in node.lower():
                    candidate_nodes.add(node)

        # Domain filter
        if domains:
            allowed = set()
            for d in domains:
                allowed |= self._domain_index.get(d, set())
            candidate_nodes &= allowed

        if not candidate_nodes:
            return []

        # Expand by hop count
        expanded = set(candidate_nodes)
        frontier = set(candidate_nodes)
        for _ in range(max_hops):
            next_frontier = set()
            for node in frontier:
                next_frontier |= set(self.g.successors(node))
                next_frontier |= set(self.g.predecessors(node))
            if domains:
                next_frontier &= allowed
            expanded |= next_frontier
            frontier = next_frontier

        # Collect triples
        triples = []
        for src in expanded:
            for _, tgt, data in self.g.out_edges(src, data=True):
                if tgt not in expanded:
                    continue
                m: EdgeMeta = data["meta"]
                triples.append({
                    "subject": src,
                    "relation": data["relation"],
                    "object": tgt,
                    "negated": m.negated,
                    "condition": m.condition,
                    "confidence": m.confidence,
                    "sources": m.sources,
                })

        # Sort by confidence, cap at top_k
        triples.sort(key=lambda x: x["confidence"], reverse=True)
        return triples[:top_k]

    def get_domains(self) -> list[str]:
        return list(self._domain_index.keys())

    def stats(self) -> dict:
        return {
            "nodes": self.g.number_of_nodes(),
            "edges": self.g.number_of_edges(),
            "domains": list(self._domain_index.keys()),
        }

    # ── persistence ────────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        data = nx.node_link_data(self.g)
        # NodeMeta / EdgeMeta are dataclasses — serialize manually
        edge_key = "links" if "links" in data else "edges"
        for node in data["nodes"]:
            if "meta" in node:
                node["meta"] = asdict(node["meta"])
        for link in data[edge_key]:
            if "meta" in link:
                link["meta"] = asdict(link["meta"])
        with open(path, "w") as f:
            json.dump({"graph": data, "domain_index": {k: list(v) for k, v in self._domain_index.items()}}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            raw = json.load(f)
        edge_key = "links" if "links" in raw["graph"] else "edges"
        for node in raw["graph"]["nodes"]:
            if "meta" in node:
                node["meta"] = NodeMeta(**node["meta"])
        for link in raw["graph"][edge_key]:
            if "meta" in link:
                link["meta"] = EdgeMeta(**link["meta"])
        self.g = nx.node_link_graph(raw["graph"])
        self._domain_index = {k: set(v) for k, v in raw["domain_index"].items()}
