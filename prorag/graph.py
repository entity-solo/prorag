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
        # Normalize strings to strip whitespaces and use lowercase
        if subject is None or relation is None or obj is None:
            return
        if isinstance(subject, list):
            subject = ", ".join(str(x) for x in subject)
        elif not isinstance(subject, str):
            subject = str(subject)

        if isinstance(relation, list):
            relation = ", ".join(str(x) for x in relation)
        elif not isinstance(relation, str):
            relation = str(relation)

        if isinstance(obj, list):
            obj = ", ".join(str(x) for x in obj)
        elif not isinstance(obj, str):
            obj = str(obj)

        subject = subject.strip().lower()
        relation = relation.strip().lower()
        obj = obj.strip().lower()
        if not subject or not relation or not obj:
            return
        domains = [d.strip().lower() for d in (domains or ["general"]) if d]
        
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

        # Domain filter (hierarchical prefix matching on index)
        allowed = set()
        if domains:
            for query_d in domains:
                query_d_lower = query_d.strip().lower()
                query_d_slash = query_d_lower if query_d_lower.endswith("/") else query_d_lower + "/"
                for index_d, nodes in self._domain_index.items():
                    if index_d == query_d_lower or index_d.startswith(query_d_slash):
                        allowed |= nodes
            candidate_nodes &= allowed

        if not candidate_nodes:
            return []

        # Dijkstra-like BFS traversal with crossing-boundary penalty
        # Initialize distances for candidate_nodes to 0.0
        distances = {node: 0.0 for node in candidate_nodes}
        import heapq
        queue = [(0.0, node) for node in candidate_nodes]
        heapq.heapify(queue)

        def get_top_levels(n):
            meta = self.g.nodes[n].get("meta")
            if not meta or not meta.domains:
                return {"general"}
            return {d.split("/")[0] for d in meta.domains}

        while queue:
            dist, node = heapq.heappop(queue)
            if dist > distances[node]:
                continue
            if dist >= max_hops:
                continue

            neighbors = set(self.g.successors(node)) | set(self.g.predecessors(node))
            node_top_levels = get_top_levels(node)
            for neighbor in neighbors:
                neighbor_top_levels = get_top_levels(neighbor)
                # If query domains are scoped, apply a +1.0 penalty if the neighbor node
                # does not belong to any of the query's active top-level domains.
                # Otherwise, check if the current node and neighbor share any top-level domain.
                if domains:
                    query_top_levels = {d.split("/")[0] for d in domains}
                    shares_query = bool(neighbor_top_levels & query_top_levels)
                    step_cost = 1.0 if shares_query else 2.0
                else:
                    shares_top = bool(node_top_levels & neighbor_top_levels)
                    step_cost = 1.0 if shares_top else 2.0

                new_dist = dist + step_cost
                if new_dist <= max_hops and (neighbor not in distances or new_dist < distances[neighbor]):
                    distances[neighbor] = new_dist
                    heapq.heappush(queue, (new_dist, neighbor))
        expanded = set(distances.keys())

        # Helper to compute keyword relevance score
        def get_relevance(triple):
            score = 0
            text = f"{triple['subject']} {triple['relation']} {triple['object']}".lower()
            for kw in keywords:
                if kw.lower() in text:
                    score += 1
            return score

        # Helper to check if a node matches any query domains (prefix check)
        def matches_query_domains(n, query_domains):
            if not query_domains:
                return True
            meta = self.g.nodes[n].get("meta")
            if not meta or not meta.domains:
                return False
            for query_d in query_domains:
                query_d_lower = query_d.strip().lower()
                query_d_slash = query_d_lower if query_d_lower.endswith("/") else query_d_lower + "/"
                for d in meta.domains:
                    if d == query_d_lower or d.startswith(query_d_slash):
                        return True
            return False

        # Collect triples
        triples = []
        for src in expanded:
            for _, tgt, data in self.g.out_edges(src, data=True):
                if tgt not in expanded:
                    continue
                m: EdgeMeta = data["meta"]
                dist = min(distances.get(src, float(max_hops)), distances.get(tgt, float(max_hops)))
                
                # Apply taxonomy match boost: reduce effective distance by 0.5
                # if either subject or object matches the query domains
                effective_dist = dist
                if domains:
                    src_match = matches_query_domains(src, domains)
                    tgt_match = matches_query_domains(tgt, domains)
                    if src_match and tgt_match:
                        effective_dist -= 0.5
                
                triples.append({
                    "subject": src,
                    "relation": data["relation"],
                    "object": tgt,
                    "negated": m.negated,
                    "condition": m.condition,
                    "confidence": m.confidence,
                    "sources": m.sources,
                    "distance": dist,
                    "effective_distance": effective_dist,
                })

        # Sort by: 1. Relevance (desc), 2. Effective Distance (asc), 3. Confidence (desc)
        triples.sort(key=lambda x: (-get_relevance(x), x["effective_distance"], -x["confidence"]))
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
