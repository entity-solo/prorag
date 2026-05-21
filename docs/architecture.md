# ProRAG System Architecture

This document describes the complete architecture of ProRAG, covering the core open-source graph engine, query routing optimizations, and the SaaS multi-tenant control plane.

```
                  ┌──────────────────────────────────────────────┐
                  │               1. INGESTION                   │
                  │  [Raw text]                                  │
                  │       │                                      │
                  │       ▼                                      │
                  │  [LLM Extractor] ──▶ Extracts explicit &     │
                  │                      nested facts            │
                  │       │                                      │
                  │       ▼                                      │
                  │  [Normalizer] ──▶ lowercase, trim, & merge   │
                  │       │                                      │
                  │       ▼                                      │
                  │  [ProRAGGraph] ──▶ In-memory MultiDiGraph    │
                  └───────┬──────────────────────────────────────┘
                          │ (persist on disk / RAM)
                          ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                        2. RETRIEVAL                           │
 │  [Question]                                                   │
 │       │                                                       │
 │       ├───────────────┐                                       │
 │       ▼               ▼                                       │
 │  [Keywords]    [Graph Route] ──▶ Zero-latency active tags     │
 │       │               │          lookup (no LLM latency)      │
 │       ▼               │                                       │
 │  [Seed Nodes] ◀───────┘                                       │
 │       │                                                       │
 │       ▼                                                       │
 │  [Dijkstra BFS] ──▶ Traverses paths up to 2 hops              │
 │                     - Crossing-boundary penalty (+1.0 hop)    │
 │                     - Taxonomy matching boost (-0.5 dist)     │
 │       │                                                       │
 │       ▼                                                       │
 │  [Context Triples] (Sorted by relevance & effective distance) │
 └───────┬───────────────────────────────────────────────────────┘
         │
         ▼
 ┌───────────────────────────────────────────────────────────────┐
 │                       3. GENERATION                           │
 │  [Clean Triples] + [Contradictions Warn]                      │
 │       │                                                       │
 │       ▼                                                       │
 │  [LLM Generator] ──▶ Synthesizes final answer with sources     │
 └───────────────────────────────────────────────────────────────┘
```

---

## 1. Core Ingestion & Representation

### Entity Normalization
To prevent duplicated nodes and broken query paths, all subjects and objects are stripped and lowercased (`strip().lower()`) during ingestion. This merges identical entities (e.g. `"Paracetamol"` and `"paracetamol"`) while keeping distinct technical terms separate.

### Nested Fact Extraction (Modifiers & Appositives)
The LLM extractor is prompted to extract both the main action and any nested background facts hidden inside noun modifiers, titles, or appositive phrases. 
- *Input*: `"CEO Apple Tim Cook cho ra mắt iPhone 17"`
- *Extracted Triples*:
  1. `(tim cook, là ceo của, apple)`
  2. `(tim cook, cho ra mắt, iphone 17)`
This ensures the physical graph structure holds the necessary bridges to support multi-hop questions (e.g., matching the company to the product via the CEO) naturally.

### Self-Building Hierarchical Taxonomy
Predefined domains are replaced by self-building, path-like hierarchical structural tags (e.g. `bo_luat_lao_dong_2019/chuong_3/dieu_49/khoan_2` or `paracetamol/chong_chi_dinh`) inferred by the LLM from the document layout.

---

## 2. Zero-Latency Routing & BFS Retrieval

### Graph-Based Routing
Rather than calling an LLM to classify query domains, ProRAG extracts keywords from the question, matches them against nodes in the graph, and aggregates the active structural tags (domains) associated with those matching nodes. This completes in sub-milliseconds (down from 8s LLM latency).

### Scoped Dijkstra-like BFS Traversal
BFS starting from seed nodes expands paths up to `max_hops = 2` with customized edge weights:
- **Explicit Edge**: Base cost of `1.0` hop.
- **Crossing-Boundary Penalty**: If the traversal hops between nodes that do not share active query domains, a penalty of `+1.0` is added to the hop cost. This prevents the search from wandering into irrelevant subgraphs.
- **Taxonomy Match Boost**: If both endpoints of a triple match the query's active taxonomy tags, its effective distance is reduced by `-0.5`, prioritizing it during context packaging.

### Query sorting
Context triples are packaged and sorted by:
1. **Keyword Relevance**: The number of query keywords appearing in the triple.
2. **Effective Distance**: The shortest distance from seeds, incorporating boundary penalties and taxonomy boosts.
3. **Confidence**: The factual credibility score.

---

## 3. Multi-Tenant SaaS Control Plane

ProRAG's SaaS wrapper provides commercial security, isolation, and billing management:

```
Developer / Client
       │ (API Key / JWT)
       ▼
[FastAPI Gateway] ──▶ [Quota & Rate Limiter] ──▶ [Database (SQLite)]
       │              (Checks user limit)         - User Accounts
       ▼                                          - API Keys / Logs
[User Graph Scope]                                - Subscriptions
       │
       ▼
[Isolated JSON File] (data/graphs/{user_id}/{graph_id}.json)
```

1. **Authentication & Isolation**: Requests are authenticated via JWT or secure developer API keys. Node operations and graphs are scoped to isolated physical directories (`data/graphs/{user_id}/`).
2. **Request Interceptor Logging**: A FastAPI middleware intercepts traffic, records method, endpoint, status code, and latency, and persists metrics to SQLite while ignoring administrative and public endpoints.
3. **Quota & Rate Limiting**: The middleware tracks monthly request usage against the user's plan limit, rejecting exceeding requests with `429 Too Many Requests`.
4. **Billing & Subscriptions**: Real-time payment simulation updates user plans (e.g., Free to Pro) to upgrade quotas instantly.
