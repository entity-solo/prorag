# ProRAG System Architecture

ProRAG converts raw text into a **flat knowledge graph** of entity-relation-entity triples, then answers questions by traversing that graph with BFS — no domain tags, no hierarchical categories.

```
┌─────────────────────────────────────────────────────────────┐
│                       1. INGESTION                          │
│                                                             │
│  [Raw Text]                                                 │
│       │                                                     │
│       ▼                                                     │
│  [LLM Extractor] ──▶ Extracts explicit & nested facts      │
│  (llama-3.3-70b)     as (subject, relation, object) triples │
│       │                                                     │
│       ▼                                                     │
│  [Normalizer] ──▶ lowercase, strip, deduplicate             │
│       │                                                     │
│       ▼                                                     │
│  [ProRAGGraph] ──▶ In-memory MultiDiGraph (NetworkX)        │
│                    Nodes = entities                         │
│                    Edges = relations + metadata             │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       2. RETRIEVAL                          │
│                                                             │
│  [Question]                                                 │
│       │                                                     │
│       ▼                                                     │
│  [Keyword Extractor] ──▶ Stopword-filtered tokens           │
│       │                                                     │
│       ▼                                                     │
│  [Seed Nodes] ──▶ Graph nodes matching any keyword          │
│       │                                                     │
│       ▼                                                     │
│  [BFS Traversal] ──▶ Uniform hop cost (1.0 per edge)        │
│                      max_hops = 2, top_k = 60               │
│       │                                                     │
│       ▼                                                     │
│  [Context Triples] sorted by:                               │
│    1. Keyword relevance (desc)                              │
│    2. Hop distance (asc)                                    │
│    3. Confidence score (desc)                               │
└─────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                       3. GENERATION                         │
│                                                             │
│  [Triples as Context] + [Contradiction Warnings]            │
│       │                                                     │
│       ▼                                                     │
│  [LLM QA Model] ──▶ Short-phrase answer (name/date/yes/no)  │
│  (llama-3.1-8b)      grounded strictly in graph context     │
└─────────────────────────────────────────────────────────────┘
```

---

## 1. Ingestion

### LLM Extractor (`prorag/extractor.py`)
The extractor calls a strong LLM (`llama-3.3-70b-versatile` by default) to decompose a text chunk into a flat list of triples:

```json
[
  { "subject": "tim cook", "relation": "is ceo of", "object": "apple", "negated": false, "confidence": 1.0 },
  { "subject": "tim cook", "relation": "announced", "object": "iphone 17", "negated": false, "confidence": 1.0 }
]
```

Key extraction rules:
- **Language consistency**: Entities and relations stay in the same language as the input text.
- **Nested facts**: Implicit facts in titles and appositive phrases are extracted explicitly (e.g., "CEO Apple Tim Cook" → two triples).
- **Negation**: Explicit denial is stored with `negated: true` instead of being dropped.
- **No tags**: The extractor does **not** output structural tags, hierarchical categories, or domain labels.

### Graph Engine (`prorag/graph.py`)
- Uses **NetworkX `MultiDiGraph`** — nodes are entity strings, edges carry `(relation, metadata)`.
- All entities are normalized to `strip().lower()` on insert to prevent duplicate nodes.
- **Contradiction detection**: When a new triple directly contradicts an existing one (same subject/relation/object but opposite `negated`), a `CONTRADICTS:relation` edge is added and the existing edge's confidence is downweighted by `×0.7`.

---

## 2. Retrieval (`prorag/pipeline.py` + `prorag/graph.py`)

### Keyword Extraction (`prorag/detector.py`)
Stopword-filtered tokens (≥3 characters) from the question are used as seed keywords — no LLM call needed, runs in sub-milliseconds.

### BFS Traversal
Starting from all graph nodes that match any keyword, a Dijkstra-like BFS expands up to `max_hops = 2` hops with **uniform edge cost = 1.0** across the entire flat graph (no domain filters, no boundary penalties).

Triples are ranked by:
1. **Keyword relevance** — how many query keywords appear in the triple
2. **Hop distance** — triples closer to seed nodes rank higher
3. **Confidence** — higher factual credibility ranks higher

---

## 3. Generation (`prorag/pipeline.py`)

A single LLM call receives the top-K context triples as a plain-text list and must respond with a **short-phrase answer only** (entity name, date, or "yes"/"no"). Contradictions are flagged in the context with a `⚠️ CONTRADICTS` prefix.

---

## Model Split (Benchmark)

| Role | Model |
|---|---|
| Triple Extraction (Ingestion) | `llama-3.3-70b-versatile` |
| Question Answering (QA) | `llama-3.1-8b-instant` |

Using a stronger model for extraction ensures high-quality graph construction while the 8B model is the evaluation target for answer generation.

---

## File Structure

```
prorag/
├── extractor.py   — LLM-based triple extraction from raw text
├── graph.py       — Knowledge graph engine (NetworkX MultiDiGraph)
├── pipeline.py    — Query pipeline: keywords → BFS → LLM answer
├── detector.py    — Keyword extractor (stopword filter)
├── llm.py         — Groq API wrapper with retry/backoff
├── cli.py         — Command-line interface
└── __init__.py    — ProRAG public API

scripts/
├── run_benchmark.py            — ProRAG vs Naive RAG benchmark (HotpotQA)
├── run_benchmark_topic.py      — Topic-scoped benchmark variant
├── run_benchmark_topic_large.py
└── download_datasets.py        — Dataset downloader

data/
├── hotpot_dev_distractor_v1.json   — HotpotQA evaluation dataset
└── extracted_triples_cache.json    — MD5-keyed extraction cache (avoids re-calling LLM)

tests/
└── test_graph.py   — Unit tests (graph engine, extractor, contradiction, persistence)
```
