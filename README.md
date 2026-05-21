# 🧠 ProRAG

**Proactive Knowledge Graph RAG** — converts text into a knowledge graph, then answers questions by traversing entity-relation paths with semantic vector retrieval.

```bash
pip install prorag
```

```python
from prorag import ProRAG

rag = ProRAG()
rag.ingest("Christopher Nolan directed Inception. Inception was filmed in Paris and Tokyo.")
result = rag.ask("Where was the film directed by Christopher Nolan filmed?")
print(result["answer"])   # → "Paris and Tokyo"
print(result["sources"])  # → ["..."]
```

---

## Why ProRAG?

| Problem with existing RAG | How ProRAG solves it |
|---|---|
| Keyword search misses synonyms and short names | **Vector similarity** matches semantically — "Ed", "AI", paraphrases all work |
| Full-text retrieval has no multi-hop reasoning | **BFS graph traversal** connects facts across multiple documents |
| GraphRAG requires 3–5 LLM calls per query | ProRAG uses **1 LLM call** per query |
| Contradictions are silently merged | Contradictions **stored explicitly** as `CONTRADICTS` edges with source tracking |
| No way to trace where an answer came from | Every answer is **traceable to specific graph edges and source documents** |

---

## How it works

```
┌──────────────────────────────────────────────────────┐
│  INGESTION                                           │
│  Raw Text → LLM Extractor (70B) → (subject, relation, object) triples │
│                                → Normalize → ProRAGGraph (NetworkX)   │
└──────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│  RETRIEVAL (2-phase vector)                          │
│                                                      │
│  Phase 1 — Entity matching                          │
│    embed(question) → cosine similarity with all nodes│
│    → top-K entity nodes as seeds                    │
│                                                      │
│  Phase 2 — Relation-guided BFS                      │
│    step_cost = 1 - sim(edge_relation, question)     │
│    → edges semantically close to question: cost ≈ 0 │
│    → irrelevant edges: cost ≈ 1 (naturally pruned)  │
│    → max_hops = 3                                   │
└──────────────────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│  GENERATION                                          │
│  Top-K context triples → LLM (8B) → short-phrase answer │
└──────────────────────────────────────────────────────┘
```

**What makes retrieval different from standard RAG:**
- **Phase 1** seeds on semantic similarity, not exact keyword match — short names, abbreviations and synonyms all resolve correctly
- **Phase 2** uses the question's semantics to guide *which paths to follow*, not just which nodes to start from — the correct branch "lights up" automatically

---

## Quick start

### 1. Install

```bash
pip install prorag
```

### 2. Set your API key

```bash
export GROQ_API_KEY=your_key_here   # free tier at console.groq.com
```

### 3. Python API

```python
from prorag import ProRAG

rag = ProRAG()

# Ingest — build the knowledge graph
rag.ingest("Marie Curie was born in Warsaw in 1867.")
rag.ingest("Marie Curie won the Nobel Prize in Physics in 1903 and in Chemistry in 1911.")
rag.ingest("The Nobel Prize in Physics is awarded by the Royal Swedish Academy of Sciences.")

# Ask — 1 LLM call, grounded in graph paths
result = rag.ask("Who awarded Marie Curie her physics prize?")
print(result["answer"])       # → "Royal Swedish Academy of Sciences"
print(result["triples_used"]) # → 5
print(result["sources"])      # → ["..."]

# Persist
rag.save("my_graph.json")
rag.load("my_graph.json")
```

### 4. CLI

```bash
# Ingest a document
prorag ingest docs/manual.txt --graph my_graph.json

# Ask a question
prorag ask "What is the refund policy?" --graph my_graph.json

# Interactive session
prorag interactive --graph my_graph.json

# Graph statistics
prorag stats --graph my_graph.json
```

---

## Key design decisions

### Proactive extraction
ProRAG extracts knowledge **at ingest time**, not at query time. When a query arrives, the graph is already complete — no on-demand LLM extraction.

### 2-phase vector retrieval
Standard BFS blindly expands all neighbors with equal cost. ProRAG's BFS uses **semantic edge scoring**: `step_cost = 1 - cosine_similarity(edge_relation, question)`. Relations close in meaning to the question get low cost and are traversed first. Irrelevant branches are naturally deprioritized without any hard cutoffs.

### Contradiction handling
When two sources disagree, ProRAG stores **both claims** with a `CONTRADICTS:relation` edge. The LLM receives the full picture — including which source says what — and can reason about the conflict rather than silently picking one side.

### One LLM call per query
The graph and vector retrieval do the heavy lifting. By the time the question reaches the LLM, context is already curated, filtered, and ranked. No multi-hop LLM reasoning chains needed.

### Explainable by design
Every answer is traced to specific graph edges and source documents. Debugging is `find wrong edge → fix extraction` instead of `try another prompt`.

---

## Benchmark

Evaluated on **HotpotQA** (multi-hop QA dataset) comparing ProRAG vs. Naive RAG:

| | Naive RAG | ProRAG |
|---|---|---|
| Extraction model | — | `llama-3.3-70b-versatile` |
| QA model | `llama-3.1-8b-instant` | `llama-3.1-8b-instant` |
| Retrieval | keyword + top-3 chunks | 2-phase vector BFS |

> Full results in [`docs/benchmark.md`](docs/benchmark.md)

---

## Configuration

```python
from prorag import ProRAG

# Separate extractor and QA models
rag = ProRAG(model="llama-3.1-8b-instant")   # QA model

# Use a different embedding model
from prorag.embeddings import EmbeddingStore
store = EmbeddingStore(model_name="all-mpnet-base-v2")  # higher accuracy, larger
```

---

## Roadmap

- [x] Core graph engine (NetworkX MultiDiGraph)
- [x] LLM-based triple extraction with negation, passive voice, nested facts
- [x] Contradiction detection and explicit storage
- [x] 2-phase vector retrieval (entity cosine seed + relation-guided BFS)
- [x] CLI + Python API
- [x] HotpotQA benchmark script (`scripts/run_benchmark.py`)
- [ ] GraphRAG baseline for benchmark comparison
- [ ] Async ingestion pipeline
- [ ] PDF / Markdown ingestion
- [ ] Graph visualization UI
- [ ] Fine-tuned extractor for higher extraction quality

---

## Contributing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Issues and PRs welcome.

---

## License

MIT
