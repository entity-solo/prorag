# ProRAG

**Proactive Knowledge Graph RAG** for entity-centric question answering.

ProRAG ingests raw text into an entity graph, resolves references like `it` / `the company` before graph writes, and answers questions with **entity-first, relation-guided retrieval** over graph evidence.

```bash
pip install prorag
```

```python
from prorag import ProRAG

rag = ProRAG()
rag.ingest("Christopher Nolan directed Inception. Inception was filmed in Paris and Tokyo.")
result = rag.ask("Where was the film directed by Christopher Nolan filmed?")
print(result["answer"])
print(result["sources"])
```

---

## Why ProRAG?

| Problem with standard RAG | How ProRAG approaches it |
|---|---|
| Chunks hide entity-to-entity structure | Stores facts in an explicit entity graph |
| Pronouns and generic mentions pollute retrieval | Resolves coreference during ingest before graph writes |
| Multi-hop questions need more than nearest chunks | Retrieves around seed entities, then follows relevant relations and short paths |
| Answers drift when too much context is dumped into the model | Uses slot-guided evidence selection (`who/what/when/where/why/how/how_many`) |
| Contradictions are easy to miss | Keeps contradiction markers in the graph and surfaces them in answer context |

---

## How It Works

```text
INGEST
Raw text
-> chunk with sentence overlap
-> LLM extracts mention-level facts
-> resolve pronouns / generic references
-> canonicalize entities
-> validate triples
-> write to NetworkX MultiDiGraph

RETRIEVE
Question
-> detect seed entities
-> detect answer slot (5W + how/how_many)
-> infer relation cues
-> retrieve candidate triples from the graph
-> rerank by entity match, relation match, slot match
-> assemble short evidence paths

GENERATE
Top graph evidence
-> one LLM call
-> concise grounded answer
```

---

## Quick Start

### 1. Install

```bash
pip install prorag
```

### 2. Set your API key

```bash
export GROQ_API_KEY=your_key_here
```

### 3. Python API

```python
from prorag import ProRAG

rag = ProRAG()

rag.ingest("Marie Curie was born in Warsaw in 1867.")
rag.ingest("Marie Curie won the Nobel Prize in Physics in 1903.")
rag.ingest("The Nobel Prize in Physics is awarded by the Royal Swedish Academy of Sciences.")

result = rag.ask("Who awarded Marie Curie her physics prize?")
print(result["answer"])
print(result["triples_used"])
print(result["sources"])

rag.save("my_graph.json")
rag.load("my_graph.json")
```

### 4. CLI

```bash
prorag ingest docs/manual.txt --graph my_graph.json
prorag ask "What is the refund policy?" --graph my_graph.json
prorag interactive --graph my_graph.json
prorag stats --graph my_graph.json
```

---

## Current Architecture

### Ingest

Ingest is no longer "extract raw triples and hope for the best".

The current pipeline:

- extracts **mention-level facts**
- resolves pronouns and generic references from local context
- canonicalizes entities before graph writes
- drops unresolved facts instead of storing nodes like `it` or `the company`
- keeps metadata such as `negated`, `condition`, `confidence`, and `sources`

This makes the graph much cleaner for downstream retrieval.

### Retrieval

Retrieval is **entity-first**:

1. detect graph entities most relevant to the question
2. detect the answer slot: `who`, `what`, `when`, `where`, `why`, `how`, or `how_many`
3. infer relation cues from the question
4. retrieve candidate graph triples
5. rerank evidence using entity match, relation match, answer-slot match, distance, and confidence
6. prefer short connected paths over isolated triples for multi-hop questions

### Generation

Generation is still intentionally simple:

- format top evidence triples into text context
- make one LLM call
- return a concise answer plus provenance

---

## Configuration

```python
from prorag import ProRAG

# The current public API uses one model name for ingest and answer generation.
rag = ProRAG(model="llama-3.3-70b-versatile")

# Optional: use a different embedding model for retrieval
from prorag.embeddings import EmbeddingStore
store = EmbeddingStore(model_name="all-mpnet-base-v2")
```

Note: benchmark scripts may experiment with different ingest/QA model combinations, but the main runtime API currently exposes a single `model` parameter.

---

## Project Layout

```text
prorag/
├── __init__.py       public API facade
├── cli.py            CLI entrypoint
├── detector.py       keyword extraction helpers
├── embeddings.py     embedding store and cache
├── entity_utils.py   entity normalization and unresolved-reference guards
├── extractor.py      mention-level ingest pipeline
├── graph.py          graph storage + candidate retrieval engine
├── llm.py            provider-agnostic LLM adapter
├── pipeline.py       answer retrieval + reranking + path assembly
└── server.py         FastAPI wrapper

docs/
├── architecture.md   as-built system architecture
└── benchmark.md      benchmark notes and historical results

tests/
└── test_graph.py     graph, ingest, and retrieval tests
```

---

## Status

- [x] Entity graph core on `NetworkX MultiDiGraph`
- [x] Mention-level ingest with coreference cleanup
- [x] Graph write guards for unresolved references
- [x] Semantic candidate retrieval
- [x] 5W-guided reranking
- [x] Path-level evidence assembly
- [x] CLI + Python API + FastAPI server
- [ ] Answer-type validation after generation
- [ ] Stronger contradiction model
- [ ] Richer entity linking / canonicalization
- [ ] Graph visualization UI

---

## Benchmarks

Benchmark notes and historical runs are in [docs/benchmark.md](docs/benchmark.md).

Some benchmark scripts in this repo predate the current ingest/retrieval pipeline and may use experimental settings for comparison. Treat them as evaluation utilities, not exact documentation of the default runtime path.

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
