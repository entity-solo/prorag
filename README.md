# 🧠 ProRAG

**Proactive Knowledge Graph RAG** — continuous learning, no hallucinations from known facts, 5x cheaper than GraphRAG.

```
pip install prorag
```

```python
from prorag import ProRAG

rag = ProRAG()
rag.ingest("Einstein developed the theory of relativity in 1905 in Bern.")
result = rag.ask("Where did Einstein develop relativity?")
# → "Einstein developed the theory of relativity in Bern, Switzerland."
```

---

## Why ProRAG?

| Problem with existing RAG | How ProRAG solves it |
|---|---|
| Graph is built *on demand* — slow, incomplete | Graph built **proactively** as you ingest — always ready |
| 3–5 LLM calls per query (GraphRAG) | **1 LLM call** per query |
| Full graph search — noisy, expensive | **Domain-partitioned** subgraph search |
| Knowledge update = retrain or rebuild index | Update = **add a node/edge** (milliseconds, no downtime) |
| Contradictions silently merged | Contradictions **stored explicitly** with sources |
| Black box — can't explain an answer | Every answer is **traceable to graph paths** |

### Benchmark vs GraphRAG (HotpotQA, 100 questions)

| Metric | GraphRAG | ProRAG | Improvement |
|---|---|---|---|
| Hallucination rate | ~18% | ~4% | **4.5× lower** |
| LLM calls / query | 3–5 | **1** | **3–5× cheaper** |
| Latency | ~4.2s | ~1.3s | **3× faster** |
| Real-time knowledge update | ❌ | **✅** | — |
| Explainable output | Partial | **Full** | — |

> Benchmark methodology: [docs/benchmark.md](docs/benchmark.md)

---

## How it works

```
Document / text
      ↓
[Proactive Extractor]        ← extracts triples continuously as you ingest
      ↓
[Domain-Partitioned Graph]   ← science / medicine / law / finance / tech / ...
      ↑↓
Query → detect domain → search subgraph → inject context → LLM (once) → Answer
                                                              ↑
                                              sources + confidence + contradictions
```

**Linguistic support out of the box:**
- Negation: *"không / chưa / not"* → `negated=True` edge flag
- Passive voice: *"bị / được / was"* → automatically flips subject/object
- Conditions: *"at 1 atm"*, *"in 1905"* → stored as edge metadata
- Contradictions: explicitly stored as `CONTRADICTS:relation` edges with separate source tracking

---

## Quick start

### 1. Install

```bash
pip install prorag
# or for local models:
pip install "prorag[ollama]"
```

### 2. Set your API key

```bash
export GROQ_API_KEY=your_key_here   # free tier available at console.groq.com
```

Supported providers: Groq (default), OpenAI, Anthropic, Ollama (local)

```bash
# Use Ollama locally (no API key needed)
export PRORAG_LLM_PROVIDER=ollama
# make sure ollama is running: ollama serve
```

### 3. Python API

```python
from prorag import ProRAG

rag = ProRAG()

# Ingest — add knowledge to the graph
rag.ingest("Water boils at 100°C at standard atmospheric pressure (1 atm).")
rag.ingest("At high altitudes, water boils below 100°C due to lower pressure.")
rag.ingest_file("my_documents/company_policy.txt")

# Ask — one LLM call, grounded answer
result = rag.ask("At what temperature does water boil?")

print(result["answer"])
# → "Water boils at 100°C at standard atmospheric pressure.
#    At high altitudes, it boils at lower temperatures."

print(result["sources"])    # ["my_file.txt"]
print(result["domains"])    # ["science"]
print(result["triples_used"])  # 4

# Persist — save graph to disk (milliseconds)
rag.save("my_graph.json")

# Load — restore later
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

### 5. Streamlit demo

```bash
pip install "prorag[demo]"
streamlit run examples/demo_app.py
```

---

## Key design decisions

### Proactive extraction
Unlike RAG systems that build context at query time, ProRAG extracts knowledge **as you ingest**. When a query arrives, the graph is already complete — no on-demand processing.

### Domain partitioning
Each node is tagged with one or more domains (`science`, `medicine`, `law`, `finance`, `tech`, `geography`, `general`). Queries are scoped to relevant subgraphs — faster retrieval, less noise, lower token cost.

### Contradiction handling
When two sources disagree, ProRAG stores **both claims** with a `CONTRADICTS:relation` edge and tracks each source independently. The LLM receives the full picture and can reason about conflicting information rather than silently picking one.

### One LLM call per query
The graph does the heavy lifting. By the time the question reaches the LLM, the context is already curated, filtered, and formatted. No multi-hop LLM traversal needed.

### Explainable by design
Every answer can be traced back to specific nodes, edges, and source documents. Debugging is `find wrong node → fix edge` instead of `try another prompt`.

---

## Configuration

```python
from prorag import ProRAG

# Use a different model
rag = ProRAG(model="mixtral-8x7b-32768")

# Use OpenAI
import os
os.environ["PRORAG_LLM_PROVIDER"] = "openai"
os.environ["OPENAI_API_KEY"] = "sk-..."
rag = ProRAG(model="gpt-4o-mini")

# Use local Ollama
os.environ["PRORAG_LLM_PROVIDER"] = "ollama"
rag = ProRAG(model="llama3")
```

---

## Roadmap

- [x] Core graph engine with domain partitioning
- [x] Proactive extractor (LLM-based triple extraction)
- [x] Negation and passive voice handling (Vietnamese + English)
- [x] Contradiction detection and storage
- [x] CLI + Streamlit demo
- [ ] Async ingestion pipeline
- [ ] PDF / Markdown ingestion
- [ ] REST API server (`prorag serve`)
- [ ] Graph visualization UI
- [x] HotpotQA / MuSiQue benchmark scripts (implemented in `scripts/run_benchmark.py`)
- [x] Topic-based & Superpowers demo scripts (implemented in `scripts/run_benchmark_topic_large.py` and `scripts/demo_superpowers.py`)
- [ ] Fine-tuned extractor for domain-specific triples
- [ ] Multi-tenant graph isolation

---

## Contributing

Issues and PRs welcome. To run tests:

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT
