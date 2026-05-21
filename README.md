# ProRAG — Proactive GraphRAG

Minimal entity-graph RAG for grounded question answering.

ProRAG ingests text into an entity graph, resolves simple references during ingest, retrieves graph evidence with entity-first ranking, and answers with one LLM call.

## About

ProRAG is a stripped-down runtime focused on the smallest useful surface:

- ingest raw text into an entity graph
- clean up pronouns and generic references before graph writes
- retrieve evidence with entity-first, relation-guided ranking
- answer from graph context with a single LLM call

The repository intentionally keeps only the core library, CLI, and tests.

## Install

```bash
pip install -e .
```

Set a provider key before using the default Groq backend:

```bash
export GROQ_API_KEY=your_key_here
```

## Quickstart

```python
from prorag import ProRAG

rag = ProRAG()
rag.ingest("Christopher Nolan directed Inception. Inception was filmed in Paris and Tokyo.")

result = rag.ask("Where was the film directed by Christopher Nolan filmed?")
print(result["answer"])
print(result["sources"])
```

## CLI

```bash
prorag ingest notes.txt --graph graph.json
prorag ask "Who directed Inception?" --graph graph.json
prorag interactive --graph graph.json
prorag stats --graph graph.json
```

## What remains in this repo

- `prorag/`: core library and CLI
- `tests/`: regression tests for ingest, graph behavior, and retrieval
- `pyproject.toml`: package metadata

Everything else has been removed to keep the repo focused on the minimal runtime.

## Development

```bash
pytest -q --basetemp C:\tmp\prorag-pytest -o cache_dir=C:\tmp\prorag-pytest-cache
python -m compileall prorag tests
```

## License

Apache License 2.0. See `LICENSE`.
