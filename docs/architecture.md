# Architecture

## Overview

ProRAG is built around four loosely coupled components:

```
┌─────────────────────────────────────────────────────────────┐
│                         ProRAG                               │
│                                                             │
│  ┌──────────────┐    ┌──────────────────────────────────┐   │
│  │   Extractor  │───▶│     ProRAGGraph                  │   │
│  │              │    │  (domain-partitioned, persistent) │   │
│  │ text → triples│   │                                  │   │
│  └──────────────┘    └──────────────────┬───────────────┘   │
│                                         │                   │
│  ┌──────────────┐    ┌──────────────────▼───────────────┐   │
│  │   Detector   │───▶│     Pipeline                     │   │
│  │              │    │  1. detect domain                │   │
│  │ question →   │    │  2. query subgraph               │   │
│  │ domain(s)   │    │  3. format context               │   │
│  └──────────────┘    │  4. single LLM call              │   │
│                      └──────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Component details

### `graph.py` — ProRAGGraph

The core data structure. A `networkx.MultiDiGraph` where:

- **Nodes** represent entities. Each node carries `NodeMeta`:
  - `sources`: which documents mention this entity
  - `confidence`: 0–1 credibility score
  - `domains`: list of knowledge domains this entity belongs to
  - `updated_at`: timestamp of last update

- **Edges** represent relations. Each edge carries `EdgeMeta`:
  - `relation`: verb or relation phrase
  - `negated`: `True` if the relation is denied (handles "không", "not", etc.)
  - `condition`: contextual qualifier ("at 1 atm", "in 1905")
  - `sources`, `confidence`, `updated_at`

**Contradiction handling:**
When a new triple directly contradicts an existing one (same subject, relation, object, but opposite `negated` flag), ProRAG:
1. Lowers the confidence of the existing edge by 30%
2. Adds a `CONTRADICTS:relation` edge pointing to the contested object
3. Keeps both claims with their respective sources

This means the LLM always sees the full picture — no silent overwriting.

**Domain index:**
An in-memory `dict[domain → set[node_id]]` allows O(1) subgraph filtering at query time.

**Relevance & Seed Distance Query Ranking:**
When query keywords are traversed, candidate seed nodes are retrieved and expanded using BFS up to `max_hops`. Triples in the expanded subgraph are ranked using a composite key:
1. **Relevance**: Count of query keywords present in the triple's subject, relation, or object text (higher matches first).
2. **Seed Distance**: The minimum BFS distance (hop count) of the triple's nodes from the initial seed nodes (closer to seeds first).
3. **Confidence**: The triple's credibility score (higher confidence first).
This prevents critical multi-hop information from being discarded arbitrarily by flat confidence sorting.

### `extractor.py`

Converts raw text to triples via a single LLM prompt. The prompt instructs the model to:
- Return a JSON array of triples (no markdown)
- Flag negation explicitly (`negated: true`)
- Assign domain labels
- Set confidence based on hedging language in the source text
- Include conditions on relations

Text is chunked at sentence boundaries to stay within context limits.

### `detector.py`

Two-stage domain detection:
1. **Keyword scan** — a hardcoded `domain → keywords` dict covers ~80% of questions for free (zero LLM tokens)
2. **LLM fallback** — used only when keyword scan returns nothing; prompts for a JSON array of domain names

### `pipeline.py`

The query path:
1. `detect_domains(question)` — keyword scan or LLM
2. `_keywords_from_question(question)` — extracts query tokens and filters them using an extended list of grammatical structural stopwords (e.g. `originally`, `which`, `also`, `did`) to prevent noisy seed node matches.
3. `graph.query(keywords, domains=...)` — BFS expansion from seed nodes, retrieving up to `max_context_triples = 60` (default).
4. If results are sparse (less than 15 triples), fallback to query the entire graph without domain filtering (`domains=None`).
5. `_format_context(triples)` — render as bullet list with conditions, confidence, and contradiction flags
6. Single LLM call with `_ANSWER_PROMPT`
7. Append contradiction warning if any `CONTRADICTS` edges appeared in context

### `llm.py`

Provider-agnostic adapter. Reads `PRORAG_LLM_PROVIDER` env var to select:
- `groq` (default)
- `openai`
- `ollama`
- `anthropic`

All providers share the same `call_llm(prompt, model, max_tokens, system)` signature.

## Data flow example

```
Input: "Einstein did not work at the patent office."

Extractor:
  → subject: "Einstein"
  → relation: "work at"
  → object: "patent office"
  → negated: true
  → confidence: 0.9
  → domains: ["history"]

Graph (existing edge):
  Einstein --[work at]--> patent office   (confidence: 1.0, source: wiki)

Contradiction detection:
  Existing edge: negated=False
  New triple: negated=True
  → Lower existing confidence to 0.7
  → Add CONTRADICTS:work at edge (confidence: 0.5)

Graph after:
  Einstein --[work at]--> patent office          (confidence: 0.7, source: wiki)
  Einstein --[CONTRADICTS:work at]--> patent office  (confidence: 0.5, source: new_doc)

Query result includes both; LLM receives:
  - Einstein work at patent office (confidence: 0.7)
  - ⚠️ CONTRADICTS work at patent office (confidence: 0.5)
  Answer includes: "⚠️ Note: conflicting information exists — see sources."
```

## Performance characteristics

| Operation | Complexity | Notes |
|---|---|---|
| `add_triple` | O(1) amortized | Hash-based node lookup |
| `query` (with domain) | O(k·h) | k = domain nodes, h = hop depth |
| `query` (no domain) | O(n·h) | n = all nodes |
| `save` / `load` | O(n+e) | JSON serialization |
| Domain detection | O(w) or 1 LLM call | w = words in question |

For graphs up to ~100K nodes, `networkx` is sufficient. Beyond that, replace the backend with Neo4j using the same `ProRAGGraph` interface.
