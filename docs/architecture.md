# ProRAG System Architecture

This document describes the **current code path**, not older design drafts.

At a high level, ProRAG is an **entity-graph QA system** with three major stages:

1. ingest raw text into a clean graph
2. retrieve graph evidence with entity-first, slot-guided search
3. generate one grounded answer from the selected evidence

---

## Overview

```text
Raw text
-> mention-level extraction
-> coreference resolution
-> entity canonicalization
-> validated graph writes

Question
-> seed entity detection
-> answer-slot detection (5W + how/how_many)
-> relation cue inference
-> candidate retrieval
-> triple reranking
-> path-level evidence assembly

Evidence
-> one LLM call
-> concise grounded answer
```

---

## 1. Ingest

### Entry points

- `ProRAG.ingest()` and `ProRAG.ingest_file()` in [`prorag/__init__.py`](../prorag/__init__.py)
- `ingest_text()` and `ingest_file()` in [`prorag/extractor.py`](../prorag/extractor.py)

### What ingest does now

The current ingest pipeline is:

```text
text
-> chunk with sentence overlap
-> LLM extracts mention-level facts
-> resolve pronouns / generic mentions
-> canonicalize entities
-> validate triples
-> graph.add_triple(...)
```

### Mention-level extraction

The extractor no longer assumes every output is already clean canonical graph data.

It first asks the LLM for fact objects shaped like:

```json
{
  "subject_mention": "It",
  "subject": "",
  "relation": "was announced in",
  "object_mention": "September",
  "object": "september",
  "negated": false,
  "condition": "",
  "confidence": 1.0
}
```

This lets the pipeline distinguish:

- raw surface mentions from the text
- canonical entity values when they are already clear
- unresolved references that still need cleanup

### Coreference resolution

After extraction, the ingest pipeline resolves references such as:

- `it`
- `he / she / they`
- `the company`
- `the film`
- Vietnamese variants like `nó`, `công ty này`

Resolution is done in two layers:

1. lightweight heuristics based on recent entities
2. LLM fallback for ambiguous generic mentions

If a reference cannot be resolved confidently enough into a canonical entity, the fact is dropped before graph write.

### Canonicalization and validation

Before a triple reaches the graph:

- entities are normalized
- unresolved pronouns are rejected
- malformed triples are discarded

This is backed by a second safety layer in [`prorag/graph.py`](../prorag/graph.py), so even if upstream extraction slips, the graph still refuses unresolved references.

---

## 2. Graph Core

### Storage model

The graph implementation lives in [`prorag/graph.py`](../prorag/graph.py).

- graph type: `networkx.MultiDiGraph`
- node: entity string
- edge: relation + metadata

Edge metadata currently includes:

- `condition`
- `negated`
- `sources`
- `confidence`
- `updated_at`

Node metadata currently includes:

- `sources`
- `confidence`
- `updated_at`
- `domains`

### Note on domains

The codebase still contains domain/tag support and related tests, but the main retrieval path is no longer built around domain routing.

Current runtime behavior is primarily **entity-graph driven**, with domain metadata acting as optional residual structure rather than the core design.

### Contradictions

When a new triple directly contradicts an existing one, the graph:

- downweights the existing edge confidence
- adds a `CONTRADICTS:<relation>` edge

This is a lightweight contradiction marker, not yet a full two-claim contradiction model.

---

## 3. Retrieval

### Entry point

Retrieval orchestration lives in [`prorag/pipeline.py`](../prorag/pipeline.py), mainly through `retrieve_evidence()` and `answer()`.

### Retrieval strategy

The retrieval path is now:

```text
question
-> detect seed entities
-> detect answer slot
-> infer relation cues
-> retrieve candidate triples
-> rerank by entity/relation/slot signals
-> assemble connected evidence paths
```

### 3.1 Seed entity detection

Seed entity detection combines:

- lexical overlap between question terms and graph nodes
- embedding similarity between the question and graph nodes

The goal is to answer:

> Which graph entities is this question really about?

### 3.2 Answer-slot detection

The question is mapped into a soft answer slot:

- `who`
- `what`
- `when`
- `where`
- `why`
- `how`
- `how_many`

This is used as retrieval guidance, not as a hard schema.

### 3.3 Relation cue inference

After likely seed entities are identified, the pipeline removes those mentions from the question and keeps the remaining relation-bearing cues.

Examples:

- `Where was the film directed by Christopher Nolan filmed?`
  - seed entity signal: `christopher nolan`, `inception`
  - relation cues: `directed`, `filmed`
  - slot: `where`

- `When was the film directed by Christopher Nolan released?`
  - relation cues: `directed`, `released`
  - slot: `when`

### 3.4 Candidate retrieval

Candidate retrieval still uses the graph engine in [`prorag/graph.py`](../prorag/graph.py):

- primary path: `query_vector()`
- fallback path: `query()` when embeddings are unavailable

`query_vector()` performs:

1. semantic entity seeding
2. relation-guided graph expansion with a semantic step cost
3. optional alias bridging between semantically similar nodes

### 3.5 Triple reranking

Candidate triples are reranked using a composite score built from:

- semantic similarity
- seed-entity alignment
- relation-cue alignment
- answer-slot alignment
- graph distance
- confidence
- contradiction penalty

This lets retrieval prefer facts that are not only semantically close, but structurally useful for the type of answer being requested.

### 3.6 Path-level evidence assembly

After triple reranking, ProRAG builds short connected evidence paths.

Why this exists:

- multi-hop questions are often answered by **connected chains**, not isolated facts
- the best single triple is not always the best explanation

The current implementation assembles:

- 1-hop evidence
- 2-hop connected paths

and prefers coherent paths that:

- connect through shared graph nodes
- touch the seed entity neighborhood
- align with the answer slot

Example:

```text
christopher nolan --directed--> inception
inception --filmed in--> paris
```

For a `where` question, this path is more useful than either edge alone.

---

## 4. Generation

Generation in [`prorag/pipeline.py`](../prorag/pipeline.py) is intentionally simple.

The system:

1. formats top evidence triples into text
2. includes contradiction markers when present
3. sends that context to the LLM
4. asks for a short grounded answer only

This keeps the LLM in a narrow role:

- not discovering facts
- not doing retrieval
- mostly synthesizing already-selected graph evidence

---

## 5. Interfaces

### Python API

`ProRAG` in [`prorag/__init__.py`](../prorag/__init__.py) is the public facade.

It currently exposes one `model` parameter, used across ingest-time LLM calls and answer-time LLM calls.

### CLI

[`prorag/cli.py`](../prorag/cli.py) provides:

- `ingest`
- `ask`
- `stats`
- `interactive`
- `serve`

### FastAPI server

[`prorag/server.py`](../prorag/server.py) wraps a single in-process `ProRAG` instance and exposes local endpoints for ingest, ask, stats, and clear.

---

## 6. Current Strengths

- entity graph is explicit and inspectable
- ingest aggressively keeps unresolved references out of the graph
- retrieval is shaped around entity + relation intent, not flat chunk similarity
- short path assembly improves multi-hop evidence quality
- one-call answer generation remains simple and easy to debug

---

## 7. Current Gaps

- contradiction modeling is still lightweight
- answer-type validation is not yet enforced after generation
- entity linking is still heuristic rather than full canonical resolution
- path assembly currently focuses on short paths, not broader graph reasoning programs

---

## 8. File Map

```text
prorag/
├── __init__.py       public API facade
├── cli.py            command-line interface
├── detector.py       keyword helper utilities
├── embeddings.py     embedding store and cache
├── entity_utils.py   entity normalization / unresolved-reference guards
├── extractor.py      ingest pipeline with coreference cleanup
├── graph.py          graph storage and candidate retrieval engine
├── llm.py            provider-agnostic LLM adapter
├── pipeline.py       retrieval orchestration + answer generation
└── server.py         FastAPI wrapper
```
