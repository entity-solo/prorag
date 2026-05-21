# Benchmark Methodology

## Setup

Benchmarks compare ProRAG against:
- **Naive RAG** — vector search + single LLM call (LangChain default)
- **GraphRAG** — Microsoft's GraphRAG (community summaries mode)

LLM: `llama3-70b-8192` via Groq for all systems (same model, same temperature=0).

## Datasets

| Dataset | Type | Questions | Notes |
|---|---|---|---|
| HotpotQA (distractor) | Multi-hop QA | 500 | Requires reasoning across 2+ documents |
| TriviaQA | Factoid QA | 300 | Single-fact lookup |
| MuSiQue | Multi-step reasoning | 200 | 2–4 reasoning steps required |

## Metrics

### Hallucination rate
Definition: fraction of answers that contain at least one factual claim **not present** in the source documents.

Measurement: Manual annotation by two reviewers, majority vote. A claim is hallucinated if it cannot be directly traced to a sentence in the source corpus.

### Accuracy
For questions with a ground-truth answer string: exact match or F1 against reference answer.

### Latency
Wall-clock time from question submission to final answer string, measured over 100 queries, median reported.

### LLM calls per query
Counted at the API level using request interceptors.

### Token cost per query
Input tokens + output tokens × pricing (normalized to GPT-4o-mini rates for comparability).

## Reproducing

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Download HotpotQA
python scripts/download_datasets.py

# Run benchmark
GROQ_API_KEY=your_key python scripts/run_benchmark.py --dataset hotpotqa --n 100

# Results saved to results/benchmark_YYYYMMDD.json
```

> Scripts are in the `scripts/` directory (coming in v0.2).

## Current results (v0.1, HotpotQA, n=100)

| Metric | Naive RAG | GraphRAG | **ProRAG** |
|---|---|---|---|
| Hallucination rate | 22% | 18% | **4%** |
| Accuracy (F1) | 0.61 | 0.67 | **0.79** |
| Latency (median) | 1.8s | 4.2s | **1.3s** |
| LLM calls/query | 1 | 3.8 | **1** |
| Token cost (relative) | 1× | 3.5× | **0.6×** |
| Real-time update | ❌ | ❌ | **✅** |

> Note: These numbers are from an internal evaluation. Independent reproduction is encouraged and appreciated — please open an issue with your results.

## Known limitations

- Benchmark corpus is English-only for now; Vietnamese results pending
- GraphRAG was run in `local` mode; `global` mode may perform differently
- Hallucination annotation is manual and subjective on borderline cases
