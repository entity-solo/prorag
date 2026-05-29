"""
Historical benchmark harness kept for reproducibility experiments.

This script patches internals for benchmarking and is not the source of truth
for the current runtime architecture.
"""

import os
import json
import time
import argparse
import hashlib
from datetime import datetime
import re
import threading

# Import ProRAG components
from prorag import ProRAG
import prorag.llm
from prorag.pipeline import _keywords_from_question

# ── LLM Interception ──────────────────────────────────────────────────────────
# Global counters to track LLM usage during benchmark execution
LLM_CALL_COUNT = 0
LLM_INPUT_CHARS = 0
LLM_OUTPUT_CHARS = 0
MOCK_MODE = False

original_call_llm = prorag.llm.call_llm
COUNTER_LOCK = threading.Lock()


def intercepted_call_llm(prompt, model="llama-3.3-70b-versatile", max_tokens=1024, system=""):
    global LLM_CALL_COUNT, LLM_INPUT_CHARS, LLM_OUTPUT_CHARS
    with COUNTER_LOCK:
        LLM_CALL_COUNT += 1
        LLM_INPUT_CHARS += len(prompt) + len(system)
    
    if MOCK_MODE:
        # Mock responses to run without API keys
        if "JSON array" in prompt or "extract ALL factual statements" in prompt:
            # We are extracting triples. Let's return some realistic ones based on the text.
            triples = []
            lower_prompt = prompt.lower()
            if "einstein" in lower_prompt:
                triples.append({"subject": "Albert Einstein", "relation": "developed", "object": "theory of relativity", "negated": False, "domains": ["science"], "confidence": 1.0})
                if "newton" in lower_prompt:
                    triples.append({"subject": "Isaac Newton", "relation": "was", "object": "physicist", "negated": False, "domains": ["science"], "confidence": 1.0})
            if "water" in lower_prompt:
                triples.append({"subject": "water", "relation": "boils at", "object": "100 degrees", "negated": False, "domains": ["science"], "confidence": 1.0})
            if "eiffel" in lower_prompt:
                triples.append({"subject": "Eiffel Tower", "relation": "located in", "object": "Paris", "negated": False, "domains": ["geography"], "confidence": 1.0})
            if "vaccine" in lower_prompt:
                triples.append({"subject": "vaccines", "relation": "cause", "object": "autism", "negated": True, "domains": ["medicine"], "confidence": 1.0})
            
            # If no matches, return a default mock triple
            if not triples:
                triples.append({"subject": "fact", "relation": "is", "object": "true", "negated": False, "domains": ["general"], "confidence": 1.0})
            response = json.dumps(triples)
        elif "list the relevant knowledge domains" in prompt:
            response = '["science"]'
        else:
            # Answer generation
            lower_prompt = prompt.lower()
            if "einstein" in lower_prompt and "newton" in lower_prompt:
                response = "yes"
            elif "einstein" in lower_prompt:
                response = "theory of relativity"
            elif "water" in lower_prompt:
                response = "100 degrees"
            elif "eiffel" in lower_prompt:
                response = "Paris"
            elif "vaccine" in lower_prompt:
                response = "no"
            else:
                response = "yes"
    else:
        # Run the original function
        response = original_call_llm(prompt, model=model, max_tokens=max_tokens, system=system)
    
    with COUNTER_LOCK:
        LLM_OUTPUT_CHARS += len(response)
    return response


# Monkey patch the LLM caller globally across all modules
prorag.llm.call_llm = intercepted_call_llm

import prorag.extractor  # noqa: E402
prorag.extractor.call_llm = intercepted_call_llm

import prorag.detector  # noqa: E402
prorag.detector.call_llm = intercepted_call_llm

import prorag.pipeline  # noqa: E402
prorag.pipeline.call_llm = intercepted_call_llm


def reset_llm_counters():
    global LLM_CALL_COUNT, LLM_INPUT_CHARS, LLM_OUTPUT_CHARS
    LLM_CALL_COUNT = 0
    LLM_INPUT_CHARS = 0
    LLM_OUTPUT_CHARS = 0


# ── Extractor Cache ───────────────────────────────────────────────────────────
# Caches LLM-based triple extraction to save API cost/time
CACHE_PATH = os.path.join("data", "extracted_triples_cache.json")
TRIPLE_CACHE = {}


def load_triple_cache():
    global TRIPLE_CACHE
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                TRIPLE_CACHE = json.load(f)
        except Exception:
            TRIPLE_CACHE = {}


def save_triple_cache():
    try:
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(TRIPLE_CACHE, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Failed to save triple cache: {e}")


# Hook into ProRAG's fact extraction to check/update cache
original_extract_facts = None
try:
    import prorag.extractor
    original_extract_facts = prorag.extractor.extract_facts
except ImportError:
    pass


CACHE_LOCK = threading.Lock()

def cached_extract_facts(text, entity_map=None, source="", llm_model="llama-3.3-70b-versatile"):
    if MOCK_MODE:
        return original_extract_facts(text, entity_map=entity_map, source=source, llm_model=llm_model)

    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    
    # If in cache, return the cached result
    with CACHE_LOCK:
        has_cache = text_hash in TRIPLE_CACHE
        if has_cache:
            facts = json.loads(json.dumps(TRIPLE_CACHE[text_hash]))
        else:
            facts = None

    if facts is not None:
        if source:
            for f in facts:
                f["source"] = source
        return facts

    # Otherwise call the original extractor LLM and cache the result
    facts = original_extract_facts(text, entity_map=entity_map, source=source, llm_model=llm_model)

    # Store in cache (clean version without runtime source side-effects)
    raw_facts = []
    for f in facts:
        f_clean = f.copy()
        if "source" in f_clean:
            del f_clean["source"]
        raw_facts.append(f_clean)

    with CACHE_LOCK:
        TRIPLE_CACHE[text_hash] = raw_facts
        save_triple_cache()

    return facts


# Apply the extractor cache patch
if original_extract_facts:
    prorag.extractor.extract_facts = cached_extract_facts


# ── Naive RAG Implementation ──────────────────────────────────────────────────
class NaiveRAG:
    """Keyword-based Naive RAG baseline."""
    
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.documents = []

    def ingest(self, text: str, source: str = ""):
        self.documents.append({"text": text, "source": source})

    def ask(self, question: str) -> dict:
        # 1. Simple keyword search over all ingested chunks
        keywords = _keywords_from_question(question)
        scored_chunks = []
        for doc in self.documents:
            score = 0
            doc_lower = doc["text"].lower()
            for kw in keywords:
                if kw in doc_lower:
                    score += 1
            scored_chunks.append((score, doc))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_docs = [chunk for score, chunk in scored_chunks[:3] if score > 0]
        
        # Fallback to first few documents if no keyword overlap
        if not top_docs:
            top_docs = self.documents[:3]

        # Format context
        context_parts = []
        sources = []
        for doc in top_docs:
            context_parts.append(doc["text"])
            if doc["source"]:
                sources.append(doc["source"])
        context = "\n\n".join(context_parts)

        # 2. Query LLM
        prompt = f"""You are a precise question-answering assistant.
Answer the question using ONLY the context below.

Rules for your answer:
1. Provide a highly concise, short-phrase answer (e.g. only the name, date, or "yes"/"no").
2. Do NOT write full sentences or conversational responses.
3. If the context does not contain enough information, say "I don't have enough information to answer this."
4. Never make up facts not present in the context.

## Context
{context}

## Question
{question}

## Answer"""
        answer_text = intercepted_call_llm(prompt, model=self.model, max_tokens=1024)
        return {
            "answer": answer_text.strip(),
            "sources": sorted(set(sources)),
        }


# ── Metric Calculations ───────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    """Lowercases, removes punctuation, articles, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        return re.sub(r'[^\w\s]', '', text)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_exact_match(prediction: str, truth: str) -> int:
    return int(normalize_answer(prediction) == normalize_answer(truth))


def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(truth).split()
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
        
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
        
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


def main():
    global MOCK_MODE
    parser = argparse.ArgumentParser(description="Evaluate ProRAG vs Naive RAG on HotpotQA")
    parser.add_argument("--dataset", type=str, default="data/hotpot_dev_distractor_v1.json", help="Path to HotpotQA dataset")
    parser.add_argument("--n", type=int, default=5, help="Number of questions to evaluate")
    parser.add_argument("--model", type=str, default="google/gemma-4-26b-a4b-it:free", help="LLM Model to use")
    parser.add_argument(
        "--extractor-model",
        type=str,
        default="google/gemma-4-26b-a4b-it:free",
        help="LLM Model to use for ProRAG ingestion/triple extraction",
    )
    parser.add_argument("--mock", action="store_true", help="Run in offline mock mode without API keys")
    parser.add_argument("--resume", type=str, default="", help="Path to partially finished benchmark JSON file to resume from")
    args = parser.parse_args()

    if args.mock:
        MOCK_MODE = True

    # Load cache
    load_triple_cache()
    print(f"Loaded triple cache with {len(TRIPLE_CACHE)} entries.")

    if not os.path.exists(args.dataset):
        print(f"Dataset not found at {args.dataset}. Please run 'python scripts/download_datasets.py' first.")
        return

    with open(args.dataset, "r", encoding="utf-8") as f:
        data = json.load(f)

    limit = min(args.n, len(data))
    eval_set = data[:limit]
    print(f"Starting evaluation on {limit} questions using model: {args.model}")

    results = []
    completed_ids = {}
    
    if args.resume and os.path.exists(args.resume):
        try:
            with open(args.resume, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            for old_res in old_data.get("details", []):
                if old_res.get("prorag", {}).get("answer") not in (None, "ERROR", "SKIPPED"):
                    completed_ids[old_res["id"]] = old_res
            print(f"Loaded {len(completed_ids)} completed items to resume from {args.resume}")
        except Exception as e:
            print(f"Failed to load resume file: {e}")

    # Store aggregated statistics
    summary = {
        "naive": {"f1": 0.0, "em": 0.0, "latency": 0.0, "calls": 0.0, "tokens": 0.0},
        "prorag": {"f1": 0.0, "em": 0.0, "latency": 0.0, "calls": 0.0, "tokens": 0.0}
    }

    for idx, item in enumerate(eval_set):
        q_id = item.get("_id", f"q_{idx}")
        question = item["question"]
        gold_answer = item["answer"]
        context_data = item["context"]  # list of [title, [sentences]]

        if q_id in completed_ids:
            print(f"\n[{idx+1}/{limit}] Question: {question} (RESUMED)")
            results.append(completed_ids[q_id])
            continue

        print(f"\n[{idx+1}/{limit}] Question: {question}")
        print(f"Ground Truth Answer: {gold_answer}")

        # ── 1. Evaluate Naive RAG (SKIPPED) ───────────────────────────────────
        print(" -> Skipping Naive RAG...")
        naive_ans = "SKIPPED"
        naive_latency = 0.0
        naive_calls = 0
        naive_tokens = 0
        naive_f1 = 0.0
        naive_em = 0
        naive_err = None

        # ── 2. Evaluate ProRAG ────────────────────────────────────────────────
        print(" -> Running ProRAG...")
        prorag_instance = ProRAG(model=args.model)
        
        # Ingest contexts
        reset_llm_counters()
        ingest_start = time.time()
        
        from concurrent.futures import ThreadPoolExecutor
        
        def extract_para(title_and_sentences):
            title, sentences = title_and_sentences
            paragraph = " ".join(sentences)
            import prorag.extractor
            print(f"    [ingest] extracting: {title[:60]}...", flush=True)
            facts = prorag.extractor.extract_facts(paragraph, source=title, llm_model=args.extractor_model)
            print(f"    [ingest] {title[:40]}: {len(facts)} facts", flush=True)
            return title, facts, paragraph

        with ThreadPoolExecutor(max_workers=1) as executor:
            extracted_results = list(executor.map(extract_para, context_data))
            
        for title, facts, paragraph in extracted_results:
            prorag_instance.graph.add_chunk(title, paragraph)
            for f in facts:
                try:
                    fact_type = f.get("type", "relation")
                    if fact_type == "relation":
                        prorag_instance.graph.add_triple(
                            subject=f["subject"],
                            relation=f["relation"],
                            obj=f["object"],
                            source=f.get("source", title),
                            condition=f.get("condition", ""),
                            negated=f.get("negated", False),
                            confidence=float(f.get("confidence", 1.0)),
                            statement_time=f.get("statement_time", ""),
                            temporal_aspect=f.get("temporal_aspect", "PRESENT"),
                        )
                    elif fact_type == "attribute":
                        prorag_instance.graph.add_attribute(
                            subject=f["subject"],
                            key=f["key"],
                            value=f["value"],
                            source=f.get("source", title),
                            confidence=float(f.get("confidence", 1.0)),
                        )
                    elif fact_type == "event":
                        prorag_instance.graph.add_event(
                            event_id=f["event_id"],
                            role=f["role"],
                            entity=f["entity"],
                            source=f.get("source", title),
                            condition=f.get("condition", ""),
                            negated=f.get("negated", False),
                            confidence=float(f.get("confidence", 1.0)),
                            statement_time=f.get("statement_time", ""),
                            temporal_aspect=f.get("temporal_aspect", "PRESENT"),
                        )
                except (KeyError, TypeError):
                    continue
        
        ingest_latency = time.time() - ingest_start
        ingest_calls = LLM_CALL_COUNT
        ingest_tokens = (LLM_INPUT_CHARS + LLM_OUTPUT_CHARS) // 4
        
        # Query
        reset_llm_counters()
        query_start = time.time()
        
        try:
            prorag_res = prorag_instance.ask(question)
            prorag_ans = prorag_res["answer"]
            prorag_err = None
        except Exception as e:
            prorag_ans = "ERROR"
            prorag_err = str(e)
            print(f"    ProRAG failed: {e}")

        query_latency = time.time() - query_start
        query_calls = LLM_CALL_COUNT
        query_tokens = (LLM_INPUT_CHARS + LLM_OUTPUT_CHARS) // 4
        
        prorag_latency = ingest_latency + query_latency
        prorag_calls = ingest_calls + query_calls
        prorag_tokens = ingest_tokens + query_tokens
        
        prorag_f1 = compute_f1(prorag_ans, gold_answer) if not prorag_err else 0.0
        prorag_em = compute_exact_match(prorag_ans, gold_answer) if not prorag_err else 0
        
        print(f"    Answer: {prorag_ans}")
        print(f"    F1: {prorag_f1:.2f} | EM: {prorag_em} | Latency: {prorag_latency:.2f}s (ingest={ingest_latency:.2f}s, query={query_latency:.2f}s) | Calls: {prorag_calls} (ingest={ingest_calls}, query={query_calls})")

        # Accumulate
        results.append({
            "id": q_id,
            "question": question,
            "truth": gold_answer,
            "naive": {
                "answer": naive_ans,
                "f1": naive_f1,
                "em": naive_em,
                "latency": naive_latency,
                "calls": naive_calls,
                "tokens": naive_tokens,
                "error": naive_err
            },
            "prorag": {
                "answer": prorag_ans,
                "f1": prorag_f1,
                "em": prorag_em,
                "latency": prorag_latency,
                "ingest_latency": ingest_latency,
                "query_latency": query_latency,
                "calls": prorag_calls,
                "ingest_calls": ingest_calls,
                "query_calls": query_calls,
                "tokens": prorag_tokens,
                "error": prorag_err,
                "graph_stats": prorag_instance.stats()
            }
        })

        # Save intermediate results to prevent data loss on crashes
        if not os.path.exists("results"):
            os.makedirs("results")
        temp_path = "results/benchmark_temp.json"
        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"    Failed to save intermediate results: {e}")

    # Calculate summary
    for r in results:
        for mode in ["naive", "prorag"]:
            summary[mode]["f1"] += r[mode]["f1"] / limit
            summary[mode]["em"] += r[mode]["em"] / limit
            summary[mode]["latency"] += r[mode]["latency"] / limit
            summary[mode]["calls"] += r[mode]["calls"] / limit
            summary[mode]["tokens"] += r[mode]["tokens"] / limit

    # Save to disk
    if not os.path.exists("results"):
        os.makedirs("results")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/benchmark_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

    # Print clean comparison table
    print("\n" + "="*60)
    print(f" BENCHMARK SUMMARY (N={limit})".center(60))
    print("="*60)
    print(f"{'Metric':<25} | {'Naive RAG':<12} | {'ProRAG':<12}")
    print("-"*60)
    print(f"{'Accuracy (F1)':<25} | {summary['naive']['f1']:.4f}       | {summary['prorag']['f1']:.4f}")
    print(f"{'Exact Match (EM)':<25} | {summary['naive']['em']:.4f}       | {summary['prorag']['em']:.4f}")
    print(f"{'Latency (Avg)':<25} | {summary['naive']['latency']:.2f}s       | {summary['prorag']['latency']:.2f}s")
    print(f"{'LLM Calls / Query':<25} | {summary['naive']['calls']:.1f}         | {summary['prorag']['calls']:.1f}")
    print(f"{'Estimated Tokens / Query':<25} | {summary['naive']['tokens']:.0f}         | {summary['prorag']['tokens']:.0f}")
    print("="*60)
    print(f"Detailed results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
