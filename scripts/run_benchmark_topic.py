"""
Historical topic benchmark kept for side-by-side comparisons.

It contains mock paths and legacy assumptions that are not representative of
the default runtime retrieval pipeline.
"""

import os
import json
import time
import argparse
import hashlib
from datetime import datetime
import re

# Import ProRAG components
from prorag import ProRAG
import prorag.llm
from prorag.pipeline import _keywords_from_question

# ── LLM Interception ──────────────────────────────────────────────────────────
LLM_CALL_COUNT = 0
LLM_INPUT_CHARS = 0
LLM_OUTPUT_CHARS = 0
MOCK_MODE = False

original_call_llm = prorag.llm.call_llm


def intercepted_call_llm(prompt, model="llama-3.3-70b-versatile", max_tokens=1024, system=""):
    global LLM_CALL_COUNT, LLM_INPUT_CHARS, LLM_OUTPUT_CHARS
    LLM_CALL_COUNT += 1
    LLM_INPUT_CHARS += len(prompt) + len(system)
    
    if MOCK_MODE:
        # Mock answers for local offline testing
        lower_prompt = prompt.lower()
        if "JSON array" in prompt or "extract ALL factual statements" in prompt:
            triples = []
            if "apollo 11" in lower_prompt or "armstrong" in lower_prompt:
                triples.append({"subject": "Apollo 11", "relation": "landed", "object": "Neil Armstrong", "negated": False, "domains": ["history"], "confidence": 1.0})
                triples.append({"subject": "Neil Armstrong", "relation": "born in", "object": "Wapakoneta", "negated": False, "domains": ["history"], "confidence": 1.0})
                triples.append({"subject": "Buzz Aldrin", "relation": "was pilot of", "object": "Apollo 11", "negated": False, "domains": ["history"], "confidence": 1.0})
                triples.append({"subject": "Apollo 11", "relation": "managed by", "object": "NASA", "negated": False, "domains": ["history"], "confidence": 1.0})
            if "nasa" in lower_prompt or "eisenhower" in lower_prompt:
                triples.append({"subject": "NASA", "relation": "established in", "object": "1958", "negated": False, "domains": ["history"], "confidence": 1.0})
                triples.append({"subject": "NASA", "relation": "established by", "object": "Dwight D. Eisenhower", "negated": False, "domains": ["history"], "confidence": 1.0})
            if "sputnik" in lower_prompt or "space race" in lower_prompt:
                triples.append({"subject": "Soviet Union", "relation": "launched", "object": "Sputnik 1", "negated": False, "domains": ["history"], "confidence": 1.0})
                triples.append({"subject": "Sputnik 1", "relation": "triggered", "object": "Space Race", "negated": False, "domains": ["history"], "confidence": 1.0})
            if "fake" in lower_prompt or "conspiracy" in lower_prompt:
                triples.append({"subject": "Apollo Moon landings", "relation": "faked by", "object": "NASA", "negated": True, "domains": ["history"], "confidence": 0.9})
                triples.append({"subject": "conspiracy theorists", "relation": "claim", "object": "Moon landing was fake", "negated": False, "domains": ["general"], "confidence": 0.8})
            
            if not triples:
                triples.append({"subject": "Space Exploration", "relation": "involves", "object": "NASA", "negated": False, "domains": ["history"], "confidence": 1.0})
            response = json.dumps(triples)
        elif "list the relevant knowledge domains" in prompt:
            response = '["history"]'
        else:
            if "first person" in lower_prompt:
                response = "Neil Armstrong"
            elif "established" in lower_prompt and "nasa" in lower_prompt:
                response = "NASA was managed by Dwight D. Eisenhower and established in 1958."
            elif "president" in lower_prompt:
                response = "Dwight D. Eisenhower"
            elif "sputnik" in lower_prompt or "triggered" in lower_prompt:
                response = "Sputnik 1 in October 1957 triggered the Space Race."
            elif "pilot" in lower_prompt:
                response = "Buzz Aldrin"
            elif "fake" in lower_prompt:
                response = "Conspiracy theorists claim Apollo Moon landings were faked, but scientific consensus denies this."
            else:
                response = "NASA"
    else:
        response = original_call_llm(prompt, model=model, max_tokens=max_tokens, system=system)
        
    LLM_OUTPUT_CHARS += len(response)
    return response


# Apply monkey patches globally
prorag.llm.call_llm = intercepted_call_llm

import prorag.extractor
prorag.extractor.call_llm = intercepted_call_llm

import prorag.detector
prorag.detector.call_llm = intercepted_call_llm

import prorag.pipeline
prorag.pipeline.call_llm = intercepted_call_llm


def reset_llm_counters():
    global LLM_CALL_COUNT, LLM_INPUT_CHARS, LLM_OUTPUT_CHARS
    LLM_CALL_COUNT = 0
    LLM_INPUT_CHARS = 0
    LLM_OUTPUT_CHARS = 0


# ── Extractor Cache ───────────────────────────────────────────────────────────
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


original_extract_triples = prorag.extractor.extract_triples


def cached_extract_triples(text, entity_map=None, source="", llm_model="llama-3.3-70b-versatile"):
    if MOCK_MODE:
        return original_extract_triples(text, entity_map=entity_map, source=source, llm_model=llm_model)
        
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if text_hash in TRIPLE_CACHE:
        triples = json.loads(json.dumps(TRIPLE_CACHE[text_hash]))
        if source:
            for t in triples:
                t["source"] = source
        return triples

    triples = original_extract_triples(text, entity_map=entity_map, source=source, llm_model=llm_model)
    raw_triples = original_extract_triples(text, entity_map=entity_map, source="", llm_model=llm_model)
    TRIPLE_CACHE[text_hash] = raw_triples
    save_triple_cache()
    return triples


prorag.extractor.extract_triples = cached_extract_triples


# ── Naive RAG Baseline ────────────────────────────────────────────────────────
class NaiveRAG:
    def __init__(self, model: str = "llama-3.3-70b-versatile"):
        self.model = model
        self.documents = []

    def ingest(self, text: str, source: str = ""):
        self.documents.append({"text": text, "source": source})

    def ask(self, question: str) -> dict:
        # Simple keyword search retrieval
        keywords = _keywords_from_question(question)
        scored_chunks = []
        for doc in self.documents:
            score = 0
            doc_lower = doc["text"].lower()
            for kw in keywords:
                if kw in doc_lower:
                    score += 1
            scored_chunks.append((score, doc))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_docs = [chunk for score, chunk in scored_chunks[:3] if score > 0]
        if not top_docs:
            top_docs = self.documents[:3]

        context = "\n\n".join([doc["text"] for doc in top_docs])
        sources = [doc["source"] for doc in top_docs if doc["source"]]

        prompt = f"""You are a precise question-answering assistant.
Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't have enough information to answer this."
Never make up facts not present in the context.

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


# ── Dataset & Topics ──────────────────────────────────────────────────────────
CORPUS = [
    {
        "source": "Apollo 11 Mission",
        "text": "The Apollo 11 mission successfully landed Neil Armstrong and Buzz Aldrin on the Moon on July 20, 1969. The Apollo program was a massive endeavor managed by NASA."
    },
    {
        "source": "Neil Armstrong Biography",
        "text": "Neil Armstrong was an American astronaut and aeronautical engineer. He was born in Wapakoneta, Ohio, and became the first person to walk on the Moon."
    },
    {
        "source": "Buzz Aldrin Role",
        "text": "Buzz Aldrin served as the Lunar Module Pilot on the Apollo 11 mission. He walked on the Moon shortly after Neil Armstrong, becoming the second person to do so."
    },
    {
        "source": "NASA Origins",
        "text": "NASA (National Aeronautics and Space Administration) was established in 1958 by President Dwight D. Eisenhower to lead US civilian space initiatives. NASA succeeded NACA."
    },
    {
        "source": "Dwight D. Eisenhower Bio",
        "text": "Dwight D. Eisenhower was the 34th President of the United States. Before his presidency, he was a five-star general in the United States Army during World War II."
    },
    {
        "source": "Sputnik 1 Launch",
        "text": "The Soviet Union launched Sputnik 1, the first artificial Earth satellite, on October 4, 1957. This technological breakthrough shocked the Western world."
    },
    {
        "source": "The Space Race",
        "text": "The launch of Sputnik 1 directly triggered the Space Race, a tense 20th-century competition between the Soviet Union and the United States for spaceflight dominance."
    },
    {
        "source": "Moon Landing Conspiracy Claims",
        "text": "Some conspiracy theorists claim that the Apollo Moon landings were faked by NASA in a studio. However, mainstream scientists, historians, and physical evidence reject these claims, proving the landings were real."
    }
]

QUESTIONS = [
    {
        "question": "Who was the first person to walk on the Moon?",
        "answer": "Neil Armstrong"
    },
    {
        "question": "What organization managed the Apollo program and when was it established?",
        "answer": "NASA, established in 1958"
    },
    {
        "question": "Which US President established NASA?",
        "answer": "Dwight D. Eisenhower"
    },
    {
        "question": "What event triggered the Space Race and when did it occur?",
        "answer": "The launch of Sputnik 1 on October 4, 1957"
    },
    {
        "question": "Who served as the Lunar Module Pilot on Apollo 11?",
        "answer": "Buzz Aldrin"
    },
    {
        "question": "Are there claims that the Apollo Moon landings were faked?",
        "answer": "Yes, some conspiracy theorists claim they were faked, but scientists and evidence reject this."
    }
]


# ── Metrics ───────────────────────────────────────────────────────────────────
def normalize_answer(s: str) -> str:
    def remove_articles(text): return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text): return ' '.join(text.split())
    def remove_punc(text): return re.sub(r'[^\w\s]', '', text)
    def lower(text): return text.lower()
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


# ── Execution ─────────────────────────────────────────────────────────────────
def main():
    global MOCK_MODE
    parser = argparse.ArgumentParser(description="Evaluate ProRAG vs Naive RAG on a Single Interconnected Topic")
    parser.add_argument("--model", type=str, default="llama-3.3-70b-versatile", help="LLM Model to use")
    parser.add_argument("--mock", action="store_true", help="Run offline mock mode")
    args = parser.parse_args()

    if args.mock:
        MOCK_MODE = True
    else:
        load_triple_cache()

    print("="*60)
    print(" TOPIC BENCHMARK: SPACE EXPLORATION HISTORY ".center(60, "="))
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Mock Mode: {args.mock}")
    print(f"Corpus Size: {len(CORPUS)} documents")
    print(f"Number of Queries: {len(QUESTIONS)}")
    print("-"*60)

    # ── INGESTION PHASE (Done ONCE) ───────────────────────────────────────────
    print("\n>>> Starting Ingestion Phase (Done ONCE for the whole topic) >>>")

    # Ingest Naive RAG
    naive_rag = NaiveRAG(model=args.model)
    reset_llm_counters()
    t_start = time.time()
    for doc in CORPUS:
        naive_rag.ingest(doc["text"], source=doc["source"])
    naive_ingest_time = time.time() - t_start
    naive_ingest_calls = LLM_CALL_COUNT
    print(f"Naive RAG Ingest: Completed in {naive_ingest_time:.2f}s | LLM Calls: {naive_ingest_calls}")

    # Ingest ProRAG
    prorag_instance = ProRAG(model=args.model)
    reset_llm_counters()
    t_start = time.time()
    for doc in CORPUS:
        prorag_instance.ingest(doc["text"], source=doc["source"])
    prorag_ingest_time = time.time() - t_start
    prorag_ingest_calls = LLM_CALL_COUNT
    print(f"ProRAG Graph Ingest: Completed in {prorag_ingest_time:.2f}s | LLM Calls: {prorag_ingest_calls}")
    print(f"ProRAG Graph Stats: {prorag_instance.stats()}")

    # ── QUERY PHASE (Run multiple queries on the pre-built knowledge) ─────────
    print("\n>>> Starting Query Phase (Executing multiple queries) >>>")
    
    results = []
    
    for idx, item in enumerate(QUESTIONS):
        question = item["question"]
        gold_answer = item["answer"]
        print(f"\nQ{idx+1}: {question}")

        # Query Naive RAG
        reset_llm_counters()
        start = time.time()
        try:
            naive_res = naive_rag.ask(question)
            naive_ans = naive_res["answer"]
            naive_err = None
        except Exception as e:
            naive_ans = "ERROR"
            naive_err = str(e)
        naive_latency = time.time() - start
        naive_calls = LLM_CALL_COUNT
        naive_tokens = (LLM_INPUT_CHARS + LLM_OUTPUT_CHARS) // 4
        naive_f1 = compute_f1(naive_ans, gold_answer) if not naive_err else 0.0
        print(f"  [Naive RAG] Latency: {naive_latency:.2f}s | LLM Calls: {naive_calls} | Answer: {naive_ans}")

        # Query ProRAG
        reset_llm_counters()
        start = time.time()
        try:
            prorag_res = prorag_instance.ask(question)
            prorag_ans = prorag_res["answer"]
            prorag_err = None
        except Exception as e:
            prorag_ans = "ERROR"
            prorag_err = str(e)
        prorag_latency = time.time() - start
        prorag_calls = LLM_CALL_COUNT
        prorag_tokens = (LLM_INPUT_CHARS + LLM_OUTPUT_CHARS) // 4
        prorag_f1 = compute_f1(prorag_ans, gold_answer) if not prorag_err else 0.0
        print(f"  [ProRAG   ] Latency: {prorag_latency:.2f}s | LLM Calls: {prorag_calls} | Answer: {prorag_ans}")

        results.append({
            "question": question,
            "truth": gold_answer,
            "naive": {"f1": naive_f1, "latency": naive_latency, "calls": naive_calls, "tokens": naive_tokens},
            "prorag": {"f1": prorag_f1, "latency": prorag_latency, "calls": prorag_calls, "tokens": prorag_tokens}
        })

    # Summary Calculations
    n_queries = len(QUESTIONS)
    naive_avg_f1 = sum(r["naive"]["f1"] for r in results) / n_queries
    prorag_avg_f1 = sum(r["prorag"]["f1"] for r in results) / n_queries
    
    naive_avg_lat = sum(r["naive"]["latency"] for r in results) / n_queries
    prorag_avg_lat = sum(r["prorag"]["latency"] for r in results) / n_queries
    
    naive_avg_calls = sum(r["naive"]["calls"] for r in results) / n_queries
    prorag_avg_calls = sum(r["prorag"]["calls"] for r in results) / n_queries

    naive_avg_tokens = sum(r["naive"]["tokens"] for r in results) / n_queries
    prorag_avg_tokens = sum(r["prorag"]["tokens"] for r in results) / n_queries

    # Output Table
    print("\n" + "="*60)
    print(" TOPIC-BASED BENCHMARK SUMMARY (N=6 queries) ".center(60))
    print("="*60)
    print(f"{'Phase / Metric':<25} | {'Naive RAG':<12} | {'ProRAG':<12}")
    print("-"*60)
    print(f"{'Ingestion Time (Total)':<25} | {naive_ingest_time:.2f}s        | {prorag_ingest_time:.2f}s")
    print(f"{'Ingestion LLM Calls':<25} | {naive_ingest_calls:<12} | {prorag_ingest_calls:<12}")
    print("-"*60)
    print(f"{'Query F1 Accuracy (Avg)':<25} | {naive_avg_f1:.4f}       | {prorag_avg_f1:.4f}")
    print(f"{'Query Latency (Avg)':<25} | {naive_avg_lat:.2f}s        | {prorag_avg_lat:.2f}s")
    print(f"{'Query LLM Calls (Avg)':<25} | {naive_avg_calls:.1f}         | {prorag_avg_calls:.1f}")
    print(f"{'Query Est. Tokens (Avg)':<25} | {naive_avg_tokens:.0f}         | {prorag_avg_tokens:.0f}")
    print("="*60)
    
    # Save results JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"results/benchmark_topic_{timestamp}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "ingest": {
                "naive": {"time": naive_ingest_time, "calls": naive_ingest_calls},
                "prorag": {"time": prorag_ingest_time, "calls": prorag_ingest_calls}
            },
            "queries": results
        }, f, ensure_ascii=False, indent=2)
    print(f"Results saved to: {out_path}\n")


if __name__ == "__main__":
    main()
