"""
Historical large-topic benchmark kept for comparison runs.

It monkey-patches retrieval behavior and should not be treated as the source
of truth for the current entity-first runtime pipeline.
"""

import os
import json
import time
import hashlib
from datetime import datetime
from prorag import ProRAG
from prorag.pipeline import answer, _keywords_from_question
from prorag.llm import call_llm

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

original_extract_triples = None
try:
    import prorag.extractor
    original_extract_triples = prorag.extractor.extract_triples
except ImportError:
    pass

def cached_extract_triples(text, source="", llm_model="llama-3.3-70b-versatile", extra_domains=None):
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()
    if text_hash in TRIPLE_CACHE:
        triples = json.loads(json.dumps(TRIPLE_CACHE[text_hash]))
        if extra_domains:
            for t in triples:
                for d in extra_domains:
                    if d not in t.get("domains", []):
                        t.setdefault("domains", []).append(d)
        if source:
            for t in triples:
                t["source"] = source
        return triples

    triples = original_extract_triples(text, source=source, llm_model=llm_model, extra_domains=extra_domains)
    raw_triples = original_extract_triples(text, source="", llm_model=llm_model, extra_domains=None)
    TRIPLE_CACHE[text_hash] = raw_triples
    save_triple_cache()
    return triples

if original_extract_triples:
    prorag.extractor.extract_triples = cached_extract_triples

load_triple_cache()


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
        answer_text = call_llm(prompt, model=self.model, max_tokens=1024)
        return {
            "answer": answer_text.strip(),
            "sources": sorted(set(sources)),
        }

if 'GROQ_API_KEY' not in os.environ:
    print("[Warning] GROQ_API_KEY is not set in your environment variables. Please set it before running this script.")
MODEL = "qwen/qwen3-32b"

# 15 Deeply Connected Documents on Space Exploration History
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
    },
    {
        "source": "Apollo 13 Accident",
        "text": "Apollo 13 was launched on April 11, 1970. An oxygen tank explosion forced the crew to abort the Moon landing. The crew returned safely to Earth, commanded by Jim Lovell."
    },
    {
        "source": "Jim Lovell Bio",
        "text": "Jim Lovell was an American astronaut who commanded Apollo 13. Before that, he flew on Apollo 8 as the Command Module Pilot. He was born in Cleveland, Ohio."
    },
    {
        "source": "Apollo 8 Mission",
        "text": "Apollo 8 was launched in December 1968 and became the first manned spacecraft to leave Earth orbit and orbit the Moon. The mission commander was Frank Borman, and the crew included Jim Lovell."
    },
    {
        "source": "Frank Borman Bio",
        "text": "Frank Borman was an American astronaut and commander of Apollo 8. He was born in Gary, Indiana, and was a career officer in the US Air Force."
    },
    {
        "source": "Yuri Gagarin Flight",
        "text": "On April 12, 1961, Soviet cosmonaut Yuri Gagarin became the first human in space, orbiting the Earth in the Vostok 1 spacecraft launched by the Soviet Union."
    },
    {
        "source": "Vostok 1 Mission",
        "text": "Vostok 1 was the first human spaceflight in history. The Vostok spacecraft was designed under the leadership of Sergei Korolev, the chief Soviet rocket engineer."
    },
    {
        "source": "Sergei Korolev Bio",
        "text": "Sergei Korolev was the lead Soviet rocket engineer and spacecraft designer during the Space Race. His identity was kept a state secret until after his death in 1966."
    },
    {
        "source": "Hubble Space Telescope",
        "text": "The Hubble Space Telescope was launched into low Earth orbit in 1990 by the Space Shuttle Discovery. Its primary mirror had a spherical aberration, which was repaired in 1993 by Space Shuttle mission STS-61."
    }
]

# 12 Complex Queries requiring multi-hop links and conflict resolution
QUESTIONS = [
    {
        "id": "Q1",
        "question": "Who established NASA and in what year?",
        "gold": "President Dwight D. Eisenhower in 1958."
    },
    {
        "id": "Q2",
        "question": "Who was the lead spacecraft designer for the mission that sent the first human into space?",
        "gold": "Sergei Korolev (lead designer for Vostok 1 which carried Yuri Gagarin)."
    },
    {
        "id": "Q3",
        "question": "What city was the commander of the first manned spacecraft to orbit the Moon born in?",
        "gold": "Gary, Indiana (Frank Borman was the commander of Apollo 8)."
    },
    {
        "id": "Q4",
        "question": "Which president established the organization that managed the mission that landed the first person on the Moon?",
        "gold": "President Dwight D. Eisenhower (established NASA, which managed Apollo 11/Apollo program)."
    },
    {
        "id": "Q5",
        "question": "What event triggered the Space Race, when did it occur, and who launched it?",
        "gold": "The launch of Sputnik 1 on October 4, 1957, by the Soviet Union."
    },
    {
        "id": "Q6",
        "question": "Is there evidence proving the Apollo Moon landings were faked?",
        "gold": "No, mainstream scientists, historians, and physical evidence reject conspiracy claims and prove the landings were real."
    },
    {
        "id": "Q7",
        "question": "Who was the commander of Apollo 13 and which previous Apollo mission did he fly on?",
        "gold": "Jim Lovell, who previously flew on Apollo 8."
    },
    {
        "id": "Q8",
        "question": "In what year was the space telescope repaired, and which spacecraft originally launched it?",
        "gold": "Repaired in 1993, originally launched by the Space Shuttle Discovery in 1990."
    },
    {
        "id": "Q9",
        "question": "Who was the first person to walk on the Moon and what city was he born in?",
        "gold": "Neil Armstrong, born in Wapakoneta, Ohio."
    },
    {
        "id": "Q10",
        "question": "Did the country that launched Vostok 1 also launch the first artificial Earth satellite?",
        "gold": "Yes, both Vostok 1 and Sputnik 1 were launched by the Soviet Union."
    },
    {
        "id": "Q11",
        "question": "Who served as Lunar Module Pilot on Apollo 11 and who did he follow on the Moon?",
        "gold": "Buzz Aldrin, who followed Neil Armstrong."
    },
    {
        "id": "Q12",
        "question": "What military rank did the president who established NASA hold during World War II?",
        "gold": "Five-star general in the United States Army (Dwight D. Eisenhower)."
    }
]


import re

def calculate_f1(pred: str, gold: str) -> float:
    # Strip <think>...</think> tags and contents from prediction
    pred_clean = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL).strip()
    
    pred_words = re.findall(r"\w+", pred_clean.lower())
    gold_words = re.findall(r"\w+", gold.lower())
    if not pred_words or not gold_words:
        return 0.0
    common = set(pred_words) & set(gold_words)
    if not common:
        return 0.0
    precision = len(common) / len(pred_words)
    recall = len(common) / len(gold_words)
    return 2 * (precision * recall) / (precision + recall)

def main():
    print("="*80)
    print("RUNNING EXTENDED TOPIC-BASED BENCHMARK (15 DOCUMENTS, 12 QUESTIONS)")
    print("="*80)

    # ──────────────────────────────────────────────────────────────────────────
    # 1. Ingestion Phase
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Ingesting to Naive RAG ---")
    naive_rag = NaiveRAG(model=MODEL)
    start_naive_ingest = time.time()
    for doc in CORPUS:
        naive_rag.ingest(doc["text"], source=doc["source"])
    naive_ingest_time = time.time() - start_naive_ingest
    print(f"Naive RAG Ingestion Time: {naive_ingest_time:.2f}s")

    print("\n--- Ingesting to ProRAG (building graph) ---")
    prorag_instance = ProRAG(model=MODEL)
    # Monkey-patch ProRAGGraph.query to use max_hops=3 for deep links
    from prorag.graph import ProRAGGraph
    original_query = ProRAGGraph.query
    def patched_query(self, keywords, domains=None, max_hops=3, top_k=40):
        return original_query(self, keywords, domains=domains, max_hops=max_hops, top_k=top_k)
    ProRAGGraph.query = patched_query

    # We also override warning emoji to keep stdout clean
    import prorag.pipeline as pr_pipeline
    pr_pipeline._CONTRADICTIONS_NOTE = "\n[Warning] Note: conflicting information exists - see sources."
    start_prorag_ingest = time.time()
    for doc in CORPUS:
        print(f"  Ingesting {doc['source']}...")
        prorag_instance.ingest(doc["text"], source=doc["source"])
    prorag_ingest_time = time.time() - start_prorag_ingest
    print(f"ProRAG Ingestion Time: {prorag_ingest_time:.2f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Querying Phase
    # ──────────────────────────────────────────────────────────────────────────
    results = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "model": MODEL,
            "corpus_size": len(CORPUS),
            "questions_size": len(QUESTIONS),
            "naive_ingest_time_sec": naive_ingest_time,
            "prorag_ingest_time_sec": prorag_ingest_time,
        },
        "runs": []
    }

    print("\n--- Starting Queries ---")
    for q_item in QUESTIONS:
        q_id = q_item["id"]
        q_text = q_item["question"]
        gold = q_item["gold"]
        print(f"\nQuestion {q_id}: {q_text}")

        # A. Naive RAG query
        t0 = time.time()
        naive_ans = naive_rag.ask(q_text)
        naive_latency = time.time() - t0
        naive_f1 = calculate_f1(naive_ans["answer"], gold)
        print(f"  [Naive RAG] Latency: {naive_latency:.2f}s | F1: {naive_f1:.4f} | Ans: {naive_ans['answer'][:80]}...")

        # B. ProRAG query
        t0 = time.time()
        prorag_ans = prorag_instance.ask(q_text)
        prorag_latency = time.time() - t0
        prorag_f1 = calculate_f1(prorag_ans["answer"], gold)
        print(f"  [ProRAG   ] Latency: {prorag_latency:.2f}s | F1: {prorag_f1:.4f} | Ans: {prorag_ans['answer'][:80]}...")

        results["runs"].append({
            "id": q_id,
            "question": q_text,
            "gold": gold,
            "naive": {
                "answer": naive_ans["answer"],
                "latency_sec": naive_latency,
                "f1": naive_f1,
                "sources": naive_ans["sources"]
            },
            "prorag": {
                "answer": prorag_ans["answer"],
                "latency_sec": prorag_latency,
                "f1": prorag_f1,
                "sources": prorag_ans["sources"],
                "domains": prorag_ans["domains"],
                "triples_used": prorag_ans["triples_used"],
                "has_contradictions": prorag_ans["has_contradictions"]
            }
        })

    # Summary Statistics
    avg_naive_lat = sum(r["naive"]["latency_sec"] for r in results["runs"]) / len(QUESTIONS)
    avg_naive_f1 = sum(r["naive"]["f1"] for r in results["runs"]) / len(QUESTIONS)
    avg_prorag_lat = sum(r["prorag"]["latency_sec"] for r in results["runs"]) / len(QUESTIONS)
    avg_prorag_f1 = sum(r["prorag"]["f1"] for r in results["runs"]) / len(QUESTIONS)

    results["summary"] = {
        "naive": {
            "avg_latency_sec": avg_naive_lat,
            "avg_f1": avg_naive_f1
        },
        "prorag": {
            "avg_latency_sec": avg_prorag_lat,
            "avg_f1": avg_prorag_f1
        }
    }

    print("\n" + "="*80)
    print(" BENCHMARK SUMMARY ".center(80, "="))
    print("="*80)
    print(f"Naive RAG - Avg Latency: {avg_naive_lat:.2f}s | Avg F1: {avg_naive_f1:.4f}")
    print(f"ProRAG    - Avg Latency: {avg_prorag_lat:.2f}s | Avg F1: {avg_prorag_f1:.4f}")
    print("="*80)

    # Save to file
    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"benchmark_large_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {filename}")


if __name__ == "__main__":
    main()
