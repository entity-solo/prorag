"""
Legacy demo script kept for historical comparison experiments.

It monkey-patches older retrieval behavior and does not reflect the current
entity-first retrieval pipeline used by `ProRAG.ask()`.
"""

import os
import time
import re
from prorag import ProRAG
from prorag.graph import ProRAGGraph
from prorag.pipeline import _keywords_from_question
import prorag.pipeline
from prorag.llm import call_llm

if 'GROQ_API_KEY' not in os.environ:
    print("[Warning] GROQ_API_KEY is not set in your environment variables. Please set it before running this script.")
MODEL = "llama-3.3-70b-versatile"

# Override warning emoji to prevent CP1252 encoding crash on Windows
prorag.pipeline._CONTRADICTIONS_NOTE = "\n[Warning] Note: conflicting information exists - see sources."

# Legacy monkey patch for the older keyword/BFS retrieval path.
original_query = ProRAGGraph.query
def patched_query(self, keywords, domains=None, max_hops=4, top_k=40):
    return original_query(self, keywords, domains=domains, max_hops=max_hops, top_k=top_k)

ProRAGGraph.query = patched_query


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


def print_banner(title):
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80)


def main():
    print_banner("DEMO: 3 CORE SUPERPOWERS OF PRORAG VS NAIVE RAG")

    # ──────────────────────────────────────────────────────────────────────────
    # SIÊU NĂNG LỰC 1: TRUY VẤN LIÊN KẾT NHIỀU BƯỚC (MULTI-HOP CHAIN REASONING)
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("SUPERPOWER 1: MULTI-HOP CHAIN REASONING (4 HOPS)")
    
    # 4 distinct docs
    docs_chain = [
        {"source": "Doc A", "text": "Alice is married to Bob."},
        {"source": "Doc B", "text": "Bob's father is Charlie."},
        {"source": "Doc C", "text": "Charlie lives in Seattle."},
        {"source": "Doc D", "text": "Seattle is located in the state of Washington."}
    ]
    
    print("Ingesting 4 disconnected documents...")
    naive_chain = NaiveRAG(model=MODEL)
    prorag_chain = ProRAG(model=MODEL)
    
    for d in docs_chain:
        naive_chain.ingest(d["text"], source=d["source"])
        prorag_chain.ingest(d["text"], source=d["source"])

    question_chain = "What state does Alice's father-in-law live in?"
    print(f"\nQuestion: '{question_chain}'")
    
    # Naive RAG top 2 docs retrieval simulation (e.g. in a large database)
    print("\n-> Running Naive RAG (simulating top 2 retrieved chunks limit):")
    keywords = ["alice", "father", "law", "live", "state"]
    scored = []
    for d in naive_chain.documents:
        score = sum(1 for kw in keywords if kw in d["text"].lower())
        scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    top_2_context = "\n\n".join([doc["text"] for score, doc in scored[:2]])
    
    naive_prompt = f"""You are a precise question-answering assistant.
Answer the question using ONLY the context below.
If the context does not contain enough information, say "I don't have enough information to answer this."

## Context
{top_2_context}

## Question
{question_chain}

## Answer"""
    naive_ans = call_llm(naive_prompt, model=MODEL).strip()
    print(f"Context Naive RAG read:\n{top_2_context}")
    print(f"Naive RAG Answer: {naive_ans}")

    print("\n-> Running ProRAG (traversing Alice -> Bob -> Charlie -> Seattle -> Washington in 4 hops):")
    prorag_res = prorag_chain.ask(question_chain)
    print(f"ProRAG Answer: {prorag_res['answer']}")

    # ──────────────────────────────────────────────────────────────────────────
    # SIÊU NĂNG LỰC 2: GIẢI QUYẾT XUNG ĐỘT THÔNG TIN (CONTRADICTION EXPLICIT RESOLUTION)
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("SUPERPOWER 2: KNOWLEDGE CONFLICT DETECTION & RESOLUTION")

    docs_conflict = [
        {"source": "Wakefield 1998", "text": "Vaccines cause autism."},
        {"source": "CDC Statement", "text": "Vaccines do not cause autism."}
    ]

    print("Ingesting conflicting vaccine claims...")
    naive_conflict = NaiveRAG(model=MODEL)
    prorag_conflict = ProRAG(model=MODEL)

    for d in docs_conflict:
        naive_conflict.ingest(d["text"], source=d["source"])
        prorag_conflict.ingest(d["text"], source=d["source"])

    question_conflict = "Do vaccines cause autism?"
    print(f"\nQuestion: '{question_conflict}'")

    print("\n-> Running Naive RAG (feeding both conflicting text chunks to LLM):")
    naive_res_conflict = naive_conflict.ask(question_conflict)
    print(f"Naive RAG Answer: {naive_res_conflict['answer']}")

    print("\n-> Running ProRAG (detecting CONTRADICTS edge and flagging conflict):")
    prorag_res_conflict = prorag_conflict.ask(question_conflict)
    print(f"ProRAG Answer: {prorag_res_conflict['answer']}")
    print(f"Conflict detected explicitly? {prorag_res_conflict['has_contradictions']}")

    # ──────────────────────────────────────────────────────────────────────────
    # SIÊU NĂNG LỰC 3: CẬP NHẬT THỜI GIAN THỰC (REAL-TIME INCREMENTAL UPDATE)
    # ──────────────────────────────────────────────────────────────────────────
    print_banner("SUPERPOWER 3: REAL-TIME INSTANT KNOWLEDGE INCREMENTAL UPDATE")

    print("Initializing systems...")
    naive_update = NaiveRAG(model=MODEL)
    prorag_update = ProRAG(model=MODEL)

    naive_update.ingest("The lead developer of Project Apollo is David.", source="Project Setup")
    prorag_update.ingest("The lead developer of Project Apollo is David.", source="Project Setup")

    print("Question: 'Who is the lead developer of Project Apollo?'")
    print(f"Naive RAG Answer before update: {naive_update.ask('Who is the lead developer of Project Apollo?')['answer']}")
    print(f"ProRAG Answer before update: {prorag_update.ask('Who is the lead developer of Project Apollo?')['answer']}")

    # Perform instant update
    print("\n--- UPDATE RECEIVED: 'Elena has just been appointed as the lead developer of Project Apollo, replacing David.' ---")
    
    # ProRAG updates graph incrementally (adds new nodes/edges instantly)
    prorag_update.ingest("Elena has just been appointed as the lead developer of Project Apollo, replacing David.", source="Update Slack")
    
    # Naive RAG index is NOT rebuilt (representing traditional static DB limitations)
    print("\n-> After update (without rebuilding Naive RAG vector/keyword index):")
    print(f"Naive RAG Answer: {naive_update.ask('Who is the lead developer of Project Apollo?')['answer']}")
    print(f"ProRAG Answer: {prorag_update.ask('Who is the lead developer of Project Apollo?')['answer']}")


if __name__ == "__main__":
    main()
