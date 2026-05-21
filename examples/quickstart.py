"""
Quickstart example — no external documents needed.

Run:
    GROQ_API_KEY=your_key python examples/quickstart.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prorag import ProRAG

SAMPLE_TEXT = """
Albert Einstein developed the special theory of relativity in 1905
while working at the patent office in Bern, Switzerland.
He later developed the general theory of relativity in 1915.
Einstein was born in Ulm, Germany in 1879.
He won the Nobel Prize in Physics in 1921 for the photoelectric effect,
not for the theory of relativity.

Water boils at 100 degrees Celsius at standard atmospheric pressure (1 atm).
At high altitudes where pressure is lower, water boils below 100 degrees.

Vaccines do not cause autism. Multiple large-scale studies involving
millions of children have found no link between vaccines and autism.
"""

def main():
    print("=" * 60)
    print("ProRAG Quickstart")
    print("=" * 60)

    rag = ProRAG()

    print("\n[1] Ingesting sample knowledge...")
    n = rag.ingest(SAMPLE_TEXT, source="sample_text")
    print(f"    → {n} triples extracted")
    print(f"    → Graph stats: {rag.stats()}")

    questions = [
        "Where did Einstein develop the theory of relativity?",
        "What did Einstein win the Nobel Prize for?",
        "At what temperature does water boil?",
        "Do vaccines cause autism?",
    ]

    print("\n[2] Answering questions (1 LLM call each):\n")
    for q in questions:
        result = rag.ask(q)
        print(f"Q: {q}")
        print(f"A: {result['answer']}")
        print(f"   domains={result['domains']} | triples_used={result['triples_used']} | sources={result['sources']}")
        if result["has_contradictions"]:
            print("   ⚠️  Contradicting info in graph")
        print()

    # Demonstrate continuous learning
    print("[3] Adding new knowledge (no retrain)...")
    rag.ingest("Einstein moved to the United States in 1933, fleeing the Nazi regime.")
    print(f"    → Graph stats after update: {rag.stats()}")
    result = rag.ask("When did Einstein move to the United States?")
    print(f"\nQ: When did Einstein move to the United States?")
    print(f"A: {result['answer']}")

    print("\n[4] Saving graph to disk...")
    rag.save("/tmp/prorag_demo.json")
    print("    → Saved to /tmp/prorag_demo.json")
    print("\nDone.")


if __name__ == "__main__":
    main()
