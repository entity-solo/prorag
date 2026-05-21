"""
Query pipeline — one LLM call from question to answer.

Flow:
  question
    → detect domains (keyword scan, LLM fallback)
    → extract keywords
    → query subgraph(s)
    → format context
    → single LLM call
    → answer + sources
"""

from .graph import ProRAGGraph
from .detector import detect_domains, _keywords_from_question
from .llm import call_llm

_ANSWER_PROMPT = """\
You are a precise question-answering assistant.
Answer the question using ONLY the knowledge graph context below.
If the context does not contain enough information, say "I don't have enough information to answer this."
Never make up facts not present in the context.

## Knowledge Graph Context
{context}

## Question
{question}

## Answer""" 

_CONTRADICTIONS_NOTE = "\n⚠️  Note: conflicting information exists — see sources."


def answer(
    question: str,
    graph: ProRAGGraph,
    llm_model: str = "llama-3.3-70b-versatile",
    max_context_triples: int = 60,
) -> dict:
    """
    Answer a question from the knowledge graph.

    Returns:
        {
          "answer": str,
          "sources": list[str],
          "domains": list[str],
          "triples_used": int,
          "has_contradictions": bool,
        }
    """
    # 1 — detect domains
    domains = detect_domains(question, graph=graph, llm_model=llm_model)

    # 2 — extract keywords from the question
    keywords = _keywords_from_question(question)

    # 3 — query scoped subgraph
    triples = graph.query(keywords, domains=domains, top_k=max_context_triples)

    # Also try without domain filter if result is sparse
    if len(triples) < 15:
        triples = graph.query(keywords, domains=None, top_k=max_context_triples)

    # 4 — format context
    context, sources, has_contradictions = _format_context(triples)

    if not context:
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": [],
            "domains": domains,
            "triples_used": 0,
            "has_contradictions": False,
        }

    # 5 — single LLM call
    prompt = _ANSWER_PROMPT.format(context=context, question=question)
    answer_text = call_llm(prompt, model=llm_model, max_tokens=1024)
    if has_contradictions:
        answer_text += _CONTRADICTIONS_NOTE

    return {
        "answer": answer_text.strip(),
        "sources": sorted(set(sources)),
        "domains": domains,
        "triples_used": len(triples),
        "has_contradictions": has_contradictions,
    }


# ── helpers ───────────────────────────────────────────────────────────────────



def _format_context(triples: list[dict]) -> tuple[str, list[str], bool]:
    """Convert triples to readable context string. Returns (context, sources, has_contradictions)."""
    lines = []
    sources = []
    has_contradictions = False

    for t in triples:
        neg = "NOT " if t.get("negated") else ""
        cond = f" [{t['condition']}]" if t.get("condition") else ""
        confidence = t.get("confidence", 1.0)
        conf_str = f" (confidence: {confidence:.1f})" if confidence < 0.8 else ""

        relation = t["relation"]
        if relation.startswith("CONTRADICTS:"):
            has_contradictions = True
            relation = f"⚠️ CONTRADICTS {relation[12:]}"

        line = f"- {t['subject']} {neg}{relation} {t['object']}{cond}{conf_str}"
        lines.append(line)
        sources.extend(t.get("sources", []))

    return "\n".join(lines), sources, has_contradictions
