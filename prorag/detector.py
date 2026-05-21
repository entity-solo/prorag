"""
Domain detector — fast, graph-based routing.

Determines which structural path tag(s) to query for a given question by
matching question keywords to existing graph entities and their associated tags.
"""

import re

def _keywords_from_question(question: str) -> list[str]:
    """Extract candidate keywords — stopword-filtered tokens."""
    stopwords = {
        "what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "did", "does", "do", "which", "also", "originally", "should", "could", "would",
        "that", "this", "these", "those", "by", "with", "from", "about", "into",
        "cái", "gì", "là", "của", "và", "ở", "tại", "khi", "nào", "có",
    }
    # Match words of length 3 or more, keeping Vietnamese characters intact
    tokens = re.findall(r"\b\w{3,}\b", question.lower())
    return [t for t in tokens if t not in stopwords]


def detect_domains(question: str, graph=None, llm_model: str = "llama-3.3-70b-versatile") -> list[str]:
    """
    Returns the most relevant domain(s)/structural tags for a question.

    Strategy:
    1. Fast entity/keyword extraction from the question.
    2. Lookup in the graph to find matching nodes.
    3. Collect all unique structural tags (domains) associated with matching nodes.
    4. Fallback to ["general"] if no graph is provided or no tags are found.
    """
    if graph is None:
        return ["general"]

    keywords = _keywords_from_question(question)
    if not keywords:
        return ["general"]

    matched_domains = set()
    for kw in keywords:
        kw_lower = kw.lower()
        for node, node_data in graph.g.nodes(data=True):
            if kw_lower in node.lower():
                meta = node_data.get("meta")
                if meta and meta.domains:
                    for d in meta.domains:
                        matched_domains.add(d)

    if not matched_domains:
        return ["general"]

    return sorted(list(matched_domains))
