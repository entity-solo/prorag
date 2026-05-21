"""
Keyword helpers for lexical retrieval fallback.
"""

from __future__ import annotations

import re


def _keywords_from_question(question: str) -> list[str]:
    stopwords = {
        "what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "do", "does", "did", "can", "could", "should", "would", "be", "been",
        "that", "this", "these", "those", "by", "with", "from", "about", "into",
        "cai", "gi", "la", "cua", "va", "o", "tai", "khi", "nao", "co",
    }
    tokens = re.findall(r"\b\w{3,}\b", question.lower())
    return [token for token in tokens if token not in stopwords]
