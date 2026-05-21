"""
Question keyword helpers.

This module is intentionally small: the current retrieval architecture is
entity-first and slot-guided, but lightweight question keywords still help
with lexical seeding and fallback retrieval.
"""

from __future__ import annotations

import re


def _keywords_from_question(question: str) -> list[str]:
    """Extract candidate content words from a question."""
    stopwords = {
        "what", "who", "when", "where", "why", "how", "is", "are", "was", "were",
        "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
        "did", "does", "do", "which", "also", "originally", "should", "could", "would",
        "that", "this", "these", "those", "by", "with", "from", "about", "into",
        "cái", "gì", "là", "của", "và", "ở", "tại", "khi", "nào", "có",
    }
    tokens = re.findall(r"\b\w{3,}\b", question.lower())
    return [token for token in tokens if token not in stopwords]
