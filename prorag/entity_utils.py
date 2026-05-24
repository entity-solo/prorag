"""
Entity normalization and unresolved-reference helpers.
"""

from __future__ import annotations

import re


def normalize_entity_name(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,;:!?()[]{}")
    return text


def is_person_like_entity(value: str) -> bool:
    text = normalize_entity_name(value)
    if not text:
        return False
    if text.startswith(("mr ", "mrs ", "ms ", "dr ", "prof ")):
        return True
    tokens = [token for token in text.split() if token.isalpha()]
    return len(tokens) >= 2
