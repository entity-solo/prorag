"""
Entity normalization and unresolved-reference helpers.

These utilities are shared by ingestion and graph-write code so we keep the
same invariant everywhere: graph entities must be canonical, not raw pronouns
or generic unresolved mentions.
"""

from __future__ import annotations

import re


_PERSON_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers",
    "anh ay", "ong ay", "co ay", "ba ay", "anh ta", "co ta",
}

_NEUTRAL_PRONOUNS = {
    "it", "its", "this", "that", "this one", "that one",
    "nó", "no", "cái này", "cai nay", "cái đó", "cai do",
}

_PLURAL_PRONOUNS = {
    "they", "them", "their", "theirs", "these", "those",
    "họ", "ho", "bọn họ", "bon ho", "những người này", "nhung nguoi nay",
}

_GENERIC_REFERENCES = {
    "the company", "this company", "that company",
    "the film", "this film", "that film",
    "the device", "this device", "that device",
    "the product", "this product", "that product",
    "the service", "this service", "that service",
    "the team", "this team", "that team",
    "the organization", "this organization", "that organization",
    "the startup", "this startup", "that startup",
    "the law", "this law", "that law",
    "the document", "this document", "that document",
    "the company itself", "the latter", "the former",
    "công ty này", "cong ty nay", "công ty đó", "cong ty do",
    "bộ phim này", "bo phim nay", "bộ phim đó", "bo phim do",
    "thiết bị này", "thiet bi nay", "thiết bị đó", "thiet bi do",
    "sản phẩm này", "san pham nay", "sản phẩm đó", "san pham do",
    "dịch vụ này", "dich vu nay", "dịch vụ đó", "dich vu do",
}

PRONOUN_REFERENCES = _PERSON_PRONOUNS | _NEUTRAL_PRONOUNS | _PLURAL_PRONOUNS
UNRESOLVED_REFERENCES = PRONOUN_REFERENCES | _GENERIC_REFERENCES


def normalize_entity_name(value: str) -> str:
    """Return a normalized entity string suitable for graph storage."""
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,;:!?()[]{}")
    return text


def is_unresolved_reference(value: str) -> bool:
    """True when the value looks like a pronoun or generic unresolved mention."""
    text = normalize_entity_name(value)
    return text in UNRESOLVED_REFERENCES


def is_plural_reference(value: str) -> bool:
    return normalize_entity_name(value) in _PLURAL_PRONOUNS


def is_person_reference(value: str) -> bool:
    return normalize_entity_name(value) in _PERSON_PRONOUNS


def is_neutral_reference(value: str) -> bool:
    return normalize_entity_name(value) in _NEUTRAL_PRONOUNS


def is_person_like_entity(value: str) -> bool:
    """
    Lightweight heuristic for person mentions.

    We bias toward multi-token alphabetic names and a few honorific patterns.
    """
    text = normalize_entity_name(value)
    if not text or is_unresolved_reference(text):
        return False
    if text.startswith(("mr ", "mrs ", "ms ", "dr ", "prof ", "ông ", "ba ", "co ", "cô ", "anh ")):
        return True
    tokens = [token for token in text.split() if token.isalpha()]
    return len(tokens) >= 2
