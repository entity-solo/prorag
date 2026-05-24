"""
Entity normalization and unresolved-reference helpers.
"""

from __future__ import annotations

import re


_PERSON_PRONOUNS = {
    "he", "him", "his", "she", "her", "hers",
}

_NEUTRAL_PRONOUNS = {
    "it", "its", "this", "that", "this one", "that one",
}

_PLURAL_PRONOUNS = {
    "they", "them", "their", "theirs", "these", "those",
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
}

PRONOUN_REFERENCES = _PERSON_PRONOUNS | _NEUTRAL_PRONOUNS | _PLURAL_PRONOUNS
UNRESOLVED_REFERENCES = PRONOUN_REFERENCES | _GENERIC_REFERENCES


def normalize_entity_name(value: str) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\r\n\"'`.,;:!?()[]{}")
    return text


def is_unresolved_reference(value: str) -> bool:
    return normalize_entity_name(value) in UNRESOLVED_REFERENCES


def is_plural_reference(value: str) -> bool:
    return normalize_entity_name(value) in _PLURAL_PRONOUNS


def is_person_reference(value: str) -> bool:
    return normalize_entity_name(value) in _PERSON_PRONOUNS


def is_neutral_reference(value: str) -> bool:
    return normalize_entity_name(value) in _NEUTRAL_PRONOUNS


def is_person_like_entity(value: str) -> bool:
    text = normalize_entity_name(value)
    if not text or is_unresolved_reference(text):
        return False
    if text.startswith(("mr ", "mrs ", "ms ", "dr ", "prof ")):
        return True
    tokens = [token for token in text.split() if token.isalpha()]
    return len(tokens) >= 2
