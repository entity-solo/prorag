"""
Ingestion pipeline: text -> entity resolution -> relation extraction -> graph writes.
"""

from __future__ import annotations

import json
import re

from .entity_utils import is_unresolved_reference, normalize_entity_name
from .graph import ProRAGGraph
from .llm import call_llm

_ENTITY_RESOLUTION_PROMPT = """\
You are an entity resolution assistant.

Find ALL noun phrases, pronouns, and references in the text.
Map each mention to its canonical entity name.

Rules:
- Named entities (people, places, orgs, products, dates, numbers): use the most complete and specific form.
- Pronouns (he, she, it, they, him, her, "ông", "cô", "họ", "anh", "chị", ...): resolve to the entity they refer to in order of appearance.
- Generic references ("the company", "the device", "công ty này", "tổ chức đó", ...): resolve to the specific entity they refer to.
- If a mention refers to a known entity from the list below: use that exact known entity name.
- If a mention is a new entity not in the known list: create a canonical name using the most complete form found in the text.
- If a mention is ambiguous and cannot be confidently resolved: map to null.
- Keep the same language as the input for canonical names.

Known entities (from previous context):
{known_entities}

Text:
\"\"\"
{text}
\"\"\"

Return ONLY JSON:
{{
  "entities": {{
    "<mention as it appears in text>": "<canonical name or null>",
    ...
  }}
}}"""

_EXTRACT_PROMPT = """\
You are a knowledge graph extractor.

Extract factual relations between entities from the text.
Use ONLY the canonical entity names provided in the entity map below.
Do not use any mention that maps to null.

Entity map:
{entity_map}

Text:
\"\"\"
{text}
\"\"\"

Rules:
- Extract only explicit facts. Do not invent facts.
- Subject and object must be canonical names from the entity map.
- Skip any fact where subject or object is null or not in the entity map.
- Keep the same language as the input.

Return ONLY a JSON array:
[
  {{
    "subject": "canonical entity name",
    "relation": "relation verb or phrase",
    "object": "canonical entity name or value",
    "negated": false,
    "condition": "",
    "confidence": 0.9
  }}
]

JSON array:"""


def resolve_entities(
    text: str,
    entity_registry: set[str],
    llm_model: str = "llama-3.3-70b-versatile",
) -> dict[str, str | None]:
    """Find all mentions in text and map each to a canonical entity name.

    Uses entity_registry (known entities from previous chunks) as candidates.
    Returns entity_map {mention -> canonical_name or None if ambiguous}.
    """
    prompt = _ENTITY_RESOLUTION_PROMPT.format(
        known_entities=json.dumps(sorted(entity_registry), ensure_ascii=False),
        text=text.strip(),
    )
    raw = call_llm(prompt, model=llm_model, max_tokens=1024)
    parsed = _parse_json_object(raw)
    raw_map = parsed.get("entities", {}) if parsed else {}

    entity_map: dict[str, str | None] = {}
    for mention, canonical in raw_map.items():
        mention_str = str(mention).strip()
        if not mention_str:
            continue
        if canonical is None:
            entity_map[mention_str] = None
        else:
            canonical_norm = normalize_entity_name(str(canonical).strip())
            entity_map[mention_str] = canonical_norm if canonical_norm else None
    return entity_map


def extract_triples(
    text: str,
    entity_map: dict[str, str | None] | None = None,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
) -> list[dict]:
    """Extract canonical triples from text using the resolved entity map.

    If entity_map is not provided, resolve_entities is called first with an
    empty registry (single-chunk usage, backward compatible).
    """
    if entity_map is None:
        entity_map = resolve_entities(text, set(), llm_model)

    valid_map = {k: v for k, v in entity_map.items() if v is not None}
    prompt = _EXTRACT_PROMPT.format(
        entity_map=json.dumps(valid_map, ensure_ascii=False),
        text=text.strip(),
    )
    raw = call_llm(prompt, model=llm_model, max_tokens=3072)
    raw_facts = _parse_json_array(raw)

    triples = []
    for fact in raw_facts:
        if not isinstance(fact, dict):
            continue
        triple = _prepare_triple(fact)
        if triple and _is_valid_triple(triple):
            if source:
                triple["source"] = source
            triples.append(triple)
    return triples


def ingest_text(
    text: str,
    graph: ProRAGGraph,
    entity_registry: set[str] | None = None,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
) -> tuple[int, set[str]]:
    """Extract canonical triples from text and add them to the graph.

    Returns (triple_count, updated_entity_registry).
    entity_registry accumulates known canonical names across chunks.
    """
    if entity_registry is None:
        entity_registry = set()

    entity_map = resolve_entities(text, entity_registry, llm_model)

    new_registry = entity_registry | {
        v for v in entity_map.values() if v is not None
    }

    triples = extract_triples(text, entity_map, source=source, llm_model=llm_model)
    for triple in triples:
        try:
            graph.add_triple(
                subject=triple["subject"],
                relation=triple["relation"],
                obj=triple["object"],
                source=triple.get("source", source),
                condition=triple.get("condition", ""),
                negated=triple.get("negated", False),
                confidence=float(triple.get("confidence", 1.0)),
            )
        except (KeyError, TypeError, ValueError):
            continue

    return len(triples), new_registry


def ingest_file(
    path: str,
    graph: ProRAGGraph,
    source: str | None = None,
    llm_model: str = "llama-3.3-70b-versatile",
    chunk_size: int = 1500,
    overlap_sentences: int = 1,
) -> int:
    """Read a plain-text file and ingest all chunks.

    Entity registry is propagated across chunks so cross-chunk references
    (pronouns, generic refs) are resolved correctly.
    """
    with open(path, encoding="utf-8") as handle:
        content = handle.read()

    entity_registry: set[str] = set()
    total = 0

    for chunk in _chunk_text(content, chunk_size, overlap_sentences=overlap_sentences):
        count, entity_registry = ingest_text(
            chunk,
            graph,
            entity_registry,
            source=source or path,
            llm_model=llm_model,
        )
        total += count

    return total


# ── helpers ───────────────────────────────────────────────────────────────────

def _prepare_triple(fact: dict) -> dict | None:
    subject = normalize_entity_name(str(fact.get("subject", "") or ""))
    relation = _normalize_relation(fact.get("relation", ""))
    obj = normalize_entity_name(str(fact.get("object", "") or ""))
    if not subject or not relation or not obj:
        return None
    return {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "negated": bool(fact.get("negated", False)),
        "condition": str(fact.get("condition", "") or "").strip(),
        "confidence": float(fact.get("confidence", 1.0) or 1.0),
    }


def _normalize_relation(relation: str) -> str:
    return re.sub(r"\s+", " ", str(relation or "").strip().lower())


def _is_valid_triple(fact: dict) -> bool:
    return bool(
        fact.get("subject")
        and fact.get("relation")
        and fact.get("object")
        and not is_unresolved_reference(fact["subject"])
        and not is_unresolved_reference(fact["object"])
    )


def _parse_json_array(text: str) -> list[dict]:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, list):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group())
            if isinstance(value, list):
                return value
        except json.JSONDecodeError:
            pass
    return []


def _parse_json_object(text: str) -> dict:
    text = re.sub(r"^```[a-z]*\n?", "", text.strip())
    text = re.sub(r"\n?```$", "", text)
    try:
        value = json.loads(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            value = json.loads(match.group())
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return {}


def _chunk_text(text: str, size: int, overlap_sentences: int = 1) -> list[str]:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    overlap = max(0, overlap_sentences)

    while start < len(sentences):
        current: list[str] = []
        current_length = 0
        end = start

        while end < len(sentences):
            sentence = sentences[end]
            extra = len(sentence) + (1 if current else 0)
            if current and current_length + extra > size:
                break
            current.append(sentence)
            current_length += extra
            end += 1

        if not current:
            current = [sentences[start]]
            end = start + 1

        chunks.append(" ".join(current).strip())
        if end >= len(sentences):
            break
        start = max(end - overlap, start + 1)

    return chunks
