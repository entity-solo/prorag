"""
Proactive ingestion pipeline: text -> mention facts -> coref resolution ->
canonical triples -> graph writes.

The graph still stores triples, but ingestion no longer lets unresolved
pronouns or generic mentions flow straight into the graph.
"""

from __future__ import annotations

import json
import re

from .entity_utils import (
    is_neutral_reference,
    is_person_like_entity,
    is_person_reference,
    is_plural_reference,
    is_unresolved_reference,
    normalize_entity_name,
)
from .graph import ProRAGGraph
from .llm import call_llm

_EXTRACT_PROMPT = """\
You are a knowledge graph extractor.

Given a text passage, extract ALL factual statements as JSON objects in text order.
Return ONLY a JSON array - no markdown, no explanation.

Each fact object must follow this schema:
{{
  "subject_mention": "surface mention from the text",
  "subject": "canonical entity name if clear, else empty string",
  "relation": "relation verb/phrase",
  "object_mention": "surface mention from the text",
  "object": "canonical entity/value if clear, else empty string",
  "negated": false,
  "condition": "",
  "confidence": 0.9
}}

Rules:
- Extract all explicit facts and strong implicit/nested facts.
- Keep entities and relations in the same language as the input.
- If a pronoun or generic mention ("it", "he", "the company", "the film", "nó", "công ty này") has a clear antecedent in the passage, write the resolved canonical entity in "subject" or "object".
- If the antecedent is unclear, keep the original mention in "*_mention" and leave the canonical "subject" or "object" field empty.
- Break compound sentences into multiple facts.
- Keep entity names concise, lowercase-friendly, and semantically precise.
- Preserve dates, quantities, laws, and products as objects when they are the factual target.
- Do not invent facts not grounded in the passage.

Text:
\"\"\"
{text}
\"\"\"

JSON array:"""

_COREF_PROMPT = """\
You are resolving a reference inside a passage.
Choose the single best antecedent for the mention from the candidate list.
Return ONLY JSON in this form:
{{"resolved": "exact candidate text or empty string", "confidence": 0.0}}

Rules:
- Use only the candidates provided.
- Return the exact candidate text, unchanged.
- If the mention is ambiguous or unsupported, return an empty string.

Passage:
\"\"\"
{text}
\"\"\"

Mention: {mention}
Candidates: {candidates}
JSON:"""


def extract_triples(
    text: str,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
    extra_domains: list[str] | None = None,
) -> list[dict]:
    """Return validated canonical triples extracted from a text passage."""
    prompt = _EXTRACT_PROMPT.format(text=text.strip())
    raw = call_llm(prompt, model=llm_model, max_tokens=3072)
    raw_facts = _parse_json_array(raw)
    triples = _normalize_extracted_facts(raw_facts, text=text, llm_model=llm_model)

    for t in triples:
        t["domains"] = t.get("domains", ["general"])

    if extra_domains:
        for t in triples:
            for d in extra_domains:
                if d not in t.get("domains", []):
                    t.setdefault("domains", []).append(d)

    if source:
        for t in triples:
            t["source"] = source

    return triples


def ingest_text(
    text: str,
    graph: ProRAGGraph,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
    extra_domains: list[str] | None = None,
) -> int:
    """
    Extract canonical triples from text and add them to the graph.
    Returns the number of validated triples added.
    """
    triples = extract_triples(text, source=source, llm_model=llm_model, extra_domains=extra_domains)
    for t in triples:
        try:
            graph.add_triple(
                subject=t["subject"],
                relation=t["relation"],
                obj=t["object"],
                domains=t.get("domains", ["general"]),
                source=t.get("source", source),
                condition=t.get("condition", ""),
                negated=t.get("negated", False),
                confidence=float(t.get("confidence", 1.0)),
            )
        except (KeyError, TypeError, ValueError):
            continue
    return len(triples)


def ingest_file(
    path: str,
    graph: ProRAGGraph,
    source: str | None = None,
    llm_model: str = "llama-3.3-70b-versatile",
    chunk_size: int = 1500,
    overlap_sentences: int = 1,
) -> int:
    """Read a plain-text file and ingest all chunks with sentence overlap."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    source = source or path
    chunks = _chunk_text(content, chunk_size, overlap_sentences=overlap_sentences)
    total = 0
    for chunk in chunks:
        total += ingest_text(chunk, graph, source=source, llm_model=llm_model)
    return total


def _normalize_extracted_facts(raw_facts: list[dict], text: str, llm_model: str) -> list[dict]:
    """Resolve mentions, canonicalize entities, and drop invalid triples."""
    prepared = [_prepare_fact(fact) for fact in raw_facts if isinstance(fact, dict)]
    recent_entities: list[str] = []
    normalized: list[dict] = []

    for fact in prepared:
        fact["subject"] = _resolve_entity(
            fact,
            field="subject",
            text=text,
            recent_entities=recent_entities,
            llm_model=llm_model,
        )
        _remember_entity(recent_entities, fact["subject"])

        fact["object"] = _resolve_entity(
            fact,
            field="object",
            text=text,
            recent_entities=recent_entities,
            llm_model=llm_model,
        )
        _remember_entity(recent_entities, fact["object"])

        fact["relation"] = _normalize_relation(fact.get("relation", ""))
        fact["condition"] = str(fact.get("condition", "") or "").strip()
        fact["confidence"] = float(fact.get("confidence", 1.0) or 1.0)
        fact["negated"] = bool(fact.get("negated", False))
        fact["unresolved_coref"] = not bool(fact["subject"] and fact["object"])

        if _is_valid_triple(fact):
            normalized.append(fact)

    return normalized


def _prepare_fact(fact: dict) -> dict:
    """Accept both old triple schema and richer mention-level schema."""
    subject_mention = fact.get("subject_mention", fact.get("subject", ""))
    object_mention = fact.get("object_mention", fact.get("object", ""))
    return {
        "subject_mention": str(subject_mention or "").strip(),
        "subject": str(fact.get("subject", "") or "").strip(),
        "relation": str(fact.get("relation", "") or "").strip(),
        "object_mention": str(object_mention or "").strip(),
        "object": str(fact.get("object", "") or "").strip(),
        "negated": fact.get("negated", False),
        "condition": fact.get("condition", ""),
        "confidence": fact.get("confidence", 1.0),
    }


def _resolve_entity(
    fact: dict,
    field: str,
    text: str,
    recent_entities: list[str],
    llm_model: str,
) -> str:
    canonical = normalize_entity_name(fact.get(field, ""))
    mention = normalize_entity_name(fact.get(f"{field}_mention", ""))

    if canonical and not is_unresolved_reference(canonical):
        return canonical

    if mention and not is_unresolved_reference(mention):
        return mention

    if not mention:
        return ""

    resolved = _resolve_reference(
        mention=mention,
        text=text,
        recent_entities=recent_entities,
        llm_model=llm_model,
    )
    return normalize_entity_name(resolved)


def _resolve_reference(
    mention: str,
    text: str,
    recent_entities: list[str],
    llm_model: str,
) -> str:
    mention = normalize_entity_name(mention)
    if not mention:
        return ""

    candidate_pool = _distinct_recent_entities(recent_entities)
    if not candidate_pool:
        return ""

    heuristic = _resolve_reference_heuristic(mention, candidate_pool)
    if heuristic and (len(candidate_pool) == 1 or is_person_reference(mention) or is_plural_reference(mention)):
        return heuristic

    llm_resolved = _resolve_reference_with_llm(mention, text, candidate_pool, llm_model)
    return llm_resolved or heuristic


def _distinct_recent_entities(recent_entities: list[str], limit: int = 8) -> list[str]:
    """Most-recent-first distinct entity list."""
    ordered: list[str] = []
    for entity in reversed(recent_entities):
        if entity and entity not in ordered:
            ordered.append(entity)
        if len(ordered) >= limit:
            break
    return ordered


def _resolve_reference_heuristic(mention: str, candidates: list[str]) -> str:
    """Cheap deterministic resolver before optional LLM disambiguation."""
    if not candidates:
        return ""

    if is_plural_reference(mention):
        pair = []
        for candidate in candidates:
            if candidate not in pair:
                pair.append(candidate)
            if len(pair) == 2:
                break
        if len(pair) == 2:
            return " and ".join(pair)
        return pair[0]

    if is_person_reference(mention):
        for candidate in candidates:
            if is_person_like_entity(candidate):
                return candidate

    if is_neutral_reference(mention):
        return candidates[0]

    return candidates[0]


def _resolve_reference_with_llm(
    mention: str,
    text: str,
    candidates: list[str],
    llm_model: str,
) -> str:
    """LLM fallback for ambiguous references when heuristics are not enough."""
    prompt = _COREF_PROMPT.format(
        text=text.strip(),
        mention=mention,
        candidates=json.dumps(candidates, ensure_ascii=False),
    )
    try:
        raw = call_llm(prompt, model=llm_model, max_tokens=256)
    except Exception:
        return ""

    parsed = _parse_json_object(raw)
    resolved = normalize_entity_name(parsed.get("resolved", "")) if parsed else normalize_entity_name(raw)
    candidate_map = {normalize_entity_name(candidate): candidate for candidate in candidates}
    return candidate_map.get(resolved, "")


def _remember_entity(recent_entities: list[str], entity: str, limit: int = 16) -> None:
    entity = normalize_entity_name(entity)
    if not entity or is_unresolved_reference(entity):
        return
    recent_entities.append(entity)
    if len(recent_entities) > limit:
        del recent_entities[:-limit]


def _normalize_relation(relation: str) -> str:
    text = str(relation or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def _is_valid_triple(fact: dict) -> bool:
    if not fact.get("subject") or not fact.get("relation") or not fact.get("object"):
        return False
    if is_unresolved_reference(fact["subject"]) or is_unresolved_reference(fact["object"]):
        return False
    return True


def _parse_json_array(text: str) -> list[dict]:
    """Best-effort JSON array parser - strips markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, list):
                return result
        except json.JSONDecodeError:
            pass

    return []


def _parse_json_object(text: str) -> dict:
    """Best-effort JSON object parser for LLM coreference fallback."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            pass

    return {}


def _chunk_text(text: str, size: int, overlap_sentences: int = 1) -> list[str]:
    """Split text into ~size-character chunks with sentence overlap."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
    if not sentences:
        return []

    chunks: list[str] = []
    start = 0
    overlap = max(0, overlap_sentences)

    while start < len(sentences):
        current: list[str] = []
        end = start
        current_len = 0

        while end < len(sentences):
            sentence = sentences[end]
            extra = len(sentence) + (1 if current else 0)
            if current and current_len + extra > size:
                break
            current.append(sentence)
            current_len += extra
            end += 1

        if not current:
            current = [sentences[start]]
            end = start + 1

        chunks.append(" ".join(current).strip())
        if end >= len(sentences):
            break
        start = max(end - overlap, start + 1)

    return chunks
