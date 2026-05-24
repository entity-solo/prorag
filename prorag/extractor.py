"""
Ingestion pipeline: text -> entity resolution -> relation extraction -> graph writes.
"""

from __future__ import annotations

import json
import re

from .entity_utils import normalize_entity_name
from .graph import ProRAGGraph
from .llm import call_llm

_ENTITY_RESOLUTION_PROMPT = """\
You are an entity resolution assistant.

Find ALL noun phrases, pronouns, references, and descriptive details (such as clothing items, physical characteristics, specific events, and objects) in the text.
Map each mention to its canonical entity name.

Rules:
- Named entities (people, places, orgs, products, dates, numbers): use the most complete and specific form.
- Physical objects, clothing, and descriptive details (e.g. "black turtleneck", "blue jeans", "iphone launch event", "steve jobs's shirt"): Treat them as distinct entities if they carry descriptive meaning. Do NOT discard or map them to null.
- Pronouns (he, she, it, they, him, her, ...): resolve to the entity they refer to in order of appearance.
- Generic references ("the company", "the device", ...): resolve to the specific entity they refer to.
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

Extract factual relations, attributes, and events from the text.
The text contains annotated canonical entities enclosed in square brackets, like [entity name].

For each fact, categorize it into exactly one of three types:
1. "relation": A standard relationship between two annotated entities.
   - Requires: "subject" (an annotated entity) and "object" (an annotated entity).
2. "attribute": A property/attribute of an annotated entity, where the value is a literal/value, not another entity (e.g. [steve jobs] has attribute "died_on" = "october 5 2011").
   - Requires: "subject" (an annotated entity), "key" (attribute key), and "value" (attribute value).
3. "event": An n-ary event involving participants, times, or places.
   - For each event, generate a unique event ID in snake_case (e.g. "iphone_launch_2007" or "company_founding_1976").
   - Extract the participant roles (e.g., actor, object, location, time) as separate event facts.
   - Requires: "event_id" (generated event identifier), "role" (the role, e.g. actor, object, location, time), and "entity" (the entity or value playing that role, which may be annotated).

Text:
\"\"\"
{text}
\"\"\"

Rules:
- Subject, object, and entity values must use the exact annotated canonical entity names from the text (without the square brackets) if they refer to annotated entities.
- Keep the same language as the input.
- Decompose complex statements into atomic facts.
- Normalize passive voice to active voice for relations (e.g., "A was released by B" -> subject: B, relation: released, object: A).
- Extract the following linguistic metadata for all "relation" and "event" facts:
  - "condition": Context/precondition under which the fact is valid (e.g., "if stock rises").
  - "negated": Boolean (true/false) indicating if the fact is negated.
  - "confidence": Float (0.0 to 1.0) indicating belief/probability (default: 1.0).
  - "temporal_aspect": Choose exactly one of "PAST" (events completed in the past), "PRESENT" (current states or general facts), or "FUTURE" (plans, projections).
  - "aspect": Aspect of the action: choose one of "perfective" (completed action), "imperfective" (ongoing), "prospective" (about to happen), "habitual" (repeating), or leave as empty string if not specified.
  - "modality": Modal semantics: choose one of "certain" (asserted fact), "possible" (might happen), "necessary" (must happen), "counterfactual" (did not happen), or leave as empty string if not specified.
  - "quantifier": Quantification: "all", "some", "most", "over", "less_than", or empty string.
  - "evidentiality": Evidentiality: choose "direct" (direct observation), "reported" (someone said), "inferred" (reasoned), or empty string.
  - "speech_act": Pragmatics: choose "assertion" (fact claim), "claim" (unverified claim), "question", or empty string.
  - "causal": Event ID which is the cause of this fact, or empty string.
  - "statement_time": Exact date or time when this statement was asserted/spoken in the document.

Return ONLY a JSON array of objects, where each object has a "type" field ("relation", "attribute", or "event"):
[
  {{
    "type": "relation",
    "subject": "canonical entity name",
    "relation": "relation verb or phrase",
    "object": "canonical entity name",
    "negated": false,
    "condition": "",
    "confidence": 0.9,
    "statement_time": "time statement was made",
    "temporal_aspect": "PAST/PRESENT/FUTURE",
    "aspect": "",
    "modality": "",
    "quantifier": "",
    "evidentiality": "",
    "speech_act": "",
    "causal": ""
  }},
  {{
    "type": "attribute",
    "subject": "canonical entity name",
    "key": "attribute key",
    "value": "attribute value",
    "confidence": 1.0
  }},
  {{
    "type": "event",
    "event_id": "iphone_launch_2007",
    "role": "actor",
    "entity": "steve jobs",
    "negated": false,
    "condition": "",
    "confidence": 1.0,
    "statement_time": "january 2007",
    "temporal_aspect": "PAST",
    "aspect": "",
    "modality": "",
    "quantifier": "",
    "evidentiality": "",
    "speech_act": "",
    "causal": ""
  }}
]

JSON array:"""


def resolve_entities(
    text: str,
    history_sentences: list[str] | set[str] | None = None,
    known_entities: list[str] | set[str] | None = None,
    llm_model: str = "llama-3.3-70b-versatile",
) -> dict[str, str | None]:
    """Find all mentions in text and map each to a canonical entity name.

    If history_sentences is a set (for backward compatibility), it is treated as entity_registry.
    Otherwise, lazy context expansion is applied on mention resolution failure (resolves to null/None).
    """
    is_old_registry = isinstance(history_sentences, set)

    if is_old_registry:
        known_str = json.dumps(sorted(history_sentences), ensure_ascii=False)
        prompt = _ENTITY_RESOLUTION_PROMPT.format(
            known_entities=known_str,
            text=text.strip(),
        )
        raw = call_llm(prompt, model=llm_model, max_tokens=1024)
        return _parse_entity_map(raw)

    history_list = list(history_sentences) if history_sentences else []
    known_list = sorted(list(known_entities)) if known_entities else []
    known_str = json.dumps(known_list, ensure_ascii=False)
    max_retries = 2
    expand_sentences = 4

    for retry in range(max_retries + 1):
        if retry == 0:
            text_input = text.strip()
        else:
            n_sentences = retry * expand_sentences
            context_part = " ".join(history_list[-n_sentences:])
            text_input = (
                f"Previous context (for reference only, do NOT resolve entities from this part):\n"
                f"\"\"\"\n{context_part}\n\"\"\"\n\n"
                f"Text:\n{text.strip()}"
            )

        prompt = _ENTITY_RESOLUTION_PROMPT.format(
            known_entities=known_str,
            text=text_input,
        )
        raw = call_llm(prompt, model=llm_model, max_tokens=1024)
        entity_map = _parse_entity_map(raw)

        has_null = any(v is None for v in entity_map.values())
        if not has_null or not history_list or retry == max_retries:
            return entity_map

    return {}


def substitute_mentions(text: str, entity_map: dict[str, str | None]) -> str:
    """Replace resolved mentions with [canonical name] in text.

    Sorts by length descending to avoid replacing substrings first.
    Uses placeholders to prevent nested replacements of overlapping entities.
    """
    resolved = {k: v for k, v in entity_map.items() if v is not None}
    sorted_mentions = sorted(resolved.keys(), key=len, reverse=True)

    placeholders = {}
    for i, mention in enumerate(sorted_mentions):
        ph = f"___ENTITY_PLACEHOLDER_{i}___"
        placeholders[ph] = resolved[mention]
        text = text.replace(mention, ph)

    for ph, canonical in placeholders.items():
        text = text.replace(ph, f"[{canonical}]")

    return text


def extract_facts(
    text: str,
    entity_map: dict[str, str | None] | None = None,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
) -> list[dict]:
    """Extract canonical relations, attributes, and events from text."""
    if entity_map is not None:
        annotated_text = substitute_mentions(text, entity_map)
        valid_canonical_names = {normalize_entity_name(v) for v in entity_map.values() if v is not None}
        null_mentions = {normalize_entity_name(k) for k, v in entity_map.items() if v is None}
    elif not re.search(r"\[.+?\]", text):
        entity_map = resolve_entities(text, None, llm_model)
        annotated_text = substitute_mentions(text, entity_map)
        valid_canonical_names = {normalize_entity_name(v) for v in entity_map.values() if v is not None}
        null_mentions = {normalize_entity_name(k) for k, v in entity_map.items() if v is None}
    else:
        annotated_text = text
        valid_canonical_names = {normalize_entity_name(name) for name in re.findall(r"\[(.+?)\]", text)}
        null_mentions = set()

    prompt = _EXTRACT_PROMPT.format(
        text=annotated_text.strip(),
    )
    raw = call_llm(prompt, model=llm_model, max_tokens=3072)
    raw_facts = _parse_json_array(raw)

    facts = []
    for raw_fact in raw_facts:
        if not isinstance(raw_fact, dict):
            continue
        fact = _prepare_fact(raw_fact)
        if not fact:
            continue

        if fact["type"] == "relation":
            if fact["subject"] not in valid_canonical_names:
                continue
            if fact["subject"] in null_mentions or fact["object"] in null_mentions:
                continue
        elif fact["type"] == "attribute":
            if fact["subject"] not in valid_canonical_names:
                continue
            if fact["subject"] in null_mentions:
                continue
        elif fact["type"] == "event":
            if fact["entity"] in null_mentions:
                continue

        if source:
            fact["source"] = source
        facts.append(fact)
    return facts


def extract_triples(
    text: str,
    entity_map: dict[str, str | None] | None = None,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
) -> list[dict]:
    """Extract canonical triples from text using the resolved entities (legacy wrapper)."""
    facts = extract_facts(text, entity_map, source, llm_model)
    return [f for f in facts if f.get("type") == "relation"]


def ingest_text(
    text: str,
    graph: ProRAGGraph,
    entity_registry: set[str] | None = None,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
) -> tuple[int, set[str]]:
    """Extract canonical triples from text and add them to the graph.

    Reads by sentence batch (size 8) and performs lazy context expansion.
    Returns (triple_count, updated_entity_registry) for backward compatibility.
    """
    if not source:
        import hashlib
        source = f"hash_{hashlib.md5(text.encode('utf-8')).hexdigest()[:12]}"

    if hasattr(graph, "add_chunk"):
        graph.add_chunk(source, text)

    sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", text) if s.strip()]
    if not sentences:
        return 0, entity_registry or set()

    total_triples = 0
    all_resolved_entities = set(entity_registry) if entity_registry is not None else set()

    batch_size = 8
    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i : i + batch_size]
        batch_text = " ".join(batch_sentences)
        history_sentences = sentences[max(0, i - 8) : i]

        known_entities = set(graph.g.nodes)
        entity_map = resolve_entities(
            batch_text,
            history_sentences,
            known_entities=known_entities,
            llm_model=llm_model,
        )

        for canonical in entity_map.values():
            if canonical is not None:
                all_resolved_entities.add(canonical)

        annotated_text = substitute_mentions(batch_text, entity_map)
        facts = extract_facts(annotated_text, source=source, llm_model=llm_model)

        for fact in facts:
            try:
                fact_type = fact.get("type")
                if fact_type == "relation":
                    graph.add_relation(
                        subject=fact["subject"],
                        relation=fact["relation"],
                        obj=fact["object"],
                        source=fact.get("source", source),
                        condition=fact.get("condition", ""),
                        negated=fact.get("negated", False),
                        confidence=float(fact.get("confidence", 1.0)),
                        statement_time=fact.get("statement_time", ""),
                        temporal_aspect=fact.get("temporal_aspect", "PRESENT"),
                        aspect=fact.get("aspect", ""),
                        modality=fact.get("modality", ""),
                        quantifier=fact.get("quantifier", ""),
                        evidentiality=fact.get("evidentiality", ""),
                        speech_act=fact.get("speech_act", ""),
                        causal=fact.get("causal", ""),
                    )
                    total_triples += 1
                elif fact_type == "attribute":
                    graph.add_attribute(
                        subject=fact["subject"],
                        key=fact["key"],
                        value=fact["value"],
                        source=fact.get("source", source),
                        confidence=float(fact.get("confidence", 1.0)),
                    )
                    total_triples += 1
                elif fact_type == "event":
                    graph.add_event(
                        event_id=fact["event_id"],
                        role=fact["role"],
                        entity=fact["entity"],
                        source=fact.get("source", source),
                        condition=fact.get("condition", ""),
                        negated=fact.get("negated", False),
                        confidence=float(fact.get("confidence", 1.0)),
                        statement_time=fact.get("statement_time", ""),
                        temporal_aspect=fact.get("temporal_aspect", "PRESENT"),
                        aspect=fact.get("aspect", ""),
                        modality=fact.get("modality", ""),
                        quantifier=fact.get("quantifier", ""),
                        evidentiality=fact.get("evidentiality", ""),
                        speech_act=fact.get("speech_act", ""),
                        causal=fact.get("causal", ""),
                    )
                    total_triples += 1
            except (KeyError, TypeError, ValueError):
                continue

    return total_triples, all_resolved_entities


def ingest_file(
    path: str,
    graph: ProRAGGraph,
    source: str | None = None,
    llm_model: str = "llama-3.3-70b-versatile",
    chunk_size: int = 1500,
    overlap_sentences: int = 1,
) -> int:
    """Read a plain-text file and ingest all content directly."""
    with open(path, encoding="utf-8") as handle:
        content = handle.read()

    chunk_source_id = source or path
    count, _ = ingest_text(
        content,
        graph,
        entity_registry=None,
        source=chunk_source_id,
        llm_model=llm_model,
    )
    return count


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_entity_map(raw: str) -> dict[str, str | None]:
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


def _prepare_fact(fact: dict) -> dict | None:
    fact_type = str(fact.get("type", "")).strip().lower()
    if not fact_type:
        fact_type = "relation"
    if fact_type == "relation":
        subject = normalize_entity_name(str(fact.get("subject", "") or ""))
        relation = _normalize_relation(fact.get("relation", ""))
        obj = normalize_entity_name(str(fact.get("object", "") or ""))
        if not subject or not relation or not obj:
            return None
        return {
            "type": "relation",
            "subject": subject,
            "relation": relation,
            "object": obj,
            "negated": bool(fact.get("negated", False)),
            "condition": str(fact.get("condition", "") or "").strip(),
            "confidence": float(fact.get("confidence", 1.0) or 1.0),
            "statement_time": str(fact.get("statement_time", "") or "").strip(),
            "temporal_aspect": str(fact.get("temporal_aspect", "PRESENT") or "PRESENT").strip().upper(),
            "aspect": str(fact.get("aspect", "") or "").strip(),
            "modality": str(fact.get("modality", "") or "").strip(),
            "quantifier": str(fact.get("quantifier", "") or "").strip(),
            "evidentiality": str(fact.get("evidentiality", "") or "").strip(),
            "speech_act": str(fact.get("speech_act", "") or "").strip(),
            "causal": str(fact.get("causal", "") or "").strip(),
        }
    elif fact_type == "attribute":
        subject = normalize_entity_name(str(fact.get("subject", "") or ""))
        key = normalize_entity_name(str(fact.get("key", "") or ""))
        value = str(fact.get("value", "") or "").strip()
        if not subject or not key or not value:
            return None
        return {
            "type": "attribute",
            "subject": subject,
            "key": key,
            "value": value,
            "confidence": float(fact.get("confidence", 1.0) or 1.0),
        }
    elif fact_type == "event":
        event_id = normalize_entity_name(str(fact.get("event_id", "") or ""))
        role = normalize_entity_name(str(fact.get("role", "") or ""))
        entity = normalize_entity_name(str(fact.get("entity", "") or ""))
        if not event_id or not role or not entity:
            return None
        return {
            "type": "event",
            "event_id": event_id,
            "role": role,
            "entity": entity,
            "negated": bool(fact.get("negated", False)),
            "condition": str(fact.get("condition", "") or "").strip(),
            "confidence": float(fact.get("confidence", 1.0) or 1.0),
            "statement_time": str(fact.get("statement_time", "") or "").strip(),
            "temporal_aspect": str(fact.get("temporal_aspect", "PRESENT") or "PRESENT").strip().upper(),
            "aspect": str(fact.get("aspect", "") or "").strip(),
            "modality": str(fact.get("modality", "") or "").strip(),
            "quantifier": str(fact.get("quantifier", "") or "").strip(),
            "evidentiality": str(fact.get("evidentiality", "") or "").strip(),
            "speech_act": str(fact.get("speech_act", "") or "").strip(),
            "causal": str(fact.get("causal", "") or "").strip(),
        }
    return None


def _prepare_triple(fact: dict) -> dict | None:
    prepared = _prepare_fact({**fact, "type": "relation"})
    return prepared


def _normalize_relation(relation: str) -> str:
    return re.sub(r"\s+", " ", str(relation or "").strip().lower())


def _is_valid_triple(fact: dict) -> bool:
    return bool(
        fact.get("subject")
        and fact.get("relation")
        and fact.get("object")
    )


def _parse_json_array(text: str) -> list[dict]:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
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

    entities = {}
    for line in text.splitlines():
        m = re.search(r'"([^"\\]*(?:\\.[^"\\]*)*)"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', line)
        if m:
            mention, canonical = m.group(1), m.group(2)
            mention = mention.replace('\\"', '"')
            canonical = canonical.replace('\\"', '"')
            entities[mention] = canonical
    if entities:
        return {"entities": entities}

    return {}
