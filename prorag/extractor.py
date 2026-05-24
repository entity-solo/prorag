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

Find ALL noun phrases, pronouns, references, and descriptive details (such as clothing items, physical characteristics, specific events, and objects) in the text.
Map each mention to its canonical entity name.

Rules:
- Named entities (people, places, orgs, products, dates, numbers): use the most complete and specific form.
- Physical objects, clothing, and descriptive details (e.g. "black turtleneck", "blue jeans", "iphone launch event", "steve jobs's shirt"): Treat them as distinct entities if they carry descriptive meaning. Do NOT discard or map them to null.
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
The text contains annotated canonical entities enclosed in square brackets, like [entity name].
Use ONLY these annotated entities as subjects and objects. Do not extract relations for any text not marked as an entity.

Text:
\"\"\"
{text}
\"\"\"

Rules:
- Extract all explicit facts, including very minor, secondary, or seemingly trivial details (e.g., clothing items worn by individuals, specific actions during an event, physical attributes, small descriptive facts, relationships between objects).
- Subject and object must be exactly one of the annotated canonical entity names from the text (without the square brackets).
- Keep the same language as the input.
- Decompose complex sentences into atomic triples (e.g., "[apple], founded by [steve jobs] in [1976], is..." becomes: subject: "steve jobs", relation: "founded", object: "apple"; subject: "steve jobs", relation: "founded in", object: "1976"; subject: "apple", relation: "is"...).
- Normalize passive voice to active voice: If a relation in the text is passive (e.g., "A was released by B", "A được phát triển bởi B"), rewrite it in the active voice by swapping the subject and object (e.g. subject: "B", relation: "released", object: "A").
- Separate event context, speech/assertion time, and temporal aspect:
  - "condition": Context/precondition under which the fact is valid (e.g., "if stock rises", "at the 2007 iphone launch").
  - "statement_time": Exact date or time when this statement was asserted, spoken, or published in the document. Leave as empty string if not specified.
  - "temporal_aspect": Choose exactly one of "PAST" (events completed in the past), "PRESENT" (current states or general facts), or "FUTURE" (plans, projections, or future predictions).
- For negations, extract the statement affirmatively and set "negated" to true.

Return ONLY a JSON array:
[
  {{
    "subject": "canonical entity name",
    "relation": "relation verb or phrase",
    "object": "canonical entity name or value",
    "negated": false,
    "condition": "",
    "confidence": 0.9,
    "statement_time": "time statement was made",
    "temporal_aspect": "PAST/PRESENT/FUTURE"
  }}
]

JSON array:"""


def resolve_entities(
    text: str,
    history_sentences: list[str] | set[str] | None = None,
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
            known_entities="[]",
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


def extract_triples(
    text: str,
    entity_map: dict[str, str | None] | None = None,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
) -> list[dict]:
    """Extract canonical triples from text using the resolved entities.

    Accepts annotated text (containing square brackets like [canonical name]).
    For backward compatibility, if entity_map is provided or if text is raw (no brackets),
    it resolves entities first and substitutes them before extracting.
    """
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

    triples = []
    for fact in raw_facts:
        if not isinstance(fact, dict):
            continue
        triple = _prepare_triple(fact)
        if triple and _is_valid_triple(triple):
            if triple["subject"] not in valid_canonical_names:
                continue
            if triple["subject"] in null_mentions or triple["object"] in null_mentions:
                continue
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

        entity_map = resolve_entities(batch_text, history_sentences, llm_model=llm_model)

        for canonical in entity_map.values():
            if canonical is not None:
                all_resolved_entities.add(canonical)

        annotated_text = substitute_mentions(batch_text, entity_map)
        triples = extract_triples(annotated_text, source=source, llm_model=llm_model)

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
                    statement_time=triple.get("statement_time", ""),
                    temporal_aspect=triple.get("temporal_aspect", "PRESENT"),
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


def _fix_passive(triple: dict) -> dict:
    relation = triple.get("relation", "")
    passive_markers = ["was ", "were ", "được ", "bị "]
    has_passive = False
    matched_marker = ""
    for marker in passive_markers:
        if relation.startswith(marker):
            has_passive = True
            matched_marker = marker
            break

    if has_passive:
        subj = triple["subject"]
        obj = triple["object"]
        triple["subject"] = obj
        triple["object"] = subj

        relation = relation[len(matched_marker):].strip()

        if relation.endswith(" by"):
            relation = relation[:-3].strip()
        elif relation.endswith(" bởi"):
            relation = relation[:-4].strip()
        elif relation.startswith("by "):
            relation = relation[3:].strip()
        elif relation.startswith("bởi "):
            relation = relation[4:].strip()

        triple["relation"] = relation
    return triple


def _prepare_triple(fact: dict) -> dict | None:
    subject = normalize_entity_name(str(fact.get("subject", "") or ""))
    relation = _normalize_relation(fact.get("relation", ""))
    obj = normalize_entity_name(str(fact.get("object", "") or ""))
    if not subject or not relation or not obj:
        return None
    triple = {
        "subject": subject,
        "relation": relation,
        "object": obj,
        "negated": bool(fact.get("negated", False)),
        "condition": str(fact.get("condition", "") or "").strip(),
        "confidence": float(fact.get("confidence", 1.0) or 1.0),
        "statement_time": str(fact.get("statement_time", "") or "").strip(),
        "temporal_aspect": str(fact.get("temporal_aspect", "PRESENT") or "PRESENT").strip().upper(),
    }
    return _fix_passive(triple)


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
