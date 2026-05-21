"""
Query pipeline - entity-first retrieval with 5W answer-slot guidance.

Flow:
  question
    -> detect seed entities
    -> detect question slot (who/what/when/where/why/how/how_many)
    -> infer relation cues from the question
    -> retrieve candidate triples from the graph
    -> rerank evidence using entity, relation, and slot signals
    -> single LLM call
"""

from __future__ import annotations

import re

from .detector import _keywords_from_question
from .entity_utils import is_person_like_entity, normalize_entity_name
from .graph import ProRAGGraph
from .llm import call_llm

_ANSWER_PROMPT = """\
You are a precise question-answering assistant.
Answer the question using ONLY the knowledge graph context below.

Rules for your answer:
1. Provide a highly concise, short-phrase answer (e.g. only the name, date, or "yes"/"no").
2. Do NOT write full sentences or conversational responses.
3. If the context does not contain enough information, say "I don't have enough information to answer this."
4. Never make up facts not present in the context.

## Knowledge Graph Context
{context}

## Question
{question}

## Answer"""

_CONTRADICTIONS_NOTE = "\nNote: conflicting information exists - see sources."

_SLOT_HINTS = {
    "who": (
        "person", "people", "founder", "found", "ceo", "president", "director",
        "author", "inventor", "actor", "starred", "led", "appointed",
        "ai", "nhung ai", "nguoi nao", "ai la", "ai da",
    ),
    "what": (
        "what", "which", "name", "title", "product", "law", "concept", "thing",
        "cai gi", "la gi", "ten gi", "thu gi",
    ),
    "when": (
        "when", "year", "date", "month", "day", "time", "born", "died",
        "released", "launched", "announced", "founded",
        "khi nao", "nam nao", "ngay nao", "thoi diem nao",
    ),
    "where": (
        "where", "location", "place", "country", "city", "headquarters",
        "born", "filmed", "located", "based", "from",
        "o dau", "tai dau", "noi nao", "thuoc dau",
    ),
    "why": (
        "why", "reason", "because", "cause", "due", "motivation", "explain",
        "tai sao", "vi sao", "ly do",
    ),
    "how": (
        "how", "method", "process", "way", "approach", "worked", "operate",
        "the nao", "bang cach nao", "ra sao",
    ),
    "how_many": (
        "how many", "how much", "number of", "amount of", "count",
        "bao nhieu", "so luong", "may",
    ),
    "yes_no": (
        "is", "are", "was", "were", "do", "does", "did", "can", "could",
        "co phai", "co", "da", "duoc khong",
    ),
}

_SLOT_RELATION_HINTS = {
    "who": ("by", "founded", "founded by", "ceo", "president", "director", "author", "invented", "created", "led", "appointed", "stars"),
    "what": ("is", "means", "called", "named", "contains", "includes", "describes", "announced", "released"),
    "when": ("in", "on", "at", "during", "since", "born", "died", "released", "launched", "announced", "founded", "created"),
    "where": ("in", "at", "from", "located", "based", "headquartered", "born", "filmed", "shot", "held", "lives"),
    "why": ("because", "caused", "causes", "due", "resulted", "led to", "reason", "motivated", "triggered"),
    "how": ("by", "using", "through", "via", "method", "process", "worked", "operates"),
    "how_many": ("number", "count", "contains", "total", "amount", "population", "size"),
    "yes_no": ("is", "was", "has", "have", "can", "supports", "contains"),
}

_QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should",
    "ai", "cai", "gi", "la", "khi", "o", "tai", "dau", "tai sao", "vi sao", "nao", "bao", "nhieu",
}


def answer(
    question: str,
    graph: ProRAGGraph,
    llm_model: str = "llama-3.3-70b-versatile",
    max_context_triples: int = 60,
) -> dict:
    """
    Answer a question from the knowledge graph.

    Returns:
        {
          "answer": str,
          "sources": list[str],
          "domains": list[str],
          "triples_used": int,
          "has_contradictions": bool,
        }
    """
    domains = ["general"]

    triples, _retrieval = retrieve_evidence(
        question,
        graph,
        top_k=max_context_triples,
    )

    context, sources, has_contradictions = _format_context(triples)
    if not context:
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": [],
            "domains": domains,
            "triples_used": 0,
            "has_contradictions": False,
        }

    prompt = _ANSWER_PROMPT.format(context=context, question=question)
    answer_text = call_llm(prompt, model=llm_model, max_tokens=1024)
    if has_contradictions:
        answer_text += _CONTRADICTIONS_NOTE

    return {
        "answer": answer_text.strip(),
        "sources": sorted(set(sources)),
        "domains": domains,
        "triples_used": len(triples),
        "has_contradictions": has_contradictions,
    }


def retrieve_evidence(
    question: str,
    graph: ProRAGGraph,
    *,
    top_k: int = 40,
    candidate_k: int | None = None,
) -> tuple[list[dict], dict]:
    """
    Full retrieval pipeline:
      1. detect seed entities
      2. detect answer slot
      3. infer relation cues
      4. fetch candidate triples
      5. rerank candidates into final evidence
    """
    slot = detect_question_slot(question)
    seed_entities = _detect_seed_entities(question, graph, limit=max(3, min(8, top_k // 4 or 3)))
    relation_cues = _infer_relation_cues(question, seed_entities, slot)
    keywords = _keywords_from_question(question)

    candidate_limit = candidate_k or max(top_k * 3, 30)
    candidates = _retrieve_candidate_triples(
        question,
        graph,
        keywords=keywords,
        top_k=candidate_limit,
        seed_k=max(10, len(seed_entities) * 3 or 10),
    )
    reranked = _rerank_triples(
        question,
        candidates,
        seed_entities=seed_entities,
        relation_cues=relation_cues,
        slot=slot,
    )
    selected, paths = _select_evidence(
        reranked,
        seed_entities=seed_entities,
        slot=slot,
        top_k=top_k,
    )
    return selected[:top_k], {
        "slot": slot,
        "seed_entities": seed_entities,
        "relation_cues": relation_cues,
        "path_count": len(paths),
    }


def detect_question_slot(question: str) -> str:
    """Map a question to a soft 5W-style answer slot."""
    q = normalize_entity_name(question)
    if not q:
        return "what"

    if any(phrase in q for phrase in ("how many", "how much", "number of", "bao nhieu", "so luong")):
        return "how_many"
    if any(q.startswith(prefix) for prefix in ("who", "ai", "whom")):
        return "who"
    if any(q.startswith(prefix) for prefix in ("when", "khi", "nam nao", "ngay nao")):
        return "when"
    if any(q.startswith(prefix) for prefix in ("where", "o dau", "tai dau", "noi nao")):
        return "where"
    if any(q.startswith(prefix) for prefix in ("why", "tai sao", "vi sao")):
        return "why"
    if any(q.startswith(prefix) for prefix in ("how", "the nao", "bang cach nao", "ra sao")):
        return "how"

    scores = {slot: 0 for slot in _SLOT_HINTS}
    for slot, hints in _SLOT_HINTS.items():
        for hint in hints:
            if hint in q:
                scores[slot] += 1

    best_slot, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score > 0:
        return best_slot
    return "what"


def _detect_seed_entities(question: str, graph: ProRAGGraph, limit: int = 5) -> list[str]:
    """Find the most relevant graph entities for the question."""
    keywords = _keywords_from_question(question)
    lexical_scores: dict[str, float] = {}

    for node in graph.g.nodes:
        score = 0.0
        node_text = normalize_entity_name(node)
        for kw in keywords:
            kw_norm = normalize_entity_name(kw)
            if not kw_norm:
                continue
            if kw_norm == node_text:
                score += 3.0
            elif kw_norm in node_text:
                score += 1.5
            elif node_text in kw_norm and len(node_text) >= 3:
                score += 1.0
        if score > 0:
            lexical_scores[node] = score

    try:
        from .embeddings import EmbeddingStore
        store = EmbeddingStore()
        semantic = store.top_k(question, list(graph.g.nodes), k=max(limit * 3, 8), threshold=0.15)
    except ImportError:
        semantic = []

    combined: dict[str, float] = {}
    for node, score in lexical_scores.items():
        combined[node] = combined.get(node, 0.0) + score
    for node, score in semantic:
        combined[node] = combined.get(node, 0.0) + max(0.0, score) * 4.0

    ranked = sorted(combined.items(), key=lambda item: -item[1])
    return [node for node, _ in ranked[:limit]]


def _infer_relation_cues(question: str, seed_entities: list[str], slot: str) -> list[str]:
    """Extract relation-oriented cues from the question after removing entity mentions."""
    q = normalize_entity_name(question)
    for entity in seed_entities:
        pattern = re.escape(normalize_entity_name(entity))
        if pattern:
            q = re.sub(pattern, " ", q)

    tokens = re.findall(r"\b\w+\b", q)
    cues = []
    for token in tokens:
        if len(token) < 3 or token in _QUESTION_WORDS:
            continue
        cues.append(token)

    # Preserve some short but high-signal relation words.
    for token in ("by", "in", "at", "on", "due"):
        if re.search(rf"\b{re.escape(token)}\b", q):
            cues.append(token)

    cues.extend(_SLOT_RELATION_HINTS.get(slot, ()))
    return _unique(cues)


def _retrieve_candidate_triples(
    question: str,
    graph: ProRAGGraph,
    *,
    keywords: list[str],
    top_k: int,
    seed_k: int,
) -> list[dict]:
    """Use the graph's main retrieval engine, then fall back if embeddings are unavailable."""
    try:
        return graph.query_vector(
            question,
            max_cost=2.2,
            top_k=top_k,
            seed_k=seed_k,
            seed_threshold=0.18,
        )
    except ImportError:
        return graph.query(keywords, domains=None, top_k=top_k)


def _rerank_triples(
    question: str,
    triples: list[dict],
    *,
    seed_entities: list[str],
    relation_cues: list[str],
    slot: str,
) -> list[dict]:
    """Rerank evidence with entity-first and 5W relation guidance."""
    if not triples:
        return []

    question_text = normalize_entity_name(question)
    reranked = []
    for triple in triples:
        entity_score = _entity_alignment_score(triple, seed_entities)
        relation_score = _relation_alignment_score(triple, relation_cues, question_text)
        slot_score = _slot_alignment_score(triple, slot)
        distance = float(triple.get("distance", 1.0))
        similarity = float(triple.get("similarity", 0.0))
        confidence = float(triple.get("confidence", 1.0))
        contradiction_penalty = -0.4 if str(triple.get("relation", "")).startswith("CONTRADICTS:") else 0.0

        retrieval_score = (
            similarity * 1.6
            + entity_score * 1.5
            + relation_score * 1.3
            + slot_score * 1.2
            + confidence * 0.2
            - distance * 0.35
            + contradiction_penalty
        )

        enriched = dict(triple)
        enriched["retrieval_score"] = retrieval_score
        reranked.append(enriched)

    reranked.sort(
        key=lambda item: (
            -item["retrieval_score"],
            -float(item.get("similarity", 0.0)),
            float(item.get("distance", 1.0)),
            -float(item.get("confidence", 1.0)),
        )
    )
    return reranked


def _select_evidence(
    reranked: list[dict],
    *,
    seed_entities: list[str],
    slot: str,
    top_k: int,
) -> tuple[list[dict], list[dict]]:
    """Prefer coherent paths first, then fill gaps with top standalone triples."""
    path_candidates = _build_evidence_paths(
        reranked[: max(top_k * 2, 12)],
        seed_entities=seed_entities,
        slot=slot,
    )

    selected: list[dict] = []
    seen = set()

    for path in path_candidates:
        for triple in path["triples"]:
            key = _triple_key(triple)
            if key in seen:
                continue
            seen.add(key)
            selected.append(triple)
            if len(selected) >= top_k:
                return selected, path_candidates

    for triple in reranked:
        key = _triple_key(triple)
        if key in seen:
            continue
        seen.add(key)
        selected.append(triple)
        if len(selected) >= top_k:
            break

    return selected, path_candidates


def _build_evidence_paths(
    triples: list[dict],
    *,
    seed_entities: list[str],
    slot: str,
) -> list[dict]:
    """Build connected 1-hop and 2-hop evidence paths from reranked triples."""
    if not triples:
        return []

    paths: list[dict] = []
    for triple in triples:
        slot_score = _slot_alignment_score(triple, slot)
        paths.append({
            "triples": [triple],
            "path_score": float(triple.get("retrieval_score", 0.0)) + slot_score * 0.2,
            "length": 1,
        })

    for i, left in enumerate(triples):
        for right in triples[i + 1:]:
            shared_nodes = _shared_nodes(left, right)
            if not shared_nodes:
                continue

            ordered_triples, chain_bonus = _orient_connected_pair(left, right, seed_entities)
            slot_bonus = max(_slot_alignment_score(t, slot) for t in ordered_triples)
            seed_bonus = 0.4 if _path_touches_seed(ordered_triples, seed_entities) else 0.0
            bridge_bonus = 0.7 + chain_bonus
            path_score = sum(float(t.get("retrieval_score", 0.0)) for t in ordered_triples)
            path_score += slot_bonus * 0.5 + seed_bonus + bridge_bonus

            paths.append({
                "triples": ordered_triples,
                "path_score": path_score,
                "length": 2,
            })

    # Prefer stronger, longer coherent paths first.
    paths.sort(key=lambda item: (-item["path_score"], -item["length"]))
    return _dedupe_paths(paths)


def _shared_nodes(left: dict, right: dict) -> set[str]:
    left_nodes = {
        normalize_entity_name(left.get("subject", "")),
        normalize_entity_name(left.get("object", "")),
    }
    right_nodes = {
        normalize_entity_name(right.get("subject", "")),
        normalize_entity_name(right.get("object", "")),
    }
    return {node for node in left_nodes & right_nodes if node}


def _orient_connected_pair(left: dict, right: dict, seed_entities: list[str]) -> tuple[list[dict], float]:
    """Order a connected pair into the most coherent path-like sequence."""
    left_subj = normalize_entity_name(left.get("subject", ""))
    left_obj = normalize_entity_name(left.get("object", ""))
    right_subj = normalize_entity_name(right.get("subject", ""))
    right_obj = normalize_entity_name(right.get("object", ""))
    seed_set = {normalize_entity_name(seed) for seed in seed_entities if normalize_entity_name(seed)}

    if left_obj and left_obj == right_subj:
        return [left, right], 0.6
    if right_obj and right_obj == left_subj:
        return [right, left], 0.6
    if left_subj and left_subj == right_subj:
        if left_subj in seed_set:
            return _sort_from_seed(left, right, seed_set), 0.35
        return _sort_by_strength(left, right), 0.25
    if left_obj and left_obj == right_obj:
        return _sort_by_strength(left, right), 0.2

    return _sort_from_seed(left, right, seed_set), 0.15


def _sort_from_seed(left: dict, right: dict, seed_set: set[str]) -> list[dict]:
    left_touch = _triple_touches_seed(left, seed_set)
    right_touch = _triple_touches_seed(right, seed_set)
    if left_touch and not right_touch:
        return [left, right]
    if right_touch and not left_touch:
        return [right, left]
    return _sort_by_strength(left, right)


def _sort_by_strength(left: dict, right: dict) -> list[dict]:
    if float(left.get("retrieval_score", 0.0)) >= float(right.get("retrieval_score", 0.0)):
        return [left, right]
    return [right, left]


def _triple_touches_seed(triple: dict, seed_set: set[str]) -> bool:
    subject = normalize_entity_name(triple.get("subject", ""))
    obj = normalize_entity_name(triple.get("object", ""))
    return subject in seed_set or obj in seed_set


def _path_touches_seed(triples: list[dict], seed_entities: list[str]) -> bool:
    seed_set = {normalize_entity_name(seed) for seed in seed_entities if normalize_entity_name(seed)}
    return any(_triple_touches_seed(triple, seed_set) for triple in triples)


def _triple_key(triple: dict) -> tuple[str, str, str, bool, str]:
    return (
        normalize_entity_name(triple.get("subject", "")),
        normalize_entity_name(triple.get("relation", "")),
        normalize_entity_name(triple.get("object", "")),
        bool(triple.get("negated", False)),
        normalize_entity_name(triple.get("condition", "")),
    )


def _dedupe_paths(paths: list[dict]) -> list[dict]:
    """Remove duplicate path permutations while preserving highest score."""
    deduped: list[dict] = []
    seen = set()
    for path in paths:
        signature = tuple(_triple_key(triple) for triple in path["triples"])
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(path)
    return deduped


def _entity_alignment_score(triple: dict, seed_entities: list[str]) -> float:
    subject = normalize_entity_name(triple.get("subject", ""))
    obj = normalize_entity_name(triple.get("object", ""))
    relation = normalize_entity_name(triple.get("relation", ""))
    if not seed_entities:
        return 0.0

    score = 0.0
    for seed in seed_entities:
        seed_norm = normalize_entity_name(seed)
        if not seed_norm:
            continue
        if seed_norm == subject:
            score += 1.6
        elif seed_norm == obj:
            score += 1.2
        elif seed_norm in subject:
            score += 0.8
        elif seed_norm in obj:
            score += 0.6
        if seed_norm in relation:
            score += 0.3
    return score


def _relation_alignment_score(triple: dict, relation_cues: list[str], question_text: str) -> float:
    if not relation_cues:
        return 0.0

    relation = normalize_entity_name(triple.get("relation", ""))
    triple_text = normalize_entity_name(
        f"{triple.get('subject', '')} {triple.get('relation', '')} {triple.get('object', '')}"
    )
    score = 0.0
    for cue in relation_cues:
        cue_norm = normalize_entity_name(cue)
        if not cue_norm:
            continue
        if cue_norm == relation:
            score += 1.4
        elif cue_norm in relation:
            score += 1.0
        elif cue_norm in triple_text:
            score += 0.4

    # If the relation text itself appears in the question, treat it as a strong signal.
    if relation and relation in question_text:
        score += 1.0
    return score


def _slot_alignment_score(triple: dict, slot: str) -> float:
    relation = normalize_entity_name(triple.get("relation", ""))
    obj = normalize_entity_name(triple.get("object", ""))
    subject = normalize_entity_name(triple.get("subject", ""))
    condition = normalize_entity_name(triple.get("condition", ""))
    combined = " ".join(part for part in (relation, obj, condition) if part)

    hints = _SLOT_RELATION_HINTS.get(slot, ())
    score = 0.0
    for hint in hints:
        hint_norm = normalize_entity_name(hint)
        if hint_norm and hint_norm in combined:
            score += 0.8

    if slot == "where" and _looks_like_location(obj):
        score += 1.2
    elif slot == "when" and _looks_like_time(obj, condition):
        score += 1.2
    elif slot == "who" and (is_person_like_entity(obj) or is_person_like_entity(subject)):
        score += 1.1
    elif slot == "how_many" and _looks_like_quantity(obj):
        score += 1.2
    elif slot == "why" and _looks_like_reason(relation, obj, condition):
        score += 1.2
    elif slot == "how" and _looks_like_method(relation, obj, condition):
        score += 1.0
    elif slot == "what":
        score += 0.2

    return score


def _looks_like_location(text: str) -> bool:
    location_terms = (
        "city", "country", "province", "state", "district", "village", "street", "road",
        "paris", "tokyo", "london", "hanoi", "saigon", "vietnam", "japan", "france",
        "located", "headquarters", "campus", "office", "region", "capital",
    )
    return any(term in text for term in location_terms)


def _looks_like_time(text: str, condition: str) -> bool:
    haystack = " ".join(part for part in (text, condition) if part)
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", haystack):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", haystack):
        return True
    months = (
        "january", "february", "march", "april", "may", "june",
        "july", "august", "september", "october", "november", "december",
    )
    return any(month in haystack for month in months)


def _looks_like_quantity(text: str) -> bool:
    return bool(re.search(r"\b\d+([.,]\d+)?\b", text))


def _looks_like_reason(relation: str, obj: str, condition: str) -> bool:
    haystack = " ".join(part for part in (relation, obj, condition) if part)
    return any(term in haystack for term in ("because", "due", "reason", "cause", "caused", "led to", "resulted"))


def _looks_like_method(relation: str, obj: str, condition: str) -> bool:
    haystack = " ".join(part for part in (relation, obj, condition) if part)
    return any(term in haystack for term in ("by", "using", "through", "via", "method", "process", "approach"))


def _unique(values: list[str]) -> list[str]:
    seen = set()
    ordered = []
    for value in values:
        normalized = normalize_entity_name(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(normalized)
    return ordered


def _format_context(triples: list[dict]) -> tuple[str, list[str], bool]:
    """Convert triples to readable context string. Returns (context, sources, has_contradictions)."""
    lines = []
    sources = []
    has_contradictions = False

    for t in triples:
        neg = "NOT " if t.get("negated") else ""
        cond = f" [{t['condition']}]" if t.get("condition") else ""
        confidence = t.get("confidence", 1.0)
        conf_str = f" (confidence: {confidence:.1f})" if confidence < 0.8 else ""

        relation = t["relation"]
        if relation.startswith("CONTRADICTS:"):
            has_contradictions = True
            relation = f"CONTRADICTS {relation[12:]}"

        line = f"- {t['subject']} {neg}{relation} {t['object']}{cond}{conf_str}"
        lines.append(line)
        sources.extend(t.get("sources", []))

    return "\n".join(lines), sources, has_contradictions
