"""
Question answering pipeline.
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

Rules:
1. Give a concise answer.
2. Do not invent facts.
3. If the context is insufficient, say "I don't have enough information to answer this."

## Knowledge Graph Context
{context}

## Question
{question}

## Answer"""

_CONTRADICTIONS_NOTE = "\nNote: conflicting information exists - see sources."

_SLOT_HINTS = {
    "who": ("person", "people", "founder", "ceo", "president", "director", "author", "inventor", "actor", "ai"),
    "what": ("what", "which", "name", "title", "product", "concept"),
    "when": ("when", "year", "date", "month", "day", "time", "born", "died", "released", "launched", "announced", "founded"),
    "where": ("where", "location", "place", "country", "city", "headquarters", "born", "filmed", "located", "based"),
    "why": ("why", "reason", "because", "cause", "due"),
    "how": ("how", "method", "process", "way", "approach"),
    "how_many": ("how many", "how much", "number of", "amount of", "count"),
}

_SLOT_RELATION_HINTS = {
    "who": ("by", "founded", "ceo", "president", "director", "author", "invented", "created", "led", "appointed", "stars"),
    "what": ("is", "means", "called", "named", "contains", "includes", "describes", "announced", "released"),
    "when": ("in", "on", "at", "during", "since", "born", "died", "released", "launched", "announced", "founded", "created"),
    "where": ("in", "at", "from", "located", "based", "headquartered", "born", "filmed", "shot", "held", "lives"),
    "why": ("because", "caused", "causes", "due", "resulted", "led to", "reason", "motivated", "triggered"),
    "how": ("by", "using", "through", "via", "method", "process", "worked", "operates"),
    "how_many": ("number", "count", "contains", "total", "amount", "population", "size"),
}

_QUESTION_WORDS = {
    "who", "what", "when", "where", "why", "how", "which", "whom", "whose",
    "is", "are", "was", "were", "do", "does", "did", "can", "could", "would", "should",
}


def answer(
    question: str,
    graph: ProRAGGraph,
    llm_model: str = "llama-3.3-70b-versatile",
    max_context_triples: int = 60,
) -> dict:
    triples, _meta = retrieve_evidence(question, graph, top_k=max_context_triples)
    context, sources, has_contradictions = _format_context(triples, graph)
    if not context:
        return {
            "answer": "I don't have enough information to answer this.",
            "sources": [],
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
    selected, paths = _select_evidence(reranked, seed_entities=seed_entities, slot=slot, top_k=top_k)
    return selected[:top_k], {
        "slot": slot,
        "seed_entities": seed_entities,
        "relation_cues": relation_cues,
        "path_count": len(paths),
    }


def detect_question_slot(question: str) -> str:
    q = normalize_entity_name(question)
    if not q:
        return "what"
    if any(phrase in q for phrase in ("how many", "how much", "number of")):
        return "how_many"
    if any(q.startswith(prefix) for prefix in ("who", "whom")):
        return "who"
    if q.startswith("when"):
        return "when"
    if q.startswith("where"):
        return "where"
    if q.startswith("why"):
        return "why"
    if q.startswith("how"):
        return "how"

    scores = {slot: 0 for slot in _SLOT_HINTS}
    for slot, hints in _SLOT_HINTS.items():
        for hint in hints:
            if hint in q:
                scores[slot] += 1
    best_slot, best_score = max(scores.items(), key=lambda item: item[1])
    return best_slot if best_score > 0 else "what"


def _detect_seed_entities(question: str, graph: ProRAGGraph, limit: int = 5) -> list[str]:
    keywords = _keywords_from_question(question)
    lexical_scores: dict[str, float] = {}

    for node in graph.g.nodes:
        node_text = normalize_entity_name(node)
        score = 0.0
        for keyword in keywords:
            keyword = normalize_entity_name(keyword)
            if not keyword:
                continue
            if keyword == node_text:
                score += 3.0
            elif keyword in node_text:
                score += 1.5
            elif node_text in keyword and len(node_text) >= 3:
                score += 1.0
        if score > 0:
            lexical_scores[node] = score

    try:
        from .embeddings import EmbeddingStore

        store = EmbeddingStore()
        semantic = store.top_k(question, list(graph.g.nodes), k=max(limit * 3, 8), threshold=0.15)
    except Exception:
        semantic = []

    combined: dict[str, float] = {}
    for node, score in lexical_scores.items():
        combined[node] = combined.get(node, 0.0) + score
    for node, score in semantic:
        combined[node] = combined.get(node, 0.0) + max(0.0, score) * 4.0

    ranked = sorted(combined.items(), key=lambda item: -item[1])
    return [node for node, _score in ranked[:limit]]


def _infer_relation_cues(question: str, seed_entities: list[str], slot: str) -> list[str]:
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
    try:
        return graph.query_vector(
            question,
            max_cost=2.2,
            top_k=top_k,
            seed_k=seed_k,
            seed_threshold=0.18,
        )
    except Exception:
        return graph.query(keywords, top_k=top_k)


def detect_question_aspect(question: str) -> str:
    q = question.lower()
    if any(word in q for word in ("plan", "predict", "forecast", "will", "would")):
        return "FUTURE"
    if any(word in q for word in ("did", "was", "were", "happened")):
        return "PAST"
    return "PRESENT"


def _rerank_triples(
    question: str,
    triples: list[dict],
    *,
    seed_entities: list[str],
    relation_cues: list[str],
    slot: str,
) -> list[dict]:
    if not triples:
        return []

    question_text = normalize_entity_name(question)
    question_years = set(re.findall(r"\b(19\d{2}|20\d{2})\b", question_text))
    question_aspect = detect_question_aspect(question)

    reranked = []
    for triple in triples:
        entity_score = _entity_alignment_score(triple, seed_entities)
        relation_score = _relation_alignment_score(triple, relation_cues, question_text)
        slot_score = _slot_alignment_score(triple, slot)
        distance = float(triple.get("distance", 1.0))
        similarity = float(triple.get("similarity", 0.0))
        confidence = float(triple.get("confidence", 1.0))
        contradiction_penalty = -0.4 if str(triple.get("relation", "")).startswith("CONTRADICTS:") else 0.0

        temporal_score = 0.0
        if question_years:
            condition_text = normalize_entity_name(triple.get("condition", ""))
            statement_time_text = normalize_entity_name(triple.get("statement_time", ""))
            for year in question_years:
                if year in condition_text or year in statement_time_text:
                    temporal_score += 1.5

        triple_aspect = triple.get("temporal_aspect", "PRESENT")
        if question_aspect == triple_aspect:
            temporal_score += 0.5

        retrieval_score = (
            similarity * 1.6
            + entity_score * 1.5
            + relation_score * 1.3
            + slot_score * 1.2
            + confidence * 0.2
            + temporal_score * 1.5
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
    path_candidates = _build_evidence_paths(reranked[: max(top_k * 2, 12)], seed_entities=seed_entities, slot=slot)
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


def _build_evidence_paths(triples: list[dict], *, seed_entities: list[str], slot: str) -> list[dict]:
    if not triples:
        return []

    paths: list[dict] = []
    for triple in triples:
        slot_score = _slot_alignment_score(triple, slot)
        paths.append(
            {
                "triples": [triple],
                "path_score": float(triple.get("retrieval_score", 0.0)) + slot_score * 0.2,
                "length": 1,
            }
        )

    for i, left in enumerate(triples):
        for right in triples[i + 1 :]:
            if not _shared_nodes(left, right):
                continue
            ordered, chain_bonus = _orient_connected_pair(left, right, seed_entities)
            slot_bonus = max(_slot_alignment_score(triple, slot) for triple in ordered)
            seed_bonus = 0.4 if _path_touches_seed(ordered, seed_entities) else 0.0
            path_score = sum(float(triple.get("retrieval_score", 0.0)) for triple in ordered)
            path_score += 0.7 + chain_bonus + slot_bonus * 0.5 + seed_bonus
            paths.append({"triples": ordered, "path_score": path_score, "length": 2})

    paths.sort(key=lambda item: (-item["path_score"], -item["length"]))
    return _dedupe_paths(paths)


def _shared_nodes(left: dict, right: dict) -> set[str]:
    left_nodes = {normalize_entity_name(left.get("subject", "")), normalize_entity_name(left.get("object", ""))}
    right_nodes = {normalize_entity_name(right.get("subject", "")), normalize_entity_name(right.get("object", ""))}
    return {node for node in left_nodes & right_nodes if node}


def _orient_connected_pair(left: dict, right: dict, seed_entities: list[str]) -> tuple[list[dict], float]:
    left_subject = normalize_entity_name(left.get("subject", ""))
    left_object = normalize_entity_name(left.get("object", ""))
    right_subject = normalize_entity_name(right.get("subject", ""))
    right_object = normalize_entity_name(right.get("object", ""))
    seed_set = {normalize_entity_name(seed) for seed in seed_entities if normalize_entity_name(seed)}

    if left_object and left_object == right_subject:
        return [left, right], 0.6
    if right_object and right_object == left_subject:
        return [right, left], 0.6
    if left_subject and left_subject == right_subject:
        if left_subject in seed_set:
            return _sort_from_seed(left, right, seed_set), 0.35
        return _sort_by_strength(left, right), 0.25
    if left_object and left_object == right_object:
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
    return [left, right] if float(left.get("retrieval_score", 0.0)) >= float(right.get("retrieval_score", 0.0)) else [right, left]


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
    score = 0.0
    for seed in seed_entities:
        seed = normalize_entity_name(seed)
        if not seed:
            continue
        if seed == subject:
            score += 1.6
        elif seed == obj:
            score += 1.2
        elif seed in subject:
            score += 0.8
        elif seed in obj:
            score += 0.6
        if seed in relation:
            score += 0.3
    return score


def _relation_alignment_score(triple: dict, relation_cues: list[str], question_text: str) -> float:
    relation = normalize_entity_name(triple.get("relation", ""))
    triple_text = normalize_entity_name(f"{triple.get('subject', '')} {triple.get('relation', '')} {triple.get('object', '')}")
    score = 0.0
    for cue in relation_cues:
        cue = normalize_entity_name(cue)
        if not cue:
            continue
        if cue == relation:
            score += 1.4
        elif cue in relation:
            score += 1.0
        elif cue in triple_text:
            score += 0.4
    if relation and relation in question_text:
        score += 1.0
    return score


def _slot_alignment_score(triple: dict, slot: str) -> float:
    relation = normalize_entity_name(triple.get("relation", ""))
    obj = normalize_entity_name(triple.get("object", ""))
    subject = normalize_entity_name(triple.get("subject", ""))
    condition = normalize_entity_name(triple.get("condition", ""))
    combined = " ".join(part for part in (relation, obj, condition) if part)

    score = 0.0
    for hint in _SLOT_RELATION_HINTS.get(slot, ()):
        hint = normalize_entity_name(hint)
        if hint and hint in combined:
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
    return any(
        term in text
        for term in (
            "city", "country", "province", "state", "district", "village", "street", "road",
            "paris", "tokyo", "london", "hanoi", "saigon", "vietnam", "japan", "france",
            "located", "headquarters", "campus", "office", "region", "capital",
        )
    )


def _looks_like_time(text: str, condition: str) -> bool:
    haystack = " ".join(part for part in (text, condition) if part)
    if re.search(r"\b(1[0-9]{3}|20[0-9]{2}|21[0-9]{2})\b", haystack):
        return True
    if re.search(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", haystack):
        return True
    return any(month in haystack for month in ("january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"))


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
        value = normalize_entity_name(value)
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _format_context(triples: list[dict], graph: ProRAGGraph | None = None) -> tuple[str, list[str], bool]:
    lines = []
    sources = []
    has_contradictions = False
    unique_sources = set()

    for triple in triples:
        negation = "NOT " if triple.get("negated") else ""
        condition = f" [{triple['condition']}]" if triple.get("condition") else ""
        confidence = triple.get("confidence", 1.0)
        confidence_suffix = f" (confidence: {confidence:.1f})" if confidence < 0.8 else ""
        relation = triple["relation"]
        if relation.startswith("CONTRADICTS:"):
            has_contradictions = True
            relation = f"CONTRADICTS {relation[12:]}"
        lines.append(f"- {triple['subject']} {negation}{relation} {triple['object']}{condition}{confidence_suffix}")
        
        for src in triple.get("sources", []):
            sources.append(src)
            if graph and hasattr(graph, "chunks") and src in graph.chunks:
                unique_sources.add(src)

    formatted_triples = "\n".join(lines)
    
    if graph and hasattr(graph, "chunks") and unique_sources:
        chunk_texts = [f"[{src}]: {graph.chunks[src]}" for src in sorted(unique_sources)]
        formatted_chunks = "\n\n".join(chunk_texts)
        context = f"""### Knowledge Graph Facts:
{formatted_triples}

### Relevant Detailed Text Chunks:
{formatted_chunks}"""
    else:
        context = formatted_triples

    return context, sources, has_contradictions
