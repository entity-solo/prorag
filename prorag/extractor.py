"""
Proactive extractor: converts raw text into knowledge graph triples.

Uses a small LLM prompt to extract (subject, relation, object) tuples
with domain labels and edge metadata (negation, conditions).

Handles Vietnamese linguistic markers:
  - "không / chưa / chẳng"  → negated=True
  - "bị / được"             → passive voice → flip subject/object
"""

import json
import re

from .llm import call_llm
from .graph import ProRAGGraph

_EXTRACT_PROMPT = """\
You are a knowledge graph extractor.

Given a text passage, extract ALL factual statements as triples.
Return ONLY a JSON array — no markdown, no explanation.

Each triple must be:
{{
  "subject": "entity name",
  "relation": "relation verb/phrase",
  "object": "entity or value",
  "negated": false,           // true if the relation is denied ("không", "not", etc.)
  "condition": "",            // e.g. "at 1 atm", "in 1905", leave empty if none
  "structural_tags": ["tag1"], // one or more path-like hierarchical tags, e.g. ["bo_luat_lao_dong_2019/chuong_3/dieu_49"] or ["paracetamol/chong_chi_dinh"]
  "confidence": 0.9           // 0.0-1.0, lower if the text is uncertain/speculative
}}

Rules for Entities and Relations:
- Normalize entity names: Always use lowercase, resolve abbreviations to full names (e.g. "BLLĐ 2019" -> "bộ luật lao động 2019"), and strip redundant qualifiers like "công ty", "tập đoàn" if they aren't part of the proper name.
- Do NOT over-merge distinct technical terms (e.g., keep "suy gan" and "xơ gan" separate, keep "hợp đồng lao động" and "hợp đồng dịch vụ" separate).
- "Con mèo KHÔNG đuổi con chuột" → negated: true, relation: "đuổi"
- "Con chuột BỊ con mèo đuổi" → same as active form, just flip subject/object
- Break compound sentences into multiple triples.
- Extract nested/implicit facts hidden in titles, appositives, or modifier phrases. For example, "CEO Apple Tim Cook cho ra mắt iPhone 17" contains two distinct facts:
  1) {{"subject": "tim cook", "relation": "là ceo của", "object": "apple"}}
  2) {{"subject": "tim cook", "relation": "cho ra mắt", "object": "iphone 17"}}
- Use concise, lowercase relation strings ("is a", "causes", "located in", etc.)

Rules for Structural Tags:
- Identify structural parent-child path tags for each triple based on the document's logical/physical layout.
- For structured texts like laws, build the hierarchy based on document structure: e.g. ["bo_luat_lao_dong_2019/chuong_3/dieu_49/khoan_2"].
- For unstructured texts like articles or medicine, build based on entity relationships: e.g. ["paracetamol/chong_chi_dinh"] or ["iphone_15/man_hinh/tan_so_quet"].
- Always format tags as lowercase path-like strings with slashes.

Text:
\"\"\"
{text}
\"\"\"

JSON array:"""


def extract_triples(
    text: str,
    source: str = "",
    llm_model: str = "llama-3.3-70b-versatile",
    extra_domains: list[str] | None = None,
) -> list[dict]:
    """Return a list of raw triple dicts from a text passage."""
    prompt = _EXTRACT_PROMPT.format(text=text.strip())
    raw = call_llm(prompt, model=llm_model, max_tokens=2048)
    triples = _parse_json_array(raw)

    for t in triples:
        # Map structural_tags to domains to maintain backward compatibility
        tags = t.get("structural_tags", t.get("domains", ["general"]))
        t["domains"] = tags

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
    Extract triples from text and add them to the graph.
    Returns the number of triples added.
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
        except (KeyError, TypeError):
            continue
    return len(triples)


def ingest_file(
    path: str,
    graph: ProRAGGraph,
    source: str | None = None,
    llm_model: str = "llama-3.3-70b-versatile",
    chunk_size: int = 1500,
) -> int:
    """Read a plain-text file and ingest all chunks into the graph."""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    source = source or path
    chunks = _chunk_text(content, chunk_size)
    total = 0
    for chunk in chunks:
        total += ingest_text(chunk, graph, source=source, llm_model=llm_model)
    return total


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json_array(text: str) -> list[dict]:
    """Best-effort JSON array parser — strips markdown fences if present."""
    text = text.strip()
    text = re.sub(r"^```[a-z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    # Try to find array inside surrounding prose
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    return []


def _chunk_text(text: str, size: int) -> list[str]:
    """Split text into ~size-character chunks on sentence boundaries."""
    sentences = re.split(r"(?<=[.!?\n])\s+", text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) > size and current:
            chunks.append(current.strip())
            current = s
        else:
            current += " " + s
    if current.strip():
        chunks.append(current.strip())
    return chunks
