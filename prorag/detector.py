"""
Domain detector — fast, rule-first, LLM fallback.

Determines which domain subgraph(s) to query for a given question.
Uses keyword heuristics first (free, instant), then LLM if ambiguous.
"""

import re
from .llm import call_llm

# Keyword → domain mapping (expand as needed)
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "science": [
        "physics", "chemistry", "biology", "quantum", "atom", "molecule",
        "vật lý", "hóa học", "sinh học", "lượng tử",
    ],
    "medicine": [
        "disease", "drug", "symptom", "diagnosis", "treatment", "vaccine",
        "bệnh", "thuốc", "triệu chứng", "chẩn đoán", "điều trị",
    ],
    "history": [
        "war", "century", "king", "empire", "revolution", "dynasty",
        "chiến tranh", "thế kỷ", "vua", "đế chế", "triều đại",
    ],
    "law": [
        "law", "regulation", "court", "contract", "legal", "statute",
        "luật", "quy định", "tòa án", "hợp đồng", "pháp lý",
    ],
    "finance": [
        "stock", "revenue", "profit", "investment", "bank", "interest",
        "cổ phiếu", "doanh thu", "lợi nhuận", "đầu tư", "ngân hàng",
    ],
    "tech": [
        "software", "algorithm", "network", "database", "api", "model",
        "phần mềm", "thuật toán", "mạng", "cơ sở dữ liệu", "mô hình",
    ],
    "geography": [
        "country", "city", "river", "mountain", "continent",
        "quốc gia", "thành phố", "sông", "núi", "châu lục",
    ],
}

_DETECT_PROMPT = """\
Given this question, list the relevant knowledge domains (1-3) from this list:
science, medicine, history, law, finance, tech, geography, general

Return only a JSON array of strings, e.g. ["science", "history"]

Question: {question}

JSON array:"""


def detect_domains(question: str, llm_model: str = "llama-3.3-70b-versatile") -> list[str]:
    """
    Returns the most relevant domain(s) for a question.

    Strategy:
    1. Fast keyword scan — covers ~80% of questions for free
    2. LLM fallback for ambiguous or cross-domain questions
    """
    domains = _keyword_scan(question)
    if domains:
        return domains

    # LLM fallback
    try:
        raw = call_llm(_DETECT_PROMPT.format(question=question), model=llm_model, max_tokens=64)
        raw = raw.strip()
        import json
        import re
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        result = json.loads(raw)
        if isinstance(result, list) and result:
            return [str(d) for d in result]
    except Exception:
        pass

    return ["general"]




def _keyword_scan(text: str) -> list[str]:
    """Return domains whose keywords appear in the text as whole words."""
    text_lower = text.lower()
    matched = []
    for domain, keywords in _DOMAIN_KEYWORDS.items():
        for kw in keywords:
            pattern = r"\b" + re.escape(kw) + r"\b"
            if re.search(pattern, text_lower):
                matched.append(domain)
                break
    return matched
