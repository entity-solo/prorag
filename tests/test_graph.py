"""
Unit tests for the minimal ProRAG runtime.
"""

from unittest.mock import patch
from pathlib import Path

import pytest

from prorag.extractor import _chunk_text, extract_triples, ingest_text
from prorag.graph import ProRAGGraph
from prorag.pipeline import detect_question_slot, retrieve_evidence


@pytest.fixture
def graph():
    g = ProRAGGraph()
    g.add_triple("Einstein", "developed", "theory of relativity", source="wiki", condition="in 1905")
    g.add_triple("Einstein", "worked at", "patent office", source="wiki")
    g.add_triple("water", "boils at", "100 degrees", source="textbook", condition="at 1 atm")
    g.add_triple("vaccine", "causes", "autism", source="retracted_paper", confidence=0.1)
    g.add_triple("vaccine", "causes", "autism", source="cdc", negated=True, confidence=0.99)
    return g


def test_basic_query(graph):
    results = graph.query(["Einstein"])
    assert results
    assert "einstein" in {item["subject"] for item in results}


def test_negation_and_contradiction(graph):
    results = graph.query(["vaccine"])
    assert any(item["negated"] for item in results)
    assert any(item["relation"].startswith("CONTRADICTS:") for item in results)


def test_stats(graph):
    stats = graph.stats()
    assert stats["nodes"] > 0
    assert stats["edges"] > 0


def test_persistence(graph):
    temp_dir = Path("C:/tmp")
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "prorag_test_graph.json"
    graph.save(str(path))
    restored = ProRAGGraph()
    restored.load(str(path))
    assert restored.stats() == graph.stats()
    path.unlink(missing_ok=True)


def test_condition_stored(graph):
    results = graph.query(["water"])
    conditions = [item["condition"] for item in results if item["subject"] == "water"]
    assert any("atm" in condition for condition in conditions)


def test_triple_dedup(graph):
    before = graph.stats()["edges"]
    graph.add_triple("Einstein", "developed", "theory of relativity", source="another_source", condition="in 1905")
    after = graph.stats()["edges"]
    assert after == before


def test_nested_fact_query_resolution():
    graph = ProRAGGraph()
    graph.add_triple("tim cook", "is ceo of", "apple")
    graph.add_triple("tim cook", "launched", "iphone 17")
    results = graph.query(["iphone 17"], max_hops=2)
    assert len(results) == 2
    assert {item["relation"] for item in results} == {"is ceo of", "launched"}


def test_alias_bridging():
    graph = ProRAGGraph()
    graph.add_triple("Kiss and Tell", "stars", "Shirley Temple")
    graph.add_triple("Shirley Temple Black", "served as", "Chief of Protocol")
    results = graph.query_vector("Kiss and Tell", alias_threshold=0.85)
    assert "chief of protocol" in {item["object"] for item in results}


def test_graph_rejects_unresolved_pronouns():
    graph = ProRAGGraph()
    graph.add_triple("it", "announced", "iphone 15")
    graph.add_triple("apple", "announced", "it")
    assert graph.stats() == {"nodes": 0, "edges": 0}


def test_chunk_text_has_sentence_overlap():
    text = "Alpha launched Beta. It shipped in September. Customers liked it. Sales increased."
    chunks = _chunk_text(text, size=45, overlap_sentences=1)
    assert len(chunks) >= 2
    assert "It shipped in September." in chunks[0]
    assert "It shipped in September." in chunks[1]


def test_extract_triples_resolves_pronoun_from_recent_entity():
    mock_response = """
    [
      {"subject_mention": "Apple", "subject": "apple", "relation": "released", "object_mention": "iPhone 15", "object": "iphone 15", "negated": false, "confidence": 1.0},
      {"subject_mention": "It", "subject": "", "relation": "was announced in", "object_mention": "September", "object": "september", "negated": false, "confidence": 1.0}
    ]
    """
    with patch("prorag.extractor.call_llm", return_value=mock_response):
        triples = extract_triples("Apple released iPhone 15. It was announced in September.")
    assert len(triples) == 2
    assert triples[1]["subject"] == "iphone 15"


def test_extract_triples_uses_llm_fallback_for_generic_reference():
    extract_response = """
    [
      {"subject_mention": "OpenAI", "subject": "openai", "relation": "partnered with", "object_mention": "Apple", "object": "apple", "negated": false, "confidence": 1.0},
      {"subject_mention": "The company", "subject": "", "relation": "launched", "object_mention": "a new model", "object": "a new model", "negated": false, "confidence": 0.9}
    ]
    """
    coref_response = '{"resolved":"apple","confidence":0.96}'
    with patch("prorag.extractor.call_llm", side_effect=[extract_response, coref_response]):
        triples = extract_triples("OpenAI partnered with Apple. The company launched a new model.")
    assert len(triples) == 2
    assert triples[1]["subject"] == "apple"


def test_extract_triples_drops_unresolved_pronoun_fact():
    mock_response = """
    [
      {"subject_mention": "It", "subject": "", "relation": "was announced in", "object_mention": "September", "object": "september", "negated": false, "confidence": 1.0}
    ]
    """
    with patch("prorag.extractor.call_llm", return_value=mock_response):
        triples = extract_triples("It was announced in September.")
    assert triples == []


def test_ingest_text_full_pipeline_avoids_pronoun_nodes():
    mock_response = """
    [
      {"subject_mention": "Apple", "subject": "apple", "relation": "released", "object_mention": "iPhone 15", "object": "iphone 15", "negated": false, "confidence": 1.0},
      {"subject_mention": "It", "subject": "", "relation": "was announced in", "object_mention": "September", "object": "september", "negated": false, "confidence": 1.0}
    ]
    """
    graph = ProRAGGraph()
    with patch("prorag.extractor.call_llm", return_value=mock_response):
        added = ingest_text("Apple released iPhone 15. It was announced in September.", graph)
    assert added == 2
    assert "it" not in graph.g.nodes
    assert "iphone 15" in graph.g.nodes


def test_detect_question_slot_5w():
    assert detect_question_slot("Where was Inception filmed?") == "where"
    assert detect_question_slot("When did Apple launch iPhone 15?") == "when"
    assert detect_question_slot("Who directed Inception?") == "who"
    assert detect_question_slot("Why was the product delayed?") == "why"
    assert detect_question_slot("How many users signed up?") == "how_many"


def test_retrieve_evidence_prefers_where_relation():
    graph = ProRAGGraph()
    graph.add_triple("christopher nolan", "directed", "inception")
    graph.add_triple("inception", "filmed in", "paris")
    graph.add_triple("inception", "released by", "warner bros")
    triples, meta = retrieve_evidence("Where was the film directed by Christopher Nolan filmed?", graph, top_k=3)
    assert meta["slot"] == "where"
    assert [triples[0]["relation"], triples[1]["relation"]] == ["directed", "filmed in"]
    assert triples[1]["object"] == "paris"


def test_retrieve_evidence_prefers_who_relation():
    graph = ProRAGGraph()
    graph.add_triple("christopher nolan", "directed", "inception")
    graph.add_triple("inception", "filmed in", "paris")
    graph.add_triple("inception", "released in", "2010")
    triples, meta = retrieve_evidence("Who directed Inception?", graph, top_k=3)
    assert meta["slot"] == "who"
    assert triples[0]["relation"] == "directed"
    assert triples[0]["subject"] == "christopher nolan"


def test_retrieve_evidence_prefers_connected_path_for_when_question():
    graph = ProRAGGraph()
    graph.add_triple("christopher nolan", "directed", "inception")
    graph.add_triple("inception", "released in", "2010")
    graph.add_triple("inception", "filmed in", "paris")
    triples, meta = retrieve_evidence("When was the film directed by Christopher Nolan released?", graph, top_k=3)
    assert meta["path_count"] > 0
    assert [triples[0]["relation"], triples[1]["relation"]] == ["directed", "released in"]
    assert triples[1]["object"] == "2010"


def test_retrieve_evidence_falls_back_to_keyword_query():
    graph = ProRAGGraph()
    graph.add_triple("apple", "launched", "iphone 15")
    original = graph.query_vector

    def raise_import_error(*args, **kwargs):
        raise ImportError("sentence-transformers missing")

    graph.query_vector = raise_import_error
    try:
        triples, meta = retrieve_evidence("What did Apple launch?", graph, top_k=3)
    finally:
        graph.query_vector = original
    assert meta["slot"] == "what"
    assert triples[0]["subject"] == "apple"
