"""
Unit tests for the minimal ProRAG runtime.
"""

from unittest.mock import patch
from pathlib import Path

import pytest

from prorag.extractor import extract_triples, ingest_text
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


def test_extract_triples_resolves_pronoun_from_recent_entity():
    mock_entity_response = '{"entities": {"Apple": "apple", "iPhone 15": "iphone 15", "It": "iphone 15", "September": "september"}}'
    mock_extract_response = """
    [
      {"subject": "apple", "relation": "released", "object": "iphone 15", "negated": false, "confidence": 1.0},
      {"subject": "iphone 15", "relation": "announced in", "object": "september", "negated": false, "confidence": 1.0}
    ]
    """
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        triples = extract_triples("Apple released iPhone 15. It was announced in September.")
    assert len(triples) == 2
    assert triples[1]["subject"] == "iphone 15"


def test_resolve_entities_basic():
    from prorag.extractor import resolve_entities
    mock_response = '{"entities": {"Apple": "apple", "It": "apple", "OpenAI": "openai", "the company": "openai"}}'
    with patch("prorag.extractor.call_llm", return_value=mock_response):
        entity_map = resolve_entities("Apple partnered with OpenAI. The company liked it.", {"apple", "openai"})
    assert entity_map["Apple"] == "apple"
    assert entity_map["It"] == "apple"
    assert entity_map["the company"] == "openai"


def test_cross_sentence_lazy_context_propagates():
    from prorag.extractor import ingest_text
    
    # We pass a text with 9 sentences so it gets split into 2 batches (size 8 and size 1).
    text = (
        "Apple released iPhone 15. "
        "S2. S3. S4. S5. S6. S7. S8. "
        "It was announced in September."
    )
    
    mock_entity_1 = '{"entities": {"Apple": "apple", "iPhone 15": "iphone 15"}}'
    mock_extract_1 = '[{"subject": "apple", "relation": "released", "object": "iphone 15"}]'
    
    # Second batch (Sentence 9) has "It was announced in September."
    # Since "It" resolves to null initially, it will retry with history.
    # Retry 0: returns "It" -> null
    mock_entity_2_try0 = '{"entities": {"It": null, "September": "september"}}'
    # Retry 1: with preceding 4 sentences, resolves "It" to "iphone 15"
    mock_entity_2_try1 = '{"entities": {"It": "iphone 15", "September": "september"}}'
    mock_extract_2 = '[{"subject": "iphone 15", "relation": "announced in", "object": "september"}]'
    
    graph = ProRAGGraph()
    with patch("prorag.extractor.call_llm", side_effect=[
        mock_entity_1, mock_extract_1, 
        mock_entity_2_try0, mock_entity_2_try1, mock_extract_2
    ]) as mock_call:
        count, registry = ingest_text(text, graph)
        
    assert "iphone 15" in graph.g.nodes
    assert "september" in graph.g.nodes
    
    # Verify call arguments to make sure context was included
    called_args = [call[0][0] for call in mock_call.call_args_list]
    assert "Previous context" in called_args[3]
    assert "S8." in called_args[3]


def test_extract_triples_drops_unresolved_pronoun_fact():
    mock_entity_response = '{"entities": {"It": null, "September": "september"}}'
    mock_extract_response = """
    [
      {"subject": "It", "relation": "announced in", "object": "september", "negated": false, "confidence": 1.0}
    ]
    """
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        triples = extract_triples("It was announced in September.")
    assert triples == []


def test_ingest_text_full_pipeline_avoids_pronoun_nodes():
    mock_entity_response = '{"entities": {"Apple": "apple", "iPhone 15": "iphone 15", "It": "iphone 15", "September": "september"}}'
    mock_extract_response = """
    [
      {"subject": "apple", "relation": "released", "object": "iphone 15", "negated": false, "confidence": 1.0},
      {"subject": "iphone 15", "relation": "announced in", "object": "september", "negated": false, "confidence": 1.0}
    ]
    """
    graph = ProRAGGraph()
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        added, registry = ingest_text("Apple released iPhone 15. It was announced in September.", graph)
    assert added == 2
    assert "it" not in graph.g.nodes
    assert "iphone 15" in graph.g.nodes


def test_ingest_file_builds_registry():
    from prorag.extractor import ingest_file
    from unittest.mock import mock_open
    
    mock_content = "Apple released iPhone 15. It was announced in September."
    mock_entity_1 = '{"entities": {"Apple": "apple", "iPhone 15": "iphone 15"}}'
    mock_extract_1 = '[{"subject": "apple", "relation": "released", "object": "iphone 15", "negated": false}]'
    
    graph = ProRAGGraph()
    with patch("builtins.open", mock_open(read_data=mock_content)), \
         patch("prorag.extractor.call_llm", side_effect=[mock_entity_1, mock_extract_1]):
        total = ingest_file("dummy.txt", graph, chunk_size=1000)
    assert total == 1
    assert "apple" in graph.g.nodes
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


def test_extract_triples_normalizes_passive_voice():
    mock_entity_response = '{"entities": {"Apple": "apple", "iPhone 15": "iphone 15"}}'
    mock_extract_response = """
    [
      {"subject": "apple", "relation": "released", "object": "iphone 15", "negated": false}
    ]
    """
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        triples = extract_triples("iPhone 15 was released by Apple.")
    assert len(triples) == 1
    assert triples[0]["subject"] == "apple"
    assert triples[0]["relation"] == "released"
    assert triples[0]["object"] == "iphone 15"


def test_extract_triples_normalizes_passive_voice_fallback():
    # Test passive voice auto-correction helper on LLM failure to follow instructions
    mock_entity_response = '{"entities": {"Apple": "apple", "iPhone 15": "iphone 15"}}'
    mock_extract_response = """
    [
      {"subject": "iphone 15", "relation": "was released by", "object": "apple", "negated": false}
    ]
    """
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        triples = extract_triples("iPhone 15 was released by Apple.")
    assert len(triples) == 1
    assert triples[0]["subject"] == "apple"
    assert triples[0]["relation"] == "released"
    assert triples[0]["object"] == "iphone 15"


def test_fix_passive():
    from prorag.extractor import _fix_passive
    
    # Test English passive voice with by
    t1 = {"subject": "iphone 15", "relation": "was released by", "object": "apple"}
    res1 = _fix_passive(t1)
    assert res1["subject"] == "apple"
    assert res1["object"] == "iphone 15"
    assert res1["relation"] == "released"
    
    # Test Vietnamese passive voice with bởi
    t2 = {"subject": "iphone 15", "relation": "được phát triển bởi", "object": "apple"}
    res2 = _fix_passive(t2)
    assert res2["subject"] == "apple"
    assert res2["object"] == "iphone 15"
    assert res2["relation"] == "phát triển"
    
    # Test passive voice prefix/suffix strip
    t3 = {"subject": "a", "relation": "bị bắt bởi", "object": "b"}
    res3 = _fix_passive(t3)
    assert res3["subject"] == "b"
    assert res3["object"] == "a"
    assert res3["relation"] == "bắt"


def test_substitute_mentions():
    from prorag.extractor import substitute_mentions
    
    entity_map = {
        "Steve Jobs": "steve jobs",
        "Steve": "steve",
        "Apple": "apple"
    }
    # Check that Steve inside Steve Jobs is not double-replaced
    text = "Steve Jobs founded Apple. Steve was CEO."
    annotated = substitute_mentions(text, entity_map)
    assert annotated == "[steve jobs] founded [apple]. [steve] was CEO."


def test_extract_triples_extracts_statement_time_and_aspect():
    mock_entity_response = '{"entities": {"Apple": "apple", "iPhone 16": "iphone 16"}}'
    mock_extract_response = """
    [
      {"subject": "apple", "relation": "release", "object": "iphone 16", "negated": false, "condition": "if stock rises", "statement_time": "May 2026", "temporal_aspect": "FUTURE"}
    ]
    """
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        triples = extract_triples("In May 2026, Apple announced it plans to release iPhone 16 if stock rises.")
    assert len(triples) == 1
    assert triples[0]["statement_time"] == "May 2026"
    assert triples[0]["temporal_aspect"] == "FUTURE"
    assert triples[0]["condition"] == "if stock rises"


def test_retrieve_evidence_prefers_matching_temporal_condition():
    graph = ProRAGGraph()
    graph.add_triple("steve jobs", "is ceo of", "apple", condition="in 1997", temporal_aspect="PAST")
    graph.add_triple("tim cook", "is ceo of", "apple", condition="in 2020", temporal_aspect="PRESENT")
    
    triples, meta = retrieve_evidence("Who was CEO of Apple in 1997?", graph, top_k=2)
    assert triples[0]["subject"] == "steve jobs"
    
    triples_2, meta_2 = retrieve_evidence("Who is CEO of Apple in 2020?", graph, top_k=2)
    assert triples_2[0]["subject"] == "tim cook"
