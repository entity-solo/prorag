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


def test_add_attribute():
    graph = ProRAGGraph()
    graph.add_attribute("steve jobs", "died_on", "october 5 2011")
    graph.add_attribute("steve jobs", "role", "ceo")
    
    assert "steve jobs" in graph.g.nodes
    meta = graph.g.nodes["steve jobs"]["meta"]
    assert meta.node_type == "entity"
    assert meta.attributes["died_on"] == "october 5 2011"
    assert meta.attributes["role"] == "ceo"


def test_add_event():
    graph = ProRAGGraph()
    graph.add_event("iphone_launch_2007", "actor", "steve jobs")
    graph.add_event("iphone_launch_2007", "time", "january 2007")

    assert "iphone_launch_2007" in graph.g.nodes
    event_meta = graph.g.nodes["iphone_launch_2007"]["meta"]
    assert event_meta.node_type == "event"

    assert "steve jobs" in graph.g.nodes
    assert "january 2007" in graph.g.nodes
    assert graph.g.nodes["january 2007"]["meta"].node_type == "temporal"

    edges = list(graph.g.out_edges("iphone_launch_2007", data=True))
    assert len(edges) == 2
    relations = {data["relation"] for _, _, data in edges}
    assert relations == {"actor", "time"}



def test_merge_entities():
    graph = ProRAGGraph()
    graph.add_relation("Steve Jobs", "founded", "Apple")
    graph.add_relation("Steve Jobs", "ceo of", "Apple")
    graph.add_relation("Steve", "founded", "Apple")
    graph.add_relation("Steve", "ceo of", "Apple")

    assert "steve jobs" in graph.g.nodes
    assert "steve" in graph.g.nodes

    graph.merge_entities(name_similarity_threshold=0.5, fingerprint_threshold=0.5)

    assert "steve jobs" in graph.g.nodes
    assert "steve" not in graph.g.nodes
    meta = graph.g.nodes["steve jobs"]["meta"]
    assert "steve" in meta.aliases


def test_resolve_entities_with_known_entities():
    from prorag.extractor import resolve_entities
    
    mock_response = '{"entities": {"Apple": "apple", "It": "apple"}}'
    
    with patch("prorag.extractor.call_llm", return_value=mock_response) as mock_call:
        entity_map = resolve_entities(
            "Apple released iPhone.",
            known_entities={"apple", "iphone"}
        )
        
    assert entity_map["Apple"] == "apple"
    
    called_prompt = mock_call.call_args[0][0]
    assert '"apple"' in called_prompt
    assert '"iphone"' in called_prompt


def test_extract_facts_mixed():
    from prorag.extractor import extract_facts
    
    mock_entity_response = '{"entities": {"Steve Jobs": "steve jobs", "Apple": "apple"}}'
    mock_extract_response = """
    [
      {"type": "relation", "subject": "steve jobs", "relation": "founded", "object": "apple", "negated": false},
      {"type": "attribute", "subject": "steve jobs", "key": "died_on", "value": "october 5 2011"},
      {"type": "event", "event_id": "iphone_launch_2007", "role": "actor", "entity": "steve jobs"}
    ]
    """
    
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        facts = extract_facts("Steve Jobs founded Apple and died on October 5 2011.")
        
    assert len(facts) == 3
    assert facts[0]["type"] == "relation"
    assert facts[1]["type"] == "attribute"
    assert facts[2]["type"] == "event"


def test_ingest_text_routes_mixed_facts():
    from prorag.extractor import ingest_text
    
    mock_entity_response = '{"entities": {"Steve Jobs": "steve jobs", "Apple": "apple"}}'
    mock_extract_response = """
    [
      {"type": "relation", "subject": "steve jobs", "relation": "founded", "object": "apple", "negated": false},
      {"type": "attribute", "subject": "steve jobs", "key": "died_on", "value": "october 5 2011"},
      {"type": "event", "event_id": "iphone_launch_2007", "role": "actor", "entity": "steve jobs"}
    ]
    """
    
    graph = ProRAGGraph()
    with patch("prorag.extractor.call_llm", side_effect=[mock_entity_response, mock_extract_response]):
        count, registry = ingest_text("Steve Jobs founded Apple.", graph)
        
    assert count == 3
    assert "steve jobs" in graph.g.nodes
    assert "apple" in graph.g.nodes
    assert "iphone_launch_2007" in graph.g.nodes
    
    assert graph.g.nodes["steve jobs"]["meta"].attributes["died_on"] == "october 5 2011"
    edges = list(graph.g.out_edges("iphone_launch_2007", data=True))
    assert len(edges) == 1
    assert edges[0][1] == "steve jobs"
    assert edges[0][2]["relation"] == "actor"


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


def test_detect_seed_entities_matches_aliases():
    from prorag.pipeline import _detect_seed_entities
    graph = ProRAGGraph()
    graph.add_relation("Steve Jobs", "founded", "Apple")
    graph.g.nodes["steve jobs"]["meta"].aliases = ["Steve", "Stephan"]
    
    seeds = _detect_seed_entities("Who is Stephan?", graph, limit=5)
    assert "steve jobs" in seeds


def test_format_context_rich_metadata():
    from prorag.pipeline import _format_context
    triples = [
        {
            "subject": "apple",
            "relation": "released",
            "object": "iphone 15",
            "negated": False,
            "condition": "in September",
            "confidence": 1.0,
            "aspect": "perfective",
            "modality": "certain",
            "quantifier": "all",
            "evidentiality": "direct",
            "speech_act": "assertion",
            "causal": "event_123",
            "temporal_aspect": "PAST"
        }
    ]
    context, sources, has_contradictions = _format_context(triples)
    assert "- apple released iphone 15 [in September]" in context
    assert "aspect: perfective" in context
    assert "modality: certain" in context
    assert "quantifier: all" in context
    assert "evidentiality: direct" in context
    assert "speech_act: assertion" in context
    assert "caused by: event_123" in context
    assert "temporal: PAST" in context
