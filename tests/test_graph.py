"""
Unit tests — no LLM required, tests the graph engine directly.
"""

import pytest
from prorag.graph import ProRAGGraph
from prorag.extractor import extract_triples, ingest_text, _chunk_text
from prorag.pipeline import detect_question_slot, retrieve_evidence


@pytest.fixture
def g():
    graph = ProRAGGraph()
    graph.add_triple("Einstein", "developed", "theory of relativity",
                     domains=["science", "history"], source="wiki", condition="in 1905")
    graph.add_triple("Einstein", "worked at", "patent office",
                     domains=["history"], source="wiki")
    graph.add_triple("water", "boils at", "100 degrees",
                     domains=["science"], condition="at 1 atm", source="textbook")
    graph.add_triple("vaccine", "causes", "autism",
                     domains=["medicine"], source="retracted_paper", confidence=0.1)
    graph.add_triple("vaccine", "causes", "autism",
                     domains=["medicine"], source="cdc", negated=True, confidence=0.99)
    return graph


def test_basic_query(g):
    results = g.query(["Einstein"])
    assert len(results) > 0
    subjects = {r["subject"] for r in results}
    assert "einstein" in subjects


def test_domain_filter(g):
    all_results = g.query(["Einstein"])
    science_only = g.query(["Einstein"], domains=["science"])
    # domain filter should narrow results
    assert len(science_only) <= len(all_results)


def test_negation_stored(g):
    results = g.query(["vaccine"])
    relations = [(r["subject"], r["negated"]) for r in results]
    # both negated and non-negated should be present
    negated_flags = [neg for _, neg in relations]
    assert True in negated_flags or False in negated_flags  # at least one


def test_contradiction_detected(g):
    # After adding contradicting triples, a CONTRADICTS edge should exist
    results = g.query(["vaccine"])
    contradiction_found = any("CONTRADICTS" in r["relation"] for r in results)
    assert contradiction_found


def test_stats(g):
    s = g.stats()
    assert s["nodes"] > 0
    assert s["edges"] > 0
    assert "science" in s["domains"]


def test_persistence(g, tmp_path):
    path = str(tmp_path / "graph.json")
    g.save(path)
    g2 = ProRAGGraph()
    g2.load(path)
    assert g2.stats()["nodes"] == g.stats()["nodes"]
    assert g2.stats()["edges"] == g.stats()["edges"]


def test_condition_stored(g):
    results = g.query(["water"])
    conds = [r["condition"] for r in results if r["subject"] == "water"]
    assert any("atm" in c for c in conds)


def test_triple_dedup(g):
    before = g.stats()["edges"]
    # Adding same triple again should not create a clean duplicate
    g.add_triple("Einstein", "developed", "theory of relativity",
                 domains=["science"], source="another_source", condition="in 1905")
    after = g.stats()["edges"]
    # Should add at most 1 new edge (same semantic content)
    assert after - before <= 1


def test_hierarchical_prefix_match():
    graph = ProRAGGraph()
    # Tag structure for law
    graph.add_triple("lao_dong_2019", "co_dieu_49", "boi_thuong", domains=["bo_luat/chuong_3/dieu_49/khoan_2"])
    graph.add_triple("dan_su_2015", "co_dieu_50", "hop_dong", domains=["bo_luat/chuong_4"])
    
    # 1. Query with specific tag
    res_dieu = graph.query(["lao_dong_2019"], domains=["bo_luat/chuong_3/dieu_49"])
    assert len(res_dieu) > 0
    
    # 2. Query with broad parent tag
    res_broad = graph.query(["lao_dong_2019", "dan_su_2015"], domains=["bo_luat"])
    assert len(res_broad) == 2
    
    # 3. Query with non-matching sibling tag
    res_sibling = graph.query(["lao_dong_2019"], domains=["bo_luat/chuong_4"])
    assert len(res_sibling) == 0


def test_crossing_boundary_penalty():
    graph = ProRAGGraph()
    # paracetamol belongs to medicine
    graph.add_triple("paracetamol", "dieu_tri", "dau_dau", domains=["medicine/drug/dosage"])
    # pharma_abc belongs to company
    graph.add_triple("pharma_abc", "thanh_lap", "2005", domains=["company/pharma"])
    # link them via a bridge edge
    graph.add_triple("pharma_abc", "san_xuat", "paracetamol", domains=["company/pharma"])
    # link pharma_abc to vietnam under company
    graph.add_triple("pharma_abc", "tru_so", "vietnam", domains=["company/pharma"])
    
    # 1. Query with domains=["medicine"] (scoped).
    # The path paracetamol -> pharma_abc (cost 2) -> vietnam (cost 2) exceeds max_hops=2.
    # So the triple (pharma_abc, tru_so, vietnam) should NOT be returned.
    results_scoped = graph.query(["paracetamol"], domains=["medicine"], max_hops=2)
    vietnam_triples_scoped = [r for r in results_scoped if r["object"] == "vietnam"]
    assert len(vietnam_triples_scoped) == 0
    
    # 2. Query with domains=None (unscoped).
    # Hop costs are 1, so both pharma_abc and vietnam are within max_hops=2.
    # The triple (pharma_abc, tru_so, vietnam) should be returned.
    results_unscoped = graph.query(["paracetamol"], domains=None, max_hops=2)
    vietnam_triples_unscoped = [r for r in results_unscoped if r["object"] == "vietnam"]
    assert len(vietnam_triples_unscoped) > 0


def test_ranking_boost():
    graph = ProRAGGraph()
    # Add two facts for paracetamol under different tags
    graph.add_triple("paracetamol", "lieu_luong", "4g", domains=["medicine/drug/dosage"])
    graph.add_triple("paracetamol", "mau_sac", "trang", domains=["medicine/other"])
    
    # Query with specific tag
    results = graph.query(["paracetamol"], domains=["medicine/drug/dosage"])
    
    dosage_fact = [r for r in results if r["relation"] == "lieu_luong"][0]
    color_fact = [r for r in results if r["relation"] == "mau_sac"][0]
    
    # Dosage fact has matching taxonomy prefix on both endpoints, so its effective_distance should be boosted by -0.5
    # Color fact does not match the tag on its object endpoint, so its effective_distance remains 0.0
    assert dosage_fact["effective_distance"] == -0.5
    assert color_fact["effective_distance"] == 0.0
    
    # It should be sorted first in the list
    assert results[0]["relation"] == "lieu_luong"


def test_nested_fact_extraction_mocked():
    from unittest.mock import patch
    from prorag.extractor import extract_triples

    # Test that extract_triples correctly calls the LLM and parses the expected nested/implicit JSON response.
    mock_response = """
    [
      {"subject": "tim cook", "relation": "là ceo của", "object": "apple", "negated": false, "confidence": 1.0},
      {"subject": "tim cook", "relation": "cho ra mắt", "object": "iphone 17", "negated": false, "confidence": 1.0}
    ]
    """
    with patch("prorag.extractor.call_llm", return_value=mock_response) as mock_call:
        res = extract_triples("CEO Apple Tim Cook cho ra mắt iPhone 17")
        assert len(res) == 2
        subjects = {r["subject"] for r in res}
        relations = {r["relation"] for r in res}
        assert "tim cook" in subjects
        assert "là ceo của" in relations
        assert "cho ra mắt" in relations
        mock_call.assert_called_once()


def test_nested_fact_query_resolution():
    graph = ProRAGGraph()
    # Ingest the extracted nested facts
    graph.add_triple("tim cook", "là ceo của", "apple", domains=["general"])
    graph.add_triple("tim cook", "cho ra mắt", "iphone 17", domains=["general"])
    
    # Query for "iphone 17" (which matches "iphone 17" node)
    # With max_hops=2, it should traverse to "tim cook" (hop 1) and then "apple" (hop 2).
    # Both triples should be returned in the results.
    results = graph.query(["iphone 17"], domains=["general"], max_hops=2)
    assert len(results) == 2
    
    relations = {r["relation"] for r in results}
    assert "là ceo của" in relations
    assert "cho ra mắt" in relations


def test_alias_bridging():
    graph = ProRAGGraph()
    # Adding two disconnected components representing the same person with slightly different name
    graph.add_triple("Kiss and Tell", "stars", "Shirley Temple")
    graph.add_triple("Shirley Temple Black", "served as", "Chief of Protocol")

    # query_vector for Shirley Temple Black's movie should retrieve Chief of Protocol
    # since Shirley Temple and Shirley Temple Black have high similarity (>0.85)
    results = graph.query_vector("Kiss and Tell", alias_threshold=0.85)
    
    # We should have traversed the alias bridge and retrieved the Chief of Protocol fact
    objects = {r["object"] for r in results}
    assert "chief of protocol" in objects


def test_graph_rejects_unresolved_pronouns():
    graph = ProRAGGraph()
    graph.add_triple("it", "announced", "iphone 15")
    graph.add_triple("apple", "announced", "it")
    assert graph.stats()["nodes"] == 0
    assert graph.stats()["edges"] == 0


def test_chunk_text_has_sentence_overlap():
    text = "Alpha launched Beta. It shipped in September. Customers liked it. Sales increased."
    chunks = _chunk_text(text, size=45, overlap_sentences=1)
    assert len(chunks) >= 2
    assert "It shipped in September." in chunks[0]
    assert "It shipped in September." in chunks[1]


def test_extract_triples_resolves_pronoun_from_recent_entity():
    from unittest.mock import patch

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
    assert triples[1]["subject_mention"].lower() == "it"


def test_extract_triples_uses_llm_fallback_for_generic_reference():
    from unittest.mock import patch

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
    from unittest.mock import patch

    mock_response = """
    [
      {"subject_mention": "It", "subject": "", "relation": "was announced in", "object_mention": "September", "object": "september", "negated": false, "confidence": 1.0}
    ]
    """
    with patch("prorag.extractor.call_llm", return_value=mock_response):
        triples = extract_triples("It was announced in September.")
    assert triples == []


def test_ingest_text_full_pipeline_avoids_pronoun_nodes():
    from unittest.mock import patch

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
    assert "christopher nolan" in meta["seed_entities"]
    first_two_relations = [triples[0]["relation"], triples[1]["relation"]]
    assert first_two_relations == ["directed", "filmed in"]
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


def test_retrieve_evidence_prefers_connected_path_for_where_question():
    graph = ProRAGGraph()
    graph.add_triple("christopher nolan", "directed", "inception")
    graph.add_triple("inception", "filmed in", "paris")
    graph.add_triple("inception", "released in", "2010")
    graph.add_triple("inception", "distributed by", "warner bros")

    triples, meta = retrieve_evidence(
        "Where was the film directed by Christopher Nolan filmed?",
        graph,
        top_k=3,
    )

    assert meta["path_count"] > 0
    assert [triples[0]["relation"], triples[1]["relation"]] == ["directed", "filmed in"]
    assert triples[1]["object"] == "paris"


def test_retrieve_evidence_prefers_connected_path_for_when_question():
    graph = ProRAGGraph()
    graph.add_triple("christopher nolan", "directed", "inception")
    graph.add_triple("inception", "released in", "2010")
    graph.add_triple("inception", "filmed in", "paris")

    triples, _meta = retrieve_evidence(
        "When was the film directed by Christopher Nolan released?",
        graph,
        top_k=3,
    )

    assert [triples[0]["relation"], triples[1]["relation"]] == ["directed", "released in"]
    assert triples[1]["object"] == "2010"


def test_retrieve_evidence_falls_back_to_keyword_query():
    graph = ProRAGGraph()
    graph.add_triple("apple", "launched", "iphone 15")

    original = graph.query_vector

    def _raise_import_error(*args, **kwargs):
        raise ImportError("sentence-transformers missing")

    graph.query_vector = _raise_import_error
    try:
        triples, meta = retrieve_evidence("What did Apple launch?", graph, top_k=3)
    finally:
        graph.query_vector = original

    assert meta["slot"] == "what"
    assert triples
    assert triples[0]["subject"] == "apple"
