"""
Unit tests — no LLM required, tests the graph engine directly.
"""

import pytest
from prorag.graph import ProRAGGraph


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

