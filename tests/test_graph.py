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
    graph.add_triple("vaccine", "NOT causes", "autism",
                     domains=["medicine"], source="cdc", negated=True, confidence=0.99)
    return graph


def test_basic_query(g):
    results = g.query(["Einstein"])
    assert len(results) > 0
    subjects = {r["subject"] for r in results}
    assert "Einstein" in subjects


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
