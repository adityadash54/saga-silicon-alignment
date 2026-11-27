"""Retrieval eval runner: agent-path cases, the request list, no-answer cases.

Uses the fake library and embedder of test_literature, so neither poppler
nor Ollama is needed.
"""

from __future__ import annotations

import json

import pytest

from evals.retrieval import run_eval
from stdiag import tracking_expert
from tests.test_literature import library  # noqa: F401  (fixture)

FINDING = {"title": "MVTX layer 0 twist-like rphi shift",
           "evidence": "weak mode suspected",
           "entries": [{"detector": "MVTX", "coord": "rphi"}]}
SUMMARY = "A twist weak mode from cosmic constraints."


def cases():
    return [
        {"id": "q", "status": "approved", "question": "twist weak mode cosmic",
         "expected": [{"file": "Paper A.pdf", "pages": [1, 2]}]},
        {"id": "agent", "status": "draft", "finding": FINDING,
         "answer_summary": SUMMARY,
         "expected": [{"file": "Paper A.pdf", "pages": [1, 2]}]},
        {"id": "none", "status": "draft",
         "question": "calorimeter energy scale with neutral pions",
         "expected": []},
    ]


def test_agent_path_query_is_the_tracking_expert_query():
    case = cases()[1]
    query = run_eval.case_query(case)
    assert query == tracking_expert.literature_query({"finding": FINDING},
                                                     SUMMARY)
    assert "MVTX" not in query and "pixel vertex detector" in query
    assert run_eval.kind(case) == "agent"
    assert run_eval.kind(cases()[2]) == "question, no answer"


@pytest.mark.parametrize("bad", [{"id": "x", "expected": []},
                                 {"id": "x", "question": "twist"}])
def test_malformed_cases_are_refused(tmp_path, bad):
    path = tmp_path / "cases.json"
    path.write_text(json.dumps({"cases": [bad]}))
    with pytest.raises(SystemExit, match="case x"):
        run_eval.load_cases(path)


def test_run_scores_the_request_list_and_skips_no_answer(library):  # noqa: F811
    folder, index, embedder, update = library
    update(embedder)
    calls = []
    search = index.search

    def spy(query, **kwargs):
        calls.append((kwargs["k"], kwargs["max_per_document"]))
        return search(query, **kwargs)

    index.search = spy
    results = run_eval.run(cases(), ["bm25", "hybrid"], index, embedder)

    assert set(calls) == {(run_eval.DEPTH, run_eval.DEPTH),
                          (run_eval.REQUEST_K, run_eval.REQUEST_CAP)}
    assert (run_eval.REQUEST_K, run_eval.REQUEST_CAP) == (6, 2)
    for mode in ("bm25", "hybrid"):
        assert results["ranks"][mode][0] == 1
        assert results["request_hits"][mode][:2] == [True, True]
        assert results["request_hits"][mode][2] is None
        scores = run_eval.summarize(cases(), results["ranks"][mode],
                                    results["request_hits"][mode])
        assert scores["recall@10"] == 1.0 and scores["hit@6"] == 1.0
    assert all(isinstance(value, float) for value in results["cosines"])
    assert results["warnings"] == []

    text = run_eval.report(cases(), results, index.status())
    assert "hit@6 (2/paper)" in text and "2 answerable, 1 no-answer" in text
    assert "| none | question, no answer | draft | n/a" in text
    assert "## Agent-path queries" in text and "pixel vertex detector" in text
    assert "## No-answer cases" in text


def test_bm25_only_run_computes_no_cosine(library):  # noqa: F811
    folder, index, embedder, update = library
    update(embedder)
    results = run_eval.run(cases(), ["bm25"], index, embedder)
    assert results["cosines"] == [None, None, None]
    lines = run_eval.cutoff_lines(cases(), results["cosines"])
    assert "not available" in lines[-1]


def test_cutoff_counts_answerable_cases_below_the_no_answer_score():
    lines = run_eval.cutoff_lines(cases(), [0.8, 0.6, 0.65])
    assert "| none | 0.650 |" in lines
    assert "median highest cosine 0.700, lowest 0.600" in lines[-1]
    assert "would also drop 1 of 2 answerable cases" in lines[-1]
    assert run_eval.cutoff_lines(cases()[:2], [0.8, 0.6]) == []
