"""Coresoftware expert eval: answer parsing and grading (no AI, no real
cases).

Uses only cases.example.json, cases built here and a small git repository
made in a temporary folder: the real cases are local and never read by the
tests.
"""

from __future__ import annotations

import copy
import json

import pytest

from evals.coresoftware_expert import run_eval
from evals.coresoftware_expert.cases import EXAMPLE_FILE, load_cases, problems_in
from evals.coresoftware_expert.grade import grade, hashes_in, parse_answer
from stdiag import coresoftware_expert
from stdiag.coresoftware_expert import NOT_FOUND

EXPECTED = "3333333333333333333333333333333333333333"


@pytest.fixture
def example():
    return load_cases(EXAMPLE_FILE)


def answer(*rows: tuple[str, str], extra: str = "") -> str:
    """Expert-format answer with (rank, commit cell) rows."""
    table = ("| rank | tier | commit | date | author | files:lines | mechanism "
             "| fit | test | confidence |\n|---|---|---|---|---|---|---|---|"
             "---|---|\n")
    table += "".join(f"| {rank} | tier 2 | {commit} | 2000-01-01 | A | f.cc:1 "
                     "| m | f | t | low |\n" for rank, commit in rows)
    return (f"### Summary\nText with 1234567 and {extra}\n\n### Hypotheses\n"
            f"{table}\n### Examined and ruled out\n- 4444444ab: no\n\n"
            "### Commands run\n`git log`\n")


# --- parsing ------------------------------------------------------------------

def test_hashes_skip_pure_digits_and_words():
    text = "run 5312345, commit abc1234 and `deadbeefcafe`; date 2026-11-25"
    assert hashes_in(text) == ["abc1234", "deadbeefcafe"]


def test_parse_table_ranks_and_commits():
    parsed = parse_answer(answer(("1", "`333333333`"),
                                 ("**2**", "aaaaaaa1, bbbbbbb2")))
    assert parsed["table_ok"] and not parsed["not_found"]
    assert parsed["rows"] == [{"rank": 1, "commits": ["333333333"]},
                              {"rank": 2, "commits": ["aaaaaaa1",
                                                      "bbbbbbb2"]}]
    # hashes outside the table count for the fabrication check only
    assert "4444444ab" in parsed["all_hashes"]


def test_parse_uncommitted_row_and_missing_table():
    parsed = parse_answer(answer(("1", "uncommitted")))
    assert parsed["table_ok"] and parsed["rows"][0]["commits"] == []
    assert not parse_answer("### Summary\nnone")["table_ok"]


def test_not_found_sentence_with_markdown_and_line_breaks():
    words = NOT_FOUND.split()
    wrapped = "> **" + " ".join(words[:6]) + "\n" + " ".join(words[6:]) + "**"
    parsed = parse_answer("### Hypotheses\n" + wrapped)
    assert parsed["not_found"] and not parsed["table_ok"]


# --- cases --------------------------------------------------------------------

def test_example_cases_load(example):
    assert [case["kind"] for case in example["cases"]] == ["code", "null"]


def test_case_checks(example):
    data = copy.deepcopy(example)
    code, null = data["cases"]
    code["request"]["finding"]["evidence"] += " Introduced in 3333333."
    code["expected_commits"] = ["3333333"]
    null["expected_commits"] = [EXPECTED]
    null["kind"] = "maybe"
    problems = "\n".join(problems_in(data))
    assert "giveaway word 'introduced'" in problems
    assert "contain the expected commit 3333333" in problems
    assert "full 40-character hash" in problems
    assert "kind must be one of" in problems


# --- grading ------------------------------------------------------------------

def test_grade_hits(example):
    case = example["cases"][0]
    first = grade(case, answer(("1", "333333333")))["grade"]
    assert first["pass"] and first["hit_at_1"] and first["hit_at_3"]
    third = grade(case, answer(("1", "aaaaaaa"), ("2", "bbbbbbb"),
                               ("3", "3333333")))["grade"]
    assert third["pass"] and not third["hit_at_1"] and third["hit_at_3"]
    fourth = grade(case, answer(*[(str(r), "aaaaaaa") for r in (1, 2, 3)],
                                ("4", "3333333")))
    assert not fourth["grade"]["pass"] and fourth["meta"]["hit_rank"] == 4
    assert grade(case, answer())["grade"]["hit_at_3"] is False


def test_grade_not_found_on_code_and_null_cases(example):
    code, null = example["cases"]
    stated = "### Hypotheses\n" + NOT_FOUND + "\n"
    assert not grade(code, stated)["grade"]["pass"]
    assert not grade(code, answer(("1", "3333333"), extra=NOT_FOUND)
                     )["grade"]["pass"]
    result = grade(null, stated)["grade"]
    assert result["pass"] and result["format_ok"] and result["hit_at_3"] is None
    assert not grade(null, answer(("1", "aaaaaaa"), extra=NOT_FOUND)
                     )["grade"]["pass"]


class FakeGit:
    real = {"3333333": EXPECTED, "aaaaaaa": "a" * 40, "bbbbbbb": "b" * 40,
            "4444444ab": "4444444ab" + "0" * 31}

    def resolve(self, commit):
        return next((full for short, full in self.real.items()
                     if commit.startswith(short)), None)

    def in_range(self, full, revision):
        return full != "a" * 40

    def in_scope(self, full):
        return full != "b" * 40


def test_grade_with_git_checks(example):
    case = example["cases"][0]
    graded = grade(case, answer(("1", "3333333"), ("2", "aaaaaaa"),
                                ("3", "bbbbbbb"), ("4", "fffffff")),
                   FakeGit())
    result = graded["grade"]
    assert result["fabricated"] == 1 and graded["meta"]["unresolved"] == [
        "fffffff"]
    assert result["outside_range"] == 1 and result["out_of_scope"] == 1
    assert result["pass"]          # the git scores are reported, not gating


# --- runner -------------------------------------------------------------------

def test_self_test_without_git_passes_on_the_example(example, capsys):
    assert run_eval.self_test(example, None) == 0


def test_claude_command_output_format():
    command = coresoftware_expert.claude_command("claude", "m", "high")
    assert command[command.index("--output-format") + 1] == "text"
    command = coresoftware_expert.claude_command("claude", "m", "high",
                                                 output_format="json")
    assert command[command.index("--output-format") + 1] == "json"


def test_later_fix_found_only_in_show_later_runs(example):
    case = copy.deepcopy(example["cases"][0])
    case["later_fix_commits"] = ["abcdef1" + "0" * 33]
    text = answer(("1", "3333333"), extra="fixed later by abcdef1")
    assert grade(case, text)["grade"]["later_fix_found"] is None
    shown = grade(case, text, show_later=True)["grade"]
    assert shown["later_fix_found"] is True and shown["pass"]
    assert grade(case, answer(("1", "3333333")), show_later=True
                 )["grade"]["later_fix_found"] is False


def test_usage_out_written(monkeypatch, tmp_path):
    (tmp_path / ".git").mkdir()
    monkeypatch.setattr(coresoftware_expert, "build_request",
                        lambda *args: "request")
    monkeypatch.setattr(coresoftware_expert, "find_claude", lambda: "claude")
    monkeypatch.setattr(coresoftware_expert, "run_claude_json",
                        lambda *args: ("answer", {"models": ["m"]}))
    usage = tmp_path / "usage.json"
    text = coresoftware_expert.ask_expert({}, {}, {}, tmp_path, "m", "high",
                                          usage_out=usage)
    assert text == "answer"
    assert json.loads(usage.read_text()) == {"models": ["m"], "effort": "high"}
