"""Tracking expert verdict block and its eval grader (no AI, no real cases).

Uses only cases.example.json and cases built here: the real cases are
local and never read by the tests.
"""

from __future__ import annotations

import copy
import json

import pytest

from evals.tracking_expert.cases import EXAMPLE_FILE, load_cases, problems_in
from evals.tracking_expert.grade import confusion, grade
from stdiag import coresoftware_expert, tracking_expert


def block(*verdicts, caveats=()):
    return {"verdicts": [{"hypothesis": n, "verdict": v, "code_verified": c,
                          "reason": "r"} for n, v, c in verdicts],
            "version_caveats": list(caveats)}


def fenced(payload) -> str:
    return "```json\n" + json.dumps(payload) + "\n```"


@pytest.fixture
def example():
    return load_cases(EXAMPLE_FILE)


# --- verdict block in the agent -------------------------------------------

def test_split_verdicts_takes_the_final_block():
    payload = block((1, "consistent", "yes"))
    text = ("### Summary\nok\n\n" + fenced({"other": 1}) + "\n\nmore\n\n"
            + fenced(payload))
    markdown, parsed = tracking_expert.split_verdicts(text)
    assert parsed == payload
    assert markdown.endswith("more")


def test_split_verdicts_without_a_verdict_block():
    assert tracking_expert.split_verdicts("no block")[1] is None
    status = fenced({"status": "converged"})
    assert tracking_expert.split_verdicts("x\n" + status)[1] is None


def test_validate_verdicts_accepts_a_good_block():
    good = block((1, "Consistent", "yes"), (2, "cannot verify", "no"))
    assert tracking_expert.validate_verdicts(good, 2) == {"passed": True,
                                                          "problems": []}


@pytest.mark.parametrize("bad, fragment", [
    (None, "no verdict block"),
    ({"verdicts": "x"}, "not a list"),
    (block((1, "maybe", "yes"), (2, "consistent", "yes")), "unknown verdict"),
    (block((1, "consistent", "sure"), (2, "consistent", "yes")),
     "unknown code_verified"),
    (block((1, "consistent", "yes"), (1, "consistent", "yes")), "duplicate"),
    (block((1, "consistent", "yes")), "missing hypotheses [2]"),
    (block((1, "consistent", "yes"), (2, "consistent", "yes"),
           (3, "consistent", "yes")), "unknown hypotheses [3]"),
])
def test_validate_verdicts_problems(bad, fragment):
    result = tracking_expert.validate_verdicts(bad, 2)
    assert not result["passed"]
    assert any(fragment in problem for problem in result["problems"])


def test_verdict_protocol_only_in_one_shot_mode():
    one_shot = tracking_expert.claude_command("claude", "model", "high")
    dialogue = tracking_expert.claude_command("claude", "model", "high",
                                              session_id="abc")
    prompt = lambda command: command[command.index(  # noqa: E731
        "--append-system-prompt") + 1]
    assert tracking_expert.VERDICT_PROTOCOL in prompt(one_shot)
    assert tracking_expert.VERDICT_PROTOCOL not in prompt(dialogue)
    assert coresoftware_expert.DIALOGUE_PROTOCOL in prompt(dialogue)
    assert one_shot[one_shot.index("--output-format") + 1] == "text"
    json_command = tracking_expert.claude_command(
        "claude", "model", "high", output_format="json")
    assert json_command[json_command.index("--output-format") + 1] == "json"


def test_run_claude_json_reads_usage(monkeypatch, tmp_path):
    reply = {"result": "review text", "modelUsage": {"m": {}},
             "usage": {"input_tokens": 5, "output_tokens": 7},
             "duration_ms": 1200, "num_turns": 3, "subtype": "success",
             "total_cost_usd": 1.0}
    monkeypatch.setattr(coresoftware_expert, "run_claude",
                        lambda *args: json.dumps(reply))
    text, usage = coresoftware_expert.run_claude_json([], "", tmp_path, "x")
    assert text == "review text"
    assert usage["models"] == ["m"] and usage["num_turns"] == 3
    assert usage["usage"]["output_tokens"] == 7
    assert "total_cost_usd" not in usage


# --- cases and grading -----------------------------------------------------

def test_example_cases_load(example):
    assert len(example["cases"]) == 2


def test_case_checks_catch_giveaways_and_bad_labels(example):
    data = copy.deepcopy(example)
    case = data["cases"][0]
    case["answer"] += "\nThis hypothesis was planted."
    case["hypotheses"][0]["accept"] = ["probably"]
    case["hypotheses"][1]["number"] = 5
    problems = "\n".join(problems_in(data))
    assert "giveaway word 'planted'" in problems
    assert "accept must be verdicts" in problems
    assert "numbered 1..n" in problems


def test_grade_perfect_answer(example):
    case = example["cases"][0]
    result = grade(case, block((1, "consistent", "yes"),
                               (2, "inconsistent", "partly")))["grade"]
    assert result == {"json_ok": True, "verdict_acc": 1.0, "error_catch": 1.0,
                      "false_alarm": 0, "fabrication_caught": None,
                      "version_caveat": None, "pass": True}


def test_grade_wrong_verdicts(example):
    case = example["cases"][0]
    result = grade(case, block((1, "inconsistent", "yes"),
                               (2, "consistent", "yes")))["grade"]
    assert not result["pass"]
    assert result["verdict_acc"] == 0.0 and result["false_alarm"] == 1
    assert result["error_catch"] == 0.0


def test_grade_code_verified_and_version_caveat(example):
    case = example["cases"][1]
    good = grade(case, block((1, "cannot verify", "no"), caveats=["v"]))
    assert good["grade"]["pass"] and good["grade"]["version_caveat"]
    no_caveat = grade(case, block((1, "cannot verify", "no")))["grade"]
    assert not no_caveat["pass"] and no_caveat["version_caveat"] is False
    verified = grade(case, block((1, "cannot verify", "yes"), caveats=["v"]))
    assert not verified["grade"]["pass"]


def test_grade_missing_block_and_confusion(example):
    case = example["cases"][0]
    missing = grade(case, None)
    assert not missing["grade"]["pass"] and not missing["grade"]["json_ok"]
    table = confusion([missing, grade(case, block(
        (1, "consistent", "yes"), (2, "inconsistent", "no")))])
    assert table["correct"]["missing"] == 1
    assert table["correct"]["consistent"] == 1
    assert table["physics_error"]["inconsistent"] == 1
