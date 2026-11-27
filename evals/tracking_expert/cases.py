"""Load and check the tracking expert eval cases.

The real cases are the evaluation set and stay local (cases.json and
PROPOSED_CASES.md are git-ignored); cases.example.json shows the format
with made-up content. File layout:

{
  "coresoftware_revision": "<full hash the cases are written against>",
  "cases": [{
    "id": "...", "status": "draft" | "approved",
    "request": {"comparison": "...", "finding": {...},
                "flags_table_rows": [...]},       # as review.py check sends
    "answer": "<coresoftware expert answer, Markdown, numbered hypotheses>",
    "expect_version_caveat": false,
    "hypotheses": [{
      "number": 1,                                # rank in the answer
      "kind": "correct" | "physics_error" | "fabricated" | "version"
              | "unverifiable",
      "accept": ["consistent", "partly consistent"],
      "code_verified": ["yes"] | null,            # null: not graded
      "cited_commits": ["<hash>"],                # must exist
      "fabricated_commits": ["<hash>"],           # must not exist
      "why": "<reasoning for the reviewer>"}]
  }]
}

"why" and "status" are for the reviewer and are never sent to the agent;
only "request" and "answer" are.
"""

from __future__ import annotations

import json
from pathlib import Path

from stdiag.tracking_expert import CODE_VERIFIED, VERDICTS

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.json"
EXAMPLE_FILE = HERE / "cases.example.json"
KINDS = ("correct", "physics_error", "fabricated", "version", "unverifiable")
# Words that would tell the agent what the answer is meant to be.
GIVEAWAY_WORDS = ("planted", "fabricat", "fake", "made-up", "made up",
                  "deliberately", "intentionally wrong", "trap")


def problems_in(data: dict) -> list[str]:
    """Format problems in a cases file (empty list when it is fine)."""
    problems = []
    if not isinstance(data.get("coresoftware_revision"), str):
        problems.append("coresoftware_revision missing")
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        return problems + ["no cases"]
    ids = [case.get("id") for case in cases]
    if len(set(ids)) != len(ids):
        problems.append("duplicate case ids")
    for case in cases:
        where = f"case {case.get('id')!r}"
        if case.get("status") not in ("draft", "approved"):
            problems.append(f"{where}: status must be draft or approved")
        request = case.get("request", {})
        if not isinstance(request.get("finding"), dict):
            problems.append(f"{where}: request.finding missing")
        answer = case.get("answer", "")
        if not answer.strip():
            problems.append(f"{where}: answer missing")
        sent = (json.dumps(request) + answer).lower()
        for word in GIVEAWAY_WORDS:
            if word in sent:
                problems.append(f"{where}: request or answer contains "
                                f"the giveaway word {word!r}")
        hypotheses = case.get("hypotheses", [])
        numbers = [h.get("number") for h in hypotheses]
        if sorted(numbers) != list(range(1, len(hypotheses) + 1)):
            problems.append(f"{where}: hypotheses must be numbered 1..n")
        for hypothesis in hypotheses:
            label = f"{where} H{hypothesis.get('number')}"
            if hypothesis.get("kind") not in KINDS:
                problems.append(f"{label}: unknown kind")
            accept = hypothesis.get("accept") or []
            if not accept or set(accept) - set(VERDICTS):
                problems.append(f"{label}: accept must be verdicts from "
                                f"{VERDICTS}")
            verified = hypothesis.get("code_verified")
            if verified is not None and (not verified or set(verified)
                                         - set(CODE_VERIFIED)):
                problems.append(f"{label}: code_verified must be null or "
                                f"values from {CODE_VERIFIED}")
    return problems


def load_cases(path: Path = CASES_FILE) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. The cases are local only: copy "
            f"{EXAMPLE_FILE.name} to {path.name} and replace the made-up "
            "examples, or pass --cases-file.")
    data = json.loads(path.read_text())
    problems = problems_in(data)
    if problems:
        raise ValueError(f"{path}:\n  " + "\n  ".join(problems))
    return data
