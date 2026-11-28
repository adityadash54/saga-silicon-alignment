"""Load and check the coresoftware expert eval cases.

The real cases are the evaluation set and stay local (cases.json and
PROPOSED_CASES.md are git-ignored); cases.example.json shows the format
with made-up content. File layout:

{
  "coresoftware_revision": "<full hash the cases are written against>",
  "cases": [{
    "id": "...", "status": "draft" | "approved",
    "kind": "code" | "null",          # null: the cause is not in the code
    "request": {"comparison": "...", "finding": {...},
                "flags_table_rows": [...]},       # as review.py ask sends
    "revision": {"from": "<rev>", "to": "<rev>"},
    "labels": {"<label>": {...}},     # rows as from labels.csv; a row's
                                      # "local_diff_text" is sent as that
                                      # build's local changes
    "expected_commits": ["<full hash>"],          # empty for null cases
    "later_fix_commits": ["<full hash>"],         # optional: a later fix or
                                      # revert in neither build, scored in
                                      # --show-later runs
    "signature": "predicted" | "measured",        # where the numbers come from
    "why": "<reasoning for the reviewer>"
  }]
}

Only "request", "revision" and "labels" are sent to the agent; "why",
"status", "signature", "expected_commits" and "later_fix_commits" never
are. "from" need not be an ancestor of "to" (builds on different
branches): the candidates are from...to. The git checks (range, scope,
later fixes) are in run_eval.py --self-test.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES_FILE = HERE / "cases.json"
EXAMPLE_FILE = HERE / "cases.example.json"
KINDS = ("code", "null")
SIGNATURES = ("predicted", "measured")
FULL_HASH = re.compile(r"^[0-9a-f]{40}$")
# Words that would tell the agent what the answer is meant to be.
GIVEAWAY_WORDS = ("commit ", "revert", "regression", "bug", "the cause",
                  "caused by", "introduced", "planted", "made-up", "made up")


def sent_text(case: dict) -> str:
    """What the agent receives from the case, lower-case."""
    return json.dumps([case.get("request", {}), case.get("labels", {})]).lower()


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
        if case.get("kind") not in KINDS:
            problems.append(f"{where}: kind must be one of {KINDS}")
        if case.get("signature") not in SIGNATURES:
            problems.append(f"{where}: signature must be one of {SIGNATURES}")
        if not isinstance(case.get("request", {}).get("finding"), dict):
            problems.append(f"{where}: request.finding missing")
        if not isinstance(case.get("labels", {}), dict):
            problems.append(f"{where}: labels must be an object")
        revision = case.get("revision", {})
        if not (isinstance(revision.get("from"), str)
                and isinstance(revision.get("to"), str)):
            problems.append(f"{where}: revision needs from and to")
        expected = case.get("expected_commits", [])
        if case.get("kind") == "code" and not expected:
            problems.append(f"{where}: a code case needs expected_commits")
        if case.get("kind") == "null" and expected:
            problems.append(f"{where}: a null case has no expected_commits")
        for field in ("expected_commits", "later_fix_commits"):
            for commit in case.get(field, []):
                if not FULL_HASH.match(str(commit)):
                    problems.append(f"{where}: {field} {commit!r} must be "
                                    "a full 40-character hash")
        sent = sent_text(case)
        for commit in expected + case.get("later_fix_commits", []):
            if str(commit)[:7].lower() in sent:
                problems.append(f"{where}: the request or labels contain "
                                f"the expected commit {commit[:7]}")
        for word in GIVEAWAY_WORDS:
            if word in sent:
                problems.append(f"{where}: the request or labels contain "
                                f"the giveaway word {word.strip()!r}")
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
