"""Code-only grading of the coresoftware expert's Markdown answer.

The expert's prompt is unchanged for the eval: its answer has a
"### Hypotheses" table with a rank and a commit column, or the exact
NOT_FOUND sentence when nothing in scope explains the finding. The grader
reads the commit hashes from that table. Per run:
- format_ok: the Hypotheses table was parsed (rank and commit columns, at
  least one row), or the NOT_FOUND sentence was given.
- hit_at_1, hit_at_3 (code cases): an expected commit is cited in a row of
  rank <= 1 or <= 3; hit_rank is the best such rank (None if missed).
- not_found_ok (null cases): the NOT_FOUND sentence was given and no
  commit is cited in the table.
- fabricated: hashes anywhere in the answer that are not commits of the
  source repository.
- out_of_scope: commits cited in the table that touch no TRACE_SCOPE path.
- outside_range: commits cited in the table that are in neither or both
  builds (not in from...to), e.g. a later fix ranked as the cause.
- later_fix_found (runs with later commits shown, cases with
  later_fix_commits): a known later fix or revert is named anywhere in
  the answer. Reported, not part of pass.
- pass: code case, hit_at_3 and no NOT_FOUND sentence; null case,
  not_found_ok.
fabricated, out_of_scope and outside_range need a git checker and are None
without one.
"""

from __future__ import annotations

import re
from typing import Protocol

from stdiag.coresoftware_expert import NOT_FOUND

HASH = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*$")
SEPARATOR_CELL = re.compile(r"^:?-{2,}:?$")


class GitChecker(Protocol):
    def resolve(self, commit: str) -> str | None: ...
    def in_range(self, full: str, revision: dict) -> bool: ...
    def in_scope(self, full: str) -> bool: ...


def hashes_in(text: str, need_letter: bool = True) -> list[str]:
    """Hex words of 7-40 characters, in order.

    In free text, pure digits are left out (run numbers and timestamps are
    not hashes); in the commit column they are kept, since about 1 in 70
    short hashes has no letter.
    """
    found = []
    for word in HASH.findall(text):
        if (not need_letter or re.search("[a-f]", word)) and word not in found:
            found.append(word)
    return found


def flat(text: str) -> str:
    return " ".join(re.sub(r"[`*_>\"]", " ", text).split()).lower()


def section(answer: str, title: str) -> str | None:
    """Text under the first heading starting with title, up to the next."""
    inside, body = False, []
    for line in answer.splitlines():
        heading = HEADING.match(line)
        if heading:
            if inside:
                break
            inside = heading.group(1).lower().startswith(title.lower())
            continue
        if inside:
            body.append(line)
    return "\n".join(body) if inside else None


def cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def parse_answer(answer: str) -> dict:
    """Hypothesis rows (rank, commits), table_ok and not_found."""
    body = section(answer, "Hypotheses") or ""
    table = [line for line in body.splitlines() if line.strip().startswith("|")]
    rows, table_ok = [], False
    if table:
        header = [cell.lower() for cell in cells(table[0])]
        rank_col = next((i for i, c in enumerate(header)
                         if c.startswith("rank")), None)
        commit_col = next((i for i, c in enumerate(header)
                           if c.startswith("commit")), None)
        for order, line in enumerate(
                (line for line in table[1:] if not all(
                    SEPARATOR_CELL.match(c) for c in cells(line) if c)),
                start=1):
            row = cells(line)
            if commit_col is None or commit_col >= len(row):
                continue
            rank_text = row[rank_col] if rank_col is not None and \
                rank_col < len(row) else ""
            digits = re.search(r"\d+", rank_text)
            rows.append({"rank": int(digits.group()) if digits else order,
                         "commits": hashes_in(row[commit_col],
                                              need_letter=False)})
        table_ok = rank_col is not None and commit_col is not None and bool(rows)
    return {"rows": rows, "table_ok": table_ok,
            "not_found": flat(NOT_FOUND) in flat(answer),
            "all_hashes": hashes_in(answer)}


def matches(cited: str, expected: list[str]) -> bool:
    return any(full.startswith(cited) for full in expected)


def grade(case: dict, answer: str, git: GitChecker | None = None,
          show_later: bool = False) -> dict:
    parsed = parse_answer(answer)
    expected = [commit.lower() for commit in case.get("expected_commits", [])]
    hit_ranks = [row["rank"] for row in parsed["rows"]
                 if any(matches(c, expected) for c in row["commits"])]
    hit_rank = min(hit_ranks) if hit_ranks else None
    cited = [c for row in parsed["rows"] for c in row["commits"]]
    code_case = case["kind"] == "code"

    result = {
        "format_ok": parsed["table_ok"] or parsed["not_found"],
        "hit_at_1": (hit_rank is not None and hit_rank <= 1)
        if code_case else None,
        "hit_at_3": (hit_rank is not None and hit_rank <= 3)
        if code_case else None,
        "not_found_ok": None if code_case
        else parsed["not_found"] and not cited,
        "later_fix_found": None,
        "fabricated": None, "out_of_scope": None, "outside_range": None,
    }
    fixes = [commit.lower() for commit in case.get("later_fix_commits", [])]
    if show_later and fixes:
        result["later_fix_found"] = any(matches(h, fixes)
                                        for h in parsed["all_hashes"])
    resolved: dict[str, str | None] = {}
    if git is not None:
        resolved = {h: git.resolve(h) for h in parsed["all_hashes"]}
        for h in cited:
            resolved.setdefault(h, git.resolve(h))
        real = {h: full for h, full in resolved.items() if full}
        result["fabricated"] = sum(full is None for full in resolved.values())
        table_real = {real[h] for h in cited if h in real}
        result["out_of_scope"] = sum(not git.in_scope(full)
                                     for full in table_real)
        result["outside_range"] = sum(
            not git.in_range(full, case["revision"]) for full in table_real)
    result["pass"] = (bool(result["hit_at_3"]) and not parsed["not_found"]
                      if code_case else bool(result["not_found_ok"]))
    return {"grade": result,
            "meta": {"kind": case["kind"], "hit_rank": hit_rank,
                     "rows": parsed["rows"],
                     "not_found_stated": parsed["not_found"],
                     "unresolved": sorted(h for h, full in resolved.items()
                                          if full is None)}}
