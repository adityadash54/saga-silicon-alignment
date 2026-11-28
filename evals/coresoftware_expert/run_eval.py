#!/usr/bin/env python3
"""Run the coresoftware expert eval: does it find the commit behind a
known effect?

For each approved (case, rep) the runner writes the case's request and
labels to a fresh work directory and runs the real coresoftware expert
(python -m stdiag.coresoftware_expert, one-shot, read-only) exactly as in
production: in the range clone of the case's two versions
(stdiag/range_clone.py: checked out at "to", only the commits in one build
but not the other). By default the later commits are hidden (--no-later),
the hard case of a new regression that has no fix yet; --show-later runs
the production setting, where later fixes and reverts are visible as
evidence. The Markdown answer is graded by grade.py (code only), with every
cited hash checked in the source checkout. Results go to
evals/coresoftware_expert/runs/<variant>/ (git-ignored):
  results.jsonl           one row per scored (case, rep); reruns skip rows
                          already present
  errors.jsonl            attempts with nothing scorable (CLI failure,
                          timeout, wrong model served)
  work/<id>_rep<k>/       request, labels, answer, usage
The range clones are cached in --clones-dir (default STDIAG_CLONES_DIR,
runs/_clones), shared with production and across repetitions.

The cases are local only (cases.json, git-ignored). Before the first paid
run, and after any change to the harness or the cases, the reviewer
approves:  python evals/coresoftware_expert/run_eval.py --approve-harness
--self-test grades scripted answers (no AI) and checks with git that every
expected commit exists, lies in from...to and touches the code in scope.
--report prints aggregate results only (no case content), for the README.

Usage:
  python evals/coresoftware_expert/run_eval.py --self-test
  python evals/coresoftware_expert/run_eval.py --cases <id>,<id> --reps 1
  python evals/coresoftware_expert/run_eval.py --reps 2
  python evals/coresoftware_expert/run_eval.py --reps 1 --show-later --variant later
  python evals/coresoftware_expert/run_eval.py --report
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.coresoftware_expert.cases import CASES_FILE, load_cases  # noqa: E402
from evals.coresoftware_expert.grade import grade  # noqa: E402
from stdiag import coresoftware_expert  # noqa: E402
from stdiag.coresoftware_expert import (DEFAULT_MODEL, NOT_FOUND,  # noqa: E402
                                        TRACE_SCOPE)
from stdiag import paths  # noqa: E402
from stdiag.paths import CORESOFTWARE  # noqa: E402
from stdiag.range_clone import RangeCloneError, range_clone  # noqa: E402

FLOW_DIR = REPO_ROOT / "evals" / "coresoftware_expert" / "runs"
HARNESS_PATHS = ["evals/coresoftware_expert/run_eval.py",
                 "evals/coresoftware_expert/cases.py",
                 "evals/coresoftware_expert/grade.py",
                 "stdiag/coresoftware_expert.py", "stdiag/range_clone.py",
                 "maps/coresoftware_code_map.md"]
SCORES = ["pass", "format_ok", "hit_at_1", "hit_at_3", "not_found_ok",
          "later_fix_found", "fabricated", "out_of_scope", "outside_range"]
SCOPE_PATHS = [path for entries in TRACE_SCOPE.values()
               for path, _ in entries]
MAX_ATTEMPTS = 2
WRITE_LOCK = threading.Lock()
CLONE_LOCK = threading.Lock()


def git(repo: Path, *arguments: str, check: bool = True
        ) -> subprocess.CompletedProcess:
    """git in repo, without the user's git config (as the agent runs it)."""
    return subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True,
        check=check, env=coresoftware_expert.agent_environment(repo))


def build_clone(source: Path, case: dict, args) -> Path:
    """The case's range clone, made once under a lock (cached)."""
    with CLONE_LOCK:
        try:
            return range_clone(source, case["revision"]["from"],
                               case["revision"]["to"], args.clones_dir,
                               later=args.show_later)
        except RangeCloneError as error:
            sys.exit(f"ERROR: {error} (inside the Claude Code sandbox, pass "
                     "--clones-dir under $TMPDIR)")


class Git:
    """Commit checks in the source checkout (full history)."""

    def __init__(self, repo: Path):
        self.repo = repo
        self._files: dict[str, list[str]] = {}

    def resolve(self, commit: str) -> str | None:
        result = git(self.repo, "rev-parse", "--verify", "--quiet",
                     f"{commit}^{{commit}}", check=False)
        return result.stdout.strip() or None

    def is_ancestor(self, older: str, newer: str) -> bool:
        return git(self.repo, "merge-base", "--is-ancestor", older, newer,
                   check=False).returncode == 0

    def in_range(self, full: str, revision: dict) -> bool:
        """In one build but not the other (from...to)."""
        return (self.is_ancestor(full, revision["to"])
                != self.is_ancestor(full, revision["from"]))

    def files(self, full: str) -> list[str]:
        if full not in self._files:
            self._files[full] = git(
                self.repo, "diff-tree", "--root", "--no-commit-id",
                "--name-only", "-r", full).stdout.split()
        return self._files[full]

    def in_scope(self, full: str) -> bool:
        return any(name == path or name.startswith(path + "/")
                   for name in self.files(full) for path in SCOPE_PATHS)

    def subject(self, full: str) -> str:
        return git(self.repo, "log", "-1", "--format=%s", full).stdout.strip()

    def is_merge(self, full: str) -> bool:
        return len(git(self.repo, "rev-list", "--parents", "-1",
                       full).stdout.split()) > 2


def harness_sha(cases_file: Path) -> str:
    """Checksum of the harness files and the (local) cases file."""
    digest = hashlib.sha256()
    for path in [REPO_ROOT / p for p in HARNESS_PATHS] + [cases_file]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def run_one(case: dict, rep: int, variant_dir: Path, args, checker: Git) -> str:
    work = variant_dir / "work" / f"{case['id']}_rep{rep}"
    work.mkdir(parents=True, exist_ok=True)
    request_path, labels_path = work / "request.json", work / "labels.json"
    request_path.write_text(json.dumps(case["request"], indent=2) + "\n")
    labels_path.write_text(json.dumps(case.get("labels", {}), indent=2) + "\n")
    build_clone(args.repo, case, args)        # cached; the CLI finds it
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT),
           "PYTHONDONTWRITEBYTECODE": "1"}

    for attempt in range(1, MAX_ATTEMPTS + 1):
        usage_path, answer_path = work / "usage.json", work / "answer.md"
        for stale in (usage_path, answer_path):
            stale.unlink(missing_ok=True)
        start = time.monotonic()
        failure, detail = None, ""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "stdiag.coresoftware_expert",
                 str(request_path), "--from", case["revision"]["from"],
                 "--to", case["revision"]["to"], "--labels", str(labels_path),
                 "--repo", str(args.repo), "--clones-dir",
                 str(args.clones_dir), "--model", args.model,
                 "--effort", args.effort, "--usage-out", str(usage_path)]
                + ([] if args.show_later else ["--no-later"]),
                cwd=REPO_ROOT, env=env, text=True, capture_output=True,
                timeout=args.timeout_s)
            if result.returncode:
                failure, detail = "cli_error", result.stderr[-1000:]
            else:
                answer_path.write_text(result.stdout)
        except subprocess.TimeoutExpired:
            failure, detail = "timeout", f"> {args.timeout_s} s"
        wall_s = time.monotonic() - start
        usage = (json.loads(usage_path.read_text())
                 if usage_path.exists() else {})
        if not failure and args.model not in (usage.get("models")
                                              or [args.model]):
            failure = "model_mismatch"
            detail = f"served {usage.get('models')}, requested {args.model}"
        if failure:
            append_jsonl(variant_dir / "errors.jsonl", {
                "prompt_id": case["id"], "rep": rep, "attempt": attempt,
                "failure_class": failure, "detail": detail,
                "usage": usage.get("usage", {}), "wall_s": round(wall_s, 1)})
            if attempt < MAX_ATTEMPTS:
                time.sleep(random.uniform(10, 30))
            continue

        graded = grade(case, answer_path.read_text(), checker,
                       show_later=args.show_later)
        tokens = usage.get("usage", {})
        append_jsonl(variant_dir / "results.jsonl", {
            "prompt_id": case["id"], "rep": rep, "status": "ok",
            "grade": graded["grade"], "meta": graded["meta"]
            | {"attempt": attempt, "num_turns": usage.get("num_turns"),
               "stop_reason": usage.get("subtype", "")},
            "model": args.model, "effort": args.effort,
            "show_later": args.show_later,
            "versions": {name: checker.resolve(case["revision"][name])
                         for name in ("from", "to")},
            "latency_s": round((usage.get("duration_ms") or wall_s * 1000)
                               / 1000, 1),
            "in_tokens": tokens.get("input_tokens", 0)
            + tokens.get("cache_read_input_tokens", 0)
            + tokens.get("cache_creation_input_tokens", 0),
            "out_tokens": tokens.get("output_tokens", 0),
            "wall_s": round(wall_s, 1)})
        verdict = "PASS" if graded["grade"]["pass"] else "fail"
        return f"{case['id']} rep{rep}: {verdict} {graded['grade']}"
    return f"{case['id']} rep{rep}: ERROR (see errors.jsonl)"


# --- self-test ---------------------------------------------------------------

TABLE_HEADER = ("| rank | tier | commit | date | author | files:lines | "
                "mechanism | fit to the signature | test to confirm | "
                "confidence |\n|---|---|---|---|---|---|---|---|---|---|\n")


def scripted_answer(commits: list[str], not_found: bool = False) -> str:
    """An answer in the expert's format citing commits in rank order."""
    rows = "".join(f"| {rank} | tier 1 | {commit[:9]} | 2000-01-01 | A | "
                   "f.cc:1 | m | f | t | medium |\n"
                   for rank, commit in enumerate(commits, start=1))
    hypotheses = (NOT_FOUND + "\n") if not_found else TABLE_HEADER + rows
    return (f"### Summary\nscripted\n\n### Hypotheses\n{hypotheses}\n"
            "### Examined and ruled out\nNone.\n\n"
            "### Non-code causes to check\nNone.\n\n"
            "### Commands run\n`git log`\n")


# Common words of findings and commit subjects that give nothing away.
COMMON_WORDS = {"residual", "residuals", "cluster", "clusters", "track",
                "tracks", "tracking", "update", "updates", "change", "changes",
                "local", "global", "alignment", "layer", "layers", "stave",
                "staves", "chip", "chips", "error", "errors", "mvtx", "intt",
                "width", "shift", "the", "with", "from", "into", "fixed"}


def subject_overlap(subject: str, sent: str) -> list[str]:
    words = {w for w in re.findall(r"[a-z][a-z0-9_]{4,}", subject.lower())
             if w not in COMMON_WORDS}
    return sorted(w for w in words if w in sent)


def git_problems(case: dict, checker: Git) -> tuple[list[str], list[str]]:
    """(problems, notes) of a case against the source checkout."""
    problems, notes = [], []
    revision = case["revision"]
    full = {name: checker.resolve(revision[name]) for name in ("from", "to")}
    for name, value in full.items():
        if value is None:
            problems.append(f"revision {name} {revision[name]} not found")
    if None in full.values():
        return problems, notes
    if not checker.is_ancestor(full["from"], full["to"]):
        notes.append("from is not an ancestor of to: builds on different "
                     "branches (both sides are candidates)")
    sent = json.dumps([case["request"], case.get("labels", {})]).lower()
    for commit in case.get("expected_commits", []):
        resolved = checker.resolve(commit)
        if resolved is None:
            problems.append(f"expected {commit[:9]} missing")
            continue
        if not checker.in_range(resolved, revision):
            problems.append(f"expected {commit[:9]} not in from...to")
        if not checker.in_scope(resolved):
            problems.append(f"expected {commit[:9]} touches no TRACE_SCOPE "
                            "path")
        if checker.is_merge(resolved):
            problems.append(f"expected {commit[:9]} is a merge (the commit "
                            "list leaves merges out)")
        overlap = subject_overlap(checker.subject(resolved), sent)
        if overlap:
            notes.append(f"subject words of {commit[:9]} in the request: "
                         f"{overlap} (check for a giveaway)")
    head = checker.resolve("HEAD")
    for commit in case.get("later_fix_commits", []):
        resolved = checker.resolve(commit)
        if resolved is None:
            problems.append(f"later fix {commit[:9]} missing")
        elif (checker.is_ancestor(resolved, full["from"])
              or checker.is_ancestor(resolved, full["to"])):
            problems.append(f"later fix {commit[:9]} is in a build")
        elif not checker.is_ancestor(resolved, head):
            problems.append(f"later fix {commit[:9]} is not in HEAD, so "
                            "--show-later cannot show it")
    try:
        listed = coresoftware_expert.traced_commits(checker.repo, revision)
        counts = {tier: len(found) for tier, found in listed.items()}
        notes.append("in-scope commits in range: " + ", ".join(
            f"{tier} {count}" for tier, count in counts.items()))
        if max(counts.values(), default=0) > coresoftware_expert.MAX_LISTED_COMMITS:
            notes.append("more commits than the request lists "
                         f"({coresoftware_expert.MAX_LISTED_COMMITS} per tier)")
    except subprocess.CalledProcessError as error:
        problems.append(f"commit list failed: {error.stderr.strip()}")
    return problems, notes


def self_test(data: dict, checker: Git | None) -> int:
    """Right answers must pass, wrong ones fail; with git, every case must
    be consistent with the checkout."""
    failures = 0
    if checker is not None and checker.resolve(
            data["coresoftware_revision"]) is None:
        print(f"coresoftware_revision {data['coresoftware_revision'][:12]} "
              f"not found in {checker.repo}")
        failures += 1
    decoy = "abcdef0" * 5 + "abcde"
    for case in data["cases"]:
        expected = case.get("expected_commits", [])
        if case["kind"] == "code":
            answers = {"right_rank1": (scripted_answer(expected[:1]), True),
                       "right_rank3": (scripted_answer(
                           [decoy, decoy[::-1], expected[0]]), True),
                       "right_rank4": (scripted_answer(
                           [decoy] * 3 + expected[:1]), False),
                       "not_found": (scripted_answer([], True), False),
                       "no_table": ("### Summary\nnothing", False)}
        else:
            answers = {"not_found": (scripted_answer([], True), True),
                       "a_commit": (scripted_answer([decoy]), False),
                       "no_table": ("### Summary\nnothing", False)}
        problems = [name for name, (answer, want) in answers.items()
                    if grade(case, answer)["grade"]["pass"] != want]
        notes: list[str] = []
        if checker is not None:
            more, notes = git_problems(case, checker)
            problems += more
        failures += bool(problems)
        print(f"{case['id']:<28} [{case['status']}] {case['kind']:<4} "
              + ("ok" if not problems else f"UNEXPECTED: {problems}"))
        for note in notes:
            print(f"{'':<30}note: {note}")
    print("self-test", "FAILED" if failures else "passed",
          f"({failures} problem case(s))")
    return 1 if failures else 0


def report(variant_dir: Path) -> None:
    """Aggregate results only: no case ids, findings or commits."""
    results = read_jsonl(variant_dir / "results.jsonl")
    errors = read_jsonl(variant_dir / "errors.jsonl")
    if not results:
        print(f"no results in {variant_dir}")
        return
    print(f"{variant_dir.name}: {len(results)} scored run(s), "
          f"{len({r['prompt_id'] for r in results})} case(s), "
          f"{len(errors)} failed attempt(s)")
    for kind in ("code", "null"):
        rows = [r for r in results if r["meta"].get("kind") == kind]
        if not rows:
            continue
        print(f"  {kind} cases ({len(rows)} run(s)):")
        for score in SCORES:
            values = [r["grade"].get(score) for r in rows
                      if r["grade"].get(score) is not None]
            if not values:
                continue
            total = sum(int(v) for v in values)
            label = (f"{total}/{len(values)}"
                     if all(isinstance(v, bool) for v in values)
                     else f"total {total} over {len(values)} run(s)")
            print(f"    {score:<17} {label}")
    for field in ("in_tokens", "out_tokens", "latency_s"):
        values = sorted(r[field] for r in results)
        print(f"  {field:<19} mean {sum(values) / len(values):.0f}, "
              f"min {values[0]:.0f}, max {values[-1]:.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases-file", type=Path, default=CASES_FILE)
    parser.add_argument("--cases", help="comma-separated case ids (default "
                        "all approved)")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--variant", default="baseline")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--effort", default="high",
                        choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--repo", type=Path, default=CORESOFTWARE,
                        help="coresoftware checkout with full history "
                        "(default STDIAG_CORESOFTWARE)")
    parser.add_argument("--clones-dir", type=Path,
                        help="where range clones are cached (default "
                        "STDIAG_CLONES_DIR or runs/_clones; the Claude Code "
                        "sandbox refuses writes to .git/config inside the "
                        "workspace, so use a folder under $TMPDIR there)")
    parser.add_argument("--show-later", action="store_true",
                        help="production setting: show the checkout's "
                        "later commits (default hidden)")
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-git", action="store_true",
                        help="self-test without the git checks")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--approve-harness", action="store_true",
                        help="record the harness checksum (reviewer only)")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.clones_dir = (args.clones_dir or paths.CLONES_DIR).resolve()

    if args.report:
        report(FLOW_DIR / args.variant)
        return
    try:
        data = load_cases(args.cases_file)
    except (FileNotFoundError, ValueError) as error:
        sys.exit(f"ERROR: {error}")
    cases = data["cases"]
    if args.cases:
        wanted = args.cases.split(",")
        unknown = set(wanted) - {case["id"] for case in cases}
        if unknown:
            sys.exit(f"unknown case ids: {sorted(unknown)}")
        cases = [case for case in cases if case["id"] in wanted]

    if not args.no_git and not (args.repo / ".git").exists():
        sys.exit(f"ERROR: {args.repo} is not a git checkout (set "
                 "STDIAG_CORESOFTWARE or pass --repo).")
    checker = None if args.no_git else Git(args.repo)
    if args.self_test:
        sys.exit(self_test({**data, "cases": cases}, checker))

    FLOW_DIR.mkdir(parents=True, exist_ok=True)
    approval = FLOW_DIR / "_harness_approval.json"
    sha = harness_sha(args.cases_file)
    if args.approve_harness:
        approval.write_text(json.dumps({"sha256": sha,
                                        "paths": HARNESS_PATHS
                                        + [args.cases_file.name]}, indent=2))
        print(f"Harness approved: {sha[:12]}")
        return
    if not approval.exists() or json.loads(approval.read_text())["sha256"] != sha:
        sys.exit("The harness or the cases are not approved or changed since "
                 "approval. The reviewer must run:\n  python "
                 "evals/coresoftware_expert/run_eval.py --approve-harness")
    if checker is None:
        sys.exit("ERROR: --no-git is for the self-test only.")

    drafts = [case["id"] for case in cases if case["status"] != "approved"]
    if drafts:
        print(f"skipping {len(drafts)} draft case(s): {', '.join(drafts)}")
    cases = [case for case in cases if case["status"] == "approved"]
    variant_dir = FLOW_DIR / args.variant
    variant_dir.mkdir(exist_ok=True)
    done = {(row["prompt_id"], row["rep"])
            for row in read_jsonl(variant_dir / "results.jsonl")}
    todo = [(case, rep) for case in cases for rep in range(args.reps)
            if (case["id"], rep) not in done]
    print(f"{len(todo)} run(s) to do ({len(done)} already done), model "
          f"{args.model}, effort {args.effort}, coresoftware {args.repo}, "
          f"later commits {'shown' if args.show_later else 'hidden'}")
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run_one, case, rep, variant_dir, args, checker)
                   for case, rep in todo]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
