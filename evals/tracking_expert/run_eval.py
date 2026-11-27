#!/usr/bin/env python3
"""Run the tracking expert eval: hypotheses with known verdicts.

For each approved (case, rep) the runner writes the case's request and
coresoftware-expert answer to a fresh work directory, runs the real
tracking expert (python -m stdiag.tracking_expert, one-shot, read-only,
in the coresoftware checkout given by STDIAG_CORESOFTWARE) and grades its
JSON verdict block with grade.py (code only). Results go to
evals/tracking_expert/runs/<variant>/ (git-ignored):
  results.jsonl           one row per scored (case, rep); reruns skip rows
                          already present
  errors.jsonl            attempts with nothing scorable (CLI failure,
                          timeout, wrong model served)
  work/<id>_rep<k>/       request, answer, review, verdicts, usage

The cases are local only (cases.json, git-ignored). Before the first paid
run, and after any change to the harness or the cases, the reviewer
approves:  python evals/tracking_expert/run_eval.py --approve-harness
--self-test grades scripted answers (no AI) and checks every cited commit
exists and every fabricated one does not. --report prints aggregate
results only (no case content), for the README.

Usage:
  python evals/tracking_expert/run_eval.py --self-test
  python evals/tracking_expert/run_eval.py --cases <id>,<id> --reps 1
  python evals/tracking_expert/run_eval.py --reps 2
  python evals/tracking_expert/run_eval.py --report
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from evals.tracking_expert.cases import CASES_FILE, load_cases  # noqa: E402
from evals.tracking_expert.grade import confusion, grade  # noqa: E402
from stdiag import fetch_tracking_docs, tracking_expert  # noqa: E402
from stdiag.coresoftware_expert import DEFAULT_MODEL  # noqa: E402
from stdiag.paths import CORESOFTWARE, TRACKING_DOCS  # noqa: E402

FLOW_DIR = REPO_ROOT / "evals" / "tracking_expert" / "runs"
HARNESS_PATHS = ["evals/tracking_expert/run_eval.py",
                 "evals/tracking_expert/cases.py",
                 "evals/tracking_expert/grade.py",
                 "stdiag/tracking_expert.py", "stdiag/coresoftware_expert.py",
                 "stdiag/fetch_tracking_docs.py", "maps/tracking_docs_map.md"]
SCORES = ["pass", "json_ok", "verdict_acc", "error_catch", "false_alarm",
          "fabrication_caught", "version_caveat"]
MAX_ATTEMPTS = 2
WRITE_LOCK = threading.Lock()


def harness_sha(cases_file: Path) -> str:
    """Checksum of the harness files and the (local) cases file."""
    digest = hashlib.sha256()
    for path in [REPO_ROOT / p for p in HARNESS_PATHS] + [cases_file]:
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def git_has_commit(repo: Path, commit: str) -> bool:
    return subprocess.run(
        ["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"],
        capture_output=True).returncode == 0


def append_jsonl(path: Path, row: dict) -> None:
    with WRITE_LOCK, path.open("a") as handle:
        handle.write(json.dumps(row) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def docs_versions() -> dict:
    """ACTS commit and Millepede page checksum the agent read."""
    sources = fetch_tracking_docs.read_sources(TRACKING_DOCS)
    return {"acts_commit": sources.get("acts", {}).get("commit"),
            "acts_version": sources.get("acts", {}).get("version_number"),
            "millepede_sha256": sources.get("millepede", {}).get("sha256")}


def run_one(case: dict, rep: int, variant_dir: Path, args) -> str:
    work = variant_dir / "work" / f"{case['id']}_rep{rep}"
    work.mkdir(parents=True, exist_ok=True)
    request_path, answer_path = work / "request.json", work / "answer.md"
    request_path.write_text(json.dumps(case["request"], indent=2) + "\n")
    answer_path.write_text(case["answer"])
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT),
           "PYTHONDONTWRITEBYTECODE": "1",
           "STDIAG_CORESOFTWARE": str(args.repo)}
    if args.no_literature:
        env["STDIAG_LITERATURE_INDEX"] = str(work / "no_literature.sqlite")

    for attempt in range(1, MAX_ATTEMPTS + 1):
        usage_path, verdicts_path = work / "usage.json", work / "verdicts.json"
        for stale in (usage_path, verdicts_path, work / "review.md"):
            stale.unlink(missing_ok=True)
        start = time.monotonic()
        failure, detail = None, ""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "stdiag.tracking_expert",
                 str(request_path), str(answer_path), "--repo", str(args.repo),
                 "--model", args.model, "--effort", args.effort,
                 "--usage-out", str(usage_path),
                 "--verdicts-out", str(verdicts_path)],
                cwd=REPO_ROOT, env=env, text=True, capture_output=True,
                timeout=args.timeout_s)
            if result.returncode:
                failure, detail = "cli_error", result.stderr[-1000:]
            else:
                (work / "review.md").write_text(result.stdout)
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

        block = json.loads(verdicts_path.read_text())["block"]
        graded = grade(case, block)
        tokens = usage.get("usage", {})
        append_jsonl(variant_dir / "results.jsonl", {
            "prompt_id": case["id"], "rep": rep, "status": "ok",
            "grade": graded["grade"], "meta": graded["meta"]
            | {"attempt": attempt, "num_turns": usage.get("num_turns"),
               "stop_reason": usage.get("subtype", "")},
            "model": args.model, "effort": args.effort,
            "no_literature": args.no_literature, "docs": docs_versions(),
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


def scripted_block(case: dict, choose) -> dict:
    """A verdict block with choose(hypothesis) -> (verdict, code_verified)."""
    verdicts = []
    for hypothesis in case["hypotheses"]:
        verdict, verified = choose(hypothesis)
        verdicts.append({"hypothesis": hypothesis["number"],
                         "verdict": verdict, "code_verified": verified,
                         "reason": "scripted"})
    caveats = ["scripted caveat"] if case.get("expect_version_caveat") else []
    return {"verdicts": verdicts, "version_caveats": caveats}


def self_test(data: dict, repo: Path | None) -> int:
    """Perfect answers must pass; uniform and missing answers must not
    (unless the case accepts them); commits must exist or not as stated."""
    failures = 0
    if repo is not None and not git_has_commit(
            repo, data["coresoftware_revision"]):
        print(f"coresoftware_revision {data['coresoftware_revision'][:12]} "
              f"not found in {repo}")
        failures += 1
    for case in data["cases"]:
        hypotheses = case["hypotheses"]
        answers = {
            "perfect": scripted_block(case, lambda h: (
                h["accept"][0], (h.get("code_verified") or ["yes"])[0])),
            "all_consistent": scripted_block(
                case, lambda h: ("consistent", "yes")),
            "all_cannot_verify": scripted_block(
                case, lambda h: ("cannot verify", "partly")),
            "no_json": None,
        }
        scores = {name: grade(case, block)["grade"]["pass"]
                  for name, block in answers.items()}
        expected = {"perfect": True, "no_json": False}
        for name, verdict, verified in (
                ("all_consistent", "consistent", "yes"),
                ("all_cannot_verify", "cannot verify", "partly")):
            expected[name] = all(
                verdict in h["accept"]
                and verified in (h.get("code_verified") or [verified])
                for h in hypotheses)
        problems = [name for name, want in expected.items()
                    if bool(scores[name]) != want]
        # Allowed (a case with only correct hypotheses), but worth seeing.
        note = ("  note: a uniform answer passes" if expected["all_consistent"]
                or expected["all_cannot_verify"] else "")
        if repo is not None:
            for hypothesis in hypotheses:
                for commit in hypothesis.get("cited_commits", []):
                    if not git_has_commit(repo, commit):
                        problems.append(f"H{hypothesis['number']} cited "
                                        f"{commit} missing")
                for commit in hypothesis.get("fabricated_commits", []):
                    if git_has_commit(repo, commit):
                        problems.append(f"H{hypothesis['number']} "
                                        f"fabricated {commit} exists")
        failures += bool(problems)
        print(f"{case['id']:<24} [{case['status']}] H={len(hypotheses)} "
              + " ".join(f"{name}={int(bool(score))}"
                         for name, score in scores.items())
              + (f"  UNEXPECTED: {problems}" if problems else "") + note)
    print("self-test", "FAILED" if failures else "passed",
          f"({failures} problem case(s))")
    return 1 if failures else 0


def report(variant_dir: Path) -> None:
    """Aggregate results only: no case ids, findings or hypotheses."""
    results = read_jsonl(variant_dir / "results.jsonl")
    errors = read_jsonl(variant_dir / "errors.jsonl")
    if not results:
        print(f"no results in {variant_dir}")
        return
    print(f"{variant_dir.name}: {len(results)} scored run(s), "
          f"{len({r['prompt_id'] for r in results})} case(s), "
          f"{len(errors)} failed attempt(s)")
    for score in SCORES:
        values = [r["grade"][score] for r in results
                  if r["grade"][score] is not None]
        if values:
            total = sum(float(v) for v in values)
            label = (f"{int(total)}/{len(values)}"
                     if all(isinstance(v, bool) for v in values)
                     else f"mean {total / len(values):.2f} over {len(values)}")
            print(f"  {score:<19} {label}")
    print("  verdicts given, per hypothesis kind:")
    for kind, counts in sorted(confusion(results).items()):
        print(f"    {kind:<14} " + "  ".join(f"{label}={count}" for label,
                                              count in counts.items()))
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
                        help="coresoftware checkout (default "
                        "STDIAG_CORESOFTWARE)")
    parser.add_argument("--no-literature", action="store_true",
                        help="run without the literature index (variant)")
    parser.add_argument("--timeout-s", type=float, default=3600)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--no-git", action="store_true",
                        help="self-test without the commit checks")
    parser.add_argument("--report", action="store_true")
    parser.add_argument("--approve-harness", action="store_true",
                        help="record the harness checksum (reviewer only)")
    args = parser.parse_args()

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

    if args.self_test:
        sys.exit(self_test({**data, "cases": cases},
                           None if args.no_git else args.repo))

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
                 "evals/tracking_expert/run_eval.py --approve-harness")

    drafts = [case["id"] for case in cases if case["status"] != "approved"]
    if drafts:
        print(f"skipping {len(drafts)} draft case(s): {', '.join(drafts)}")
    cases = [case for case in cases if case["status"] == "approved"]
    tracking_expert.require_inputs(args.repo)
    variant_dir = FLOW_DIR / args.variant
    variant_dir.mkdir(exist_ok=True)
    done = {(row["prompt_id"], row["rep"])
            for row in read_jsonl(variant_dir / "results.jsonl")}
    todo = [(case, rep) for case in cases for rep in range(args.reps)
            if (case["id"], rep) not in done]
    print(f"{len(todo)} run(s) to do ({len(done)} already done), model "
          f"{args.model}, effort {args.effort}, coresoftware {args.repo}"
          + (", no literature" if args.no_literature else ""))
    with concurrent.futures.ThreadPoolExecutor(args.concurrency) as pool:
        futures = [pool.submit(run_one, case, rep, variant_dir, args)
                   for case, rep in todo]
        for future in concurrent.futures.as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()
