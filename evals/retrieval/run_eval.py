#!/usr/bin/env python3
"""Retrieval eval for the literature index (stdiag-literature).

For every case in cases.json and every mode (bm25, dense, hybrid), runs the
search and checks whether an expected file/page range is among the top k
passages. Reports per mode:

- recall@5, recall@10 and mean reciprocal rank (MRR, over the top 10) of the
  raw ranking, with no limit on passages per paper;
- hit@6 (2/paper): whether the list the tracking expert's request gets (top
  6, at most 2 passages per paper) contains an expected passage.

Three kinds of case:

- question: "question" is searched as written;
- agent path: "finding" (title, evidence, entries) and optionally
  "answer_summary" instead of "question"; the query is built by
  tracking_expert.literature_query(), exactly as for the agent's request;
- no answer: "expected": [] (nothing in the collection answers it). Left out
  of recall and MRR; their highest cosine similarity is compared with the
  answerable cases' to see whether a "no relevant literature" cut-off could
  work.

No AI model is called apart from the local Ollama embedding model; nothing
leaves the machine.

Results go to evals/retrieval/runs/<model>_<UTC time>.md (git-ignored).
Cases with status "draft" are included but flagged, since their expected
pages have not been checked by the reviewer yet.

The cases name your papers and pages, so cases.json is git-ignored; copy
cases.example.json to cases.json and write your own.

Usage:
  python evals/retrieval/run_eval.py                    # all modes
  python evals/retrieval/run_eval.py --modes bm25       # without Ollama
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stdiag.literature import (DEFAULT_MODEL, Hit, LiteratureIndex,  # noqa: E402
                               OllamaEmbedder, OllamaError)
from stdiag.paths import LITERATURE_INDEX  # noqa: E402
from stdiag.tracking_expert import (LITERATURE_PASSAGES,  # noqa: E402
                                    literature_query)

HERE = Path(__file__).resolve().parent
DEPTH = 10
# What the tracking expert's request gets: search_literature() defaults
REQUEST_K = LITERATURE_PASSAGES
REQUEST_CAP = 2


def case_query(case: dict) -> str:
    """The search text: the question, or the agent's query for a finding."""
    if case.get("question"):
        return case["question"]
    if case.get("finding"):
        return literature_query({"finding": case["finding"]},
                                case.get("answer_summary", ""))
    raise ValueError(f"case {case.get('id', '?')} has neither 'question' "
                     "nor 'finding'")


def answerable(case: dict) -> bool:
    return bool(case["expected"])


def kind(case: dict) -> str:
    text = "question" if case.get("question") else "agent"
    return text if answerable(case) else f"{text}, no answer"


def matches(hit: Hit, expected: list[dict]) -> bool:
    name = Path(hit.path).name
    return any(name == item["file"] and hit.page_start <= item["pages"][1]
               and hit.page_end >= item["pages"][0] for item in expected)


def first_rank(hits: list[Hit], expected: list[dict]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if matches(hit, expected):
            return rank
    return None


def top_cosine(index: LiteratureIndex, embedder: OllamaEmbedder,
               query: str) -> float | None:
    """Highest cosine similarity of the query with any chunk, or None."""
    try:
        if index.check_embedder(embedder):
            return None
        ids, matrix = index._embeddings()
        if not len(ids):
            return None
        vector = np.asarray(embedder.embed([query])[0], dtype=np.float32)
        return float(np.max(matrix @ vector))
    except OllamaError:
        return None


def summarize(cases: list[dict], ranks: list[int | None],
              request_hits: list[bool | None]) -> dict[str, float]:
    keep = [number for number, case in enumerate(cases) if answerable(case)]
    n = len(keep) or 1
    kept = [ranks[number] for number in keep]
    return {"recall@5": sum(1 for r in kept if r and r <= 5) / n,
            "recall@10": sum(1 for r in kept if r and r <= 10) / n,
            "MRR@10": sum(1 / r for r in kept if r) / n,
            "hit@6": sum(1 for number in keep if request_hits[number]) / n}


def run(cases: list[dict], modes: list[str], index: LiteratureIndex,
        embedder: OllamaEmbedder) -> dict:
    ranks: dict[str, list[int | None]] = {mode: [] for mode in modes}
    request_hits: dict[str, list[bool | None]] = {mode: [] for mode in modes}
    cosines: list[float | None] = []
    warnings: list[str] = []
    dense = any(mode in ("dense", "hybrid") for mode in modes)
    for case in cases:
        query = case_query(case)
        for mode in modes:
            # max_per_document=DEPTH: score raw ranking, no per-paper cap
            hits, notes = index.search(query, k=DEPTH, mode=mode,
                                       embedder=embedder,
                                       max_per_document=DEPTH)
            request, more = index.search(query, k=REQUEST_K, mode=mode,
                                         embedder=embedder,
                                         max_per_document=REQUEST_CAP)
            warnings.extend(f"{case['id']} {mode}: {note}"
                            for note in [*notes, *more])
            ranks[mode].append(first_rank(hits, case["expected"]))
            request_hits[mode].append(
                first_rank(request, case["expected"]) is not None
                if answerable(case) else None)
        cosines.append(top_cosine(index, embedder, query) if dense else None)
    return {"ranks": ranks, "request_hits": request_hits,
            "cosines": cosines, "warnings": sorted(set(warnings))}


def cutoff_lines(cases: list[dict], cosines: list[float | None]) -> list[str]:
    """No-answer cases against answerable ones, by highest cosine."""
    negative = [(case["id"], value) for case, value in zip(cases, cosines)
                if not answerable(case)]
    if not negative:
        return []
    lines = ["", "## No-answer cases", ""]
    if all(value is None for _, value in negative):
        return lines + ["Highest cosine not available (needs the dense or "
                        "hybrid mode and the index's embedding model)."]
    lines += ["| case | highest cosine |", "|---|---|"]
    lines += [f"| {name} | " + ("-" if value is None else f"{value:.3f}")
              + " |" for name, value in negative]
    positive = [value for case, value in zip(cases, cosines)
                if answerable(case) and value is not None]
    worst = max(value for _, value in negative if value is not None)
    if positive:
        dropped = sum(1 for value in positive if value <= worst)
        lines += ["", f"Answerable cases: median highest cosine "
                  f"{statistics.median(positive):.3f}, lowest "
                  f"{min(positive):.3f}. A cut-off at the highest no-answer "
                  f"value ({worst:.3f}) would also drop {dropped} of "
                  f"{len(positive)} answerable cases."]
    return lines


def report(cases: list[dict], results: dict, meta: dict) -> str:
    ranks, request_hits = results["ranks"], results["request_hits"]
    modes = list(ranks)
    drafts = sum(1 for case in cases if case.get("status") != "approved")
    positives = sum(1 for case in cases if answerable(case))
    agent = sum(1 for case in cases if not case.get("question"))
    lines = [f"# Retrieval eval, {meta.get('embed_model')} "
             f"({dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC)", "",
             f"Index: {meta.get('index')}; {meta.get('chunks')} chunks, "
             f"chunk_words {meta.get('chunk_words')}; {len(cases)} cases "
             f"({positives} answerable, {len(cases) - positives} no-answer; "
             f"{agent} agent-path)"
             + (f" (**{drafts} still drafts**: expected pages not yet "
                "checked by the reviewer)" if drafts else ""), "",
             "Recall and MRR score the raw ranking with no limit per paper; "
             f"hit@{REQUEST_K} ({REQUEST_CAP}/paper) scores the list the "
             "tracking expert's request gets. Both count answerable cases "
             "only.", "",
             f"| mode | recall@5 | recall@10 | MRR@10 | hit@{REQUEST_K} "
             f"({REQUEST_CAP}/paper) |", "|---|---|---|---|---|"]
    for mode in modes:
        scores = summarize(cases, ranks[mode], request_hits[mode])
        lines.append(f"| {mode} | {scores['recall@5']:.2f} | "
                     f"{scores['recall@10']:.2f} | {scores['MRR@10']:.2f} | "
                     f"{scores['hit@6']:.2f} |")
    lines += ["", f"First matching rank per case (- = not in top {DEPTH}) "
              f"and hit@{REQUEST_K} ({REQUEST_CAP}/paper):", "",
              "| case | type | status | "
              + " | ".join(f"{mode} rank" for mode in modes) + " | "
              + " | ".join(f"{mode} hit@{REQUEST_K}" for mode in modes)
              + " | highest cosine |",
              "|---|---|---|" + "---|" * (2 * len(modes) + 1)]
    for number, case in enumerate(cases):
        if answerable(case):
            rank_cells = [str(ranks[mode][number] or "-") for mode in modes]
            hit_cells = ["yes" if request_hits[mode][number] else "no"
                         for mode in modes]
        else:
            rank_cells = hit_cells = ["n/a"] * len(modes)
        cosine = results["cosines"][number]
        lines.append(f"| {case['id']} | {kind(case)} | "
                     f"{case.get('status', '?')} | " + " | ".join(rank_cells)
                     + " | " + " | ".join(hit_cells) + " | "
                     + ("-" if cosine is None else f"{cosine:.3f}") + " |")
    agent_cases = [case for case in cases if not case.get("question")]
    if agent_cases:
        lines += ["", "## Agent-path queries (built by literature_query)", ""]
        lines += [f"- {case['id']}: {case_query(case)}"
                  for case in agent_cases]
    lines += cutoff_lines(cases, results["cosines"])
    if results["warnings"]:
        lines += ["", "Warnings:",
                  *[f"- {warning}" for warning in results["warnings"]]]
    return "\n".join(lines) + "\n"


def load_cases(path: Path) -> list[dict]:
    """Cases from cases.json; exits with the reason if one is malformed."""
    cases = json.loads(path.read_text())["cases"]
    for case in cases:
        try:
            case_query(case)
        except ValueError as error:
            sys.exit(f"ERROR: {error} in {path}.")
        if not isinstance(case.get("expected"), list):
            sys.exit(f"ERROR: case {case.get('id', '?')} needs an 'expected' "
                     f"list ([] for a no-answer case) in {path}.")
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", type=Path, default=HERE / "cases.json")
    parser.add_argument("--index", type=Path, default=LITERATURE_INDEX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modes", default="bm25,dense,hybrid")
    parser.add_argument("--out-dir", type=Path, default=HERE / "runs")
    args = parser.parse_args()

    if not args.cases.exists():
        sys.exit(f"ERROR: no test cases at {args.cases}. Copy "
                 f"{HERE / 'cases.example.json'} to {HERE / 'cases.json'} "
                 "and replace the examples with questions about your papers.")
    cases = load_cases(args.cases)
    index = LiteratureIndex(args.index, readonly=True)
    status = index.status()
    results = run(cases, args.modes.split(","), index,
                  OllamaEmbedder(args.model))
    text = report(cases, results, status)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{str(status.get('embed_model')).replace(':', '-')}_"
            f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.md")
    (args.out_dir / name).write_text(text)
    print(text)
    print(f"Written to {args.out_dir / name}")


if __name__ == "__main__":
    main()
