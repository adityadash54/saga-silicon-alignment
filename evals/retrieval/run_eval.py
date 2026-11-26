#!/usr/bin/env python3
"""Retrieval eval for the literature index (stdiag-literature).

For every case in cases.json and every mode (bm25, dense, hybrid), runs the
search and checks whether an expected file/page range is among the top k
passages. Reports recall@5, recall@10 and mean reciprocal rank (MRR, over
the top 10) per mode, plus the first matching rank per case. No AI model is
called apart from the local Ollama embedding model; nothing leaves the
machine.

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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from stdiag.literature import (DEFAULT_MODEL, Hit, LiteratureIndex,  # noqa: E402
                               OllamaEmbedder)
from stdiag.paths import LITERATURE_INDEX  # noqa: E402

HERE = Path(__file__).resolve().parent
DEPTH = 10


def matches(hit: Hit, expected: list[dict]) -> bool:
    name = Path(hit.path).name
    return any(name == item["file"] and hit.page_start <= item["pages"][1]
               and hit.page_end >= item["pages"][0] for item in expected)


def first_rank(hits: list[Hit], expected: list[dict]) -> int | None:
    for rank, hit in enumerate(hits, start=1):
        if matches(hit, expected):
            return rank
    return None


def summarize(ranks: list[int | None]) -> dict[str, float]:
    n = len(ranks) or 1
    return {"recall@5": sum(1 for r in ranks if r and r <= 5) / n,
            "recall@10": sum(1 for r in ranks if r and r <= 10) / n,
            "MRR@10": sum(1 / r for r in ranks if r) / n}


def run(cases: list[dict], modes: list[str], index: LiteratureIndex,
        embedder: OllamaEmbedder) -> tuple[dict, list[str]]:
    ranks: dict[str, list[int | None]] = {mode: [] for mode in modes}
    warnings: list[str] = []
    for case in cases:
        for mode in modes:
            # max_per_document=DEPTH: score raw ranking, no per-paper cap
            hits, notes = index.search(case["question"], k=DEPTH, mode=mode,
                                       embedder=embedder,
                                       max_per_document=DEPTH)
            warnings.extend(f"{case['id']} {mode}: {note}" for note in notes)
            ranks[mode].append(first_rank(hits, case["expected"]))
    return ranks, sorted(set(warnings))


def report(cases: list[dict], ranks: dict, warnings: list[str],
           meta: dict) -> str:
    modes = list(ranks)
    drafts = sum(1 for case in cases if case.get("status") != "approved")
    lines = [f"# Retrieval eval, {meta.get('embed_model')} "
             f"({dt.datetime.now(dt.timezone.utc):%Y-%m-%d %H:%M} UTC)", "",
             f"Index: {meta.get('index')}; {meta.get('chunks')} chunks, "
             f"chunk_words {meta.get('chunk_words')}; {len(cases)} cases"
             + (f" (**{drafts} still drafts**: expected pages not yet "
                "checked by the reviewer)" if drafts else ""), "",
             "| mode | recall@5 | recall@10 | MRR@10 |", "|---|---|---|---|"]
    for mode in modes:
        scores = summarize(ranks[mode])
        lines.append(f"| {mode} | {scores['recall@5']:.2f} | "
                     f"{scores['recall@10']:.2f} | {scores['MRR@10']:.2f} |")
    lines += ["", "First matching rank per case (- = not in top "
              f"{DEPTH}):", "",
              "| case | status | " + " | ".join(modes) + " |",
              "|---|---|" + "---|" * len(modes)]
    for number, case in enumerate(cases):
        cells = [str(ranks[mode][number] or "-") for mode in modes]
        lines.append(f"| {case['id']} | {case.get('status', '?')} | "
                     + " | ".join(cells) + " |")
    if warnings:
        lines += ["", "Warnings:", *[f"- {warning}" for warning in warnings]]
    return "\n".join(lines) + "\n"


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
    cases = json.loads(args.cases.read_text())["cases"]
    index = LiteratureIndex(args.index, readonly=True)
    status = index.status()
    ranks, warnings = run(cases, args.modes.split(","), index,
                          OllamaEmbedder(args.model))
    text = report(cases, ranks, warnings, status)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    name = (f"{str(status.get('embed_model')).replace(':', '-')}_"
            f"{dt.datetime.now(dt.timezone.utc):%Y%m%dT%H%M%SZ}.md")
    (args.out_dir / name).write_text(text)
    print(text)
    print(f"Written to {args.out_dir / name}")


if __name__ == "__main__":
    main()
