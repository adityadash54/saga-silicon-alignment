# Evaluation layer

Measures the AI agents, one eval per agent. Each eval needs the reviewer's
sign-off on its test cases and on its grading before any paid run. Graders
are code wherever the output allows it; AI judges only where judgment is
genuinely needed.

| Eval | Measures | Inputs | Grading | Status |
|---|---|---|---|---|
| `triage/` | Does the triage group flags correctly without inventing findings? | Synthetic comparisons with known injected changes, including no-change and distractor cases | Code, from `findings_*.json` against the injected truth | Cases and grading approved (`triage/PROPOSED_CASES.md`, harness `4d598628dc6c`); self-test passes; live run 21/21 pass, 7 key cases repeated with identical findings (2026-11-26) |
| `coresoftware_expert/` | Does the expert find the commit behind a known effect? | Past coresoftware commits with a known residual effect, described as findings | Code: commit hit in top 1 / top 3; every cited hash checked with git | Planned |
| `tracking_expert/` | Are its verdicts right? | Hypotheses with known verdicts (correct, sign-flipped residual, ACTS API absent in v45) | Code: verdict label | Planned |
| `retrieval/` | Does the literature search find the right paper and page, also in the list the tracking expert's request gets? | Your questions, or findings searched as the agent does, with expected file and page range, plus no-answer questions (`retrieval/cases.json`, git-ignored because it names your papers; `cases.example.json` shows the format) | Code: recall@5, recall@10, MRR, hit@6 (2/paper) for bm25, dense, hybrid; highest cosine of no-answer against answerable cases | Runner built; not yet run |
| `permissions/` | Do the read-only and web-domain limits hold? | Prompts that try to edit files, run forbidden commands or fetch disallowed sites | End state: checkout unchanged, nothing forbidden fetched | Planned |
| `dialogue/` | How do the two experts converge? | Findings with known causes | Convergence rate, turns used, questions to the reviewer, status-format compliance | Planned |

Run outputs (transcripts, reports) are git-ignored; cases, runners and
graders are committed.

## Triage eval

```bash
python evals/triage/run_eval.py --self-test        # grader check, no AI
python evals/triage/run_eval.py --approve-harness  # reviewer only; records a checksum
python evals/triage/run_eval.py --cases null-1,stave-mvtx-rphi,mixed-1 --reps 1   # pilot
python evals/triage/run_eval.py --reps 2           # full run: 21 cases x 2
```

The runner refuses to start until the reviewer has approved the current
harness files (runner, cases, grader, triage, comparison, generator); any
change to them requires a new approval. Results are in
`evals/triage/runs/<variant>/` (`baseline` by default; `results.jsonl`,
`traces/`, `errors.jsonl`). A rerun skips the (case, repetition) pairs
already in `results.jsonl`, so repetitions can be added later for chosen
cases. With 21 cases × 2 repetitions, the pass rate has a noise floor of
roughly ±15 percentage points, so smaller differences between variants are
not meaningful.

### Results (2026-11-26, harness `4d598628dc6c`, `claude-opus-5-5`)

Run as the `pilot` variant, once per case, then a second repetition of 7
key cases:

```bash
python evals/triage/run_eval.py --reps 1 --variant pilot
python evals/triage/run_eval.py --reps 2 --variant pilot \
  --cases mixed-1,mixed-2,two-adjacent-chips,half-layer-z,opposite-staves,null-2,sub-threshold-stave
```

- **21 of 21 cases pass**: recall and group accuracy 1.0 wherever changes
  were injected, no false or merged findings. With 21 runs, the pass rate
  is at least 84% at 95% confidence (Clopper–Pearson).
- **Consistency**: the 7 repeated cases passed again, and every second
  answer had exactly the same findings (same groups, same rows).
- **Behaviour on the hard cases**: two adjacent chips were one single-chip
  finding (B); a half layer was one stave-wide finding (A) of 90 rows; two
  opposite stave shifts stayed separate; entries-noise rows, the
  sub-threshold shift and low-statistics rows went to group E as notes.
- **Size**: about 47k tokens per case on average (21k for a no-change case,
  145k for the whole-layer shift with 145 flagged rows), 53 s on average.
- **Not covered**: real productions; a per-case consistency rate (that
  needs more repetitions).

## Retrieval eval

```bash
python evals/retrieval/run_eval.py                 # bm25, dense and hybrid
python evals/retrieval/run_eval.py --modes bm25    # without Ollama
```

Needs a built literature index (`stdiag-literature index`) and, for the dense
and hybrid modes, Ollama with the embedding model. A case passes at k if a
top-k passage comes from an expected file and overlaps its page range.
The cases name your papers and pages, so they live in `cases.json`, which is
git-ignored: copy `cases.example.json` to `cases.json` and replace the
made-up examples. Check each expected page and set `"status": "approved"`.

Two scores per mode, both over answerable cases only:

- recall@5, recall@10 and MRR@10 score the raw ranking, with no limit on
  passages per paper;
- hit@6 (2/paper) scores the list the tracking expert's request actually
  gets: top 6, at most 2 passages per paper.

Three kinds of case:

| Kind | Fields | Query |
|---|---|---|
| Question | `question`, `expected` | The question as written |
| Agent path | `finding` (title, evidence, entries), optional `answer_summary`, `expected` | Built by `tracking_expert.literature_query()`, as for the request (sPHENIX detector names replaced by generic terms) |
| No answer | `question` or `finding`, `"expected": []` | Not scored; its highest cosine similarity is compared with the answerable cases' to show whether a "no relevant literature" cut-off could work (needs dense or hybrid mode) |


Phrase questions the way you would ask them: questions written while reading
the page tend to share its words and favour BM25. With n cases one miss moves
recall by 1/n, so only large differences between modes or models are
meaningful. Reports go to `evals/retrieval/runs/` (git-ignored); they quote
paper names and stay local.
