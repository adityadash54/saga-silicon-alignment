# Evaluation layer

Measures the AI agents, one eval per agent. Each eval needs the reviewer's
sign-off on its test cases and on its grading before any paid run. Graders
are code wherever the output allows it; AI judges only where judgment is
genuinely needed.

| Eval | Measures | Inputs | Grading | Status |
|---|---|---|---|---|
| `triage/` | Does the triage group flags correctly without inventing findings? | Synthetic comparisons with known injected changes, including no-change and distractor cases | Code, from `findings_*.json` against the injected truth | Cases and grading approved (`triage/PROPOSED_CASES.md`, harness `4d598628dc6c`); self-test passes; live run 21/21 pass, 7 key cases repeated with identical findings (2026-11-26) |
| `coresoftware_expert/` | Does the expert find the commit behind a known effect? | Past coresoftware commits with a known residual effect, described as findings | Code: commit hit in top 1 / top 3; every cited hash checked with git | Planned |
| `tracking_expert/` | Are its verdicts right, and does it catch wrong physics and citations that don't check out? | Coresoftware-expert answers whose hypotheses have known verdicts (`cases.json`, local only; `cases.example.json` shows the format) | Code, from the JSON verdict block: verdict per hypothesis, code-verified flag, version caveat | Harness built, self-test passes on the examples; cases being drafted |
| `retrieval/` | Does the literature search find the right paper and page, also in the list the tracking expert's request gets? | Your questions, or findings searched as the agent does, with expected file and page range, plus no-answer questions (`retrieval/cases.json`, git-ignored because it names your papers; `cases.example.json` shows the format) | Code: recall@5, recall@10, MRR, hit@6 (2/paper) for bm25, dense, hybrid; highest cosine of no-answer against answerable cases | Run on 32 approved cases (2026-11-27): questions fine (hybrid hit@6 18/18), agent path weaker in hybrid than in BM25 alone (hit@6 4/8 against 7/8); see Results below |
| `permissions/` | Do the read-only and no-network limits hold? | Prompts that try to edit files, run forbidden commands or reach the network | End state: checkout unchanged, nothing fetched | Planned |
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

## Tracking expert eval

```bash
export STDIAG_CORESOFTWARE=/path/to/coresoftware       # full history
python evals/tracking_expert/run_eval.py --self-test   # grader and commit check, no AI
python evals/tracking_expert/run_eval.py --approve-harness   # reviewer only
python evals/tracking_expert/run_eval.py --cases <id>,<id> --reps 1   # pilot
python evals/tracking_expert/run_eval.py --reps 2      # full run
python evals/tracking_expert/run_eval.py --report      # aggregate results
```

Each case is a finding with a coresoftware-expert answer whose hypotheses
have verdicts agreed by the reviewer. The tracking expert runs exactly as
in `review.py check` (one-shot, read-only, no network access, in the
coresoftware checkout, with the ACTS / Millepede-II snapshot and the
literature index) and ends its answer with a JSON verdict block, which is
graded by code. Needs a complete `tracking_docs/` snapshot
(`python -m stdiag.fetch_tracking_docs --check`). Every result row records
the ACTS commit and the Millepede-II page checksum the agent read.

The cases are the evaluation set and are not published: `cases.json`
and `PROPOSED_CASES.md` are git-ignored, and this README reports only
aggregate results. Copy `cases.example.json` to `cases.json` and replace
the made-up examples; only cases with `"status": "approved"` are run. The
harness checksum covers the runner, grader, case loader, the tracking
expert, the docs map and the local cases file, so any change to the
cases needs a new approval.

Hypothesis kinds: correct mechanism, physics error, citation that does
not check out, ACTS version problem, claim that cannot be checked from
the available sources. Scores per run:

- **pass** (headline): the verdict block is valid, every hypothesis gets
  an accepted verdict (and code-verified value, where graded), and an
  expected version caveat is given;
- **verdict accuracy**: fraction of hypotheses with an accepted verdict;
- **error catch**: physics errors and unverifiable citations judged
  inconsistent or cannot verify;
- **false alarms**: correct hypotheses judged inconsistent;
- **citation check**: unverifiable citations marked `code_verified: no`;
- tokens and latency per run, and verdict counts per hypothesis kind.

Variants: `--no-literature` (without the literature index) and
`--effort medium|high`. A rerun skips (case, repetition) pairs already in
`results.jsonl`. With about 14 cases × 2 repetitions only large
differences between variants are meaningful.

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

### Results (2026-11-27, `granite-embedding:30m`)

32 cases, all approved by the reviewer: 18 questions, 8 agent-path findings,
6 no-answer questions. Scores over the answerable cases:

| Cases | Mode | recall@5 | recall@10 | MRR@10 | hit@6 (2/paper) |
|---|---|---|---|---|---|
| Questions (18) | bm25 | 17/18 | 17/18 | 0.92 | 17/18 |
| Questions (18) | dense | 14/18 | 15/18 | 0.69 | 15/18 |
| Questions (18) | hybrid | 18/18 | 18/18 | 0.91 | 18/18 |
| Agent path (8) | bm25 | 5/8 | 8/8 | 0.58 | 7/8 |
| Agent path (8) | dense | 2/8 | 3/8 | 0.08 | 3/8 |
| Agent path (8) | hybrid | 4/8 | 5/8 | 0.39 | 4/8 |
| All answerable (26) | bm25 | 0.85 | 0.96 | 0.81 | 0.92 |
| All answerable (26) | dense | 0.62 | 0.69 | 0.50 | 0.69 |
| All answerable (26) | hybrid | 0.85 | 0.88 | 0.75 | 0.85 |

- **Questions**: hybrid puts every expected page in the top 5 and in the
  agent's capped list (18/18; 95% interval 81–100%). Dense alone is
  weakest. The questions were drafted from the extracted text, which
  favours BM25.
- **Agent path** (what the tracking expert's request gets): still the weak
  point. BM25 alone puts the expected page in the request list for 7 of 8
  findings, hybrid for 4 of 8. The dense ranking finds 3 of 8, and fusion
  pushes BM25's hits down (rank 5 and 6 to out of the top 10, rank 1 to 8).
  BM25 was better on 3 findings and hybrid on none (exact McNemar p = 0.25
  at n = 8), the same direction as in the draft run.
- **Per-paper cap**: no expected page lost to the 2-per-paper cap any more.
  The cap losses of the draft run came from page ranges that were too
  narrow.
- **No answer**: highest cosine 0.67–0.79, against 0.77–0.93 (median 0.82)
  for the answerable cases. A cut-off at 0.785 would also drop 4 of the 26
  answerable cases, so there is no reliable "nothing relevant" threshold.
- **Draft run, before review** (same day): hybrid hit@6 16/18 questions and
  2/8 findings. Correcting the page ranges and findings raised most scores;
  the gap between BM25 and hybrid on the agent path remained.
- **Open**: the agent-path gap lies in the dense half of the fusion, either
  the long query (about 120 words mixing the finding, numbers and generic
  terms) or the small embedding model. Candidates to test, cheapest first:
  more BM25 weight in the fusion (or BM25 alone) for the request, a shorter
  query, then Granite R2.
- **Cost**: no Claude tokens; embeddings run locally.
