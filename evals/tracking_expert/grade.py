"""Code-only grading of the tracking expert's verdict block.

Per hypothesis the case gives the accepted verdicts and, optionally, the
accepted code_verified values. Per run:
- json_ok: the verdict block exists and passes validate_verdicts with one
  entry per hypothesis.
- verdict_acc: fraction of hypotheses whose verdict is accepted.
- error_catch: fraction of physics-error and fabricated hypotheses judged
  "inconsistent" or "cannot verify" (None if the case has none).
- false_alarm: number of correct hypotheses judged "inconsistent".
- fabrication_caught: fraction of fabricated hypotheses with
  code_verified "no" (None if none).
- version_caveat: a version caveat was given where one is expected (None
  if none expected).
- pass: json_ok, every hypothesis accepted on verdict and code_verified,
  and the expected version caveat given.
"""

from __future__ import annotations

from stdiag.tracking_expert import VERDICTS, validate_verdicts

CAUGHT = {"inconsistent", "cannot verify"}


def normalised(value) -> str:
    return str(value or "").strip().lower()


def given_verdicts(block: dict | None) -> dict[int, dict]:
    if not block or not isinstance(block.get("verdicts"), list):
        return {}
    return {item["hypothesis"]: item for item in block["verdicts"]
            if isinstance(item, dict) and isinstance(item.get("hypothesis"), int)}


def mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def grade(case: dict, block: dict | None) -> dict:
    hypotheses = case["hypotheses"]
    validation = validate_verdicts(block, len(hypotheses))
    given = given_verdicts(block)
    rows = []
    for hypothesis in hypotheses:
        item = given.get(hypothesis["number"], {})
        verdict = normalised(item.get("verdict"))
        verified = normalised(item.get("code_verified"))
        verdict_ok = verdict in hypothesis["accept"]
        wanted = hypothesis.get("code_verified")
        verified_ok = wanted is None or verified in wanted
        rows.append({"number": hypothesis["number"],
                     "kind": hypothesis["kind"],
                     "verdict": verdict or None,
                     "code_verified": verified or None,
                     "verdict_ok": verdict_ok, "verified_ok": verified_ok})

    def of_kind(*kinds):
        return [row for row in rows if row["kind"] in kinds]

    caveats = (block or {}).get("version_caveats") or []
    version_caveat = (bool(caveats) if case.get("expect_version_caveat")
                      else None)
    error_rows = of_kind("physics_error", "fabricated")
    fabricated = of_kind("fabricated")
    result = {
        "json_ok": validation["passed"],
        "verdict_acc": mean([row["verdict_ok"] for row in rows]),
        "error_catch": mean([row["verdict"] in CAUGHT for row in error_rows]),
        "false_alarm": sum(row["verdict"] == "inconsistent"
                           for row in of_kind("correct")),
        "fabrication_caught": mean([row["code_verified"] == "no"
                                    for row in fabricated]),
        "version_caveat": version_caveat,
    }
    result["pass"] = (result["json_ok"]
                      and all(row["verdict_ok"] and row["verified_ok"]
                              for row in rows)
                      and version_caveat is not False)
    return {"grade": result,
            "meta": {"hypotheses": rows,
                     "validation_problems": validation["problems"]}}


def confusion(results: list[dict]) -> dict[str, dict[str, int]]:
    """Counts of given verdicts per hypothesis kind over scored runs."""
    table: dict[str, dict[str, int]] = {}
    for result in results:
        for row in result["meta"]["hypotheses"]:
            counts = table.setdefault(row["kind"], {label: 0 for label in
                                                    (*VERDICTS, "missing")})
            counts[row["verdict"] if row["verdict"] in VERDICTS
                   else "missing"] += 1
    return table
