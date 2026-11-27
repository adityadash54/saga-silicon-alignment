"""Triage output handling with canned AI responses (no AI calls)."""

from __future__ import annotations

import json
import sys

import pytest

from stdiag import triage_agent

STAVE = [{"detector": "MVTX", "layer": 1, "stave": 4, "chip": chip,
          "coord": "rphi"} for chip in range(9)]
REST = [{"detector": "MVTX", "layer": 0, "stave": 2, "chip": 5, "coord": "rphi"},
        {"detector": "MVTX", "layer": 0, "stave": 2, "chip": 5, "coord": "z"},
        {"detector": "MVTX", "layer": 2, "stave": 10, "chip": 3, "coord": "z"},
        {"detector": "INTT", "layer": 4, "stave": 6, "chip": 1, "coord": "rphi"}]
# Entries flags: the INTT ladder missing from Iter1 (-100%) and one MVTX chip
# whose cluster count moved by Poisson noise beyond z = 3 (seed 20261001).
COVERAGE = ([{"detector": "INTT", "layer": 5, "stave": 9, "chip": chip,
              "coord": coord} for chip in range(4) for coord in ("rphi", "z")]
            + [{"detector": "MVTX", "layer": 2, "stave": 13, "chip": 6,
                "coord": coord} for coord in ("rphi", "z")])
GOOD = [{"id": "F1", "group": "A", "title": "stave shift", "priority": 1,
         "entries": STAVE},
        {"id": "F2", "group": "B", "title": "the rest", "priority": 2,
         "entries": REST},
        {"id": "F3", "group": "E", "title": "entries changes", "priority": 3,
         "entries": COVERAGE}]
GHOST = {"id": "F4", "group": "B", "title": "ghost", "priority": 4,
         "entries": [{"detector": "MVTX", "layer": 9, "stave": 0, "chip": 0,
                      "coord": "z"}]}


def response(findings):
    return ("### Overview\nText.\n\n```json\n"
            + json.dumps({"findings": findings}) + "\n```\n")


@pytest.mark.parametrize("text, passed, unassigned, unknown, invalid", [
    (response(GOOD), True, 0, 0, 0),
    (response(GOOD[:1]), False, 4 + 10, 0, 0),
    (response(GOOD + [GHOST]), False, 0, 1, 0),
    (response([GOOD[0], {**GOOD[1], "group": "not_comparable"}, GOOD[2]]),
     False, 0, 0, 1),
    ("### Overview\nNo block.", False, 0, 0, 0),
    (response([{**GOOD[0], "entries": [{**entry, "chip": float(entry["chip"])}
                                       for entry in STAVE]}, *GOOD[1:]]),
     True, 0, 0, 0),
    (response([GOOD[0], {**GOOD[1], "entries": REST + STAVE[:1]}, GOOD[2]]),
     False, 0, 0, 0),
])
def test_findings_validation(tmp_path, synthetic_outputs, monkeypatch, text,
                             passed, unassigned, unknown, invalid):
    for name in ("flags_Iter0_vs_Iter1.csv", "summary_Iter0_vs_Iter1.json"):
        (tmp_path / name).write_bytes((synthetic_outputs / name).read_bytes())
    monkeypatch.setattr(triage_agent, "run_triage_claude_code",
                        lambda summary, model, effort: (text, "canned", {}))
    monkeypatch.setattr(sys, "argv", [
        "triage_agent", str(tmp_path / "flags_Iter0_vs_Iter1.csv"),
        str(tmp_path / "summary_Iter0_vs_Iter1.json"),
        "--backend", "claude-code"])
    triage_agent.main()

    findings = json.loads((tmp_path / "findings_Iter0_vs_Iter1.json").read_text())
    validation = findings["validation"]
    assert validation["passed"] is passed
    assert len(validation.get("unassigned_flagged", [])) == unassigned
    assert len(validation.get("unknown_entries", [])) == unknown
    assert len(validation.get("invalid_groups", [])) == invalid
    report = (tmp_path / "triage_report_Iter0_vs_Iter1.md").read_text()
    assert "```json" not in report
    assert ("validation passed" in report) is passed


@pytest.mark.parametrize("stdout, expected_text, expected_models", [
    (json.dumps({"result": "Report text", "is_error": False,
                 "modelUsage": {"claude-opus-5-5": {}},
                 "usage": {"input_tokens": 10, "output_tokens": 5},
                 "total_cost_usd": 0.01, "duration_ms": 1200,
                 "num_turns": 1, "subtype": "success"}),
     "Report text", ["claude-opus-5-5"]),
    ("Plain text answer", "Plain text answer", []),
])
def test_claude_code_output_parsing(monkeypatch, synthetic_outputs, stdout,
                                    expected_text, expected_models):
    class Result:
        returncode, stderr = 0, ""

    Result.stdout = stdout
    triage_agent.ComparisonData.flags = __import__("pandas").read_csv(
        synthetic_outputs / "flags_Iter0_vs_Iter1.csv")
    summary = json.loads(
        (synthetic_outputs / "summary_Iter0_vs_Iter1.json").read_text())
    monkeypatch.setattr(triage_agent.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(triage_agent.subprocess, "run",
                        lambda *args, **kwargs: Result)
    text, label, usage = triage_agent.run_triage_claude_code(
        summary, "claude-opus-5-5", None)
    assert text == expected_text
    assert usage["models"] == expected_models
    assert label == "claude-opus-5-5 via Claude Code CLI"


def test_notable_unflagged_with_nothing_notable(monkeypatch,
                                                synthetic_outputs):
    """No notable rows (as in a no-change comparison) gives an empty list,
    not an error (pandas >= 3 returns a DataFrame from an empty apply)."""
    flags = __import__("pandas").read_csv(
        synthetic_outputs / "flags_Iter0_vs_Iter1.csv",
        dtype={"detector": str, "coord": str, "reason": str})
    quiet = flags.loc[
        flags["comparable"] & ~flags["flagged"]
        & (flags["mu_z"] <= triage_agent.NOTABLE_MU_Z)
        & (flags["sigma_rel_significance"]
           <= triage_agent.NOTABLE_WIDTH_SIGNIFICANCE)
        & (flags[["entries_1", "entries_2"]].min(axis=1)
           >= triage_agent.NOTABLE_LOW_ENTRIES)]
    monkeypatch.setattr(triage_agent.ComparisonData, "flags", quiet)
    result = json.loads(triage_agent.list_notable_unflagged())
    assert result["total_matching"] == 0 and result["rows"] == []


@pytest.mark.parametrize("effort", ["xhigh", None])
def test_claude_code_effort_flag(monkeypatch, synthetic_outputs, effort):
    class Result:
        returncode, stderr = 0, ""
        stdout = json.dumps({"result": "Report text", "is_error": False})

    commands = []

    def run(command, **kwargs):
        commands.append(command)
        return Result

    triage_agent.ComparisonData.flags = __import__("pandas").read_csv(
        synthetic_outputs / "flags_Iter0_vs_Iter1.csv")
    summary = json.loads(
        (synthetic_outputs / "summary_Iter0_vs_Iter1.json").read_text())
    monkeypatch.setattr(triage_agent.shutil, "which", lambda name: "claude")
    monkeypatch.setattr(triage_agent.subprocess, "run", run)
    _, _, usage = triage_agent.run_triage_claude_code(
        summary, "claude-opus-5-5", effort)
    command = commands[0]
    if effort is None:
        assert "--effort" not in command
    else:
        assert command[command.index("--effort") + 1] == effort
    assert usage["effort"] == effort


def test_stave_overview_keeps_stave_row_apart(monkeypatch,
                                              flags_with_stave_rows):
    flags, quiet = flags_with_stave_rows
    keys = ["detector", "layer", "stave", "coord"]

    def overview(table, **kwargs):
        monkeypatch.setattr(triage_agent.ComparisonData, "flags", table)
        rows = json.loads(triage_agent.get_stave_overview(**kwargs))
        return {tuple(row[key] for key in keys): row for row in rows}

    with_stave = overview(flags, detector="MVTX", only_with_flags=False)
    without_stave = overview(flags.loc[flags["chip"] >= 0], detector="MVTX",
                             only_with_flags=False)
    stave = with_stave[("MVTX", 1, 4, "rphi")]
    reference = without_stave[("MVTX", 1, 4, "rphi")]
    for column in ("chips", "comparable", "flagged", "median_delta_mu_mm",
                   "std_delta_mu_mm", "fraction_positive_shift"):
        assert stave[column] == reference[column], column
    assert stave["stave_flagged"] is True
    assert stave["stave_delta_mu_mm"] == pytest.approx(0.012)
    assert reference["stave_flagged"] is False
    assert reference["stave_delta_mu_mm"] is None

    # A stave whose only flag is its stave row is still listed.
    flagged_only = overview(flags, detector="MVTX")
    quiet_key = tuple(value.item() if hasattr(value, "item") else value
                      for value in quiet)
    assert flagged_only[quiet_key]["flagged"] == 0
    assert flagged_only[quiet_key]["stave_flagged"] is True


def test_dry_run_sends_no_injection_truth(synthetic_outputs, capsys,
                                          monkeypatch):
    monkeypatch.setattr(sys, "argv", [
        "triage_agent", str(synthetic_outputs / "flags_Iter0_vs_Iter1.csv"),
        str(synthetic_outputs / "summary_Iter0_vs_Iter1.json"),
        "--backend", "claude-code", "--dry-run"])
    triage_agent.main()
    prompt = capsys.readouterr().out
    assert "list_notable_unflagged" in prompt
    for leaked in ("inject", "stave_shift_rphi", "synthetic"):
        assert leaked not in prompt.lower()


@pytest.mark.parametrize("prefix, suffix", [
    ("", ""),
    ('Example:\n```json\n{"example": 1}\n```\n', ""),
    ("", "\n\n"),
])
def test_split_findings_takes_the_final_block(prefix, suffix):
    report, findings = triage_agent.split_findings(
        prefix + response(GOOD) + suffix)
    assert findings == {"findings": GOOD}
    assert report.startswith(prefix.rstrip())
    assert report.endswith("Text.")


def test_split_findings_keeps_fences_inside_strings():
    tricky = [{**GOOD[0], "evidence": "see ```json``` above"}, GOOD[1]]
    assert triage_agent.split_findings(response(tricky))[1] == {
        "findings": tricky}
