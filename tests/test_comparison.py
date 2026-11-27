"""The steps without AI: fit tables -> flags, checked against injected truth."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stdiag.compare_runs import (DEFAULT_MIN_SHIFT_MM, build_summary,
                                 compare_tables, entries_rule, load_fit_table,
                                 parse_min_shift_overrides)
from tests.conftest import run_module


@pytest.mark.parametrize("comparison", ["Iter1", "Iter2"])
def test_flags_match_injected_truth(synthetic_outputs, comparison):
    stem = f"Iter0_vs_{comparison}"
    output = run_module("synthetic.check_against_truth",
                        str(synthetic_outputs / f"flags_{stem}.csv"),
                        str(synthetic_outputs / f"summary_{stem}.json"))
    assert "PASS" in output


def test_expected_counts(synthetic_outputs):
    counts = {comparison: json.loads((synthetic_outputs /
              f"summary_Iter0_vs_{comparison}.json").read_text())["counts"]
              for comparison in ("Iter1", "Iter2")}
    # Iter1: 13 mean/width flags + 8 rows of the ladder missing from Iter1
    # (entries -100%) + 2 entries-noise rows; Iter2: 93 + 3 entries-noise rows
    # (Poisson, z > 3; fixed by the seed).
    assert (counts["Iter1"]["flagged"], counts["Iter1"]["not_comparable"]) == (23, 16)
    assert (counts["Iter2"]["flagged"], counts["Iter2"]["not_comparable"]) == (96, 8)
    assert counts["Iter1"]["flag_entries"] == 10
    # 3 entries-only rows + 1 whose chip also has an injected z shift.
    assert counts["Iter2"]["flag_entries"] == 4


def test_sub_threshold_shift_is_significant_but_not_flagged(synthetic_outputs):
    flags = pd.read_csv(synthetic_outputs / "flags_Iter0_vs_Iter1.csv")
    row = flags.query("detector == 'MVTX' and layer == 2 and stave == 7 "
                      "and chip == 0 and coord == 'rphi'").iloc[0]
    assert row["mu_z"] > 5 and not row["flagged"]


def test_entries_rule_worked_examples():
    n1 = np.array([3000, 30000, 300, 3000, 0, 0])
    n2 = np.array([3160, 31600, 240, 0, 50, 0])
    fires, rel, z = entries_rule(n1, n2, 1.0, 1.0, 0.05, 3.0)
    np.testing.assert_allclose(z[:3], [2.04, 6.45, 2.58], atol=0.01)
    assert rel[3] == -1.0 and np.isinf(rel[4]) and np.isnan(rel[5])
    # +5.3% at 3000 is noise (z = 2.0); at 30000 it is real; a vanished or a
    # new chip fires; a chip empty in both inputs never does.
    assert fires.tolist() == [False, True, False, True, True, False]


def test_missing_ladder_is_an_entries_flag(synthetic_outputs):
    flags = pd.read_csv(synthetic_outputs / "flags_Iter0_vs_Iter1.csv",
                        dtype={"detector": str, "coord": str, "reason": str})
    ladder = flags.query("detector == 'INTT' and layer == 5 and stave == 9")
    assert len(ladder) == 8
    assert ladder["flag_entries"].all() and not ladder["comparable"].any()
    assert (ladder["comparison_status"] == "flagged").all()
    assert ladder["reason"].str.startswith("entries -100%").all()


def comparison_with_fewer_events(synthetic_outputs, factor):
    reference = load_fit_table(synthetic_outputs / "residual_fits_Iter0.csv")
    comparison = reference.assign(
        label="Fewer", entries=(reference["entries"] * factor).round())
    return reference, comparison


@pytest.mark.parametrize("normalization, expect_flags",
                         [("raw", True), ("detector", False),
                          ("layer", False)])
def test_entries_normalization_and_event_count(synthetic_outputs,
                                               normalization, expect_flags):
    reference, comparison = comparison_with_fewer_events(synthetic_outputs,
                                                         0.8)
    flags = compare_tables(reference, comparison, 0.20, 1.96,
                           DEFAULT_MIN_SHIFT_MM,
                           entries_normalization=normalization)
    # 20% fewer events everywhere: raw counts flag every well-populated row
    # (z > 3 needs about 500 entries at -20%), shares flag none.
    populated = flags["entries_1"] >= 1000
    assert flags.loc[populated, "flag_entries"].all() == expect_flags
    assert flags["flag_entries"].any() == expect_flags


def test_entries_test_skipped_for_absent_detector(synthetic_outputs, capsys):
    reference = load_fit_table(synthetic_outputs / "residual_fits_Iter0.csv")
    comparison = reference.loc[reference["detector"] != "INTT"].assign(
        label="NoINTT")
    flags = compare_tables(reference, comparison, 0.20, 1.96,
                           DEFAULT_MIN_SHIFT_MM)
    intt = flags["detector"] == "INTT"
    assert not flags.loc[intt, "flag_entries"].any()
    assert flags.loc[intt, "entries_rel"].isna().all()
    assert (flags.loc[intt, "comparison_status"] == "not_comparable").all()


def test_report_carries_label_configuration_table(synthetic_outputs):
    report = (synthetic_outputs / "report_Iter0_vs_Iter1.md").read_text()
    assert "Detector configuration (fill in)" in report
    assert "| reference | Iter0 | ntuple_list1.txt" in report


def test_summary_keeps_stave_rows_apart(flags_with_stave_rows):
    flags, _ = flags_with_stave_rows
    without_stave_rows = flags.loc[(flags["chip"] >= 0)
                                   | (flags["detector"] == "VERTEX")]
    summary = build_summary(flags, {}, {})
    reference = build_summary(without_stave_rows, {}, {})

    assert summary["counts_by_level"]["stave"]["rows"] == 2
    assert summary["counts_by_level"]["stave"]["flagged"] == 2
    assert (summary["counts_by_level"]["chip"]
            == reference["counts_by_level"]["chip"])

    def chip_level(rows):
        return [row for row in rows if row["level"] == "chip"]

    assert (chip_level(summary["by_detector_layer_coord"])
            == chip_level(reference["by_detector_layer_coord"]))


def test_min_shift_override():
    shifts = parse_min_shift_overrides(["INTT:z=0.5"])
    assert shifts[("INTT", "z")] == 0.5 and shifts[("MVTX", "rphi")] == 0.005
    with pytest.raises(SystemExit):
        parse_min_shift_overrides(["TPC:z=1"])


def copy_tables(synthetic_outputs, folder, names):
    """Copy the Iter0/1/2 tables into folder as residual_fits_<name>.csv."""
    folder.mkdir(parents=True, exist_ok=True)
    for label, name in names.items():
        (folder / f"residual_fits_{name}.csv").write_bytes(
            (synthetic_outputs / f"residual_fits_{label}.csv").read_bytes())
    return [str(folder / f"residual_fits_{name}.csv") for name in names.values()]


def test_default_output_is_a_fresh_run_folder(synthetic_outputs, tmp_path):
    runs = tmp_path / "runs"
    env = {"STDIAG_RUNS_DIR": str(runs)}
    tables = copy_tables(synthetic_outputs, tmp_path / "inputs",
                         {"Iter0": "Iter0", "Iter1": "Iter1", "Iter2": "Iter2"})
    run_module("stdiag.compare_runs", *tables, env=env)
    assert sorted(path.name for path in runs.iterdir()) == [
        "Iter0_vs_Iter1", "Iter0_vs_Iter2"]
    folder = runs / "Iter0_vs_Iter1"
    assert sorted(path.name for path in folder.iterdir()) == [
        "flags_Iter0_vs_Iter1.csv", "report_Iter0_vs_Iter1.md",
        "summary_Iter0_vs_Iter1.json"]

    (folder / "findings_Iter0_vs_Iter1.json").write_text("{}")
    output = run_module("stdiag.compare_runs", *tables[:2], env=env)
    assert "Replaced" in output
    assert not (folder / "findings_Iter0_vs_Iter1.json").exists()
    assert (folder / "flags_Iter0_vs_Iter1.csv").exists()


def test_run_folder_is_named_from_the_files(synthetic_outputs, tmp_path):
    runs = tmp_path / "runs"
    tables = copy_tables(synthetic_outputs, tmp_path / "inputs",
                         {"Iter0": "ana.499", "Iter1": "ana.500"})
    output = run_module("stdiag.compare_runs", *tables,
                        env={"STDIAG_RUNS_DIR": str(runs)})
    assert "Note: residual_fits_ana.499.csv holds label 'Iter0'" in output
    assert (runs / "ana.499_vs_ana.500" / "flags_Iter0_vs_Iter1.csv").exists()


@pytest.mark.parametrize("problem", ["bad_name", "input_in_run_folder"])
def test_refused_before_anything_is_deleted(synthetic_outputs, tmp_path,
                                            problem):
    runs = tmp_path / "runs"
    keep = runs / "Iter0_vs_Iter1" / "keep.txt"
    keep.parent.mkdir(parents=True)
    keep.write_text("earlier run")
    tables = copy_tables(synthetic_outputs, tmp_path / "inputs",
                         {"Iter0": "Iter0", "Iter1": "Iter1"})
    if problem == "bad_name":
        tables[1] = str(Path(tables[1]).rename(tmp_path / "inputs" / "iter1.csv"))
        expected = "is not named residual_fits_<name>.csv"
    else:
        tables[1] = str(Path(tables[1]).rename(keep.parent / "residual_fits_Iter1.csv"))
        expected = "which is replaced on every run"
    result = run_module("stdiag.compare_runs", *tables,
                        env={"STDIAG_RUNS_DIR": str(runs)}, check=False)
    assert result.returncode != 0 and expected in result.stderr
    assert keep.read_text() == "earlier run"
