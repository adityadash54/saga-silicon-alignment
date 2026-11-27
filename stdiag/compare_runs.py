#!/usr/bin/env python3
"""Layer 2: compare per-chip residual fits between inputs and flag changed chips.

Runs entirely locally: it reads the CSV tables written by
macros/plot_fits_overlaid_stavechip.C and writes CSV, JSON and Markdown files. It imports no
network libraries and sends nothing to any external service.

Criteria (input 1 = reference, input 2 = comparison), evaluated separately for
every (detector, layer, stave, chip, coord):
  width : |(sigma2 - sigma1) / sigma1| > --sigma-rel-threshold      (default 0.10)
          AND its significance > --width-significance-threshold      (default 3)
  mean  : |mu2 - mu1| / sqrt(dmu1^2 + dmu2^2) > --z-threshold        (default 1.96)
          AND |mu2 - mu1| > minimum shift for (detector, coord)
  entries: |a2/a1 - 1| > --entries-rel-threshold                     (default 0.05)
          AND |a2 - a1| / sqrt(N1/t1^2 + N2/t2^2) > --entries-z-threshold (default 3)
          with a = N/t: t = 1 (--entries-normalization raw, default), the
          detector total or the layer total of chip rows of the same
          coordinate in that input (detector, layer)
  A chip is flagged if the width, the mean OR the entries criterion fires.
  The width and mean tests need fit_status == "ok" in both inputs; anything
  else is reported as not comparable. The entries test also applies to not
  comparable rows, and a row missing from one input counts as 0 entries; it is
  skipped for a detector with no entries at all in either input (e.g. VERTEX
  rows without track_ntp).
MVTX/INTT rows with chip = -1 are fits to the sum of all chips of the
stave/ladder ("stave rows"); they get the same criteria and minimum shifts as
the chips and are counted separately (level "stave") in the summary.

Usage:
  python -m stdiag.compare_runs inputs/residual_fits_Iter0.csv \
      inputs/residual_fits_Iter1.csv [inputs/residual_fits_Iter2.csv ...]
  python -m stdiag.compare_runs TABLE TABLE [...] --out-dir DIR
The first table is the reference; every other table is compared with it.
Without --out-dir, each comparison of residual_fits_<a>.csv with
residual_fits_<b>.csv goes to its own folder runs/<a>_vs_<b> (see
stdiag.paths.RUNS_DIR). That folder is deleted and written afresh on every
run, so earlier triage and review files in it are removed with it. With
--out-dir, all outputs go to DIR and nothing is deleted.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from stdiag.paths import RUNS_DIR

# chip is the chip/sensor id for MVTX/INTT (-1 = whole stave/ladder) and the
# track-phi bin for VERTEX rows (-1 = all tracks, 0-13 = phi bins over
# [-pi, pi)).
KEY_COLUMNS = ["detector", "layer", "stave", "chip", "coord"]
STAVE_CHIP = -1
FIT_COLUMNS = [
    "entries", "entries_in_fit_window", "populated_bins", "fit_status",
    "mu_mm", "mu_err_mm", "sigma_mm", "sigma_err_mm", "chi2", "ndf",
    "fit_low_mm", "fit_high_mm",
]
REQUIRED_COLUMNS = ["label", "listfile", *KEY_COLUMNS, *FIT_COLUMNS]
TEXT_COLUMNS = {"label": str, "listfile": str, "detector": str, "coord": str,
                "fit_status": str}

DEFAULT_MIN_SHIFT_MM = {
    ("MVTX", "rphi"): 0.005,
    ("MVTX", "z"): 0.005,
    ("INTT", "rphi"): 0.020,
    ("INTT", "z"): 1.0,
    # Track-to-vertex residuals (-DCA) from track_ntp; their chip id is the
    # track-phi bin (chip -1 = all tracks, chip 0-13 = phi bins).
    ("VERTEX", "dca_xy"): 0.010,
    ("VERTEX", "dca_z"): 0.020,
}


def stave_row_mask(frame: pd.DataFrame) -> pd.Series:
    """True for MVTX/INTT whole-stave rows (chip = -1)."""
    return (frame["detector"] != "VERTEX") & (frame["chip"] == STAVE_CHIP)


def row_level(frame: pd.DataFrame) -> pd.Series:
    """Return the level of every row: chip, stave or vertex."""
    return pd.Series(
        np.select([frame["detector"] == "VERTEX", stave_row_mask(frame)],
                  ["vertex", "stave"], default="chip"),
        index=frame.index)


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sanitize_for_filename(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", text) or "input"


TABLE_NAME = re.compile(r"residual_fits_(.+)\.csv")


def table_name(path: Path) -> str:
    """Return <a> of residual_fits_<a>.csv (the name used for run folders)."""
    match = TABLE_NAME.fullmatch(path.name)
    if not match:
        sys.exit(f"ERROR: {path.name} is not named residual_fits_<name>.csv; "
                 "rename it or give --out-dir")
    return match.group(1)


def run_dir_for(reference: Path, comparison: Path) -> Path:
    """runs/<a>_vs_<b> for residual_fits_<a>.csv vs residual_fits_<b>.csv."""
    name = (f"{sanitize_for_filename(table_name(reference))}_vs_"
            f"{sanitize_for_filename(table_name(comparison))}")
    return RUNS_DIR / name


def fresh_run_dir(run_dir: Path, inputs: list[Path]) -> None:
    """Delete and recreate one comparison folder directly under RUNS_DIR."""
    if run_dir.resolve().parent != RUNS_DIR:
        sys.exit(f"ERROR: refusing to replace {run_dir}: not directly in "
                 f"{RUNS_DIR}")
    for path in inputs:
        if run_dir.resolve() in path.resolve().parents:
            sys.exit(f"ERROR: input {path} is inside {run_dir}, which is "
                     "replaced on every run; keep inputs in inputs/")
    if run_dir.exists():
        shutil.rmtree(run_dir)
        print(f"Replaced {run_dir} (its earlier contents, including any "
              "triage and review files, were deleted)")
    run_dir.mkdir(parents=True)


def load_fit_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, dtype=TEXT_COLUMNS)
    missing = [column for column in REQUIRED_COLUMNS
               if column not in table.columns]
    if missing:
        sys.exit(f"ERROR: {path} is missing columns: {', '.join(missing)}")

    labels = table["label"].dropna().unique()
    if len(labels) != 1:
        sys.exit(f"ERROR: {path} must contain exactly one label, "
                 f"found {list(labels)}")
    if table.duplicated(KEY_COLUMNS).any():
        sys.exit(f"ERROR: {path} has duplicate (detector, layer, stave, chip, "
                 "coord) rows")
    return table


def parse_min_shift_overrides(overrides: list[str]) -> dict:
    minimum_shift = dict(DEFAULT_MIN_SHIFT_MM)
    for override in overrides:
        match = re.fullmatch(r"(\w+):(\w+)=([0-9.eE+-]+)", override)
        if not match or (match.group(1), match.group(2)) not in minimum_shift:
            valid = ", ".join(f"{detector}:{coord}"
                              for detector, coord in DEFAULT_MIN_SHIFT_MM)
            sys.exit(f"ERROR: --min-shift expects DETECTOR:COORD=MM with "
                     f"DETECTOR:COORD one of {valid} (e.g. INTT:z=0.5), "
                     f"got '{override}'")
        minimum_shift[(match.group(1), match.group(2))] = float(match.group(3))
    return minimum_shift


ENTRIES_NORMALIZATIONS = ("raw", "detector", "layer")


def entries_rule(n1, n2, t1, t2, rel_cut: float, z_cut: float):
    """Entries test on counts n1 (reference) and n2 (comparison).

    a = n / t, with t = 1 (raw), the detector total or the layer total of the
    same input. Returns (fires, rel, z):
      rel = a2 / a1 - 1    (inf when a row appears, -1 when it vanishes)
      z   = |a2 - a1| / sqrt(n1 / t1^2 + n2 / t2^2)   (Poisson errors on the
            counts; the totals' own errors are neglected)
    A row fires when |rel| > rel_cut and z > z_cut; a row with no entries in
    either input never fires (rel and z are NaN).
    """
    n1, n2 = np.asarray(n1, float), np.asarray(n2, float)
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        a1, a2 = n1 / t1, n2 / t2
        rel = np.where(a1 > 0, a2 / a1 - 1.0, np.where(a2 > 0, np.inf, np.nan))
        sigma = np.sqrt(n1 / t1 ** 2 + n2 / t2 ** 2)
        z = np.where(sigma > 0, np.abs(a2 - a1) / sigma, np.nan)
        fires = (np.abs(rel) > rel_cut) & (z > z_cut)
    return fires, rel, z


def entries_columns(merged: pd.DataFrame, normalization: str,
                    rel_cut: float, z_cut: float) -> pd.DataFrame:
    """entries_rel, entries_z and flag_entries for every row of merged.

    Missing rows count as 0 entries. Totals sum the chip rows (chip >= 0) of
    the same detector and coordinate (and layer for "layer"); stave rows
    (chip = -1) are divided by the same total. A detector with no entries at
    all in either input is not tested.
    """
    if normalization not in ENTRIES_NORMALIZATIONS:
        raise ValueError(f"entries normalization must be one of "
                         f"{ENTRIES_NORMALIZATIONS}, got {normalization!r}")
    n1 = merged["entries_1"].fillna(0.0).astype(float)
    n2 = merged["entries_2"].fillna(0.0).astype(float)
    chips = merged["chip"] >= 0
    detector_keys = [merged["detector"], merged["coord"]]

    def totals(counts, keys):
        return (counts.where(chips, 0.0).groupby(keys).transform("sum"))

    present = ((totals(n1, detector_keys) > 0)
               & (totals(n2, detector_keys) > 0))
    if normalization == "raw":
        t1 = t2 = pd.Series(1.0, index=merged.index)
    else:
        keys = detector_keys + ([merged["layer"]] if normalization == "layer"
                                else [])
        t1, t2 = totals(n1, keys), totals(n2, keys)
    fires, rel, z = entries_rule(n1, n2, t1, t2, rel_cut, z_cut)
    return pd.DataFrame({
        "entries_rel": np.where(present, rel, np.nan),
        "entries_z": np.where(present, z, np.nan),
        "flag_entries": present.to_numpy() & fires}, index=merged.index)


def describe_entries(row: pd.Series) -> str:
    n1 = 0.0 if pd.isna(row["entries_1"]) else row["entries_1"]
    n2 = 0.0 if pd.isna(row["entries_2"]) else row["entries_2"]
    change = ("new" if np.isinf(row["entries_rel"])
              else f"{100.0 * row['entries_rel']:+.0f}%")
    return (f"entries {change} ({n1:.0f} -> {n2:.0f}, "
            f"z={row['entries_z']:.1f})")


def describe_row(row: pd.Series, reference_label: str,
                 comparison_label: str) -> str:
    if not row["comparable"]:
        problems = [f"{label} {status}" for label, status in
                    ((reference_label, row["fit_status_1"]),
                     (comparison_label, row["fit_status_2"]))
                    if status != "ok"]
        text = "not comparable: " + (", ".join(problems)
                                     or "invalid fit errors")
        if row["flag_entries"]:
            return f"{describe_entries(row)}; {text}"
        return text

    parts = []
    if row["flag_mean"]:
        parts.append(f"mean shift {1000.0 * row['delta_mu_mm']:+.1f} um "
                     f"(z={row['mu_z']:.1f})")
    if row["flag_width"]:
        direction = "broader" if row["sigma_rel"] > 0 else "narrower"
        parts.append(f"width {100.0 * row['sigma_rel']:+.0f}% ({direction})")
    if row["flag_entries"]:
        parts.append(describe_entries(row))
    return "; ".join(parts)


def compare_tables(reference: pd.DataFrame, comparison: pd.DataFrame,
                   sigma_rel_threshold: float, z_threshold: float,
                   minimum_shift: dict, entries_rel_threshold: float = 0.05,
                   entries_z_threshold: float = 3.0,
                   entries_normalization: str = "raw",
                   width_significance_threshold: float = 3.0) -> pd.DataFrame:
    reference_label = reference["label"].iloc[0]
    comparison_label = comparison["label"].iloc[0]
    merged = reference[KEY_COLUMNS + FIT_COLUMNS].merge(
        comparison[KEY_COLUMNS + FIT_COLUMNS], on=KEY_COLUMNS, how="outer",
        suffixes=("_1", "_2"))
    merged = merged.rename(columns={
        f"{column}{suffix}": f"{column.replace('_mm', '')}{suffix}_mm"
        if column.endswith("_mm") else f"{column}{suffix}"
        for column in FIT_COLUMNS for suffix in ("_1", "_2")})
    merged["fit_status_1"] = merged["fit_status_1"].fillna("missing")
    merged["fit_status_2"] = merged["fit_status_2"].fillna("missing")

    mu1, mu2 = merged["mu_1_mm"], merged["mu_2_mm"]
    mu_err1, mu_err2 = merged["mu_err_1_mm"], merged["mu_err_2_mm"]
    sigma1, sigma2 = merged["sigma_1_mm"], merged["sigma_2_mm"]
    sigma_err1, sigma_err2 = merged["sigma_err_1_mm"], merged["sigma_err_2_mm"]

    mu_err = np.sqrt(mu_err1 ** 2 + mu_err2 ** 2)
    merged["comparable"] = (
        (merged["fit_status_1"] == "ok") & (merged["fit_status_2"] == "ok")
        & (sigma1 > 0) & (sigma2 > 0) & np.isfinite(mu_err) & (mu_err > 0))

    with np.errstate(divide="ignore", invalid="ignore"):
        merged["delta_mu_mm"] = mu2 - mu1
        merged["mu_z"] = np.abs(mu2 - mu1) / mu_err
        merged["separation"] = np.abs(mu2 - mu1) / np.sqrt(
            sigma1 ** 2 + sigma2 ** 2)
        merged["sigma_rel"] = (sigma2 - sigma1) / sigma1
        sigma_rel_err = (sigma2 / sigma1) * np.sqrt(
            (sigma_err1 / sigma1) ** 2 + (sigma_err2 / sigma2) ** 2)
        merged["sigma_rel_err"] = sigma_rel_err
        merged["sigma_rel_significance"] = (
            np.abs(merged["sigma_rel"]) / sigma_rel_err)

    merged["min_shift_mm"] = [
        minimum_shift.get((detector, coord), np.nan)
        for detector, coord in zip(merged["detector"], merged["coord"])]
    merged["flag_mean"] = (
        merged["comparable"] & (merged["mu_z"] > z_threshold)
        & (np.abs(merged["delta_mu_mm"]) > merged["min_shift_mm"]))
    merged["flag_width"] = (
        merged["comparable"]
        & (np.abs(merged["sigma_rel"]) > sigma_rel_threshold)
        & (merged["sigma_rel_significance"] > width_significance_threshold))
    merged = merged.join(entries_columns(
        merged, entries_normalization, entries_rel_threshold,
        entries_z_threshold))
    merged["flagged"] = (merged["flag_mean"] | merged["flag_width"]
                         | merged["flag_entries"])
    # Only the entries test can flag a row that is not comparable.
    merged["comparison_status"] = np.select(
        [merged["flagged"], ~merged["comparable"]],
        ["flagged", "not_comparable"], default="unchanged")
    merged["reason"] = merged.apply(
        describe_row, axis=1, args=(reference_label, comparison_label))
    merged.insert(0, "comparison_label", comparison_label)
    merged.insert(0, "reference_label", reference_label)
    return merged.sort_values(KEY_COLUMNS).reset_index(drop=True)


def count_flags(frame: pd.DataFrame) -> dict:
    return {
        "rows": int(len(frame)),
        "comparable": int(frame["comparable"].sum()),
        "not_comparable": int((~frame["comparable"]).sum()),
        "flagged": int(frame["flagged"].sum()),
        "flag_mean": int(frame["flag_mean"].sum()),
        "flag_width": int(frame["flag_width"].sum()),
        "flag_mean_and_width": int(
            (frame["flag_mean"] & frame["flag_width"]).sum()),
        "flag_entries": int(frame["flag_entries"].sum()),
    }


def build_summary(flags: pd.DataFrame, inputs: dict, criteria: dict) -> dict:
    # level keeps whole-stave rows out of the per-layer chip counts.
    levels = flags.assign(level=row_level(flags))
    by_layer = []
    for (detector, layer, coord, level), group in levels.groupby(
            ["detector", "layer", "coord", "level"], sort=True):
        by_layer.append({"detector": detector, "layer": int(layer),
                         "coord": coord, "level": level,
                         **count_flags(group)})
    by_level = {level: count_flags(group)
                for level, group in levels.groupby("level", sort=True)}

    not_comparable = (
        flags.loc[~flags["comparable"]]
        .groupby(["detector", "fit_status_1", "fit_status_2"])
        .size().reset_index(name="rows"))

    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "tool": "compare_runs.py",
        "inputs": inputs,
        "criteria": criteria,
        "counts": count_flags(flags),
        "counts_by_level": by_level,
        "by_detector_layer_coord": by_layer,
        "not_comparable_by_status": not_comparable.to_dict(orient="records"),
    }


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(value) for value in row) + " |"
              for row in rows]
    return "\n".join(lines)


def format_um(value_mm: float, digits: int = 1) -> str:
    return "" if pd.isna(value_mm) else f"{1000.0 * value_mm:.{digits}f}"


def build_report(flags: pd.DataFrame, summary: dict) -> str:
    reference = summary["inputs"]["reference"]
    comparison = summary["inputs"]["comparison"]
    criteria = summary["criteria"]
    counts = summary["counts"]

    sections = [
        f"# Residual comparison: {comparison['label']} vs "
        f"{reference['label']}",
        f"Generated {summary['generated_utc']} by compare_runs.py "
        "(deterministic, no AI).",
        "## Inputs and detector configuration",
        "Record the detector/reconstruction configuration of each label "
        "below (geometry/alignment tag, field, run conditions, track "
        "selection).",
        markdown_table(
            ["Role", "Label", "List file", "Fit table (sha256, first 12)",
             "Detector configuration (fill in)"],
            [["reference", reference["label"], reference["listfile"],
              f"{reference['csv']} ({reference['sha256'][:12]})", ""],
             ["comparison", comparison["label"], comparison["listfile"],
              f"{comparison['csv']} ({comparison['sha256'][:12]})", ""]]),
        "## Criteria",
        f"- Width: \\|(sigma2 - sigma1)/sigma1\\| > "
        f"{criteria['sigma_rel_threshold']} and its significance > "
        f"{criteria['width_significance_threshold']}",
        f"- Mean: \\|mu2 - mu1\\| / sqrt(dmu1^2 + dmu2^2) > "
        f"{criteria['z_threshold']} and \\|mu2 - mu1\\| > minimum shift "
        "(" + ", ".join(f"{key} {1000.0 * value:g} um" for key, value in
                        criteria["min_shift_mm"].items()) + ")",
        f"- Entries ({criteria['entries_normalization']}): \\|a2/a1 - 1\\| > "
        f"{criteria['entries_rel_threshold']} and z > "
        f"{criteria['entries_z_threshold']}, a = N"
        + ("" if criteria["entries_normalization"] == "raw" else
           f" / {criteria['entries_normalization']} total")
        + "; missing rows count as 0 entries.",
        "- Flagged if width OR mean OR entries fires; width and mean need "
        "both fits with status ok, entries applies to every row.",
        "## Counts",
        markdown_table(list(counts.keys()), [list(counts.values())]),
        markdown_table(
            ["Detector", "Layer", "Coord", "Level", "Comparable", "Flagged",
             "Mean", "Width", "Entries", "Not comparable"],
            [[row["detector"], row["layer"], row["coord"], row["level"],
              row["comparable"], row["flagged"], row["flag_mean"],
              row["flag_width"], row["flag_entries"], row["not_comparable"]]
             for row in summary["by_detector_layer_coord"]]),
    ]

    flagged = flags.loc[flags["flagged"]]
    sections.append(f"## Flagged entries ({len(flagged)})")
    if flagged.empty:
        sections.append("No entries passed the flagging criteria.")
    else:
        sections.append(markdown_table(
            ["Det", "Layer", "Stave", "Chip", "Coord", "dmu [um]", "z",
             "Separation", "sigma1 [um]", "sigma2 [um]", "sigma rel",
             "sigma rel signif.", "N1", "N2", "N rel", "N z", "Reason"],
            [[row.detector, row.layer, row.stave, row.chip, row.coord,
              format_um(row.delta_mu_mm), f"{row.mu_z:.1f}",
              f"{row.separation:.2f}", format_um(row.sigma_1_mm),
              format_um(row.sigma_2_mm), f"{row.sigma_rel:+.2f}",
              f"{row.sigma_rel_significance:.1f}", f"{row.entries_1:.0f}",
              f"{row.entries_2:.0f}", f"{row.entries_rel:+.3f}",
              f"{row.entries_z:.1f}", row.reason]
             for row in flagged.itertuples()]))

    sections.append("## Not comparable")
    if summary["not_comparable_by_status"]:
        sections.append(markdown_table(
            ["Detector", f"{reference['label']} status",
             f"{comparison['label']} status", "Rows"],
            [[row["detector"], row["fit_status_1"], row["fit_status_2"],
              row["rows"]] for row in summary["not_comparable_by_status"]]))
    else:
        sections.append("All entries were comparable.")

    sections += [
        "## Caveats",
        "- About 1300 chip/coordinate tests (plus about 210 stave/ladder "
        "tests when the table has chip = -1 stave rows, and 30 VERTEX tests "
        "when track_ntp is present) are made per comparison; at z > 1.96 "
        "about 5% of unchanged entries pass the significance test by chance, "
        "so the minimum-shift requirement carries most of the rejection of "
        "chance flags.",
        "- MVTX/INTT rows with chip = -1 are fits to the sum of all chips of "
        "the stave/ladder, so they are not independent of the chip rows. A "
        "whole-stave shift moves the stave row and most chips together. A "
        "single chip shifted within the core of the stave distribution pulls "
        "the stave mean by about its share of the stave entries (about 1/9 "
        "of its shift for MVTX, 1/4 for INTT); a chip shifted beyond the core "
        "barely moves it. Stave rows have small errors, so the minimum shift "
        "decides most of their flags. INTT z stave rows sum the sensors' "
        "strip-length boxes and are only indicative.",
        "- VERTEX rows (layer = stave = -1) are track-to-vertex residuals "
        "whose chip id is the track-phi bin: chip -1 is the distribution of "
        "all tracks and chip 0-13 are equal track-phi bins over [-pi, pi), "
        "so they are not independent of each other. A transverse silicon-to-vertex offset appears as a "
        "sinusoidal dca_xy shift versus phi that can average to almost zero "
        "in chip -1. The vertex is reconstructed from the same tracks, which "
        "biases these residuals toward zero.",
        "- The width test requires a significance above the threshold "
        "('sigma rel signif.' column), so width changes compatible with "
        "statistical fluctuation are not flagged; at low statistics only "
        "large width changes can pass it.",
        "- The entries test with raw counts flags nearly every row when the "
        "two inputs have different numbers of events or tracks; then use "
        "--entries-normalization detector (or layer), which compares each "
        "row's share of its detector (or layer) total and so only finds local "
        "changes. At z > 3 about 0.3% of unchanged rows pass by chance.",
        "- INTT z residuals are not Gaussian (strip length sets the width), "
        "so INTT z fit parameters are only indicative.",
        "- Different run conditions (field, luminosity, track momentum "
        "spectrum, track selection, alignment/geometry tag) change sigma "
        "without any detector change; compare like with like.",
    ]
    return "\n\n".join(sections) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="+", type=Path,
                        help="residual_fits_<label>.csv files; first is the "
                             "reference")
    parser.add_argument("--out-dir", type=Path,
                        help="write every output here (default: a fresh "
                             "runs/<a>_vs_<b> per comparison)")
    parser.add_argument("--sigma-rel-threshold", type=float, default=0.10,
                        help="width test: minimum relative width change "
                             "(default 0.10)")
    parser.add_argument("--width-significance-threshold", type=float,
                        default=3.0,
                        help="width test: minimum significance of the width "
                             "change (default 3)")
    parser.add_argument("--z-threshold", type=float, default=1.96)
    parser.add_argument("--min-shift", action="append", default=[],
                        metavar="DETECTOR:COORD=MM",
                        help="override a minimum mean shift, e.g. "
                             "INTT:z=0.5 or VERTEX:dca_xy=0.02 (repeatable)")
    parser.add_argument("--entries-rel-threshold", type=float, default=0.05,
                        help="entries test: minimum relative change "
                             "(default 0.05)")
    parser.add_argument("--entries-z-threshold", type=float, default=3.0,
                        help="entries test: minimum significance (default 3)")
    parser.add_argument("--entries-normalization",
                        choices=ENTRIES_NORMALIZATIONS, default="raw",
                        help="entries test: compare raw counts (default), or "
                             "each row's share of its detector or layer total")
    args = parser.parse_args()

    if len(args.tables) < 2:
        sys.exit("ERROR: give at least two fit tables (reference first)")

    minimum_shift = parse_min_shift_overrides(args.min_shift)
    tables = [load_fit_table(path) for path in args.tables]
    labels = [table["label"].iloc[0] for table in tables]
    if len(set(labels)) != len(labels):
        sys.exit(f"ERROR: input labels must be distinct, got {labels}")

    criteria = {
        "sigma_rel_threshold": args.sigma_rel_threshold,
        "width_significance_threshold": args.width_significance_threshold,
        "z_threshold": args.z_threshold,
        "min_shift_mm": {f"{detector}:{coord}": value for
                         (detector, coord), value in minimum_shift.items()},
        "entries_rel_threshold": args.entries_rel_threshold,
        "entries_z_threshold": args.entries_z_threshold,
        "entries_normalization": args.entries_normalization,
        "logic": "flagged = (width change AND width significance) OR (z AND "
                 "minimum shift) OR (entries change AND entries z); "
                 "reference = first table",
    }

    def describe_input(path: Path, table: pd.DataFrame) -> dict:
        return {"label": table["label"].iloc[0],
                "listfile": table["listfile"].iloc[0],
                "csv": path.name, "sha256": sha256sum(path)}

    # Check every name before anything is deleted or written.
    if args.out_dir is None:
        out_dirs = [run_dir_for(args.tables[0], path)
                    for path in args.tables[1:]]
        for path, table in zip(args.tables, tables):
            if sanitize_for_filename(table_name(path)) != sanitize_for_filename(
                    table["label"].iloc[0]):
                print(f"Note: {path.name} holds label "
                      f"'{table['label'].iloc[0]}'; the folder is named from "
                      "the file, the files inside from the label")
    else:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        out_dirs = [args.out_dir] * (len(args.tables) - 1)

    reference = tables[0]
    for path, comparison, out_dir in zip(args.tables[1:], tables[1:],
                                         out_dirs):
        if args.out_dir is None:
            fresh_run_dir(out_dir, args.tables)
        flags = compare_tables(reference, comparison,
                               args.sigma_rel_threshold, args.z_threshold,
                               minimum_shift, args.entries_rel_threshold,
                               args.entries_z_threshold,
                               args.entries_normalization,
                               args.width_significance_threshold)
        for detector in sorted(set(flags["detector"])):
            rows = flags["detector"] == detector
            if flags.loc[rows, "entries_rel"].isna().all() and (
                    flags.loc[rows, ["entries_1", "entries_2"]].fillna(0)
                    .sum().gt(0).any()):
                print(f"Warning: {detector} has no entries in one input; "
                      "the entries test is skipped for it")
        inputs = {"reference": describe_input(args.tables[0], reference),
                  "comparison": describe_input(path, comparison)}
        summary = build_summary(flags, inputs, criteria)

        stem = (f"{sanitize_for_filename(inputs['reference']['label'])}_vs_"
                f"{sanitize_for_filename(inputs['comparison']['label'])}")
        flags_path = out_dir / f"flags_{stem}.csv"
        summary_path = out_dir / f"summary_{stem}.json"
        report_path = out_dir / f"report_{stem}.md"
        flags.to_csv(flags_path, index=False)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        report_path.write_text(build_report(flags, summary))

        counts = summary["counts"]
        print(f"{inputs['comparison']['label']} vs "
              f"{inputs['reference']['label']}: {counts['flagged']} flagged "
              f"of {counts['comparable']} comparable "
              f"({counts['not_comparable']} not comparable) -> "
              f"{flags_path}, {summary_path}, {report_path}")


if __name__ == "__main__":
    main()
