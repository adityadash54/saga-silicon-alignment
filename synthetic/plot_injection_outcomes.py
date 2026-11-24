#!/usr/bin/env python3
"""Plot which injected changes were caught, at two levels, and which were not.

For each synthetic comparison with known injected changes this shows, per
injection:
  1. flags: did the fixed criteria of stdiag.compare_runs flag its rows
     (or, for decoys, correctly leave them unflagged)?
  2. AI triage: did a finding hold its flagged rows with the expected group,
     or, for unflagged injections, did the triage at least mention them?

Datasets (--datasets, comma-separated):
  full  the nine full-geometry injections, Iter1 and Iter2 vs Iter0, from the
        committed ROOT-pipeline tables in synthetic/reference/
  demo  the stave/chip demo (synthetic/stavechip_demo/reference/after/)
  eval  the 21 triage eval cases (evals/triage/cases.py); their AI results
        come from a benchmark run (evals/triage/runs/<variant>/work/), if any

Outputs in --out-dir (default runs/injection_plots, git-ignored), one folder
per comparison:
  map_rphi.png, map_z.png   detector maps: change relative to the flagging
                            threshold per chip/sensor, injected cells outlined
                            by outcome
  shifts.png                measured vs injected mean shift, in units of the
                            minimum shift
  vertex.png                (demo) dca_xy shift per track-phi bin vs truth
  outcomes.csv, outcomes.png  one row per injection with both outcomes
and summary_outcomes.csv/png across all comparisons.

--triage runs the AI triage (claude-code backend, about $0.50 per comparison;
sends the synthetic flag summary to Anthropic) for the full and demo
comparisons. Without it, existing findings files are used if present.

Usage:
  python -m synthetic.plot_injection_outcomes
  python -m synthetic.plot_injection_outcomes --datasets full,demo --triage
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

from stdiag.paths import REPO_ROOT, RUNS_DIR  # noqa: E402
from synthetic import check_against_truth  # noqa: E402
from synthetic.make_synthetic_fit_tables import (  # noqa: E402
    GEOMETRY, INJECTIONS, matches)
from synthetic.stavechip_demo import check_demo  # noqa: E402

KEY = ["detector", "layer", "stave", "chip", "coord"]
FLAG_GROUPS = {"A", "B", "C", "D"}
COLORS = {"caught": "#1a9850", "missed": "#d73027", "decoy ok": "#4575b4",
          "flagged anyway": "#f46d43", "not comparable": "#fdae61",
          "either": "#984ea3", "false flag": "#e7298a"}

# Expected triage group of the full-geometry injections (None: no flags
# expected, so the triage can only mention it).
FULL_GROUPS = {"stave_shift_rphi": "A", "chip_broadened": "C",
               "chip_shift_z": "B", "intt_sensor_shift": "B",
               "half_layer_shift_z": "A"}


def demo_injections() -> list[dict]:
    """The stave/chip demo truth (make_synth.C) in the injection format."""
    def inj(name, detector, layer, chips, coord, expected, should, **extra):
        return {"name": name, "labels": ["Iter1"], "detector": detector,
                "layer": layer, "staves": [0] if detector != "VERTEX" else
                [-1], "chips": chips, "coord": coord, "expected": expected,
                "should": should, **extra}
    return [
        inj("chip3_shift_+10um", "MVTX", 0, [3], "rphi", "B", "flag",
            offset_mm=0.010),
        inj("chip5_back_from_+300um", "MVTX", 0, [5], "rphi", "B", "flag",
            offset_mm=-0.300),
        inj("chip6_+30um_in_both", "MVTX", 0, [6], "rphi", None, "no flag"),
        inj("chip7_6_entries", "MVTX", 0, [7], None, None, "not comparable"),
        inj("chip8_back_from_+800um", "MVTX", 0, [8], "rphi", None,
            "not comparable", offset_mm=-0.800),
        inj("intt_sensor2_z_+1.5mm", "INTT", 3, [2], "z", "B", "flag",
            offset_mm=1.5),
        inj("vertex_offset_change", "VERTEX", -1, list(range(14)), "dca_xy",
            "A", "some flags"),
    ]


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def run(module: str, *args: str) -> None:
    result = subprocess.run([sys.executable, "-m", module, *args],
                            cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode:
        sys.exit(f"ERROR: {module} failed:\n{result.stderr[-2000:]}")


def read_flags(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"detector": str, "coord": str,
                                    "reason": str})


def key_of(row) -> tuple:
    return (str(row["detector"]), int(row["layer"]), int(row["stave"]),
            int(row["chip"]), str(row["coord"]))


def comparisons(datasets: list[str], out: Path) -> list[dict]:
    """Build the flags of every requested comparison (no AI)."""
    items = []
    if "full" in datasets:
        tables = REPO_ROOT / "synthetic" / "reference"
        folder = out / "full"
        run("stdiag.compare_runs", *(str(tables / f"residual_fits_Iter{i}.csv")
                                     for i in range(3)),
            "--out-dir", str(folder))
        for comparison in ("Iter1", "Iter2"):
            injections = [dict(inj, expected=FULL_GROUPS.get(inj["name"]))
                          for inj in INJECTIONS]
            items.append({"name": f"full_Iter0_vs_{comparison}",
                          "folder": folder, "reference": "Iter0",
                          "comparison": comparison,
                          "stem": f"Iter0_vs_{comparison}",
                          "injections": injections, "kind": "full"})
    if "demo" in datasets:
        tables = REPO_ROOT / "synthetic" / "stavechip_demo" / "reference" / "after"
        folder = out / "demo"
        run("stdiag.compare_runs", str(tables / "residual_fits_Iter0.csv"),
            str(tables / "residual_fits_Iter1.csv"), "--out-dir", str(folder))
        items.append({"name": "demo_Iter0_vs_Iter1", "folder": folder,
                      "reference": "Iter0", "comparison": "Iter1",
                      "stem": "Iter0_vs_Iter1",
                      "injections": demo_injections(), "kind": "demo"})
    if "eval" in datasets:
        from evals.triage.cases import CASES
        from evals.triage.run_eval import build_inputs
        for case in CASES:
            folder = out / "eval" / case["id"]
            build_inputs(case, folder)
            items.append({"name": f"eval_{case['id']}", "folder": folder,
                          "reference": "Ref", "comparison": "Cmp",
                          "stem": "Ref_vs_Cmp", "case": case["id"],
                          "injections": case["injections"], "kind": "eval"})
    return items


def find_findings(item: dict) -> Path | None:
    own = item["folder"] / f"findings_{item['stem']}.json"
    if own.exists():
        return own
    if item["kind"] == "eval":
        runs = sorted((REPO_ROOT / "evals" / "triage" / "runs").glob(
            f"*/work/{item['case']}_rep*/findings_Ref_vs_Cmp.json"))
        return runs[0] if runs else None
    return None


def run_triage(item: dict) -> None:
    folder, stem = item["folder"], item["stem"]
    print(f"Triage (AI) for {item['name']} ...", flush=True)
    run("stdiag.triage_agent", str(folder / f"flags_{stem}.csv"),
        str(folder / f"summary_{stem}.json"), "--backend", "claude-code",
        "--usage-out", str(folder / f"triage_usage_{stem}.json"))


# ---------------------------------------------------------------------------
# Outcomes
# ---------------------------------------------------------------------------

def injection_rows(injection: dict, item: dict, keys: list[tuple]) -> set:
    """Rows of one injection in this comparison (empty if it is not active
    in either input). An entry override applies to every input and to both
    coordinates (a cluster carries both residuals), as in the generators."""
    labels = {item["reference"], item["comparison"]}
    if "entries" in injection:
        selector, active = dict(injection, coord=None), injection["labels"]
    else:
        selector, active = injection, labels & set(injection["labels"])
    return {key for key in keys
            if any(matches(selector, label, *key) for label in active)}


def expected_row_status(item: dict, flags: pd.DataFrame,
                        criteria: dict) -> dict:
    """Expected status of every row where the truth gives one."""
    if item["kind"] == "full":
        return {key_of(row): check_against_truth.expected_status(
                    row, item["reference"], item["comparison"], criteria)
                for _, row in flags.iterrows()}
    if item["kind"] == "demo":
        return {key_of(row): check_demo.expected_status(row)
                for _, row in flags.iterrows()}
    return {}


def should_flag(injection: dict, item: dict, rows: set,
                expected: dict) -> str:
    if "should" in injection:
        return injection["should"]
    if item["kind"] == "eval":
        group = injection["expected"]
        return ("flag" if group in FLAG_GROUPS else "either"
                if group == "either" else "not comparable"
                if injection.get("missing") or injection.get("entries", 999) < 10
                else "no flag")
    statuses = [expected.get(row) for row in rows if row[3] >= 0]
    if not statuses:
        return "not comparable"
    if all(status == "not_comparable" for status in statuses):
        return "not comparable"
    if "flagged" in statuses:
        return "flag"
    if "either" in statuses:
        return "either"
    return "no flag"


def flag_outcome(should: str, n_rows: int, n_flagged: int, n_nc: int) -> str:
    if should in ("flag", "some flags"):
        return "caught" if n_flagged else "missed"
    if should == "not comparable":
        return "not comparable" if n_nc and not n_flagged else "flagged anyway"
    if should == "either":
        return "either"
    return "decoy ok" if not n_flagged else "flagged anyway"


def ai_outcome(injection: dict, rows: set, flagged_rows: set,
               findings: dict | None) -> str:
    if findings is None:
        return "not run"
    entries = []
    for finding in findings.get("findings", []):
        keys = set()
        for entry in finding.get("entries", []):
            try:
                keys.add(key_of(entry))
            except (TypeError, ValueError, KeyError):
                continue
        entries.append((finding, keys))
    own_flags = rows & flagged_rows
    if own_flags:
        best, best_overlap = None, 0
        for finding, keys in entries:
            overlap = len(own_flags & keys)
            if overlap > best_overlap:
                best, best_overlap = finding, overlap
        if best is None or best_overlap < 0.5 * len(own_flags):
            return "missed"
        group = best.get("group")
        expected = injection.get("expected")
        if expected in FLAG_GROUPS and group != expected:
            return f"found, group {group} (expected {expected})"
        return f"found, group {group}"
    mentioned = [finding for finding, keys in entries if rows & keys]
    if mentioned:
        return "mentioned, group " + ",".join(
            sorted({str(finding.get("group")) for finding in mentioned}))
    return "not mentioned"


def ai_verdict(ai: str, flag_outcome: str, why: str) -> str:
    """Colour class of the AI outcome: good, mention, bad or neutral."""
    if ai.startswith("found, group") and "expected" not in ai:
        return "good"
    if ai.startswith("mentioned"):
        return "mention"
    if ai == "not mentioned" and flag_outcome == "decoy ok" and \
            "significant" not in why:
        return "neutral"
    if ai in ("missed", "not mentioned", "invented finding") or \
            "expected" in ai:
        return "bad"
    return "neutral"


def analyse(item: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    flags = read_flags(item["folder"] / f"flags_{item['stem']}.csv")
    summary = json.loads((item["folder"] / f"summary_{item['stem']}.json")
                         .read_text())
    keys = [key_of(row) for _, row in flags.iterrows()]
    status = dict(zip(keys, flags["comparison_status"]))
    flagged_rows = {key for key in keys if status[key] == "flagged"}
    expected = expected_row_status(item, flags, summary["criteria"])
    findings_path = find_findings(item)
    findings = (json.loads(findings_path.read_text())
                if findings_path else None)

    rows_out, injected_all = [], set()
    for injection in item["injections"]:
        rows = injection_rows(injection, item, keys)
        if not rows:
            continue
        chip_rows = {row for row in rows if row[3] >= 0 or row[0] == "VERTEX"}
        injected_all |= rows
        n_flagged = sum(status[row] == "flagged" for row in chip_rows)
        n_nc = sum(status[row] == "not_comparable" for row in chip_rows)
        should = should_flag(injection, item, chip_rows, expected)
        rows_out.append({
            "comparison": item["name"], "injection": injection["name"],
            "rows": len(chip_rows), "flagged": n_flagged,
            "not_comparable": n_nc, "should": should,
            "flag_outcome": flag_outcome(should, len(chip_rows), n_flagged,
                                         n_nc),
            "expected_group": injection.get("expected") or "-",
            "ai_outcome": ai_outcome(injection, chip_rows, flagged_rows,
                                     findings),
            "reason": reason(injection, chip_rows, flags, status)})
    # A whole-stave row (chip -1) of a stave with an injected chip follows
    # that chip (a shifted chip widens the summed distribution), so it is
    # not counted as a false flag.
    injected_staves = {(d, l, s, c) for d, l, s, chip, c in injected_all
                       if chip >= 0}
    false_flags = sorted(
        row for row in flagged_rows if row not in injected_all and not (
            row[0] != "VERTEX" and row[3] == -1
            and (row[0], row[1], row[2], row[4]) in injected_staves))
    if false_flags:
        rows_out.append({
            "comparison": item["name"], "injection": "(not injected)",
            "rows": len(false_flags), "flagged": len(false_flags),
            "not_comparable": 0, "should": "no flag",
            "flag_outcome": "false flag", "expected_group": "-",
            "ai_outcome": "-", "reason": ", ".join(
                f"{d} L{l} S{s} c{c} {co}" for d, l, s, c, co in false_flags[:4])})
    if not rows_out:
        rows_out.append({
            "comparison": item["name"], "injection": "(no injection)",
            "rows": 0, "flagged": 0, "not_comparable": 0, "should": "no flag",
            "flag_outcome": "decoy ok", "expected_group": "-",
            "ai_outcome": "-" if findings is None else (
                "invented finding" if any(
                    finding.get("group") in FLAG_GROUPS
                    for finding in findings.get("findings", []))
                else "no finding"),
            "reason": ""})
    item["findings_path"] = findings_path
    return pd.DataFrame(rows_out), flags


def reason(injection, rows, flags, status) -> str:
    """Short explanation for rows that are not flagged."""
    unflagged = [row for row in rows if status[row] != "flagged"]
    if not unflagged:
        return ""
    sub = flags.set_index(KEY).loc[unflagged]
    texts = set()
    for _, row in sub.iterrows():
        if row["comparison_status"] == "not_comparable":
            texts.add(str(row["reason"]).replace("not comparable: ", ""))
        elif abs(row["delta_mu_mm"]) <= row["min_shift_mm"] and abs(
                row["mu_z"]) > 1.96:
            texts.add("significant but below the minimum shift")
        elif abs(row["sigma_rel"]) <= 0.2 and abs(row["delta_mu_mm"]) <= row[
                "min_shift_mm"]:
            texts.add("change below both thresholds")
    return "; ".join(sorted(texts))[:90]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def cell_outcome(key, status, injected_should) -> str | None:
    flagged = status.get(key) == "flagged"
    nc = status.get(key) == "not_comparable"
    if key in injected_should:
        should = injected_should[key]
        if should in ("flag", "some flags"):
            return "caught" if flagged else ("not comparable" if nc
                                             else "missed")
        if should == "either":
            return "either"
        if should == "not comparable":
            return "not comparable" if nc else (
                "flagged anyway" if flagged else "decoy ok")
        return "flagged anyway" if flagged else "decoy ok"
    return "false flag" if flagged else None


def plot_maps(item, flags, outcomes, out: Path) -> None:
    status = {key_of(row): row["comparison_status"]
              for _, row in flags.iterrows()}
    should = {}
    for injection in item["injections"]:
        rows = injection_rows(injection, item, list(status))
        verdict = outcomes.loc[outcomes["injection"] == injection["name"],
                               "should"]
        for row in rows:
            should[row] = verdict.iloc[0] if len(verdict) else "flag"
    ratio = flags.assign(
        ratio=np.maximum(flags["delta_mu_mm"].abs() / flags["min_shift_mm"],
                         flags["sigma_rel"].abs() / 0.2))
    for coord in ("rphi", "z"):
        panels = [(det, layer) for det in ("MVTX", "INTT")
                  for layer in GEOMETRY[det][0]
                  if ((ratio["detector"] == det) & (ratio["layer"] == layer)
                      & (ratio["coord"] == coord)).any()]
        if not panels:
            continue
        fig, axes = plt.subplots(1, len(panels), figsize=(
            2.0 + 2.3 * len(panels), 4.2), squeeze=False)
        image = None
        for ax, (det, layer) in zip(axes[0], panels):
            sub = ratio[(ratio["detector"] == det) & (ratio["layer"] == layer)
                        & (ratio["coord"] == coord) & (ratio["chip"] >= 0)]
            n_staves = int(sub["stave"].max()) + 1
            n_chips = GEOMETRY[det][2]
            grid = np.full((n_chips, n_staves), np.nan)
            for _, row in sub.iterrows():
                if row["comparison_status"] != "not_comparable":
                    grid[int(row["chip"]), int(row["stave"])] = row["ratio"]
            cmap = plt.get_cmap("Greys").copy()
            cmap.set_bad("#fde0c5")
            image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap,
                              vmin=0, vmax=3, interpolation="nearest")
            for _, row in sub.iterrows():
                key = key_of(row)
                outcome = cell_outcome(key, status, should)
                if outcome:
                    ax.add_patch(Rectangle(
                        (row["stave"] - 0.5, row["chip"] - 0.5), 1, 1,
                        fill=False, lw=1.6, ec=COLORS[outcome]))
            ax.set_title(f"{det} L{layer}", fontsize=9)
            ax.set_xlabel("stave" if det == "MVTX" else "ladder", fontsize=8)
            ax.set_ylabel("chip" if det == "MVTX" else "sensor", fontsize=8)
            ax.set_yticks(range(n_chips))
            ax.tick_params(labelsize=7)
        bar = fig.colorbar(image, cax=fig.add_axes([0.875, 0.3, 0.012, 0.56]))
        bar.set_label("change / threshold\n(max of |Δμ|/min shift, |σ rel|/0.2)",
                      fontsize=8)
        handles = [Patch(fill=False, ec=color, lw=1.6, label=label)
                   for label, color in COLORS.items()]
        handles.append(Patch(fc="#fde0c5", label="not comparable (no fit)"))
        fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
                   frameon=False)
        fig.suptitle(f"{item['name']}: local {coord} — injected cells "
                     "outlined by outcome", fontsize=10)
        fig.subplots_adjust(bottom=0.27, top=0.86, left=0.04, right=0.85, wspace=0.35)
        fig.savefig(out / f"map_{coord}.png", dpi=130)
        plt.close(fig)


def plot_shifts(item, flags, out: Path) -> None:
    points = []
    status = {key_of(row): row for _, row in flags.iterrows()}
    for injection in item["injections"]:
        offset = injection.get("offset_mm", 0.0)
        if not offset or injection["detector"] == "VERTEX":
            continue
        for key in injection_rows(injection, item, list(status)):
            row = status[key]
            if key[3] < 0 or row["comparison_status"] == "not_comparable":
                continue
            sign = 1.0 if item["comparison"] in injection["labels"] else -1.0
            points.append((sign * offset / row["min_shift_mm"],
                           row["delta_mu_mm"] / row["min_shift_mm"],
                           row["comparison_status"] == "flagged",
                           injection["name"]))
    if not points:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    names = sorted({p[3] for p in points})
    for index, name in enumerate(names):
        sub = [p for p in points if p[3] == name]
        color = plt.get_cmap("tab10")(index % 10)
        flagged = [p for p in sub if p[2]]
        if flagged:
            ax.scatter([p[0] for p in flagged], [p[1] for p in flagged],
                       color=color, marker="o", s=22, label=f"{name} (flagged)")
        unflagged = [p for p in sub if not p[2]]
        if unflagged:
            ax.scatter([p[0] for p in unflagged], [p[1] for p in unflagged],
                       facecolors="none", edgecolors=color, marker="o", s=40,
                       label=f"{name} (not flagged)")
    limit = max(2.5, max(abs(p[0]) for p in points) * 1.15)
    ax.plot([-limit, limit], [-limit, limit], color="0.6", lw=0.8, ls=":")
    for y in (-1, 1):
        ax.axhline(y, color="#d73027", lw=0.8, ls="--")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_xlabel("injected shift / minimum shift")
    ax.set_ylabel("measured Δμ / minimum shift")
    ax.set_title(f"{item['name']}: mean-shift injections "
                 "(dashed: flagging threshold)", fontsize=9)
    ax.legend(fontsize=6, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "shifts.png", dpi=130)
    plt.close(fig)


def plot_vertex(flags, out: Path) -> None:
    sub = flags[(flags["detector"] == "VERTEX") & (flags["coord"] == "dca_xy")
                & (flags["chip"] >= 0)].sort_values("chip")
    if sub.empty:
        return
    phi = -math.pi + (sub["chip"] + 0.5) * 2 * math.pi / 14
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    curve = np.linspace(-math.pi, math.pi, 200)
    ax.plot(curve, -10 * np.sin(curve) - 10 * np.cos(curve), color="0.4",
            label="injected: −10 sin φ − 10 cos φ µm")
    flagged = sub["comparison_status"] == "flagged"
    error = (sub["delta_mu_mm"] / sub["mu_z"]).abs() * 1000
    ax.errorbar(phi[flagged], sub["delta_mu_mm"][flagged] * 1000,
                yerr=error[flagged], fmt="o", color=COLORS["caught"],
                label="flagged")
    ax.errorbar(phi[~flagged], sub["delta_mu_mm"][~flagged] * 1000,
                yerr=error[~flagged], fmt="o", mfc="none",
                color="0.3", label="not flagged")
    for y in (-10, 10):
        ax.axhline(y, color="#d73027", lw=0.8, ls="--")
    ax.set_xlabel("track φ [rad]")
    ax.set_ylabel("Δ⟨−DCA_xy⟩ [µm]")
    ax.set_title("demo: vertex offset (30,0) → (20,10) µm, per φ bin "
                 "(dashed: 10 µm minimum shift)", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "vertex.png", dpi=130)
    plt.close(fig)


def plot_table(outcomes: pd.DataFrame, path: Path, title: str) -> None:
    columns = ["comparison", "injection", "rows", "flagged", "not_comparable",
               "flag_outcome", "expected_group", "ai_outcome", "reason"]
    shown = outcomes[columns].copy()
    shown.columns = ["comparison", "injection", "rows", "flagged", "NC",
                     "flag level", "exp. group", "AI triage", "why not flagged"]
    height = 0.5 + 0.2 * len(shown)
    fig, ax = plt.subplots(figsize=(16, height))
    ax.axis("off")
    table = ax.table(cellText=shown.values, colLabels=shown.columns,
                     loc="upper center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.auto_set_column_width(list(range(len(shown.columns))))
    for row_index, (_, row) in enumerate(outcomes.iterrows(), start=1):
        table[row_index, 5].set_facecolor(
            COLORS.get(row["flag_outcome"], "white") + "55")
        verdict = ai_verdict(row["ai_outcome"], row["flag_outcome"],
                             str(row["reason"]))
        table[row_index, 7].set_facecolor(
            {"good": "#1a985055", "mention": "#4575b455",
             "bad": "#d7302755"}.get(verdict, "white"))
    ax.set_title(title, fontsize=10)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", default="full,demo,eval")
    parser.add_argument("--triage", action="store_true",
                        help="run the AI triage for the full and demo "
                             "comparisons (claude-code backend)")
    parser.add_argument("--out-dir", type=Path,
                        default=RUNS_DIR / "injection_plots")
    args = parser.parse_args()

    datasets = [name.strip() for name in args.datasets.split(",")]
    unknown = set(datasets) - {"full", "demo", "eval"}
    if unknown:
        sys.exit(f"ERROR: unknown datasets {sorted(unknown)}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = comparisons(datasets, args.out_dir)
    if args.triage:
        for item in items:
            if item["kind"] in ("full", "demo"):
                run_triage(item)

    all_outcomes = []
    for item in items:
        outcomes, flags = analyse(item)
        folder = args.out_dir / item["name"]
        folder.mkdir(parents=True, exist_ok=True)
        outcomes.to_csv(folder / "outcomes.csv", index=False)
        plot_maps(item, flags, outcomes, folder)
        plot_shifts(item, flags, folder)
        if item["kind"] == "demo":
            plot_vertex(flags, folder)
        source = item["findings_path"]
        plot_table(outcomes, folder / "outcomes.png",
                   f"{item['name']}: injected changes, flag level and AI "
                   f"triage ({'findings: ' + source.name if source else 'no AI results'})")
        all_outcomes.append(outcomes)
        print(f"{item['name']}: " + ", ".join(
            f"{name} {count}" for name, count in
            outcomes["flag_outcome"].value_counts().items()))

    summary = pd.concat(all_outcomes, ignore_index=True)
    summary.to_csv(args.out_dir / "summary_outcomes.csv", index=False)
    non_eval = summary[~summary["comparison"].str.startswith("eval_")]
    if not non_eval.empty:
        plot_table(non_eval, args.out_dir / "summary_outcomes.png",
                   "Injected changes: flag level and AI triage")
    evals = summary[summary["comparison"].str.startswith("eval_")]
    if not evals.empty:
        plot_table(evals, args.out_dir / "summary_eval_cases.png",
                   "Triage eval cases: flag level (AI: from benchmark runs, "
                   "if any)")
    print(f"Plots and tables in {args.out_dir}")


if __name__ == "__main__":
    main()
