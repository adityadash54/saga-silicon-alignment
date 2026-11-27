#!/usr/bin/env python3
"""Sensitivity of the compare_runs flagging rule to small mean and width changes.

Level 1 (fit tables only; no ROOT, no AI, nothing leaves the machine).
Synthetic fit tables from synthetic.make_synthetic_fit_tables.build_tables are
compared with stdiag.compare_runs.compare_tables while sweeping
  - the injected mean shift, -4 to +4 x the minimum shift (steps of 0.05
    up to 2 x, then 0.1),
  - the injected width scale sigma_Cmp / sigma_Ref, 0.70 to 1.40 in steps of
    0.01,
for several entry counts per chip. Every MVTX chip and INTT sensor of a
coordinate gets the same injection, so each one is an independent trial
(432 MVTX chips, 224 INTT sensors per coordinate, per point and seed).

The fit errors start from the ideal ones (mu_err = sigma / sqrt(0.95 N),
sigma_err = sigma / sqrt(2 * 0.95 N), Gaussian noise) and are scaled per
detector and coordinate by the ratio measured from the ROOT fits of
synthetic/reference (about 1.13 for the mean and 1.56 for the width of MVTX
and INTT r-phi; --ideal-errors turns this off). Fit biases of the ROOT macro
(level 2) and the track fit's response (biased residuals, weak modes; needs
full simulation) are not included.

Alternative thresholds are evaluated on the same comparisons by re-applying
the rule to the columns compare_tables writes; on every comparison the
default set is checked to reproduce compare_tables' own flags exactly.

Entries test (stdiag.compare_runs.entries_rule): a counting toy, since the
test only uses the entry counts. Each MVTX chip and INTT sensor gets one
count per input (r-phi and z rows share their clusters), drawn from Poisson
with a chip-to-chip acceptance spread of 0.8-1.2. One chip per stave
(MVTX chip 4) or ladder (INTT sensor 1) has its expected count scaled by
0.80-1.20; the others measure the false-flag rate. The comparison input can
also have fewer events overall (--event-ratios, default 1.00 and 0.90).
compare_runs (default 5% and z > 3) and the variants here: a row fires when
|a2/a1 - 1| > 5% (and, optionally, z > 1.96 or 3), with a = N (raw), N / its
detector total, or N / its layer total, and
z = |a2 - a1| / sqrt(N1 / t1^2 + N2 / t2^2) (t = 1 for raw).

Outputs in --out-dir (default synthetic/sensitivity_output, git-ignored):
  efficiency_mean.csv, efficiency_width.csv  toy efficiency per threshold set,
                                             detector, coordinate, N and
                                             injected value, with 68% Wilson
                                             intervals and the analytic value
  efficiency_entries.csv                     the same for the entries test, for
                                             injected and control chips
  sensitivity_summary.md                     50% and 90% points, false-flag
                                             rates, toy vs analytic
  mean_efficiency.png, width_efficiency.png, mean_thresholds.png,
  false_width_rate.png, entries_efficiency.png,
  entries_false_rate.png                     plots (need matplotlib)
  mean_cuts.png, width_cuts.png,
  entries_cuts.png, cut_tradeoff.png         cut choice: threshold options
                                             side by side at the entries per
                                             chip of --plot-entries

Usage:
  python -m synthetic.scan_flag_sensitivity
  python -m synthetic.scan_flag_sensitivity --seeds 10 --entries 20,100,nominal
  python -m synthetic.scan_flag_sensitivity --entries 100,300,800,1000,1500,3000,5000 \
      --plot-entries 100,300,800,1500,5000
  python -m synthetic.scan_flag_sensitivity --replot --plot-entries 100,300,1000
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stdiag.compare_runs import DEFAULT_MIN_SHIFT_MM, compare_tables
from stdiag.compare_runs import entries_rule as compare_runs_entries_rule
from synthetic.make_synthetic_fit_tables import (GEOMETRY, NOMINAL_ENTRIES,
                                                 NOMINAL_SIGMA_MM,
                                                 build_tables)

REPO_ROOT = Path(__file__).resolve().parent.parent
LABELS = {"Ref": "reference.txt", "Cmp": "comparison.txt"}
COMBOS = [("MVTX", "rphi"), ("MVTX", "z"), ("INTT", "rphi"), ("INTT", "z")]
COMBO_NAME = {("MVTX", "rphi"): "MVTX r-phi", ("MVTX", "z"): "MVTX z",
              ("INTT", "rphi"): "INTT r-phi", ("INTT", "z"): "INTT z"}
# Display units: INTT z in mm, everything else in um.
UNIT = {combo: ("mm", 1.0) if combo == ("INTT", "z") else ("um", 1000.0)
        for combo in COMBOS}
# x minimum shift: steps of 0.05 up to 2, then 0.1 up to 4 (the min_x2 set
# and low-statistics INTT z only turn on above 2).
_MEAN_HALF = np.concatenate([np.arange(0, 41) * 0.05, 2.0 + np.arange(1, 21) * 0.1])
MEAN_GRID = np.round(np.unique(np.concatenate([-_MEAN_HALF, _MEAN_HALF])), 2)
WIDTH_GRID = np.round(np.arange(70, 141) * 0.01, 2)  # sigma_Cmp / sigma_Ref
DEFAULT_ENTRIES = "15,20,40,100,300,1000,nominal"
# make_synthetic_fit_tables: fit errors use 0.95 N entries, chip widths
# spread uniformly by 0.9-1.1 around the nominal width.
CORE_FRACTION = 0.95
CHIP_SIGMA_SPREAD = np.linspace(0.9, 1.1, 41)
WILSON_Z = 1.0  # 68.3% intervals

# Fit errors of the toy relative to the ideal ones (mu_err = sigma /
# sqrt(0.95 N), sigma_err = sigma / sqrt(1.9 N)), per (detector, coord):
# (mu factor, sigma factor). Measured by main() from the committed ROOT
# reference unless --ideal-errors; empty = ideal errors.
REFERENCE_TABLE = REPO_ROOT / "synthetic" / "reference" / "residual_fits_Iter0.csv"
ERROR_SCALE: dict = {}


def measured_error_scale(path: Path = REFERENCE_TABLE) -> dict:
    """Median ratio of the ROOT-fit errors to the ideal ones, per (detector,
    coord), over the chip rows with fit_status ok of a fit table written by
    the fit macro from synthetic ntuples (Gaussian core refit, INTT z box)."""
    table = pd.read_csv(path, dtype={"detector": str, "coord": str,
                                     "fit_status": str})
    table = table[(table["fit_status"] == "ok") & (table["chip"] >= 0)
                  & table["detector"].isin(list(GEOMETRY))]
    core = CORE_FRACTION * table["entries"]
    mu = table["mu_err_mm"] / (table["sigma_mm"] / np.sqrt(core))
    sigma = table["sigma_err_mm"] / (table["sigma_mm"] / np.sqrt(2.0 * core))
    return {(detector, coord): (float(mu[index].median()),
                                float(sigma[index].median()))
            for (detector, coord), index in
            table.groupby(["detector", "coord"]).groups.items()}


def error_factors(combo) -> tuple[float, float]:
    return ERROR_SCALE.get(combo, (1.0, 1.0)) if combo else (1.0, 1.0)


def error_model_text() -> str:
    if not ERROR_SCALE:
        return ("ideal (mu_err = sigma / sqrt(0.95 N), sigma_err = sigma / "
                "sqrt(1.9 N)).")
    return ("ideal errors scaled by the factors measured from the ROOT fits "
            "of synthetic/reference (mean, width): " + "; ".join(
                f"{COMBO_NAME[combo]} {mu:.2f}, {sigma:.2f}"
                for combo, (mu, sigma) in sorted(ERROR_SCALE.items()))
            + ".")

# "default" = the compare_runs defaults (checked against its flags on every
# comparison); the width variants differ only in the width test, the mean
# variants only in the mean test.
THRESHOLD_SETS = {
    "default": dict(z=1.96, min_factor=1.0, width=0.10, width_sig=3.0),
    "min_x0.5": dict(z=1.96, min_factor=0.5, width=0.10, width_sig=3.0),
    "min_x2": dict(z=1.96, min_factor=2.0, width=0.10, width_sig=3.0),
    "z3": dict(z=3.0, min_factor=1.0, width=0.10, width_sig=3.0),
    "w0.20": dict(z=1.96, min_factor=1.0, width=0.20, width_sig=None),
    "w0.15": dict(z=1.96, min_factor=1.0, width=0.15, width_sig=None),
    "w0.10": dict(z=1.96, min_factor=1.0, width=0.10, width_sig=None),
    "w0.20_sig3": dict(z=1.96, min_factor=1.0, width=0.20, width_sig=3.0),
}
SET_NAMES = list(THRESHOLD_SETS)
MEAN_SETS = ["default", "min_x0.5", "min_x2", "z3"]
WIDTH_SETS = ["default", "w0.20", "w0.15", "w0.10", "w0.20_sig3"]
SET_DESCRIPTION = {
    "default": "z > 1.96 and |dmu| > min shift; |dsigma/sigma| > 0.10 "
               "and width significance > 3",
    "min_x0.5": "minimum shift x 0.5",
    "min_x2": "minimum shift x 2",
    "z3": "z > 3 instead of 1.96",
    "w0.20": "width cut 20%, no significance (old default)",
    "w0.15": "width cut 15%, no significance",
    "w0.10": "width cut 10%, no significance",
    "w0.20_sig3": "width cut 20% and width significance > 3",
}

# Entries test (compare_runs.entries_rule; variants of its thresholds).
ENTRIES_GRID = np.round(np.arange(80, 121) * 0.01, 2)  # N_Cmp / N_Ref expected
ENTRIES_REL_CUT = 0.05
NORMALIZATIONS = ["raw", "detector", "layer"]
ENTRIES_Z_CUTS = [None, 1.96, 3.0]
DEFAULT_EVENT_RATIOS = "1.00,0.90"
ACCEPTANCE_SPREAD = (0.8, 1.2)
INJECTED_CHIP = {"MVTX": 4, "INTT": 1}  # one chip per stave / ladder


def entries_set_name(normalization: str, z_cut) -> str:
    return f"{normalization}_rel5" + ("" if z_cut is None else f"_z{z_cut:g}")


ENTRIES_SETS = [(entries_set_name(norm, z_cut), norm, z_cut)
                for norm in NORMALIZATIONS for z_cut in ENTRIES_Z_CUTS]


# --------------------------------------------------------------- the toy ---

def injections_for(scan: str, value: float, entries: int | None) -> list:
    """One injection per (detector, layer[, coord]) covering every chip."""
    injections = []
    for detector, (layers, _, _) in GEOMETRY.items():
        for layer in layers:
            base = dict(labels=["Cmp"], detector=detector, layer=layer,
                        staves=None, chips=None, description="scan")
            if scan == "mean":
                for coord in ("rphi", "z"):
                    injections.append(dict(
                        base, name=f"mean_{detector}_L{layer}_{coord}",
                        coord=coord, offset_mm=value * DEFAULT_MIN_SHIFT_MM[
                            (detector, coord)]))
            else:
                injections.append(dict(
                    base, name=f"width_{detector}_L{layer}", coord=None,
                    sigma_scale=value))
    if entries is not None:
        # Entry overrides apply to both labels (make_synthetic_fit_tables).
        for injection in injections:
            injection["entries"] = entries
    return injections


def apply_rule(flags: pd.DataFrame, settings: dict):
    """The compare_runs rule with the given thresholds, on its own columns."""
    comparable = flags["comparable"].to_numpy(bool)
    minimum = flags["min_shift_mm"].to_numpy(float) * settings["min_factor"]
    with np.errstate(invalid="ignore"):
        mean = (comparable
                & (flags["mu_z"].to_numpy(float) > settings["z"])
                & (np.abs(flags["delta_mu_mm"].to_numpy(float)) > minimum))
        width = (comparable & (np.abs(flags["sigma_rel"].to_numpy(float))
                               > settings["width"]))
        if settings["width_sig"] is not None:
            width &= (flags["sigma_rel_significance"].to_numpy(float)
                      > settings["width_sig"])
    return mean, width


def run_scan(scan: str, grid: np.ndarray, entries: list, seeds: int,
             seed_counter) -> dict:
    """counts[(entries_label, value)] = array[set, combo, (trials, mean,
    width, any)] summed over seeds."""
    combo_lookup = {combo: index for index, combo in enumerate(COMBOS)}
    defaults = THRESHOLD_SETS["default"]
    counts = {}
    for label, n_entries in entries:
        start = time.time()
        for value in grid:
            injections = injections_for(scan, float(value), n_entries)
            total = np.zeros((len(SET_NAMES), len(COMBOS), 4))
            for _ in range(seeds):
                tables = build_tables(next(seed_counter), injections, LABELS,
                                      error_scale=ERROR_SCALE or None)
                flags = compare_tables(
                    tables["Ref"], tables["Cmp"], defaults["width"],
                    defaults["z"], DEFAULT_MIN_SHIFT_MM,
                    width_significance_threshold=defaults["width_sig"])
                flags = flags[flags["chip"] >= 0]
                combo = np.array([combo_lookup[(detector, coord)] for
                                  detector, coord in zip(flags["detector"],
                                                         flags["coord"])])
                trial = flags["comparable"].to_numpy(bool)
                for set_index, name in enumerate(SET_NAMES):
                    mean, width = apply_rule(flags, THRESHOLD_SETS[name])
                    if name == "default" and not (
                            np.array_equal(mean, flags["flag_mean"].to_numpy(bool))
                            and np.array_equal(width,
                                               flags["flag_width"].to_numpy(bool))):
                        raise RuntimeError(
                            "re-applied default rule differs from "
                            "compare_tables; the scan would be wrong")
                    for column, mask in enumerate((trial, mean, width,
                                                   mean | width)):
                        total[set_index, :, column] += np.bincount(
                            combo, weights=mask.astype(float),
                            minlength=len(COMBOS))
            counts[(label, float(value))] = total
        print(f"  {scan:5s} N={label:>7s}: {len(grid)} points x {seeds} "
              f"seeds in {time.time() - start:.0f} s", flush=True)
    return counts


# ------------------------------------------------------------ statistics ---

_erfc = np.vectorize(math.erfc)


def phi(x):
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def wilson(k, n, z: float = WILSON_Z):
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = k / n
        denominator = 1.0 + z * z / n
        centre = (p + z * z / (2.0 * n)) / denominator
        half = z * np.sqrt(p * (1.0 - p) / n
                           + z * z / (4.0 * n * n)) / denominator
    return p, centre - half, centre + half


def analytic_mean(delta_mm, combo, n_entries: float, settings: dict):
    """P(mean test fires) for true shift delta, averaged over chip widths."""
    delta = np.atleast_1d(np.asarray(delta_mm, float))[:, None]
    sigma = NOMINAL_SIGMA_MM[combo] * CHIP_SIGMA_SPREAD[None, :]
    sigma_delta = (math.sqrt(2.0) * sigma * error_factors(combo)[0]
                   / math.sqrt(CORE_FRACTION * n_entries))
    threshold = np.maximum(settings["min_factor"] * DEFAULT_MIN_SHIFT_MM[combo],
                           settings["z"] * sigma_delta)
    probability = (phi((delta - threshold) / sigma_delta)
                   + phi((-delta - threshold) / sigma_delta))
    return probability.mean(axis=1)


def analytic_width(scale, n_entries: float, settings: dict, combo=None):
    """P(width test fires) for true scale k.

    The measured ratio r = sigma2/sigma1 fires above T_up or below T_down.
    With a significance cut s, the significance is about |r - 1| / (r /
    sqrt(c)) (c = 0.95 N), so it also needs r > 1 / (1 - s/sqrt(c)) or
    r < 1 / (1 + s/sqrt(c)); the stricter bound of each side applies."""
    scale = np.atleast_1d(np.asarray(scale, float))
    # Width errors scaled by f behave like f^2 times fewer entries.
    core = CORE_FRACTION * n_entries / error_factors(combo)[1] ** 2
    cut = settings["width"]
    t_up, t_down = 1.0 + cut, 1.0 - cut
    if settings["width_sig"] is not None:
        step = settings["width_sig"] / math.sqrt(core)
        t_up = max(t_up, 1.0 / (1.0 - step)) if step < 1.0 else math.inf
        t_down = min(t_down, 1.0 / (1.0 + step))
    upper = (np.zeros_like(scale) if math.isinf(t_up) else
             phi((scale - t_up)
                 / np.sqrt((scale ** 2 + t_up ** 2) / (2.0 * core))))
    lower = phi((t_down - scale)
                / np.sqrt((scale ** 2 + t_down ** 2) / (2.0 * core)))
    return upper + lower


def crossing(x, y, level: float) -> float:
    """First x at which y reaches level (linear interpolation); NaN if never."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    above = np.nonzero(y >= level)[0]
    if len(above) == 0:
        return math.nan
    i = above[0]
    if i == 0:
        return float(x[0])
    x0, x1, y0, y1 = x[i - 1], x[i], y[i - 1], y[i]
    return float(x0 + (level - y0) * (x1 - x0) / (y1 - y0))


# ------------------------------------------- entries test (compare_runs) ---

def entries_rule(n1, n2, t1, t2, rel_cut: float = ENTRIES_REL_CUT,
                 z_cut: float | None = None):
    """The compare_runs entries rule; z_cut None = no significance
    requirement (z > 0 holds whenever |rel| > rel_cut)."""
    return compare_runs_entries_rule(n1, n2, t1, t2, rel_cut,
                                     0.0 if z_cut is None else z_cut)


def entries_geometry():
    """One row per MVTX chip / INTT sensor: detector, layer, injected flag."""
    detector, layer, injected = [], [], []
    for name, (layers, staves, chips) in GEOMETRY.items():
        for layer_id, stave_count in zip(layers, staves):
            for _ in range(stave_count):
                for chip in range(chips):
                    detector.append(name)
                    layer.append(layer_id)
                    injected.append(chip == INJECTED_CHIP[name])
    return np.array(detector), np.array(layer), np.array(injected)


def group_totals(counts, keys):
    """Sum of counts over each group of keys, returned per row."""
    _, inverse = np.unique(keys, return_inverse=True)
    return np.bincount(inverse, weights=counts)[inverse]


def run_entries_scan(entries: list, event_ratios: list, seeds: int,
                     first_seed: int) -> pd.DataFrame:
    """Counting toy for the entries test; one row per set x detector x N x
    event ratio x scale x group (injected or control chips)."""
    detector, layer, injected = entries_geometry()
    layer_key = np.char.add(detector, layer.astype(str))
    detectors = list(GEOMETRY)
    rng = np.random.default_rng(first_seed)
    rows = []
    for label, n_entries in entries:
        nominal = np.array([float(NOMINAL_ENTRIES[d]) if n_entries is None
                            else float(n_entries) for d in detector])
        start = time.time()
        for ratio in event_ratios:
            for scale in ENTRIES_GRID:
                counts = {}  # (set, detector, group) -> [trials, fired]
                for _ in range(seeds):
                    acceptance = rng.uniform(*ACCEPTANCE_SPREAD, len(detector))
                    expected1 = nominal * acceptance
                    expected2 = expected1 * ratio * np.where(injected, scale, 1.0)
                    n1 = rng.poisson(expected1).astype(float)
                    n2 = rng.poisson(expected2).astype(float)
                    totals = {"raw": (np.ones_like(n1), np.ones_like(n2)),
                              "detector": (group_totals(n1, detector),
                                           group_totals(n2, detector)),
                              "layer": (group_totals(n1, layer_key),
                                        group_totals(n2, layer_key))}
                    for name, norm, z_cut in ENTRIES_SETS:
                        t1, t2 = totals[norm]
                        fires, _, _ = entries_rule(n1, n2, t1, t2,
                                                   ENTRIES_REL_CUT, z_cut)
                        for det in detectors:
                            for group, mask in (("injected", injected),
                                                ("control", ~injected)):
                                select = (detector == det) & mask
                                total = counts.setdefault((name, det, group),
                                                          [0, 0])
                                total[0] += int(select.sum())
                                total[1] += int((fires & select).sum())
                for (name, det, group), (trials, fired) in counts.items():
                    norm, z_cut = next((n, z) for s, n, z in ENTRIES_SETS
                                       if s == name)
                    n_mean = (float(NOMINAL_ENTRIES[det]) if n_entries is None
                              else float(n_entries))
                    p, low, high = wilson(fired, trials)
                    rows.append(dict(
                        scan="entries", threshold_set=name, normalization=norm,
                        z_cut=z_cut, detector=det, entries=label,
                        entries_per_chip=n_mean, event_ratio=ratio,
                        entries_scale=float(scale), group=group,
                        trials=trials, n_flagged=fired, eff=float(p),
                        eff_lo=float(low), eff_hi=float(high),
                        analytic=analytic_entries(
                            n_mean, scale if group == "injected" else 1.0,
                            scale, ratio, det, norm, z_cut)))
        print(f"  entries N={label:>7s}: {len(event_ratios)} event ratios x "
              f"{len(ENTRIES_GRID)} points x {seeds} seeds in "
              f"{time.time() - start:.0f} s", flush=True)
    return pd.DataFrame(rows)


def analytic_entries(n_mean: float, chip_scale: float, injected_scale: float,
                     ratio: float, det: str, normalization: str,
                     z_cut) -> float:
    """Normal approximation to P(entries test fires) for one chip whose
    expected count is scaled by chip_scale (1 for a control chip), averaged
    over the chip acceptance spread. In units of reference counts:
    d = lam2 / R - lam1, s = sqrt(lam1 + lam2 / R^2), threshold
    max(cut * lam1, z_cut * s), with R the expected ratio of the comparison to
    the reference total (1 for raw)."""
    acceptance = np.linspace(*ACCEPTANCE_SPREAD, 41)
    lam1 = n_mean * acceptance
    lam2 = lam1 * ratio * chip_scale
    if normalization == "raw":
        total_ratio = 1.0
    else:
        # One injected chip per stave or ladder in every layer, so the
        # detector and layer totals move by the same expected factor.
        chips = GEOMETRY[det][2]
        total_ratio = ratio * (1.0 + (injected_scale - 1.0) / chips)
    d = lam2 / total_ratio - lam1
    s = np.sqrt(lam1 + lam2 / total_ratio ** 2)
    threshold = np.maximum(ENTRIES_REL_CUT * lam1,
                           0.0 if z_cut is None else z_cut * s)
    return float(np.mean(phi((d - threshold) / s) + phi((-d - threshold) / s)))


# ---------------------------------------------------------------- tables ---

def entries_value(label: str, combo) -> float:
    return float(NOMINAL_ENTRIES[combo[0]]) if label == "nominal" else float(label)


def to_frame(scan: str, counts: dict) -> pd.DataFrame:
    rows = []
    for (label, value), total in counts.items():
        for set_index, name in enumerate(SET_NAMES):
            settings = THRESHOLD_SETS[name]
            for combo_index, combo in enumerate(COMBOS):
                trials, n_mean, n_width, n_any = total[set_index, combo_index]
                n_entries = entries_value(label, combo)
                row = dict(scan=scan, threshold_set=name, detector=combo[0],
                           coord=combo[1], entries=label,
                           entries_per_chip=n_entries, trials=int(trials),
                           n_flag_mean=int(n_mean), n_flag_width=int(n_width),
                           n_flagged=int(n_any))
                if scan == "mean":
                    shift = value * DEFAULT_MIN_SHIFT_MM[combo]
                    row.update(shift_over_min=value, injected_shift_mm=shift,
                               analytic_mean=float(analytic_mean(
                                   shift, combo, n_entries, settings)[0]))
                else:
                    row.update(sigma_scale=value,
                               analytic_width=float(analytic_width(
                                   value, n_entries, settings, combo)[0]))
                for test, k in (("mean", n_mean), ("width", n_width)):
                    p, low, high = wilson(k, trials)
                    row.update({f"eff_{test}": float(p),
                                f"eff_{test}_lo": float(low),
                                f"eff_{test}_hi": float(high)})
                row["eff_flagged"] = n_any / trials if trials else math.nan
                rows.append(row)
    return pd.DataFrame(rows)


def entries_order(labels: list[str]) -> list[str]:
    return sorted(labels, key=lambda label: math.inf if label == "nominal"
                  else float(label))


def fmt(value_mm: float, combo, digits: int = 2) -> str:
    if not np.isfinite(value_mm):
        return "not reached"
    unit, factor = UNIT[combo]
    return f"{value_mm * factor:.{digits}f}"


def fmt_rate(value: float) -> str:
    return "-" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def mean_points(frame: pd.DataFrame, name: str, combo, label: str):
    """Toy and analytic 50%/90% points (mm, positive side) and false rate."""
    part = frame[(frame["threshold_set"] == name)
                 & (frame["detector"] == combo[0]) & (frame["coord"] == combo[1])
                 & (frame["entries"] == label)].sort_values("shift_over_min")
    positive = part[part["shift_over_min"] >= 0]
    x, y = positive["injected_shift_mm"], positive["eff_mean"]
    fine = np.linspace(0.0, x.max(), 4001)
    analytic = analytic_mean(fine, combo, entries_value(label, combo),
                             THRESHOLD_SETS[name])
    false_rate = part.loc[part["shift_over_min"] == 0, "eff_mean"].iloc[0]
    return dict(false=false_rate,
                toy50=crossing(x, y, 0.5), an50=crossing(fine, analytic, 0.5),
                toy90=crossing(x, y, 0.9), an90=crossing(fine, analytic, 0.9))


def width_points(frame: pd.DataFrame, name: str, label: str):
    """Pooled over detectors and coordinates (the width rule is relative)."""
    part = frame[(frame["threshold_set"] == name)
                 & (frame["entries"] == label)]
    pooled = part.groupby("sigma_scale").agg(
        trials=("trials", "sum"), k=("n_flag_width", "sum"),
        k_any=("n_flagged", "sum")).reset_index()
    weights = part.pivot_table(index="sigma_scale", columns=["detector", "coord"],
                               values="trials", aggfunc="sum")
    analytic_cols = part.pivot_table(index="sigma_scale",
                                     columns=["detector", "coord"],
                                     values="analytic_width", aggfunc="first")
    pooled["analytic"] = ((analytic_cols * weights).sum(axis=1, min_count=1)
                          / weights.sum(axis=1)).to_numpy()
    pooled["eff"], pooled["lo"], pooled["hi"] = wilson(pooled["k"],
                                                       pooled["trials"])
    up = pooled[pooled["sigma_scale"] >= 1.0]
    down = pooled[pooled["sigma_scale"] <= 1.0].iloc[::-1]
    null = pooled[pooled["sigma_scale"] == 1.0].iloc[0]
    fine_up = np.linspace(1.0, WIDTH_GRID.max(), 2001)
    fine_down = np.linspace(1.0, WIDTH_GRID.min(), 2001)
    settings = THRESHOLD_SETS[name]

    def analytic_curve(scales):
        values = [analytic_width(scales, entries_value(label, combo), settings,
                                 combo)
                  for combo in COMBOS]
        return np.mean(values, axis=0)

    curve_up, curve_down = analytic_curve(fine_up), analytic_curve(fine_down)
    return dict(
        pooled=pooled, false=null["eff"], false_lo=null["lo"],
        false_hi=null["hi"], false_an=null["analytic"],
        false_any=null["k_any"] / null["trials"],
        up50=crossing(up["sigma_scale"], up["eff"], 0.5),
        up90=crossing(up["sigma_scale"], up["eff"], 0.9),
        down50=crossing(down["sigma_scale"], down["eff"], 0.5),
        down90=crossing(down["sigma_scale"], down["eff"], 0.9),
        an_up50=crossing(fine_up, curve_up, 0.5),
        an_up90=crossing(fine_up, curve_up, 0.9),
        an_down50=crossing(fine_down, curve_down, 0.5),
        an_down90=crossing(fine_down, curve_down, 0.9))


def fmt_scale(value: float) -> str:
    return "not reached" if not np.isfinite(value) else f"{value:.3f}"


def markdown_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(cell).replace("|", "\\|")
                                for cell in row) + " |"
              for row in rows]
    return "\n".join(lines)


def build_summary(mean_frame: pd.DataFrame, width_frame: pd.DataFrame,
                  args, labels: list[str]) -> str:
    out = ["# Flag sensitivity of compare_runs (level 1: fit tables)", ""]
    out += [
        f"Seeds per point: {args.seeds} (first seed {args.first_seed}). "
        "Each point injects the same change into every MVTX chip (432 per "
        "coordinate) and INTT sensor (224 per coordinate). Efficiencies are "
        "toy fractions; intervals in the CSV files are 68% Wilson. Fit "
        "errors: " + error_model_text() + " Chip widths spread uniformly by "
        "0.9-1.1. 'nominal' is a Poisson "
        f"mean of {NOMINAL_ENTRIES['MVTX']} entries (MVTX) and "
        f"{NOMINAL_ENTRIES['INTT']} (INTT) per chip.",
        "",
        "This measures the decision rule with the fit precision above. Fit "
        "biases of the ROOT macro (level 2) and the track fit's response to "
        "a real displacement (biased residuals, weak modes) are not "
        "included.", "",
        "## Threshold sets", "",
        markdown_table(["set", "meaning"],
                       [[name, SET_DESCRIPTION[name]] for name in SET_NAMES]),
        "", "## Mean test", "",
        "Points are on the positive side, for the mean test alone "
        "(`flag_mean`). The false rate is the mean-test flag rate with no "
        "injected shift.", ""]

    for combo in COMBOS:
        unit = UNIT[combo][0]
        rows = []
        for label in labels:
            p = mean_points(mean_frame, "default", combo, label)
            rows.append([label, fmt_rate(p["false"]),
                         fmt(p["toy50"], combo), fmt(p["an50"], combo),
                         fmt(p["toy90"], combo), fmt(p["an90"], combo)])
        out += [f"### {COMBO_NAME[combo]}, default thresholds "
                f"(min shift {fmt(DEFAULT_MIN_SHIFT_MM[combo], combo)} {unit})",
                "", markdown_table(
                    ["N per chip", "false rate", f"50% toy [{unit}]",
                     f"50% analytic [{unit}]", f"90% toy [{unit}]",
                     f"90% analytic [{unit}]"], rows), ""]

    out += ["### 90% point across mean threshold sets (toy)", ""]
    rows = []
    for combo in COMBOS:
        for label in labels:
            rows.append([COMBO_NAME[combo], label] + [
                fmt(mean_points(mean_frame, name, combo, label)["toy90"], combo)
                for name in MEAN_SETS])
    out += [markdown_table(["detector, coord", "N per chip"]
                           + MEAN_SETS, rows), "",
            "Units: um, except INTT z in mm.", ""]

    asymmetry = []
    default = mean_frame[mean_frame["threshold_set"] == "default"]
    for combo in COMBOS:
        part = default[(default["detector"] == combo[0])
                       & (default["coord"] == combo[1])]
        table = part.pivot_table(index="shift_over_min", columns="entries",
                                 values="eff_mean")
        positive = table.loc[table.index > 0]
        negative = table.loc[-positive.index]
        asymmetry.append(float(np.nanmax(np.abs(positive.to_numpy()
                                                - negative.to_numpy()))))
    out += [f"Largest |efficiency(+shift) - efficiency(-shift)| over the grid "
            f"(default): {max(asymmetry):.3f}.", ""]

    out += ["## Width test", "",
            "Pooled over detectors and coordinates: the width rule is relative, "
            "so it depends only on the entries per chip. The false rate is the "
            "width-test flag rate with no injected change (scale 1.00); "
            "'any test' adds the mean test (default thresholds).", ""]
    rows = []
    for label in labels:
        row = [label]
        for name in WIDTH_SETS:
            p = width_points(width_frame, name, label)
            row.append(fmt_rate(p["false"]))
        default_point = width_points(width_frame, "default", label)
        row += [fmt_rate(default_point["false_an"]),
                fmt_rate(default_point["false_any"])]
        rows.append(row)
    out += ["### False width-flag rate at scale 1.00 (toy)", "",
            markdown_table(["N per chip"] + WIDTH_SETS
                           + ["default analytic", "default, any test"], rows),
            ""]

    rows = []
    for label in labels:
        p = width_points(width_frame, "default", label)
        rows.append([label, fmt_scale(p["up50"]), fmt_scale(p["an_up50"]),
                     fmt_scale(p["up90"]), fmt_scale(p["an_up90"]),
                     fmt_scale(p["down50"]), fmt_scale(p["an_down50"]),
                     fmt_scale(p["down90"]), fmt_scale(p["an_down90"])])
    out += ["### Width scale at 50% and 90% efficiency, default (10% and significance > 3)", "",
            markdown_table(["N per chip", "up 50% toy", "up 50% an.",
                            "up 90% toy", "up 90% an.", "down 50% toy",
                            "down 50% an.", "down 90% toy", "down 90% an."],
                           rows), ""]

    rows = []
    for label in labels:
        row = [label]
        for name in WIDTH_SETS:
            p = width_points(width_frame, name, label)
            row.append(f"{fmt_scale(p['down90'])} / {fmt_scale(p['up90'])}")
        rows.append(row)
    out += ["### Width scale at 90% efficiency across width sets "
            "(narrower / broader, toy)", "",
            markdown_table(["N per chip"] + WIDTH_SETS, rows), "",
            "A scale reached at 1.000 means the test already fires at 90% "
            "with no change: the false rate dominates.", ""]
    return "\n".join(out)


def entries_points(frame: pd.DataFrame, name: str, det: str, label: str,
                   ratio: float):
    """Scale at 50% / 90% efficiency on the injected chip, loss and gain side."""
    part = frame[(frame["threshold_set"] == name) & (frame["detector"] == det)
                 & (frame["entries"] == label)
                 & np.isclose(frame["event_ratio"], ratio)
                 & (frame["group"] == "injected")].sort_values("entries_scale")
    up = part[part["entries_scale"] >= 1.0]
    down = part[part["entries_scale"] <= 1.0].iloc[::-1]
    return {f"{side}{level}": crossing(data["entries_scale"], data["eff"],
                                       level / 100.0)
            for side, data in (("down", down), ("up", up))
            for level in (50, 90)}


def entries_false(frame: pd.DataFrame, name: str, label: str, ratio: float,
                  det: str | None = None):
    """Flag rate of unchanged control chips at scale 1 (MVTX + INTT pooled
    unless det is given)."""
    part = frame[(frame["threshold_set"] == name) & (frame["entries"] == label)
                 & np.isclose(frame["event_ratio"], ratio)
                 & (frame["group"] == "control")
                 & np.isclose(frame["entries_scale"], 1.0)]
    if det is not None:
        part = part[part["detector"] == det]
    trials, fired = part["trials"].sum(), part["n_flagged"].sum()
    p, low, high = wilson(fired, trials)
    return float(p), float(low), float(high)


def build_entries_summary(frame: pd.DataFrame, labels: list[str],
                          ratios: list[float]) -> str:
    names = [name for name, _, _ in ENTRIES_SETS]
    out = ["## Entries test (compare_runs default: raw, 5% and z > 3)", "",
           "Counting toy: one Poisson count per chip and input, chip "
           "acceptance spread 0.8-1.2, one injected chip per stave (MVTX "
           "chip 4) or ladder (INTT sensor 1); the other chips are controls. "
           "Event ratio = events in the comparison / events in the reference. "
           "Set names: normalization (raw, detector share, layer share), "
           "`rel5` = |a2/a1 - 1| > 5%, `zX` = and z > X.", ""]
    big = frame[frame["trials"] >= 500]
    deviation = np.nanmax(np.abs(big["eff"] - big["analytic"]))
    out += [f"Largest |toy - analytic| on points with >= 500 trials: "
            f"{deviation:.3f} (analytic = normal approximation).", ""]
    for ratio in ratios:
        rows = [[label] + [fmt_rate(entries_false(frame, name, label, ratio)[0])
                           for name in names] for label in labels]
        out += [f"### False-flag rate on unchanged chips, event ratio "
                f"{ratio:.2f} (MVTX + INTT pooled)", "",
                markdown_table(["N per chip"] + names, rows), ""]
    first = ratios[0]
    rows = []
    for det in GEOMETRY:
        for label in labels:
            row = [det, label]
            for name in names:
                if entries_false(frame, name, label, first, det)[0] > 0.5:
                    row.append("noise")
                    continue
                p = entries_points(frame, name, det, label, first)
                row.append(f"{fmt_scale(p['down90'])} / {fmt_scale(p['up90'])}")
            rows.append(row)
    out += [f"### Entries scale flagged 90% of the time (loss / gain), event "
            f"ratio {first:.2f}", "",
            markdown_table(["detector", "N per chip"] + names, rows), "",
            "'noise' = more than half of the unchanged chips of that detector "
            "already fire, so no 90% point is meaningful. 'not reached' = "
            "below 90% over the whole 0.80-1.20 range.", ""]
    return "\n".join(out)


# ----------------------------------------------------------------- plots ---

SURFACE, INK, INK2, MUTED, GRID = ("#fcfcfb", "#0b0b0b", "#52514e",
                                   "#898781", "#e1e0d9")
# Validated with the dataviz palette script: ordinal blue ramp (5 steps,
# light -> dark = few -> many entries) and the first 5 categorical slots.
ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
# Second encoding for the threshold sets: coinciding curves stay visible
# (largest marker drawn first, smaller ones on top).
MARKERS = ["o", "s", "^", "D", "v"]
MARKER_SIZE = {"o": 9, "s": 7, "^": 7, "D": 5, "v": 5}


def plot_labels(labels: list[str]) -> list[str]:
    if len(labels) <= len(ORDINAL):
        return labels
    picks = np.unique(np.round(np.linspace(0, len(labels) - 1,
                                           len(ORDINAL))).astype(int))
    return [labels[i] for i in picks]


def style(ax, plt) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(INK2)
    ax.yaxis.label.set_color(INK2)


def error_bars(value, low, high):
    """Asymmetric error bars, clipped at 0 (rounding at p = 0 or 1)."""
    value, low, high = (np.asarray(array, float) for array in (value, low, high))
    return [np.clip(value - low, 0.0, None), np.clip(high - value, 0.0, None)]


def entries_legend_label(label: str) -> str:
    return (f"N = nominal ({NOMINAL_ENTRIES['MVTX']} / "
            f"{NOMINAL_ENTRIES['INTT']})" if label == "nominal"
            else f"N = {label}")


def make_plots(mean_frame, width_frame, labels, out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    written = []
    shown = plot_labels(labels)
    errorbar_style = dict(fmt="o", ms=4, elinewidth=1.0, capsize=0,
                          markeredgecolor=SURFACE, markeredgewidth=0.6)

    def curve_legend(fig):
        handles = [Line2D([], [], color=color, marker="o", ms=5, lw=1.5,
                          label=entries_legend_label(label))
                   for label, color in zip(shown, ORDINAL)]
        handles.append(Line2D([], [], color=INK2, marker="o", ms=0, lw=0,
                              label="points: toy (68% Wilson); "
                                    "lines: analytic"))
        legend = fig.legend(handles=handles, loc="lower center",
                            ncol=len(handles), frameon=False, fontsize=8)
        for text in legend.get_texts():
            text.set_color(INK2)

    # Mean efficiency, default thresholds.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    default = mean_frame[mean_frame["threshold_set"] == "default"]
    for ax, combo in zip(axes.flat, COMBOS):
        unit, factor = UNIT[combo]
        part = default[(default["detector"] == combo[0])
                       & (default["coord"] == combo[1])
                       & (default["shift_over_min"] >= 0)]
        fine = np.linspace(0.0, part["injected_shift_mm"].max(), 801)
        for label, color in zip(shown, ORDINAL):
            points = part[part["entries"] == label].sort_values(
                "shift_over_min")
            ax.plot(fine * factor, analytic_mean(
                fine, combo, entries_value(label, combo),
                THRESHOLD_SETS["default"]), color=color, lw=1.5)
            ax.errorbar(points["injected_shift_mm"] * factor,
                        points["eff_mean"],
                        yerr=error_bars(points["eff_mean"],
                                        points["eff_mean_lo"],
                                        points["eff_mean_hi"]),
                        color=color, ecolor=color, **errorbar_style)
        minimum = DEFAULT_MIN_SHIFT_MM[combo] * factor
        ax.axvline(minimum, color=MUTED, lw=0.8, ls="--")
        ax.text(minimum, 1.03, " min shift", color=MUTED, fontsize=8,
                ha="left", va="bottom")
        ax.set_ylim(-0.02, 1.08)
        ax.set_title(COMBO_NAME[combo], color=INK, fontsize=10, loc="left")
        ax.set_xlabel(f"Injected mean shift [{unit}]")
        ax.set_ylabel("Flag efficiency (mean test)")
        style(ax, plt)
    fig.suptitle("compare_runs mean test, default thresholds "
                 "(z > 1.96 and |Δμ| > min shift)", color=INK, fontsize=11)
    curve_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = out_dir / "mean_efficiency.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)

    # Width efficiency, default thresholds.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    default = width_frame[width_frame["threshold_set"] == "default"]
    fine = np.linspace(WIDTH_GRID.min(), WIDTH_GRID.max(), 701)
    for ax, combo in zip(axes.flat, COMBOS):
        part = default[(default["detector"] == combo[0])
                       & (default["coord"] == combo[1])]
        for label, color in zip(shown, ORDINAL):
            points = part[part["entries"] == label].sort_values("sigma_scale")
            ax.plot(fine, analytic_width(fine, entries_value(label, combo),
                                         THRESHOLD_SETS["default"], combo),
                    color=color, lw=1.5)
            ax.errorbar(points["sigma_scale"], points["eff_width"],
                        yerr=error_bars(points["eff_width"],
                                        points["eff_width_lo"],
                                        points["eff_width_hi"]),
                        color=color, ecolor=color, **errorbar_style)
        for cut in (0.9, 1.1):
            ax.axvline(cut, color=MUTED, lw=0.8, ls="--")
        ax.text(1.1, 1.03, " ±10% cut", color=MUTED, fontsize=8,
                ha="left", va="bottom")
        ax.set_ylim(-0.02, 1.08)
        ax.set_title(COMBO_NAME[combo], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Injected width scale σ_Cmp / σ_Ref")
        ax.set_ylabel("Flag efficiency (width test)")
        style(ax, plt)
    fig.suptitle("compare_runs width test, default threshold "
                 "(|σ₂/σ₁ − 1| > 0.10 and significance > 3)", color=INK, fontsize=11)
    curve_legend(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = out_dir / "width_efficiency.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)

    # 90% point of the mean test vs N, per threshold set.
    positions = np.arange(len(labels))
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), facecolor=SURFACE)
    for ax, combo in zip(axes.flat, COMBOS):
        unit, factor = UNIT[combo]
        for name, color, marker in zip(MEAN_SETS, CATEGORICAL, MARKERS):
            values = [mean_points(mean_frame, name, combo, label)["toy90"]
                      * factor for label in labels]
            ax.plot(positions, values, color=color, lw=1.5, marker=marker,
                    ms=MARKER_SIZE[marker], markeredgecolor=SURFACE,
                    markeredgewidth=0.6,
                    label=f"{name}: {SET_DESCRIPTION[name]}")
        ax.set_xticks(positions, labels)
        ax.set_ylim(bottom=0)
        ax.set_title(COMBO_NAME[combo], color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Entries per chip")
        ax.set_ylabel(f"Shift at 90% efficiency [{unit}]")
        style(ax, plt)
    handles, names = axes.flat[0].get_legend_handles_labels()
    legend = fig.legend(handles, names, loc="lower center", ncol=2,
                        frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK2)
    fig.suptitle("Mean test: smallest shift flagged 90% of the time "
                 "(toy; not reached within 4 x min shift is not drawn)",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.97))
    path = out_dir / "mean_thresholds.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)

    # False width-flag rate vs N, per width threshold set.
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    for name, color, marker in zip(WIDTH_SETS, CATEGORICAL, MARKERS):
        points = [width_points(width_frame, name, label) for label in labels]
        rate = np.array([p["false"] for p in points]) * 100.0
        low = np.array([p["false_lo"] for p in points]) * 100.0
        high = np.array([p["false_hi"] for p in points]) * 100.0
        ax.errorbar(positions, rate, yerr=error_bars(rate, low, high),
                    color=color, ecolor=color, lw=1.5, marker=marker,
                    ms=MARKER_SIZE[marker],
                    elinewidth=1.0, capsize=0, markeredgecolor=SURFACE,
                    markeredgewidth=0.6,
                    label=f"{name}: {SET_DESCRIPTION[name]}")
    ax.set_xticks(positions, labels)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Entries per chip")
    ax.set_ylabel("Chips flagged with no width change [%]")
    ax.set_title("Width test: false-flag rate at σ_Cmp = σ_Ref "
                 "(toy, all detectors and coordinates pooled)",
                 color=INK, fontsize=11, loc="left")
    style(ax, plt)
    legend = ax.legend(frameon=False, fontsize=8, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK2)
    fig.tight_layout()
    path = out_dir / "false_width_rate.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)
    return written


def z_label(z_cut) -> str:
    return "5% only" if z_cut is None else f"5% and z > {z_cut:g}"


def make_entries_plots(frame: pd.DataFrame, labels: list[str],
                       ratios: list[float], out_dir: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    written = []
    shown = plot_labels(labels)
    first = ratios[0]

    # Efficiency on the injected chip vs scale: raw counts, first event ratio.
    fig, axes = plt.subplots(len(GEOMETRY), len(ENTRIES_Z_CUTS),
                             figsize=(13, 7.5), facecolor=SURFACE,
                             sharey=True)
    fine = np.linspace(ENTRIES_GRID.min(), ENTRIES_GRID.max(), 401)
    for row_axes, det in zip(axes, GEOMETRY):
        for ax, z_cut in zip(row_axes, ENTRIES_Z_CUTS):
            name = entries_set_name("raw", z_cut)
            part = frame[(frame["threshold_set"] == name)
                         & (frame["detector"] == det)
                         & np.isclose(frame["event_ratio"], first)
                         & (frame["group"] == "injected")]
            for label, color in zip(shown, ORDINAL):
                points = part[part["entries"] == label].sort_values(
                    "entries_scale")
                n_mean = (float(NOMINAL_ENTRIES[det]) if label == "nominal"
                          else float(label))
                ax.plot(fine, [analytic_entries(n_mean, f, f, first, det,
                                                "raw", z_cut) for f in fine],
                        color=color, lw=1.5)
                ax.errorbar(points["entries_scale"], points["eff"],
                            yerr=error_bars(points["eff"], points["eff_lo"],
                                            points["eff_hi"]),
                            fmt="o", ms=4, color=color, ecolor=color,
                            elinewidth=1.0, capsize=0,
                            markeredgecolor=SURFACE, markeredgewidth=0.6)
            for cut in (1.0 - ENTRIES_REL_CUT, 1.0 + ENTRIES_REL_CUT):
                ax.axvline(cut, color=MUTED, lw=0.8, ls="--")
            ax.set_ylim(-0.02, 1.05)
            ax.set_title(f"{det}: {z_label(z_cut)}", color=INK, fontsize=10,
                         loc="left")
            ax.set_xlabel("Injected entries scale N_Cmp / N_Ref")
            if z_cut is None:
                ax.set_ylabel("Flag efficiency (entries test)")
            style(ax, plt)
    handles = [Line2D([], [], color=color, marker="o", ms=5, lw=1.5,
                      label=entries_legend_label(label))
               for label, color in zip(shown, ORDINAL)]
    handles.append(Line2D([], [], lw=0, label="points: toy (68% Wilson); "
                                              "lines: analytic"))
    legend = fig.legend(handles=handles, loc="lower center",
                        ncol=len(handles), frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK2)
    fig.suptitle(f"Entries test on raw counts, event ratio "
                 f"{first:.2f}: one chip per stave or ladder scaled",
                 color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    path = out_dir / "entries_efficiency.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)

    # False-flag rate of unchanged chips vs N, per normalization.
    positions = np.arange(len(labels))
    styles = ["-", "--", ":"]
    fig, axes = plt.subplots(1, len(NORMALIZATIONS), figsize=(14, 5),
                             facecolor=SURFACE, sharey=True)
    for ax, norm in zip(axes, NORMALIZATIONS):
        for z_cut, color, marker in zip(ENTRIES_Z_CUTS, CATEGORICAL, MARKERS):
            name = entries_set_name(norm, z_cut)
            for ratio, line_style in zip(ratios, styles):
                rates = [entries_false(frame, name, label, ratio)
                         for label in labels]
                rate = np.array([r[0] for r in rates]) * 100.0
                low = np.array([r[1] for r in rates]) * 100.0
                high = np.array([r[2] for r in rates]) * 100.0
                ax.errorbar(positions, rate, yerr=error_bars(rate, low, high),
                            color=color, ecolor=color, lw=1.5, ls=line_style,
                            marker=marker, ms=MARKER_SIZE[marker],
                            elinewidth=1.0, capsize=0,
                            markeredgecolor=SURFACE, markeredgewidth=0.6)
        ax.set_xticks(positions, labels)
        ax.set_ylim(0, 102)
        ax.set_title({"raw": "Raw counts", "detector": "Share of detector "
                      "total", "layer": "Share of layer total"}[norm],
                     color=INK, fontsize=10, loc="left")
        ax.set_xlabel("Entries per chip")
        if norm == NORMALIZATIONS[0]:
            ax.set_ylabel("Unchanged chips flagged [%]")
        style(ax, plt)
    handles = [Line2D([], [], color=color, marker=marker,
                      ms=MARKER_SIZE[marker], lw=1.5, label=z_label(z_cut))
               for z_cut, color, marker in zip(ENTRIES_Z_CUTS, CATEGORICAL,
                                               MARKERS)]
    handles += [Line2D([], [], color=INK2, lw=1.5, ls=line_style,
                       label=f"event ratio {ratio:.2f}")
                for ratio, line_style in zip(ratios, styles)]
    legend = fig.legend(handles=handles, loc="lower center",
                        ncol=len(handles), frameon=False, fontsize=8)
    for text in legend.get_texts():
        text.set_color(INK2)
    fig.suptitle("Entries test: false-flag rate of unchanged chips "
                 "(MVTX + INTT pooled)", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    path = out_dir / "entries_false_rate.png"
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    written.append(path)
    return written


# ------------------------------------------------------ cut-choice plots ---

def column_title(label: str) -> str:
    return (f"N = nominal ({NOMINAL_ENTRIES['MVTX']} / "
            f"{NOMINAL_ENTRIES['INTT']})" if label == "nominal"
            else f"N = {label} per chip")


def make_cut_plots(frames: dict, columns: list[str], ratio: float,
                   out_dir: Path) -> list[Path]:
    """Threshold options side by side at fixed entries per chip (columns):
    mean_cuts, width_cuts, entries_cuts and cut_tradeoff."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    written = []
    width_in = 2.9 * len(columns) + 1.0

    def finish(fig, handles, title, name, bottom=0.08, ncol=5):
        legend = fig.legend(handles=handles, loc="lower center",
                            ncol=min(len(handles), ncol), frameon=False,
                            fontsize=8)
        for text in legend.get_texts():
            text.set_color(INK2)
        fig.suptitle(title, color=INK, fontsize=11)
        fig.tight_layout(rect=(0, bottom, 1, 0.96))
        path = out_dir / name
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        written.append(path)

    def set_handles(names, describe):
        return [Line2D([], [], color=color, marker=marker,
                       ms=MARKER_SIZE[marker] * 0.7, lw=1.5,
                       label=describe(name))
                for name, color, marker in zip(names, CATEGORICAL, MARKERS)]

    def pad_false_axis(ax):
        """False-flag axis from 0, at least 0-1%, with a little margin."""
        right = max(ax.get_xlim()[1], 1.0)
        ax.set_xlim(-0.04 * right, 1.06 * right)

    def curve(ax, x, p, low, high, color, marker):
        ax.errorbar(x, p, yerr=error_bars(p, low, high), color=color,
                    ecolor=color, lw=1.2, marker=marker,
                    ms=MARKER_SIZE[marker] * 0.45, elinewidth=0.8,
                    capsize=0, markeredgecolor=SURFACE,
                    markeredgewidth=0.4)

    # Mean test: options per detector/coordinate.
    if "mean" in frames:
        frame = frames["mean"]
        fig, axes = plt.subplots(len(COMBOS), len(columns),
                                 figsize=(width_in, 11), facecolor=SURFACE,
                                 sharey=True, squeeze=False)
        for row, combo in enumerate(COMBOS):
            unit, factor = UNIT[combo]
            for col, label in enumerate(columns):
                ax = axes[row][col]
                for name, color, marker in zip(MEAN_SETS, CATEGORICAL,
                                               MARKERS):
                    part = frame[(frame["threshold_set"] == name)
                                 & (frame["detector"] == combo[0])
                                 & (frame["coord"] == combo[1])
                                 & (frame["entries"] == label)
                                 & (frame["shift_over_min"] >= 0)
                                 ].sort_values("shift_over_min")
                    curve(ax, part["injected_shift_mm"] * factor,
                          part["eff_mean"], part["eff_mean_lo"],
                          part["eff_mean_hi"], color, marker)
                ax.axvline(DEFAULT_MIN_SHIFT_MM[combo] * factor, color=MUTED,
                           lw=0.8, ls="--")
                ax.set_ylim(-0.02, 1.05)
                if row == 0:
                    ax.set_title(column_title(label), color=INK, fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{COMBO_NAME[combo]}\nflag efficiency",
                                  fontsize=8)
                ax.set_xlabel(f"Injected mean shift [{unit}]", fontsize=8)
                style(ax, plt)
        finish(fig, set_handles(MEAN_SETS, lambda n: f"{n}: "
                                f"{SET_DESCRIPTION[n]}"),
               "Mean test: threshold options at fixed entries per chip "
               "(dashed: default minimum shift)", "mean_cuts.png", 0.05)

    # Width test: options, detectors and coordinates pooled.
    if "width" in frames:
        frame = frames["width"]
        fig, axes = plt.subplots(1, len(columns), figsize=(width_in, 4.2),
                                 facecolor=SURFACE, sharey=True,
                                 squeeze=False)
        for col, label in enumerate(columns):
            ax = axes[0][col]
            for name, color, marker in zip(WIDTH_SETS, CATEGORICAL, MARKERS):
                pooled = width_points(frame, name, label)["pooled"]
                curve(ax, pooled["sigma_scale"], pooled["eff"],
                      pooled["lo"], pooled["hi"], color, marker)
            for cut in (0.9, 1.1):
                ax.axvline(cut, color=MUTED, lw=0.8, ls="--")
            ax.set_ylim(-0.02, 1.05)
            ax.set_title(column_title(label), color=INK, fontsize=9)
            ax.set_xlabel("Injected width scale σ_Cmp / σ_Ref", fontsize=8)
            if col == 0:
                ax.set_ylabel("Flag efficiency (width test)", fontsize=8)
            style(ax, plt)
        finish(fig, set_handles(WIDTH_SETS, lambda n: f"{n}: "
                                f"{SET_DESCRIPTION[n]}"),
               "Width test: threshold options (all detectors pooled; value "
               "at 1.00 = unchanged chips flagged; dashed: ±10%, the default cut)",
               "width_cuts.png", 0.2, ncol=2)

    # Entries test: significance options, raw counts.
    entries_names = [entries_set_name("raw", z_cut)
                     for z_cut in ENTRIES_Z_CUTS]
    if "entries" in frames:
        frame = frames["entries"]
        fig, axes = plt.subplots(len(GEOMETRY), len(columns),
                                 figsize=(width_in, 6.5), facecolor=SURFACE,
                                 sharey=True, squeeze=False)
        for row, det in enumerate(GEOMETRY):
            for col, label in enumerate(columns):
                ax = axes[row][col]
                for name, color, marker in zip(entries_names, CATEGORICAL,
                                               MARKERS):
                    part = frame[(frame["threshold_set"] == name)
                                 & (frame["detector"] == det)
                                 & (frame["entries"] == label)
                                 & np.isclose(frame["event_ratio"], ratio)
                                 & (frame["group"] == "injected")
                                 ].sort_values("entries_scale")
                    curve(ax, part["entries_scale"], part["eff"],
                          part["eff_lo"], part["eff_hi"], color, marker)
                for cut in (1.0 - ENTRIES_REL_CUT, 1.0 + ENTRIES_REL_CUT):
                    ax.axvline(cut, color=MUTED, lw=0.8, ls="--")
                ax.set_ylim(-0.02, 1.05)
                if row == 0:
                    ax.set_title(column_title(label), color=INK, fontsize=9)
                if col == 0:
                    ax.set_ylabel(f"{det}\nflag efficiency", fontsize=8)
                ax.set_xlabel("Injected entries scale N_Cmp / N_Ref",
                              fontsize=8)
                style(ax, plt)
        finish(fig, [Line2D([], [], color=color, marker=marker,
                            ms=MARKER_SIZE[marker] * 0.7, lw=1.5,
                            label=z_label(z_cut))
                     for z_cut, color, marker in zip(ENTRIES_Z_CUTS,
                                                     CATEGORICAL, MARKERS)],
               f"Entries test on raw counts: significance options (event "
               f"ratio {ratio:.2f}; value at 1.00 = unchanged chips flagged; "
               f"dashed: ±5%)", "entries_cuts.png", 0.08)

    # Trade-off: false-flag rate vs smallest change flagged 90% of the time.
    if "width" in frames and "entries" in frames:
        fig, axes = plt.subplots(2, len(columns), figsize=(width_in, 7.5),
                                 facecolor=SURFACE, squeeze=False)
        cap = 45.0  # % change drawn for "not reached within the scan"
        for col, label in enumerate(columns):
            ax = axes[0][col]
            for name, color, marker in zip(WIDTH_SETS, CATEGORICAL, MARKERS):
                p = width_points(frames["width"], name, label)
                change = 100.0 * (p["up90"] - 1.0)
                reached = np.isfinite(change)
                ax.plot(100.0 * p["false"], change if reached else cap,
                        marker=marker, ms=MARKER_SIZE[marker], ls="none",
                        color=color, markerfacecolor=color if reached
                        else "none", markeredgewidth=1.2)
            ax.set_title(column_title(label), color=INK, fontsize=9)
            ax.set_xlabel("Unchanged chips flagged [%]", fontsize=8)
            if col == 0:
                ax.set_ylabel("Width test: broadening flagged\n90% of the "
                              "time [%]", fontsize=8)
            ax.set_ylim(0, cap + 5)
            pad_false_axis(ax)
            style(ax, plt)
            ax = axes[1][col]
            for name, color, marker in zip(entries_names, CATEGORICAL,
                                           MARKERS):
                false_rate = entries_false(frames["entries"], name, label,
                                           ratio)[0]
                for det, fill in (("MVTX", True), ("INTT", False)):
                    p = entries_points(frames["entries"], name, det, label,
                                       ratio)
                    change = 100.0 * (1.0 - p["down90"])
                    reached = np.isfinite(change) and false_rate <= 0.5
                    ax.plot(100.0 * false_rate,
                            change if reached else cap,
                            marker="o" if det == "MVTX" else "^",
                            ms=7, ls="none", color=color,
                            markerfacecolor=color if fill and reached
                            else "none", markeredgewidth=1.2)
            ax.set_xlabel("Unchanged chips flagged [%]", fontsize=8)
            if col == 0:
                ax.set_ylabel("Entries test: loss flagged\n90% of the time "
                              "[%]", fontsize=8)
            ax.set_ylim(0, cap + 5)
            pad_false_axis(ax)
            style(ax, plt)
        handles = set_handles(WIDTH_SETS, lambda n: f"width {n}")
        handles += [Line2D([], [], color=color, marker="o", ms=6, lw=0,
                           label=f"entries {z_label(z_cut)}")
                    for z_cut, color in zip(ENTRIES_Z_CUTS, CATEGORICAL)]
        handles += [Line2D([], [], color=INK2, marker="o", ms=6, lw=0,
                           label="MVTX (filled)"),
                    Line2D([], [], color=INK2, marker="^", ms=6, lw=0,
                           markerfacecolor="none", label="INTT (open)")]
        finish(fig, handles,
               f"Cut trade-off: lower left is better (marker at "
               f"{cap:.0f}% = not reached within the scan, or noise-"
               f"dominated)", "cut_tradeoff.png", 0.12)
    return written


def load_frames(out_dir: Path) -> dict:
    """The efficiency tables of an earlier run, for --replot."""
    frames = {}
    for scan in ("mean", "width", "entries"):
        path = out_dir / f"efficiency_{scan}.csv"
        if path.exists():
            frames[scan] = pd.read_csv(path, dtype={"entries": str})
    if not frames:
        raise SystemExit(f"--replot: no efficiency_*.csv in {out_dir}")
    return frames


# ------------------------------------------------------------------ main ---

def parse_entries(text: str) -> list:
    entries = []
    for item in text.split(","):
        item = item.strip()
        if item == "nominal":
            entries.append(("nominal", None))
            continue
        value = int(item)
        if CORE_FRACTION * value < 10:
            raise SystemExit(f"--entries {value}: below the low_stats cut "
                             "(0.95 N < 10), nothing would be comparable")
        entries.append((str(value), value))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "synthetic" / "sensitivity_output")
    parser.add_argument("--seeds", type=int, default=5,
                        help="independent tables per grid point (default 5)")
    parser.add_argument("--first-seed", type=int, default=20261126)
    parser.add_argument("--entries", default=DEFAULT_ENTRIES,
                        help="entries per chip, comma-separated; 'nominal' = "
                             "the generator's Poisson default")
    parser.add_argument("--scans", default="mean,width,entries")
    parser.add_argument("--entries-seeds", type=int, default=50,
                        help="entries scan: toys per point (default 50; it "
                             "is fast, and only one chip per stave carries "
                             "the injection)")
    parser.add_argument("--event-ratios", default=DEFAULT_EVENT_RATIOS,
                        help="entries scan: events in the comparison / "
                             "events in the reference, comma-separated")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--replot", action="store_true",
                        help="only redraw the plots from the efficiency_*.csv "
                             "files already in --out-dir (no sweep; the "
                             "summary is kept as it is)")
    parser.add_argument("--ideal-errors", action="store_true",
                        help="use ideal fit errors instead of the ones "
                             "measured from the ROOT fits of "
                             "synthetic/reference (default: measured)")
    parser.add_argument("--plot-entries",
                        help="entries per chip shown as columns of the "
                             "cut-choice plots, comma-separated, from the "
                             "swept values (default: up to 5 of them)")
    args = parser.parse_args()

    if not args.ideal_errors:
        ERROR_SCALE.update(measured_error_scale())
    print("Fit errors: " + error_model_text(), flush=True)

    if args.replot:
        frames = load_frames(args.out_dir)
        labels = entries_order(sorted({label for frame in frames.values()
                                       for label in frame["entries"]}))
        event_ratios = (sorted(frames["entries"]["event_ratio"].unique(),
                               reverse=True)
                        if "entries" in frames else [1.0])
        draw_plots(frames, labels, event_ratios, args)
        return

    entries = parse_entries(args.entries)
    labels = entries_order([label for label, _ in entries])
    scans = args.scans.split(",")
    event_ratios = [float(item) for item in args.event_ratios.split(",")]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seed_counter = itertools.count(args.first_seed)

    frames = {}
    for scan, grid in (("mean", MEAN_GRID), ("width", WIDTH_GRID)):
        if scan not in scans:
            continue
        print(f"{scan} scan: {len(grid)} points x {len(entries)} N values x "
              f"{args.seeds} seeds", flush=True)
        frames[scan] = to_frame(scan, run_scan(scan, grid, entries,
                                               args.seeds, seed_counter))
        path = args.out_dir / f"efficiency_{scan}.csv"
        frames[scan].to_csv(path, index=False, float_format="%.6g")
        print(f"Wrote {len(frames[scan])} rows to {path}")

    if "entries" in scans:
        print(f"entries scan: {len(ENTRIES_GRID)} points x {len(entries)} N "
              f"values x {len(event_ratios)} event ratios x "
              f"{args.entries_seeds} seeds", flush=True)
        frames["entries"] = run_entries_scan(entries, event_ratios,
                                             args.entries_seeds,
                                             args.first_seed)
        path = args.out_dir / "efficiency_entries.csv"
        frames["entries"].to_csv(path, index=False, float_format="%.6g")
        print(f"Wrote {len(frames['entries'])} rows to {path}")

    parts = []
    if "mean" in frames and "width" in frames:
        parts.append(build_summary(frames["mean"], frames["width"], args,
                                   labels))
    if "entries" in frames:
        parts.append(build_entries_summary(frames["entries"], labels,
                                           event_ratios))
    if parts:
        path = args.out_dir / "sensitivity_summary.md"
        path.write_text("\n\n".join(parts) + "\n")
        print(f"Wrote {path}")
    if args.no_plots or not parts:
        return
    draw_plots(frames, labels, event_ratios, args)


def draw_plots(frames: dict, labels: list[str], event_ratios: list[float],
               args) -> None:
    if args.plot_entries:
        columns = [item.strip() for item in args.plot_entries.split(",")]
        unknown = [label for label in columns if label not in labels]
        if unknown:
            raise SystemExit(f"--plot-entries {unknown}: not swept; "
                             f"available: {', '.join(labels)}")
    else:
        columns = plot_labels(labels)
    try:
        written = []
        if "mean" in frames and "width" in frames:
            written += make_plots(frames["mean"], frames["width"], labels,
                                  args.out_dir)
        if "entries" in frames:
            written += make_entries_plots(frames["entries"], labels,
                                          event_ratios, args.out_dir)
        written += make_cut_plots(frames, columns, event_ratios[0],
                                  args.out_dir)
    except ImportError:
        print("matplotlib is not installed: plots skipped "
              "(pip install matplotlib)")
    else:
        for path in written:
            print(f"Wrote {path}")


if __name__ == "__main__":
    main()
