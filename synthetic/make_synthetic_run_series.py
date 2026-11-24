#!/usr/bin/env python3
"""Write a synthetic run-by-run series of fit tables with known injected changes.

One residual_fits_<run>.csv per run, in the format written by
macros/plot_fits_overlaid_stavechip.C, for the full MVTX/INTT geometry, plus
injections_runs.csv (the truth) and run_info.csv (run number, luminosity
factor, synthetic notes). stdiag.run_stability scans these tables and
synthetic.check_run_stability compares its change points with the truth.

Model per (run, detector, layer, stave, chip, coord), all lengths in mm:
  true mean  = static sensor offset (shared by all runs)
             + per-run sensor jitter + per-run layer jitter
             + injected offset(run)
  true sigma = nominal width * static spread * (1 + per-run width jitter)
             * injected scale(run)
  entries    = Poisson(static entries * luminosity(run))
  fitted mu, sigma = true values + Gaussian noise with the fit errors
  mu_err = sigma / sqrt(0.95 N),  sigma_err = sigma / sqrt(2 * 0.95 N)
The static offsets, widths and errors follow make_synthetic_fit_tables.py
(which is imported, not changed). The run-to-run jitter stands in for real
run-to-run systematics (track selection, beam, distortions); its size is a
guess, not a measurement. Every random number is drawn whether or not an
injection uses it, so --null gives the same series without the injections.

Usage:
  python -m synthetic.make_synthetic_run_series [--out-dir DIR] [--seed N]
      [--n-runs 60] [--null]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.make_synthetic_fit_tables import (
    FIT_HALF_WIDTH_SIGMAS, GEOMETRY, NOMINAL_ENTRIES, NOMINAL_SIGMA_MM)

DEFAULT_SEED = 20261124
DEFAULT_RUNS = 60
FIRST_RUN = 53000
# Per-run sensor jitter of the residual mean (mm); the per-run layer jitter
# (common to every sensor of a layer and coordinate) is LAYER_JITTER_FRACTION
# of it.
RUN_JITTER_MM = {("MVTX", "rphi"): 0.001, ("MVTX", "z"): 0.001,
                 ("INTT", "rphi"): 0.004, ("INTT", "z"): 0.3}
LAYER_JITTER_FRACTION = 0.5
WIDTH_JITTER = 0.02
LUMINOSITY_RANGE = (0.3, 1.0)
BAD_RUN_NOISE_MM = 0.015

# Selectors: detector/layer None = all, staves/chips None = all, coord None =
# both. Run indices count from 0 in run order.
#   step    offset_mm from run `start` on
#   window  offset_mm in runs start..end (inclusive), then back
#   ramp    offset rising linearly from 0 at `start` to offset_mm at `end`,
#           constant afterwards
#   width   sigma_scale from run `start` on
#   missing rows absent in runs start..end
#   lumi    luminosity factor `lumi` in the runs listed in `runs`
#   bad_run in run `start`: sigma x sigma_scale and extra mean noise
# expected = the class stdiag.run_stability should report (None: must not be
# reported; "either": statistically limited, never a failure).
INJECTIONS = [
    dict(name="step_chip", kind="step", detector="MVTX", layer=0, staves=[5],
         chips=[3], coord="rphi", start=20, offset_mm=0.010, expected="step",
         description="One MVTX chip moves +10 um in local r-phi"),
    dict(name="step_sensor", kind="step", detector="INTT", layer=4,
         staves=[6], chips=[1], coord="rphi", start=15, offset_mm=0.040,
         expected="step",
         description="One INTT sensor moves +40 um in local r-phi"),
    dict(name="step_stave", kind="step", detector="MVTX", layer=1,
         staves=[4], chips=None, coord="rphi", start=30, offset_mm=0.008,
         expected="coherent_stave",
         description="Whole MVTX stave moves +8 um in local r-phi"),
    dict(name="step_halflayer", kind="step", detector="MVTX", layer=2,
         staves=list(range(10)), chips=None, coord="z", start=40,
         offset_mm=0.006, expected="multi_stave",
         description="Half of MVTX layer 2 (staves 0-9) moves +6 um in "
                     "local z"),
    dict(name="layer_rotation", kind="step", detector="MVTX", layer=1,
         staves=None, chips=None, coord="rphi", start=52, offset_mm=0.007,
         expected="layer_mode",
         description="All of MVTX layer 1 shifts +7 um in local r-phi "
                     "(layer rotation or global alignment change)"),
    dict(name="transient", kind="window", detector="MVTX", layer=0,
         staves=[8], chips=[2], coord="rphi", start=22, end=27,
         offset_mm=0.015, expected="transient",
         description="One MVTX chip is +15 um off in runs 22-27, then back"),
    dict(name="step_near_threshold", kind="step", detector="MVTX", layer=1,
         staves=[2], chips=[7], coord="z", start=35, offset_mm=0.006,
         expected="either",
         description="+6 um chip step, just above the 5 um minimum shift"),
    dict(name="step_sub_threshold", kind="step", detector="MVTX", layer=2,
         staves=[7], chips=[0], coord="rphi", start=25, offset_mm=0.003,
         expected=None,
         description="+3 um chip step: significant over many runs but below "
                     "the 5 um minimum shift (must NOT be reported)"),
    dict(name="drift", kind="ramp", detector="MVTX", layer=1, staves=[10],
         chips=[4], coord="rphi", start=10, end=50, offset_mm=0.012,
         expected="drift",
         description="One MVTX chip drifts 0 -> +12 um over runs 10-50"),
    dict(name="drift_sub_threshold", kind="ramp", detector="INTT", layer=5,
         staves=[2], chips=[0], coord="rphi", start=0, end=59,
         offset_mm=0.010, expected=None,
         description="INTT sensor drifts +10 um over the whole series, below "
                     "the 20 um minimum shift (must NOT be reported)"),
    dict(name="outlier", kind="window", detector="MVTX", layer=0,
         staves=[1], chips=[1], coord="rphi", start=33, end=33,
         offset_mm=0.030, expected="outlier_run",
         description="One MVTX chip is +30 um off in run 33 only"),
    dict(name="bad_run", kind="bad_run", detector="MVTX", layer=None,
         staves=None, chips=None, coord=None, start=47, sigma_scale=1.5,
         noise_mm=BAD_RUN_NOISE_MM, expected="bad_run",
         description="Run 47 is bad for all of MVTX: widths x1.5 and 15 um "
                     "random mean offsets"),
    dict(name="low_stats", kind="lumi", detector=None, layer=None,
         staves=None, chips=None, coord=None, runs=[12, 44], lumi=0.05,
         expected=None,
         description="Runs 12 and 44 have 5% luminosity (large errors; no "
                     "change may be reported)"),
    dict(name="missing_ladder", kind="missing", detector="INTT", layer=5,
         staves=[9], chips=None, coord=None, start=18, end=24,
         expected=None,
         description="INTT layer 5 ladder 9 absent in runs 18-24 (gaps; no "
                     "change may be reported)"),
    dict(name="width", kind="width", detector="MVTX", layer=0, staves=[2],
         chips=[5], coord=None, start=28, sigma_scale=1.4,
         expected="width_step",
         description="One MVTX chip 40% broader from run 28 on, both "
                     "coordinates"),
]
# Kinds that remain in --null: they change no position or width.
NULL_KINDS = {"lumi", "missing"}

RUN_NOTES = {15: "synthetic note: INTT access",
             30: "synthetic note: magnet ramp",
             52: "synthetic note: new global alignment tag"}


def selects(injection: dict, detector: str, layer: int, stave: int,
            chip: int, coord: str) -> bool:
    """True if the injection's selectors cover this sensor and coordinate."""
    return ((injection["detector"] is None
             or detector == injection["detector"])
            and (injection["layer"] is None or layer == injection["layer"])
            and (injection["staves"] is None or stave in injection["staves"])
            and (injection["chips"] is None or chip in injection["chips"])
            and (injection["coord"] is None or coord == injection["coord"]))


def offset_at(injection: dict, run: int) -> float:
    kind, start = injection["kind"], injection.get("start", 0)
    if kind == "step":
        return injection["offset_mm"] if run >= start else 0.0
    if kind == "window":
        return (injection["offset_mm"]
                if start <= run <= injection["end"] else 0.0)
    if kind == "ramp":
        fraction = (run - start) / max(1, injection["end"] - start)
        return injection["offset_mm"] * float(np.clip(fraction, 0.0, 1.0))
    return 0.0


def geometry_keys() -> list[tuple]:
    return [(detector, layer, stave, chip, coord)
            for detector, (layers, staves, chips) in GEOMETRY.items()
            for layer, stave_count in zip(layers, staves)
            for stave in range(stave_count)
            for chip in range(chips)
            for coord in ("rphi", "z")]


def run_labels(n_runs: int, rng: np.random.Generator) -> list[int]:
    """Non-contiguous run numbers, as in real data taking."""
    return list(FIRST_RUN + np.cumsum(rng.integers(1, 8, n_runs)))


def build_run_series(seed: int = DEFAULT_SEED, n_runs: int = DEFAULT_RUNS,
                     injections: list[dict] | None = None
                     ) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Return ({label: fit table}, run_info) for the given injections."""
    injections = INJECTIONS if injections is None else injections
    rng = np.random.default_rng(seed)
    keys = geometry_keys()
    n_keys = len(keys)
    numbers = run_labels(n_runs, rng)
    luminosity = rng.uniform(*LUMINOSITY_RANGE, n_runs)
    for injection in injections:
        if injection["kind"] == "lumi":
            for run in injection["runs"]:
                if run < n_runs:
                    luminosity[run] = injection["lumi"]

    nominal = np.array([NOMINAL_SIGMA_MM[(key[0], key[4])] for key in keys])
    jitter = np.array([RUN_JITTER_MM[(key[0], key[4])] for key in keys])
    static_offset = rng.normal(0.0, 0.25 * nominal)
    static_sigma = nominal * rng.uniform(0.9, 1.1, n_keys)
    static_entries = rng.poisson([NOMINAL_ENTRIES[key[0]] for key in keys])

    layer_groups = sorted({(key[0], key[1], key[4]) for key in keys})
    layer_index = np.array([layer_groups.index((key[0], key[1], key[4]))
                            for key in keys])
    layer_jitter_scale = np.array([RUN_JITTER_MM[(group[0], group[2])]
                                   for group in layer_groups])

    # Every random number is drawn here, in a fixed order.
    sensor_jitter = rng.normal(0.0, 1.0, (n_runs, n_keys)) * jitter
    layer_jitter = (rng.normal(0.0, 1.0, (n_runs, len(layer_groups)))
                    * layer_jitter_scale * LAYER_JITTER_FRACTION)
    width_jitter = rng.normal(0.0, WIDTH_JITTER, (n_runs, n_keys))
    entries_draw = rng.poisson(static_entries[None, :]
                               * luminosity[:, None]).astype(float)
    mu_noise = rng.normal(0.0, 1.0, (n_runs, n_keys))
    sigma_noise = rng.normal(0.0, 1.0, (n_runs, n_keys))
    bad_noise = rng.normal(0.0, 1.0, (n_runs, n_keys))
    chi2 = rng.chisquare(30, (n_runs, n_keys))

    selected = [np.array([selects(injection, *key) for key in keys])
                for injection in injections]

    tables = {}
    for run in range(n_runs):
        offset = np.zeros(n_keys)
        scale = np.ones(n_keys)
        absent = np.zeros(n_keys, bool)
        for injection, mask in zip(injections, selected):
            kind = injection["kind"]
            if kind in ("step", "window", "ramp"):
                offset[mask] += offset_at(injection, run)
            elif kind == "width" and run >= injection["start"]:
                scale[mask] *= injection["sigma_scale"]
            elif kind == "missing" and (injection["start"] <= run
                                        <= injection["end"]):
                absent |= mask
            elif kind == "bad_run" and run == injection["start"]:
                scale[mask] *= injection["sigma_scale"]
                offset[mask] += bad_noise[run, mask] * injection["noise_mm"]

        true_mu = (static_offset + sensor_jitter[run]
                   + layer_jitter[run, layer_index] + offset)
        true_sigma = static_sigma * (1.0 + width_jitter[run]) * scale
        entries = entries_draw[run]
        core = 0.95 * entries
        ok = core >= 10
        with np.errstate(divide="ignore", invalid="ignore"):
            mu_err = np.where(ok, true_sigma / np.sqrt(core), np.nan)
            sigma_err = np.where(ok, true_sigma / np.sqrt(2.0 * core), np.nan)
        mu = np.where(ok, true_mu + mu_noise[run] * mu_err, np.nan)
        sigma = np.where(ok, np.abs(true_sigma + sigma_noise[run] * sigma_err),
                         np.nan)
        label = f"run{numbers[run]}"
        frame = pd.DataFrame(keys, columns=["detector", "layer", "stave",
                                            "chip", "coord"])
        frame.insert(0, "listfile", f"list_{label}.txt")
        frame.insert(0, "label", label)
        frame["entries"] = entries.astype(int)
        frame["entries_in_fit_window"] = np.round(core).astype(int)
        frame["populated_bins"] = np.minimum(entries, 60).astype(int)
        frame["fit_status"] = np.where(ok, "ok", "low_stats")
        frame["mu_mm"] = mu
        frame["mu_err_mm"] = mu_err
        frame["sigma_mm"] = sigma
        frame["sigma_err_mm"] = sigma_err
        frame["chi2"] = np.where(ok, chi2[run], np.nan)
        frame["ndf"] = np.where(ok, 30, 0)
        frame["fit_low_mm"] = mu - FIT_HALF_WIDTH_SIGMAS * sigma
        frame["fit_high_mm"] = mu + FIT_HALF_WIDTH_SIGMAS * sigma
        tables[label] = frame.loc[~absent].reset_index(drop=True)

    run_info = pd.DataFrame({
        "run_index": range(n_runs),
        "run": [f"run{number}" for number in numbers],
        "run_number": numbers,
        "luminosity": np.round(luminosity, 4),
        "note": [RUN_NOTES.get(run, "") for run in range(n_runs)],
    })
    return tables, run_info


def active_injections(null: bool) -> list[dict]:
    return [injection for injection in INJECTIONS
            if not null or injection["kind"] in NULL_KINDS]


def injection_table(injections: list[dict]) -> pd.DataFrame:
    def listed(values):
        return "all" if values is None else ";".join(map(str, values))

    return pd.DataFrame([dict(
        name=injection["name"], kind=injection["kind"],
        detector=injection["detector"] or "all",
        layer="all" if injection["layer"] is None else injection["layer"],
        staves=listed(injection["staves"]), chips=listed(injection["chips"]),
        coord=injection["coord"] or "both",
        start=injection.get("start", ""), end=injection.get("end", ""),
        runs=listed(injection.get("runs")) if "runs" in injection else "",
        offset_mm=injection.get("offset_mm", 0.0),
        sigma_scale=injection.get("sigma_scale", 1.0),
        expected=injection["expected"] or "none",
        description=injection["description"]) for injection in injections])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path,
                        default=Path("synthetic/run_series_output"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--n-runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--null", action="store_true",
                        help="no position or width changes (luminosity and "
                             "missing-ladder effects stay)")
    args = parser.parse_args()

    injections = active_injections(args.null)
    tables, run_info = build_run_series(args.seed, args.n_runs, injections)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for old in args.out_dir.glob("residual_fits_run*.csv"):
        old.unlink()
    for label, table in tables.items():
        table.to_csv(args.out_dir / f"residual_fits_{label}.csv",
                     index=False, float_format="%.8g")
    run_info.to_csv(args.out_dir / "run_info.csv", index=False)
    injection_table(injections).to_csv(
        args.out_dir / "injections_runs.csv", index=False)
    print(f"Wrote {len(tables)} run tables ({len(next(iter(tables.values())))}"
          f" rows each), run_info.csv and injections_runs.csv "
          f"({len(injections)} injections{', null series' if args.null else ''})"
          f" to {args.out_dir}")


if __name__ == "__main__":
    main()
