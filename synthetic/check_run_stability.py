#!/usr/bin/env python3
"""Check stdiag.run_stability change points against the injected truth.

For every injection of synthetic.make_synthetic_run_series:
  - an expected class must be found at the right place and run
    (+-WINDOW runs; +-DRIFT_WINDOW for the start and end of a drift);
  - expected None ("must not be reported") must produce no change point in
    its sensors and runs;
  - "either" (statistically limited) is never a failure.
Every change point not explained by an injection counts against the false
alarm budget (--budget, default 2). With --null only the budget applies.

Usage:
  python -m synthetic.check_run_stability CHANGEPOINTS_CSV [--null]
      [--budget 2]
Exit status is 1 if a check fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic.make_synthetic_run_series import INJECTIONS, active_injections

WINDOW = 2
DRIFT_WINDOW = 6
LAYER_MODE_MAX_SENSOR_EVENTS = 3
MULTI_STAVE_FRACTION = 0.8
POSITION_KINDS = {"step", "window", "ramp"}


def covers(injection: dict, row) -> bool:
    """True if the change point's place lies inside the injection's
    selectors. Aggregated rows (stave or chip = -1) match any stave/chip."""
    def inside(selection, value):
        return selection is None or value == -1 or value in selection

    return ((injection["detector"] is None
             or row["detector"] == injection["detector"])
            and (injection["layer"] is None or row["layer"] == -1
                 or row["layer"] == injection["layer"])
            and inside(injection["staves"], row["stave"])
            and inside(injection["chips"], row["chip"])
            and (injection["coord"] is None or row["coord"] == ""
                 or row["coord"] == injection["coord"]))


def run_window(injection: dict) -> tuple[int, int]:
    kind = injection["kind"]
    start = injection.get("start", 0)
    if kind == "ramp":
        return start - DRIFT_WINDOW, injection["end"] + DRIFT_WINDOW
    if kind == "window":
        return start - WINDOW, injection["end"] + 1 + WINDOW
    if kind in ("missing",):
        return start - WINDOW, injection["end"] + 1 + WINDOW
    return start - WINDOW, start + WINDOW


def explains(injection: dict, row) -> bool:
    """True if the injection can explain this change point."""
    kind = injection["kind"]
    if kind == "lumi":
        return (row["run_index"] in injection["runs"]
                and covers(injection, row))
    if kind == "bad_run":
        return (row["run_index"] == injection["start"]
                and covers(injection, row))
    if kind == "width" and row["quantity"] != "sigma":
        return False
    if kind in POSITION_KINDS and row["quantity"] == "sigma":
        return False
    low, high = run_window(injection)
    return covers(injection, row) and low <= row["run_index"] <= high


def near(value, target, window):
    return abs(int(value) - int(target)) <= window


def sensors(injection: dict, events: pd.DataFrame) -> pd.DataFrame:
    return events.loc[events.apply(lambda row: covers(injection, row),
                                   axis=1)]


def check_injection(injection: dict, events: pd.DataFrame) -> str | None:
    """Return None if the injection's expectation holds, else a message."""
    expected = injection["expected"]
    start = injection.get("start", 0)
    mine = sensors(injection, events)

    def having(name, quantity="mu"):
        part = mine.loc[mine["class"] == name]
        return part.loc[part["quantity"] == quantity] if quantity else part

    if expected == "either":
        return None
    if expected is None:
        others = [other for other in INJECTIONS if other is not injection
                  and other["expected"] is not None]
        hits = mine.loc[mine.apply(
            lambda row: explains(injection, row)
            and not any(explains(other, row) for other in others), axis=1)]
        if injection["kind"] in ("step", "ramp"):
            # A single-sensor position change below the minimum shift: any
            # sensor-level mean change on it, in any run, is a failure.
            hits = mine.loc[(mine["quantity"] == "mu") & (mine["chip"] >= 0)]
        return (None if hits.empty else
                f"must not be reported, found {len(hits)}: "
                + ", ".join(f"{r['class']}@{r['run_index']}"
                            for _, r in hits.head(5).iterrows()))
    if expected in ("step", "width_step"):
        quantity = "mu" if expected == "step" else "sigma"
        problems = []
        coords = ([injection["coord"]] if injection["coord"]
                  else ["rphi", "z"])
        for coord in coords:
            found = having(expected, quantity)
            found = found.loc[(found["coord"] == coord)
                              & found["run_index"].apply(
                                  lambda r: near(r, start, WINDOW))]
            sign = np.sign(injection.get("offset_mm",
                                         injection.get("sigma_scale", 1) - 1))
            if found.empty or not (np.sign(found["delta_mm"]) == sign).any():
                problems.append(f"no {expected} in {coord} at run {start}")
        return "; ".join(problems) or None
    if expected == "coherent_stave":
        found = having("coherent_stave")
        found = found.loc[found["run_index"].apply(
            lambda r: near(r, start, WINDOW))]
        return None if len(found) else f"no coherent_stave at run {start}"
    if expected == "multi_stave":
        multi = having("multi_stave")
        multi = multi.loc[multi["run_index"].apply(
            lambda r: near(r, start, WINDOW))]
        staves = having("coherent_stave")
        staves = staves.loc[staves["run_index"].apply(
            lambda r: near(r, start, WINDOW))]
        layer = having("layer_mode")
        layer = layer.loc[layer["run_index"].apply(
            lambda r: near(r, start, WINDOW))]
        enough = (staves["stave"].nunique()
                  >= MULTI_STAVE_FRACTION * len(injection["staves"]))
        return (None if len(multi) or enough or len(layer) else
                f"only {staves['stave'].nunique()} coherent staves and no "
                f"multi_stave/layer_mode at run {start}")
    if expected == "layer_mode":
        layer = having("layer_mode")
        layer = layer.loc[layer["run_index"].apply(
            lambda r: near(r, start, WINDOW))]
        if layer.empty:
            return f"no layer_mode at run {start}"
        flood = having("step")
        flood = flood.loc[flood["run_index"].apply(
            lambda r: near(r, start, WINDOW))]
        if len(flood) > LAYER_MODE_MAX_SENSOR_EVENTS:
            return (f"layer_mode found but {len(flood)} sensor steps were "
                    "not absorbed")
        return None
    if expected == "transient":
        found = having("transient")
        found = found.loc[found["run_index"].apply(
            lambda r: near(r, start, WINDOW))
            & found["end_run_index"].apply(
                lambda r: near(r, injection["end"] + 1, WINDOW))]
        return None if len(found) else (
            f"no transient from run {start} to {injection['end'] + 1}")
    if expected == "drift":
        found = having("drift")
        found = found.loc[found["run_index"].apply(
            lambda r: near(r, start, DRIFT_WINDOW))
            & found["end_run_index"].apply(
                lambda r: near(r, injection["end"], DRIFT_WINDOW))]
        return None if len(found) else (
            f"no drift from run {start} to {injection['end']}")
    if expected == "outlier_run":
        found = having("outlier_run")
        found = found.loc[found["run_index"] == start]
        steps = mine.loc[mine["class"].isin(["step", "transient", "drift"])
                         & (mine["quantity"] == "mu")]
        if found.empty:
            return f"no outlier_run at run {start}"
        return (None if steps.empty else
                f"outlier found but also {len(steps)} step-like events")
    if expected == "bad_run":
        found = events.loc[(events["class"] == "bad_run")
                           & (events["detector"] == injection["detector"])
                           & (events["run_index"] == start)]
        return None if len(found) else f"no bad_run at run {start}"
    return f"unknown expectation {expected!r}"


def unexplained(events: pd.DataFrame, injections: list[dict]) -> pd.DataFrame:
    reporting = [injection for injection in injections
                 if injection["expected"] is not None]
    mask = events.apply(lambda row: not any(explains(injection, row)
                                            for injection in reporting),
                        axis=1) if len(events) else pd.Series(dtype=bool)
    return events.loc[mask] if len(events) else events


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("changepoints", type=Path)
    parser.add_argument("--null", action="store_true",
                        help="the series was made with --null")
    parser.add_argument("--budget", type=int, default=2,
                        help="allowed number of unexplained change points")
    args = parser.parse_args()

    events = pd.read_csv(args.changepoints, keep_default_na=False,
                         na_values={"delta_mm": [""], "z": [""],
                                    "delta_err_mm": [""]})
    for column in ("layer", "stave", "chip", "run_index", "end_run_index"):
        events[column] = events[column].astype(int)
    injections = active_injections(args.null)
    top = events.loc[events["group_id"] == ""]

    failures = []
    print(f"== {args.changepoints.name}: {len(top)} events "
          f"({len(events)} rows with grouped members) ==")
    print(f"{'injection':24} {'expected':16} result")
    for injection in injections:
        problem = check_injection(injection, events)
        status = "ok" if problem is None else f"FAIL: {problem}"
        print(f"{injection['name']:24} {str(injection['expected']):16} "
              f"{status}")
        if problem is not None:
            failures.append(f"{injection['name']}: {problem}")

    extra = unexplained(events, injections)
    print(f"\nUnexplained change points: {len(extra)} (budget {args.budget})")
    if len(extra):
        print(extra[["event_id", "class", "detector", "layer", "stave",
                     "chip", "coord", "quantity", "frame", "run_index",
                     "delta_mm", "z", "methods"]].to_string(index=False))
    if len(extra) > args.budget:
        failures.append(f"{len(extra)} unexplained change points "
                        f"(budget {args.budget})")

    if failures:
        print(f"\nFAIL: {len(failures)} problem(s):")
        print("\n".join(f"  {failure}" for failure in failures))
        sys.exit(1)
    print("\nPASS: every injection found as expected, "
          f"{len(extra)} unexplained change point(s).")


if __name__ == "__main__":
    main()
