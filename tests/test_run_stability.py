"""Run-by-run stability: change-point methods and the synthetic run series."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stdiag import run_stability as rs
from tests.conftest import run_module

UM = 0.001


def noisy_step(n=60, at=20, size=10 * UM, noise=1 * UM, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, noise, n)
    x[at:] += size
    return x, np.full(n, noise)


def methods_on(x, s, min_shift=5 * UM, h=4.0, p_threshold=0.95):
    X, S = x[None, :], s[None, :]
    Xc, Sc, lengths, _ = rs.pack(X, S)
    cfg = rs.Config()
    cusum = rs.cusum_batch(Xc, Sc, lengths, np.array([min_shift]),
                           np.array([h]), cfg.baseline_runs)[0]
    beta = 2.0 * np.log(len(x) * 3000)
    pelt = rs.pelt(x, s, beta)
    mu0 = np.array([rs.weighted_mean(x[:5], s[:5])[0]])
    q, r_map = rs.bocpd_batch(X, S, mu0, np.array([5 * min_shift]),
                              1 / 100, cfg.bocpd_lag)
    bocpd = rs.bocpd_locations(q[0], r_map[0], p_threshold, cfg.bocpd_lag)
    return cusum, pelt, bocpd


def test_methods_find_a_known_step():
    cusum, pelt, bocpd = methods_on(*noisy_step())
    assert pelt == [20]
    assert [abs(point - 20) <= 1 for point in cusum] == [True]
    assert [abs(point - 20) <= 1 for point in bocpd] == [True]


def test_methods_quiet_on_a_flat_series():
    x, s = noisy_step(size=0.0, seed=3)
    assert methods_on(x, s) == ([], [], [])


def test_cusum_trace_carries_over_gaps():
    """A missing run must not reset the CUSUM (max(0, nan) would)."""
    x, s = noisy_step()
    x[40] = np.nan
    plus, _, level = rs.cusum_trace(x, s, 5 * UM)
    assert np.isnan(plus[40]) and plus[41] > plus[39] > 0
    assert np.isnan(level[:5]).all() and np.isfinite(level[5])


def test_self_starting_cusum_threshold_stays_moderate():
    """At low statistics (k small) a fixed 5-run baseline made the null
    CUSUM drift and the threshold explode; the self-starting one does not."""
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((20000, 60))
    h = rs.tail_quantile(rs.cusum_null_statistic(Z, 0.75, 5), 1 / 5000)
    assert 4.0 < h < 15.0


def test_pelt_scan_equals_pelt():
    """The vectorised scan solves the same recursion as PELT."""
    rng = np.random.default_rng(7)
    rows, n = 40, 50
    X = rng.normal(0.0, 1.0, (rows, n))
    X[::2, 15:] += 4.0
    X[::3, 35:] -= 3.0
    S = rng.uniform(0.5, 2.0, (rows, n))
    X[rng.random((rows, n)) < 0.05] = np.nan          # gaps
    Xc, Sc, lengths, _ = rs.pack(X, S)
    beta = np.full(rows, 12.0)
    batch = rs.segment_batch(Xc, Sc, lengths, beta, 2)
    for row in range(rows):
        length = lengths[row]
        assert batch[row] == rs.pelt(Xc[row, :length], Sc[row, :length],
                                     12.0, 2)


def test_unequal_errors_are_respected():
    """A big deviation in a run with a large error is not a change."""
    x, s = noisy_step(size=0.0, seed=4)
    x[30:33] += 30 * UM
    s[30:33] = 40 * UM
    assert rs.pelt(x, s, 2.0 * np.log(60 * 3000)) == []


def test_gaps_keep_run_indices():
    x, s = noisy_step(at=30)
    x[[5, 6, 7]] = np.nan
    X, S = x[None, :], s[None, :]
    Xc, Sc, lengths, Ic = rs.pack(X, S)
    points = rs.pelt(Xc[0, :lengths[0]], Sc[0, :lengths[0]], 30.0)
    assert [int(Ic[0, point]) for point in points] == [30]


def test_minimum_shift_removes_a_significant_small_step():
    x, s = noisy_step(size=3 * UM, noise=0.3 * UM)
    points, delta, error = rs.prune_changes(x, s, [20], 5 * UM, 5.0)
    assert points == []
    points, delta, error = rs.prune_changes(x, s, [20], 2 * UM, 5.0)
    assert points == [20] and abs(delta[0] - 3 * UM) < 0.5 * UM


@pytest.mark.parametrize("shape, expected", [
    ("step", "step"), ("transient", "transient"), ("ramp", "drift")])
def test_classification(shape, expected):
    rng = np.random.default_rng(11)
    n = 60
    x = rng.normal(0.0, 1 * UM, n)
    if shape == "step":
        x[25:] += 10 * UM
    elif shape == "transient":
        x[25:31] += 10 * UM
    else:
        x += 12 * UM * np.clip((np.arange(n) - 10) / 40, 0, 1)
    s = np.full(n, 1 * UM)
    cusum, pelt, bocpd = methods_on(x, s)
    events = rs.classify_row(x, s, {"cusum": cusum, "pelt": pelt,
                                    "bocpd": bocpd}, 5 * UM, rs.Config(),
                             "mu")
    assert [event["class"] for event in events] == [expected]
    if expected == "drift":
        assert abs(events[0]["start"] - 10) <= 6
        assert abs(events[0]["end"] - 50) <= 6


def test_outliers_are_spikes_not_steps():
    x, s = noisy_step(at=40, size=20 * UM, seed=5)
    x[15] += 30 * UM
    mask, delta, _ = rs.find_outliers(x[None, :], s[None, :], 6.0)
    assert np.flatnonzero(mask[0]).tolist() == [15]
    assert abs(delta[0, 15] - 30 * UM) < 4 * UM


def test_common_mode_offset_and_harmonic():
    """Per-run common mode: offset follows a uniform layer shift, harmonic
    also follows a sin(phi) modulation."""
    staves, chips, n = 16, 9, 10
    keys = pd.DataFrame([("MVTX", 1, stave, chip, "rphi")
                         for stave in range(staves) for chip in range(chips)],
                        columns=rs.KEY_COLUMNS)
    phi = 2 * np.pi * keys["stave"].to_numpy() / staves
    rng = np.random.default_rng(2)
    X = rng.normal(0.0, 0.5 * UM, (len(keys), n))
    X[:, 5:] += 7 * UM + (3 * UM * np.sin(phi))[:, None]
    E = np.full_like(X, 0.1 * UM)
    rows, *_ , values, _ = rs.common_mode(X, E, keys, rs.Config())
    assert abs(values[0, 7] - 7 * UM) < 0.5 * UM
    rows, _, cm_keys, values, _ = rs.common_mode(
        X, E, keys, rs.Config(common_mode="harmonic"))
    assert cm_keys["component"].tolist() == ["offset", "sin", "cos"]
    assert abs(values[1, 7] - 3 * UM) < 0.5 * UM
    assert np.nanstd(X[:, 7] - rows[:, 7]) < 1 * UM


def test_run_order_from_labels():
    assert rs.run_number("run53007") == 53007
    assert rs.run_number("ana.499_run00053007") == 53007
    assert rs.run_number("Iter") is None


# ---------------------------------------------------- synthetic run series

@pytest.fixture(scope="module")
def run_series(tmp_path_factory):
    """The default synthetic series and its null, scanned by the CLI."""
    out = {}
    for name, extra in (("series", []), ("null", ["--null"])):
        folder = tmp_path_factory.mktemp(name)
        run_module("synthetic.make_synthetic_run_series", "--out-dir",
                   str(folder), *extra)
        tables = sorted(str(path) for path in
                        folder.glob("residual_fits_run*.csv"))
        run_module("stdiag.run_stability", *tables, "--out-dir",
                   str(folder / "scan"), "--run-info",
                   str(folder / "run_info.csv"))
        out[name] = folder / "scan"
    return out


@pytest.mark.parametrize("name, extra", [("series", []),
                                         ("null", ["--null"])])
def test_synthetic_series_matches_truth(run_series, name, extra):
    changepoints = next(run_series[name].glob("changepoints_*.csv"))
    output = run_module("synthetic.check_run_stability", str(changepoints),
                        *extra)
    assert "PASS" in output


def test_synthetic_outputs_and_notes(run_series):
    folder = run_series["series"]
    events = pd.read_csv(next(folder.glob("changepoints_*.csv")),
                         keep_default_na=False)
    assert list(events.columns) == rs.EVENT_COLUMNS
    layer = events.loc[events["class"] == "layer_mode"].iloc[0]
    assert layer["notes"] == "synthetic note: new global alignment tag"
    assert layer["n_members"] > 100          # absorbed sensor steps
    report = next(folder.glob("stability_report_*.md")).read_text()
    assert "no AI" in report and "## Caveats" in report
    for pattern in ("series_*.csv", "detections_*.csv",
                    "stability_summary_*.json"):
        assert next(folder.glob(pattern)).stat().st_size > 0
