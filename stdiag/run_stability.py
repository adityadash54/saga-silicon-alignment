#!/usr/bin/env python3
"""Run-by-run stability of every MVTX chip and INTT sensor (no AI).

Reads one residual_fits_<run>.csv per run (written by
macros/plot_fits_overlaid_stavechip.C), builds for every (detector, layer,
stave, chip, coord) the series of the fitted residual mean (mu) and width
(sigma) versus run, and finds the runs where they change. It runs entirely
locally and sends nothing anywhere.

Steps:
  1. Errors: sigma_tot^2 = fit error^2 + extra run-to-run scatter^2, the
     extra scatter estimated per series from the MAD of run-to-run
     differences (fit errors alone ignore run-to-run systematics).
  2. Outlier runs: a point more than --outlier-nsigma sigma_tot from the
     runs on BOTH sides is an outlier and is left out. A run in which more
     than --bad-run-fraction of a detector's series are outliers is a bad run
     for that detector and is left out everywhere.
  3. Common mode (--common-mode offset|harmonic|none): per run, detector,
     layer and coordinate, a clipped mean over the layer's sensors (offset),
     or a + b sin(phi) + c cos(phi) over stave azimuth (harmonic: layer
     rotation and x/y translation). Its series is scanned on its own
     (layer_mode), and the sensors are scanned both relative to it and raw.
  4. Change points, three methods per series:
       cusum  two-sided tabular CUSUM (Page 1954), reference value
              k = min_shift / 2, restarted after every alarm
       pelt   exact penalised segmentation with a weighted Gaussian mean
              cost (Killick et al. 2012); the scan solves the same
              recursion without pruning, vectorised over series, which
              gives identical change points (tested against pelt())
       bocpd  Bayesian online change-point detection (Adams and MacKay
              2007) with known per-run errors; alarm when the probability
              of a change within the last --bocpd-lag runs passes a
              threshold
     Thresholds are set so that about --expected-false false alarms per
     method are expected in the whole scan for Gaussian noise (Monte Carlo
     calibration; PELT penalty 2 ln(n N / E)).
  5. A change is kept if at least --min-methods methods find it within
     +-window runs, it is larger than the minimum shift (the compare_runs
     values; for sigma --sigma-rel-threshold of the reference width) and its
     significance is above --z-min.
  6. Classes: step, width_step, transient (a step and its return), drift
     (a ramp fits better than steps), outlier_run, bad_run, coherent_stave
     (at least half the chips of a stave step together), multi_stave (three
     or more coherent staves of a layer) and layer_mode (common mode).

Not scanned yet: stave rows (chip = -1) and VERTEX rows.

Usage:
  python -m stdiag.run_stability inputs/residual_fits_run*.csv
  python -m stdiag.run_stability TABLES... --out-dir DIR [--run-info runs.csv]
      [--order runs.txt] [--common-mode harmonic]
Without --out-dir the outputs go to a fresh runs/stability_<first>_to_<last>.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stdiag.compare_runs import (
    DEFAULT_MIN_SHIFT_MM, KEY_COLUMNS, fresh_run_dir, load_fit_table,
    parse_min_shift_overrides, sanitize_for_filename, sha256sum,
    stave_row_mask)
from stdiag.paths import RUNS_DIR

EVENT_COLUMNS = [
    "event_id", "class", "detector", "layer", "stave", "chip", "coord",
    "quantity", "frame", "component", "run_index", "run", "end_run_index",
    "end_run", "delta_mm", "delta_err_mm", "z", "min_shift_mm", "methods",
    "n_methods", "n_members", "group_id", "notes"]
METHODS = ("cusum", "pelt", "bocpd")
MIN_RUNS = 8


@dataclass
class Config:
    baseline_runs: int = 5
    expected_false: float = 1.0
    z_min: float = 5.0
    sigma_rel_threshold: float = 0.20
    min_shift_mm: dict = field(
        default_factory=lambda: dict(DEFAULT_MIN_SHIFT_MM))
    common_mode: str = "offset"
    common_mode_nsigma: float = 3.0
    outlier_nsigma: float = 6.0
    bad_run_fraction: float = 0.20
    window: int = 2
    min_methods: int = 2
    pelt_min_size: int = 2
    bocpd_hazard_runs: float = 100.0
    bocpd_prior_scale: float = 5.0
    bocpd_lag: int = 3
    min_drift_runs: int = 6
    # The ramp's start and end are searched over all run pairs, so it needs
    # this much better BIC than the step model to count as a drift.
    drift_bic_margin: float = 10.0
    max_transient_runs: int = 10
    multi_stave_min: int = 3
    n_sims: int = 20000
    n_sims_bocpd: int = 10000
    seed: int = 12345


# ---------------------------------------------------------------- helpers

def robust_sigma(values: np.ndarray, axis=None) -> np.ndarray:
    """1.4826 x median absolute deviation (NaN-aware)."""
    median = np.nanmedian(values, axis=axis, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=axis)
    return 1.4826 * mad


def logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    peak = np.max(values, axis=axis, keepdims=True)
    safe = np.where(np.isfinite(peak), peak, 0.0)
    with np.errstate(divide="ignore"):
        total = np.log(np.sum(np.exp(values - safe), axis=axis,
                              keepdims=True)) + safe
    return np.squeeze(total, axis=axis)


def pack(X: np.ndarray, S: np.ndarray):
    """Move every row's valid points to the left (gaps removed).

    Returns (Xc, Sc, lengths, Ic): packed values and errors (NaN after each
    row's length) and the run index of every packed point (-1 after it)."""
    valid = np.isfinite(X) & np.isfinite(S) & (S > 0)
    order = np.argsort(~valid, axis=1, kind="stable")
    lengths = valid.sum(axis=1)
    Xc = np.take_along_axis(X, order, axis=1).astype(float)
    Sc = np.take_along_axis(S, order, axis=1).astype(float)
    inside = np.arange(X.shape[1])[None, :] < lengths[:, None]
    Xc[~inside] = np.nan
    Sc[~inside] = np.nan
    Ic = np.where(inside, order, -1)
    return Xc, Sc, lengths, Ic


def unpack(values: np.ndarray, Ic: np.ndarray, n: int, fill=np.nan):
    out = np.full((values.shape[0], n), fill,
                  dtype=float if fill is np.nan else type(fill))
    rows, cols = np.nonzero(Ic >= 0)
    out[rows, Ic[rows, cols]] = values[rows, cols]
    return out


def extra_scatter(X: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Run-to-run scatter beyond the fit errors, per row (mm)."""
    Xc, Ec, lengths, _ = pack(X, E)
    diffs = Xc[:, 1:] - Xc[:, :-1]
    with np.errstate(all="ignore"), _quiet():
        variance = robust_sigma(diffs, axis=1) ** 2 / 2.0
        fit = np.nanmedian(Ec ** 2, axis=1)
    extra = np.sqrt(np.clip(variance - fit, 0.0, None))
    return np.where((lengths >= 4) & np.isfinite(extra), extra, 0.0)


class _quiet:
    """Silence 'All-NaN slice' and 'Mean of empty slice' warnings."""

    def __enter__(self):
        import warnings
        self._context = warnings.catch_warnings()
        self._context.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *args):
        self._context.__exit__(*args)


def total_error(X: np.ndarray, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    extra = extra_scatter(X, E)
    return np.sqrt(E ** 2 + extra[:, None] ** 2), extra


def weighted_mean(x: np.ndarray, s: np.ndarray) -> tuple[float, float]:
    w = 1.0 / s ** 2
    return float(np.sum(w * x) / np.sum(w)), float(1.0 / np.sqrt(np.sum(w)))


# ------------------------------------------------------------- outliers

def find_outliers(X: np.ndarray, S: np.ndarray, nsigma: float):
    """Points that disagree with the runs on both sides.

    Each side is the mean of up to two neighbouring valid runs. Returns
    (mask, delta, z) as (rows, runs) arrays; delta is the point minus the
    mean of both sides."""
    n = X.shape[1]
    Xc, Sc, _, Ic = pack(X, S)

    def shifted(A, k):
        out = np.full_like(A, np.nan)
        if k > 0:
            out[:, k:] = A[:, :-k]
        else:
            out[:, :k] = A[:, -k:]
        return out

    with _quiet(), np.errstate(invalid="ignore"):
        sides = []
        for near, far in ((1, 2), (-1, -2)):
            values = np.stack([shifted(Xc, near), shifted(Xc, far)])
            errors = np.stack([shifted(Sc, near), shifted(Sc, far)])
            sides.append((np.nanmean(values, axis=0),
                          np.sqrt(np.nanmean(errors ** 2, axis=0))))
        flags = []
        for centre, spread in sides:
            flags.append(np.abs(Xc - centre)
                         > nsigma * np.sqrt(Sc ** 2 + spread ** 2))
        mask = flags[0] & flags[1]
        centre = (sides[0][0] + sides[1][0]) / 2.0
        delta = Xc - centre
        z = delta / np.sqrt(Sc ** 2 + (sides[0][1] ** 2
                                       + sides[1][1] ** 2) / 4.0)
    return (unpack(mask.astype(float), Ic, n, 0.0) > 0,
            unpack(delta, Ic, n), unpack(z, Ic, n))


# ------------------------------------------------------------ common mode

def common_mode(X: np.ndarray, E: np.ndarray, keys: pd.DataFrame,
                cfg: Config):
    """Per run, detector, layer and coordinate common mode.

    Returns (cm_rows, cm_err_rows, cm_keys, CM, CME): the common mode at
    every sensor (rows x runs), its error, and the common-mode series
    (one row per detector, layer, coord and component)."""
    n = X.shape[1]
    harmonic = cfg.common_mode == "harmonic"
    components = ["offset", "sin", "cos"] if harmonic else ["offset"]
    cm_rows = np.full_like(X, np.nan)
    cm_err_rows = np.full_like(X, np.nan)
    series_keys, series_values, series_errors = [], [], []
    groups = keys.groupby(["detector", "layer", "coord"], sort=True).indices
    for (detector, layer, coord), index in groups.items():
        staves = keys["stave"].to_numpy()[index]
        phi = 2.0 * np.pi * staves / (staves.max() + 1)
        design = (np.column_stack([np.ones_like(phi), np.sin(phi),
                                   np.cos(phi)]) if harmonic
                  else np.ones((len(index), 1)))
        values = np.full((len(components), n), np.nan)
        errors = np.full((len(components), n), np.nan)
        for run in range(n):
            column = X[index, run]
            good = np.isfinite(column)
            if good.sum() < 3 * design.shape[1]:
                continue
            keep = good.copy()
            for _ in range(3):
                A, y = design[keep], column[keep]
                coef, *_ = np.linalg.lstsq(A, y, rcond=None)
                residual = column - design @ coef
                spread = robust_sigma(residual[good])
                if not spread > 0:
                    break
                new_keep = good & (np.abs(residual)
                                   <= cfg.common_mode_nsigma * spread)
                if np.array_equal(new_keep, keep):
                    break
                keep = new_keep
            A, y = design[keep], column[keep]
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            dof = max(1, keep.sum() - A.shape[1])
            variance = np.sum((y - A @ coef) ** 2) / dof
            covariance = variance * np.linalg.inv(A.T @ A)
            values[:, run] = coef
            errors[:, run] = np.sqrt(np.diag(covariance))
            cm_rows[index, run] = design @ coef
            cm_err_rows[index, run] = np.sqrt(np.einsum(
                "ij,jk,ik->i", design, covariance, design))
        for position, component in enumerate(components):
            series_keys.append((detector, layer, coord, component))
            series_values.append(values[position])
            series_errors.append(errors[position])
    cm_keys = pd.DataFrame(series_keys,
                           columns=["detector", "layer", "coord", "component"])
    cm_keys["stave"] = -1
    cm_keys["chip"] = -1
    return (cm_rows, cm_err_rows, cm_keys, np.array(series_values),
            np.array(series_errors))


# --------------------------------------------------------------- methods

# The CUSUM is self-starting (Hawkins 1987): the reference level of run t is
# the weighted mean of all runs since the last restart (before t), and runs
# are tested only after `baseline` warm-up runs. A fixed baseline from the
# first few runs carries its own error (sigma/sqrt(baseline)) into every
# later run; when that error is comparable to k, the null CUSUM grows
# steadily and the calibrated threshold becomes very large.

def cusum_trace(x: np.ndarray, s: np.ndarray, min_shift: float,
                baseline: int = 5):
    """S+, S- and the reference level of a self-starting two-sided CUSUM
    without restarts (for plots). Runs without a value (NaN) are skipped:
    the sums carry over and the returned trace is NaN there."""
    n = len(x)
    plus, minus, level = (np.full(n, np.nan) for _ in range(3))
    upper = lower = sum_w = sum_wx = 0.0
    count = 0
    for t in range(n):
        if not (np.isfinite(x[t]) and np.isfinite(s[t])):
            continue
        if count >= baseline:
            mu0 = sum_wx / sum_w
            z = (x[t] - mu0) / s[t]
            k = 0.5 * min_shift / s[t]
            upper = max(0.0, upper + z - k)
            lower = max(0.0, lower - z - k)
            plus[t], minus[t], level[t] = upper, lower, mu0
        sum_w += 1.0 / s[t] ** 2
        sum_wx += x[t] / s[t] ** 2
        count += 1
    return plus, minus, level


def cusum_batch(Xc, Sc, lengths, min_shift, h, baseline):
    """Self-starting two-sided CUSUM with restarts on packed rows.

    Returns, per row, the packed indices where an excursion that crossed h
    started (the first run after the CUSUM was last zero). After an alarm
    the reference level restarts from that run."""
    rows, n = Xc.shape
    starts = [[] for _ in range(rows)]
    sum_w, sum_wx = np.zeros(rows), np.zeros(rows)
    count = np.zeros(rows, int)
    upper, lower = np.zeros(rows), np.zeros(rows)
    last_upper = np.full(rows, baseline)
    last_lower = np.full(rows, baseline)
    for t in range(n):
        active = t < lengths
        if not active.any():
            break
        ready = active & (count >= baseline)
        with np.errstate(invalid="ignore", divide="ignore"):
            mu0 = sum_wx / sum_w
            z = (Xc[:, t] - mu0) / Sc[:, t]
            k = 0.5 * min_shift / Sc[:, t]
            upper = np.where(ready, np.maximum(0.0, upper + z - k), upper)
            lower = np.where(ready, np.maximum(0.0, lower - z - k), lower)
            w = np.where(active, 1.0 / Sc[:, t] ** 2, 0.0)
            sum_w += w
            sum_wx += np.where(active, w * Xc[:, t], 0.0)
        count += active
        last_upper = np.where(ready & (upper == 0), t + 1, last_upper)
        last_lower = np.where(ready & (lower == 0), t + 1, last_lower)
        alarm = ready & ((upper > h) | (lower > h))
        for row in np.flatnonzero(alarm):
            start = int(last_upper[row] if upper[row] >= lower[row]
                        else last_lower[row])
            start = min(start, t)
            starts[row].append(start)
            weights = 1.0 / Sc[row, start:t + 1] ** 2
            sum_w[row] = weights.sum()
            sum_wx[row] = (weights * Xc[row, start:t + 1]).sum()
            count[row] = t + 1 - start
            upper[row] = lower[row] = 0.0
            last_upper[row] = last_lower[row] = max(t + 1, start + baseline)
    return starts


def cusum_null_statistic(Z: np.ndarray, k: float, baseline: int,
                         S: np.ndarray | None = None):
    """Largest self-starting CUSUM excursion of unit-normal rows
    (calibration, no restarts).

    k is half the minimum shift in units of the true noise. With S (one
    estimated error per row), z and k are computed with the estimated error,
    as the scan does."""
    S = np.ones(len(Z)) if S is None else S
    upper = np.zeros(len(Z))
    lower = np.zeros(len(Z))
    best = np.zeros(len(Z))
    total = np.zeros(len(Z))
    for t in range(Z.shape[1]):
        if t >= baseline:
            z = (Z[:, t] - total / t) / S
            upper = np.maximum(0.0, upper + z - k / S)
            lower = np.maximum(0.0, lower - z - k / S)
            best = np.maximum(best, np.maximum(upper, lower))
        total += Z[:, t]
    return best


def simulated_errors(Z: np.ndarray, fit_fraction: float) -> np.ndarray:
    """Per-row error estimated as the scan does (extra_scatter), for rows of
    unit-normal noise whose fit error is fit_fraction of the true noise."""
    with _quiet():
        scatter2 = robust_sigma(np.diff(Z, axis=1), axis=1) ** 2 / 2.0
    extra2 = np.clip(scatter2 - fit_fraction ** 2, 0.0, None)
    return np.sqrt(fit_fraction ** 2 + extra2)


def pelt_null_statistic(Z: np.ndarray, S: np.ndarray, min_size: int):
    """Largest cost reduction of a single change (PELT penalty units)."""
    n = Z.shape[1]
    w = np.broadcast_to((1.0 / S ** 2)[:, None], Z.shape)
    centred = Z - Z.mean(axis=1, keepdims=True)
    zero = np.zeros((len(Z), 1))
    S0 = np.concatenate([zero, np.cumsum(w, axis=1)], axis=1)
    S1 = np.concatenate([zero, np.cumsum(w * centred, axis=1)], axis=1)
    S2 = np.concatenate([zero, np.cumsum(w * centred ** 2, axis=1)], axis=1)
    whole = _cost_matrix(S0, S1, S2, np.array([0]), n)[:, 0]
    splits = np.arange(min_size, n - min_size + 1)
    left = np.stack([_cost_matrix(S0, S1, S2, np.array([0]), t)[:, 0]
                     for t in splits], axis=1)
    right = np.stack([_cost_matrix(S0, S1, S2, np.array([t]), n)[:, 0]
                      for t in splits], axis=1)
    return whole - np.min(left + right, axis=1)


def _segment_cost(S0, S1, S2, a, b):
    weight = S0[..., b] - S0[..., a]
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = (S2[..., b] - S2[..., a]
                - (S1[..., b] - S1[..., a]) ** 2 / weight)
    return np.where(weight > 0, np.maximum(cost, 0.0), np.inf)


def pelt(x: np.ndarray, s: np.ndarray, beta: float,
         min_size: int = 2) -> list[int]:
    """PELT (Killick, Fearnhead and Eckley 2012) for one series.

    Cost of a segment: sum w (x - weighted mean)^2 with w = 1/s^2. Returns
    the indices where new segments start."""
    n = len(x)
    w = 1.0 / s ** 2
    centred = x - np.average(x, weights=w)
    S0 = np.concatenate([[0.0], np.cumsum(w)])
    S1 = np.concatenate([[0.0], np.cumsum(w * centred)])
    S2 = np.concatenate([[0.0], np.cumsum(w * centred ** 2)])
    F = np.full(n + 1, np.inf)
    F[0] = -beta
    last = np.zeros(n + 1, int)
    candidates = np.array([0])
    for t in range(min_size, n + 1):
        eligible = candidates[t - candidates >= min_size]
        totals = (F[eligible] + _segment_cost(S0, S1, S2, eligible, t)
                  + beta)
        best = int(np.argmin(totals))
        F[t], last[t] = totals[best], eligible[best]
        cost = np.where(t - candidates >= min_size,
                        F[candidates]
                        + _segment_cost(S0, S1, S2, candidates, t),
                        -np.inf)
        candidates = np.append(candidates[cost <= F[t]], t - min_size + 1)
    return _backtrack(last, n)


def _backtrack(last: np.ndarray, end: int) -> list[int]:
    points = []
    while end > 0:
        end = int(last[end])
        if end > 0:
            points.append(end)
    return sorted(points)


def segment_batch(Xc, Sc, lengths, beta, min_size):
    """The PELT recursion without pruning, vectorised over packed rows.

    Gives the same optimum as pelt() for every row; beta is per row."""
    rows, n = Xc.shape
    inside = np.isfinite(Xc)
    w = np.where(inside, 1.0 / np.where(inside, Sc, 1.0) ** 2, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.nansum(w * np.nan_to_num(Xc), axis=1) / w.sum(axis=1)
    centred = np.where(inside, Xc - mean[:, None], 0.0)
    zero = np.zeros((rows, 1))
    S0 = np.concatenate([zero, np.cumsum(w, axis=1)], axis=1)
    S1 = np.concatenate([zero, np.cumsum(w * centred, axis=1)], axis=1)
    S2 = np.concatenate([zero, np.cumsum(w * centred ** 2, axis=1)], axis=1)
    F = np.full((rows, n + 1), np.inf)
    F[:, 0] = -beta
    last = np.zeros((rows, n + 1), int)
    for t in range(min_size, n + 1):
        taus = np.arange(0, t - min_size + 1)
        cost = _cost_matrix(S0, S1, S2, taus, t)
        totals = F[:, taus] + cost + beta[:, None]
        best = np.argmin(totals, axis=1)
        F[:, t] = totals[np.arange(rows), best]
        last[:, t] = taus[best]
    return [_backtrack(last[row], int(lengths[row])) if lengths[row] > 0
            else [] for row in range(rows)]


def _cost_matrix(S0, S1, S2, taus, t):
    weight = S0[:, t:t + 1] - S0[:, taus]
    with np.errstate(divide="ignore", invalid="ignore"):
        cost = (S2[:, t:t + 1] - S2[:, taus]
                - (S1[:, t:t + 1] - S1[:, taus]) ** 2 / weight)
    return np.where(weight > 0, np.maximum(cost, 0.0), np.inf)


def bocpd_batch(X, S, mu0, tau, hazard, lag, return_posterior=False):
    """Bayesian online change-point detection (Adams and MacKay 2007).

    Gaussian observations with known per-run errors S and a Normal(mu0,
    tau^2) prior on each segment mean; constant hazard. The run length r
    counts the observations of the current segment including the latest
    one. Runs where X is NaN are skipped. Returns q (rows x runs): the
    probability that a change happened within the last lag + 1
    observations (r <= lag + 1), and the most probable r; with
    return_posterior also the run-length posterior of row 0."""
    rows, n = X.shape
    log_h, log_1mh = math.log(hazard), math.log1p(-hazard)
    log_p = np.full((rows, n + 1), -np.inf)
    log_p[:, 0] = 0.0
    mean = np.repeat(mu0[:, None], n + 1, axis=1).astype(float)
    var = np.repeat((tau ** 2)[:, None], n + 1, axis=1).astype(float)
    q = np.full((rows, n), np.nan)
    r_map = np.full((rows, n), -1)
    posterior = np.full((n, n + 1), np.nan) if return_posterior else None
    for t in range(n):
        ok = np.isfinite(X[:, t]) & np.isfinite(S[:, t])
        if not ok.any():
            continue
        x = X[ok, t][:, None]
        s2 = S[ok, t][:, None] ** 2
        lp, m, v = log_p[ok], mean[ok], var[ok]
        predictive_var = v + s2
        log_pred = -0.5 * (np.log(2.0 * np.pi * predictive_var)
                           + (x - m) ** 2 / predictive_var)
        prior_var = tau[ok][:, None] ** 2 + s2
        log_pred0 = -0.5 * (np.log(2.0 * np.pi * prior_var)
                            + (x - mu0[ok][:, None]) ** 2 / prior_var)[:, 0]
        new = np.full_like(lp, -np.inf)
        new[:, 1:] = lp[:, :-1] + log_pred[:, :-1] + log_1mh
        change = logsumexp(lp[:, 1:], axis=1) + log_h + log_pred0
        new[:, 1] = np.logaddexp(new[:, 1], change)
        new -= logsumexp(new, axis=1)[:, None]
        post_var = 1.0 / (1.0 / v + 1.0 / s2)
        post_mean = post_var * (m / v + x / s2)
        m_new, v_new = np.empty_like(m), np.empty_like(v)
        m_new[:, 0], v_new[:, 0] = mu0[ok], tau[ok] ** 2
        m_new[:, 1:], v_new[:, 1:] = post_mean[:, :-1], post_var[:, :-1]
        log_p[ok], mean[ok], var[ok] = new, m_new, v_new
        q[ok, t] = np.exp(logsumexp(new[:, :lag + 2], axis=1))
        r_map[ok, t] = np.argmax(new, axis=1)
        if return_posterior and ok[0]:
            posterior[t] = np.exp(new[0])
    if return_posterior:
        return q, r_map, posterior
    return q, r_map


def bocpd_locations(q: np.ndarray, r_map: np.ndarray, threshold: float,
                    lag: int) -> list[int]:
    """Packed indices of changes from one packed row of bocpd_batch.

    An alarm episode is a run of consecutive points with q > threshold;
    its change starts at the last point of the episode minus its most
    probable run length + 1. Points 0..lag are skipped (q = 1 there)."""
    alarm = np.zeros(len(q), bool)
    alarm[lag + 1:] = np.nan_to_num(q[lag + 1:]) > threshold
    locations = []
    t = 0
    while t < len(q):
        if alarm[t]:
            end = t
            while end + 1 < len(q) and alarm[end + 1]:
                end += 1
            locations.append(max(0, end - int(r_map[end]) + 1))
            t = end + 1
        else:
            t += 1
    return sorted(set(locations))


def bocpd_null_statistic(Z: np.ndarray, tau: float, cfg: Config,
                         S: np.ndarray | None = None):
    """Largest BOCPD change probability of unit-normal rows; tau in units
    of the true noise; S = one estimated error per row (default 1)."""
    rows = len(Z)
    S = np.ones(rows) if S is None else S
    mu0 = Z[:, :cfg.baseline_runs].mean(axis=1)
    q, _ = bocpd_batch(Z, np.broadcast_to(S[:, None], Z.shape), mu0,
                       np.full(rows, tau), 1.0 / cfg.bocpd_hazard_runs,
                       cfg.bocpd_lag)
    return np.nanmax(q[:, cfg.bocpd_lag + 1:], axis=1)


def tail_quantile(samples: np.ndarray, p: float) -> float:
    """The (1 - p) quantile; below 10 expected exceedances, an exponential
    tail fitted above the 99% quantile."""
    samples = np.sort(np.asarray(samples, float))
    if p * len(samples) >= 10:
        return float(np.quantile(samples, 1.0 - p))
    u = float(np.quantile(samples, 0.99))
    excess = samples[samples > u] - u
    if len(excess) < 5 or excess.mean() <= 0:
        return float(samples[-1])
    return u + float(excess.mean()) * math.log(0.01 / p)


def calibrate(n_runs: int, k: float, tau_over_s: float, p: float,
              cfg: Config, rng: np.random.Generator,
              fit_fraction: float = 1.0) -> dict:
    """Thresholds giving a false-alarm probability p per series for Gaussian
    noise: CUSUM h, BOCPD probability threshold and a Monte Carlo PELT
    penalty. The simulated series use an error estimated from the series
    itself, like the scan (fit_fraction = fit error / true noise), so the
    thresholds include the uncertainty of that estimate."""
    Z = rng.standard_normal((cfg.n_sims, n_runs))
    S = simulated_errors(Z, fit_fraction)
    h = max(2.0, tail_quantile(cusum_null_statistic(Z, k, cfg.baseline_runs,
                                                    S), p))
    beta = tail_quantile(pelt_null_statistic(Z, S, cfg.pelt_min_size), p)
    rows = cfg.n_sims_bocpd
    stat = bocpd_null_statistic(Z[:rows], tau_over_s, cfg, S[:rows])
    clipped = np.clip(stat, 1e-12, 1.0 - 1e-12)
    logit = tail_quantile(np.log(clipped / (1.0 - clipped)), p)
    p_threshold = float(np.clip(1.0 / (1.0 + math.exp(-logit)), 0.5,
                                1.0 - 1e-9))
    return {"cusum_h": h, "bocpd_threshold": p_threshold,
            "pelt_beta_mc": float(beta)}


# ----------------------------------------------------- per-series logic

def segment_means(x, s, points):
    bounds = [0, *points, len(x)]
    means, errors = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        mean, error = weighted_mean(x[a:b], s[a:b])
        means.append(mean)
        errors.append(error)
    return np.array(means), np.array(errors)


def prune_changes(x, s, points, min_shift, z_min):
    """Drop, one at a time, the weakest change that is not larger than the
    minimum shift or not significant, re-fitting the segments each time."""
    points = sorted(points)
    while points:
        means, errors = segment_means(x, s, points)
        delta = np.diff(means)
        error = np.sqrt(errors[:-1] ** 2 + errors[1:] ** 2)
        weak = (np.abs(delta) <= min_shift) | (np.abs(delta) / error <= z_min)
        if not weak.any():
            return points, delta, error
        candidates = np.flatnonzero(weak)
        points.pop(int(candidates[np.argmin(np.abs(delta[candidates])
                                            / error[candidates])]))
    return [], np.array([]), np.array([])


def chi2_steps(x, s, points):
    means, _ = segment_means(x, s, points)
    bounds = [0, *points, len(x)]
    model = np.concatenate([np.full(b - a, m) for a, b, m in
                            zip(bounds[:-1], bounds[1:], means)])
    return float(np.sum(((x - model) / s) ** 2))


def ramp_fit(x, s):
    """Best ramp: constant a, linear from t1 to t2, constant a + d after.

    Returns (t1, t2, a, d, d_err, chi2), packed indices."""
    n = len(x)
    t = np.arange(n)
    t1, t2 = np.triu_indices(n, k=1)
    g = np.clip((t[None, :] - t1[:, None]) / (t2 - t1)[:, None], 0.0, 1.0)
    w = 1.0 / s ** 2
    Sw, Sx, Sxx = w.sum(), (w * x).sum(), (w * x * x).sum()
    Sg, Sgg, Sgx = (g * w).sum(1), (g * g * w).sum(1), (g * w * x).sum(1)
    det = Sw * Sgg - Sg ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (Sw * Sgx - Sg * Sx) / det
        a = (Sx - d * Sg) / Sw
        chi2 = np.where(det > 0, Sxx - a * Sx - d * Sgx, np.inf)
    best = int(np.argmin(chi2))
    return (int(t1[best]), int(t2[best]), float(a[best]), float(d[best]),
            float(np.sqrt(Sw / det[best])), float(chi2[best]))


def cluster_detections(found: dict[str, list[int]], window: int):
    """Group the methods' change locations that lie within +-window."""
    items = sorted((location, method) for method, locations in found.items()
                   for location in locations)
    clusters = []
    for location, method in items:
        if clusters and location - clusters[-1]["max"] <= window:
            cluster = clusters[-1]
            cluster["locations"].append(location)
            cluster["methods"].add(method)
            cluster["max"] = location
            if method == "pelt":
                cluster["pelt"] = location
        else:
            clusters.append({"locations": [location], "methods": {method},
                             "max": location,
                             "pelt": location if method == "pelt" else None})
    for cluster in clusters:
        cluster["location"] = (cluster["pelt"] if cluster["pelt"] is not None
                               else int(np.median(cluster["locations"])))
    return clusters


def classify_row(x, s, found, min_shift, cfg, quantity):
    """Turn one packed series and its method detections into events.

    Returns a list of dicts with packed indices start/end, delta, error,
    methods and class."""
    if not any(found.values()):
        return []
    clusters = cluster_detections(found, cfg.window)
    confirmed = [cluster for cluster in clusters
                 if len(cluster["methods"]) >= cfg.min_methods]
    points = sorted({cluster["location"] for cluster in confirmed
                     if 0 < cluster["location"] < len(x)})
    points, delta, error = prune_changes(x, s, points, min_shift, cfg.z_min)
    methods_at = {}
    for point in points:
        nearest = min(confirmed, key=lambda c: abs(c["location"] - point))
        methods_at[point] = ";".join(m for m in METHODS
                                     if m in nearest["methods"])

    n = len(x)
    if n >= 2 * cfg.min_drift_runs:
        t1, t2, a, d, d_err, chi2_ramp = ramp_fit(x, s)
        bic_ramp = chi2_ramp + 4.0 * math.log(n)
        bic_steps = (chi2_steps(x, s, points)
                     + (2 * len(points) + 1) * math.log(n))
        if (t2 - t1 >= cfg.min_drift_runs
                and bic_ramp + cfg.drift_bic_margin < bic_steps
                and abs(d) > min_shift and abs(d) / d_err > cfg.z_min):
            all_methods = set().union(*(c["methods"] for c in clusters))
            return [{"class": "drift", "start": t1, "end": t2, "delta": d,
                     "error": d_err,
                     "methods": ";".join(m for m in METHODS
                                         if m in all_methods)}]

    events = [{"class": "step" if quantity == "mu" else "width_step",
               "start": point, "end": None, "delta": float(dl),
               "error": float(er), "methods": methods_at[point]}
              for point, dl, er in zip(points, delta, error)]
    merged, i = [], 0
    while i < len(events):
        first = events[i]
        if i + 1 < len(events):
            second = events[i + 1]
            if (np.sign(first["delta"]) != np.sign(second["delta"])
                    and abs(first["delta"] + second["delta"]) < min_shift
                    and second["start"] - first["start"]
                    <= cfg.max_transient_runs):
                methods = set(first["methods"].split(";")) | set(
                    second["methods"].split(";"))
                merged.append({**first, "class": "transient",
                               "end": second["start"],
                               "methods": ";".join(m for m in METHODS
                                                   if m in methods)})
                i += 2
                continue
        merged.append(first)
        i += 1
    return merged


# ------------------------------------------------------------- scanning

@dataclass
class Frame:
    """A set of series scanned together."""
    name: str            # relative, raw or common_mode
    quantity: str        # mu or sigma
    keys: pd.DataFrame   # detector, layer, stave, chip, coord (+component)
    X: np.ndarray        # rows x runs, NaN = no value
    E: np.ndarray        # fit (or common-mode) errors
    min_shift: np.ndarray
    S: np.ndarray = None
    extra: np.ndarray = None
    thresholds: pd.DataFrame = None


def calibrate_frames(frames: list[Frame], n_runs: int, cfg: Config) -> list:
    """Set per-row thresholds; returns the calibration records."""
    n_series = sum(len(frame.keys) for frame in frames)
    p = cfg.expected_false / max(1, n_series)
    records, cache = [], {}
    for frame in frames:
        frame.S, frame.extra = total_error(frame.X, frame.E)
        with _quiet():
            typical_s = np.nanmedian(frame.S, axis=1)
            fit_fraction = np.nanmedian(frame.E / frame.S, axis=1)
        groups = frame.keys.groupby(["detector", "coord"], sort=True).indices
        h = np.full(len(frame.keys), np.nan)
        p_thr = np.full(len(frame.keys), np.nan)
        beta_mc = np.full(len(frame.keys), np.nan)
        for (detector, coord), index in groups.items():
            ratio = frame.min_shift[index] / typical_s[index]
            good = np.isfinite(ratio)
            if not good.any():
                continue
            # Groups with similar parameters share one calibration; k and
            # the fit fraction are rounded down (larger thresholds).
            k = math.floor(2.0 * np.median(ratio[good])) / 4.0
            fraction = math.floor(10.0 * float(np.nanmedian(
                fit_fraction[index]))) / 10.0
            key = (k, fraction)
            if key not in cache:
                cache[key] = calibrate(
                    n_runs, max(k, 0.25), cfg.bocpd_prior_scale * 2.0 * k, p,
                    cfg, np.random.default_rng(cfg.seed), max(fraction, 0.05))
            result = cache[key]
            h[index] = result["cusum_h"]
            p_thr[index] = result["bocpd_threshold"]
            beta_mc[index] = result["pelt_beta_mc"]
            records.append({"frame": frame.name, "quantity": frame.quantity,
                            "detector": detector, "coord": coord,
                            "series": len(index), "k": k,
                            "tau_over_sigma": cfg.bocpd_prior_scale * 2 * k,
                            "fit_fraction": fraction, **result})
        lengths = np.isfinite(frame.X).sum(axis=1)
        beta = 2.0 * np.log(np.maximum(lengths, 2) * n_series
                            / cfg.expected_false)
        beta = np.fmax(beta, beta_mc)
        frame.thresholds = pd.DataFrame({"cusum_h": h, "bocpd_threshold":
                                         p_thr, "pelt_beta": beta})
    return records


def scan_frame(frame: Frame, runs: pd.DataFrame, cfg: Config):
    """Run the three methods and the classification on one frame.

    Returns (events, detections, q): event dicts with run indices, per-method
    detections and the BOCPD probability (rows x runs)."""
    n = frame.X.shape[1]
    Xc, Sc, lengths, Ic = pack(frame.X, frame.S)
    thresholds = frame.thresholds
    usable = lengths >= MIN_RUNS
    h = np.where(np.isfinite(thresholds["cusum_h"]), thresholds["cusum_h"],
                 np.inf)
    cusum = cusum_batch(Xc, Sc, lengths, frame.min_shift, h,
                        cfg.baseline_runs)
    pelt_points = segment_batch(Xc, Sc, lengths,
                                thresholds["pelt_beta"].to_numpy(),
                                cfg.pelt_min_size)
    mu0 = np.array([weighted_mean(Xc[r, :min(lengths[r], cfg.baseline_runs)],
                                  Sc[r, :min(lengths[r], cfg.baseline_runs)])[0]
                    if lengths[r] > 0 else 0.0 for r in range(len(Xc))])
    tau = cfg.bocpd_prior_scale * frame.min_shift
    q, r_map = bocpd_batch(Xc, Sc, mu0, tau, 1.0 / cfg.bocpd_hazard_runs,
                           cfg.bocpd_lag)
    events, detections = [], []
    for row in np.flatnonzero(usable):
        length = lengths[row]
        found = {
            "cusum": [p for p in cusum[row] if 0 < p < length],
            "pelt": pelt_points[row],
            "bocpd": [p for p in bocpd_locations(
                q[row, :length], r_map[row, :length],
                thresholds["bocpd_threshold"].iat[row], cfg.bocpd_lag)
                if 0 < p < length],
        }
        for method, points in found.items():
            for point in points:
                detections.append({"row": row, "method": method,
                                   "run_index": int(Ic[row, point])})
        x, s = Xc[row, :length], Sc[row, :length]
        for event in classify_row(x, s, found, frame.min_shift[row], cfg,
                                  frame.quantity):
            event["row"] = row
            event["run_index"] = int(Ic[row, event["start"]])
            event["end_run_index"] = (int(Ic[row, event["end"]])
                                      if event["end"] is not None else -1)
            events.append(event)
    return events, detections, unpack(q, Ic, n)


def build_matrices(long: pd.DataFrame, n_runs: int):
    """Chip/sensor rows only (no stave rows, no VERTEX)."""
    chips = long.loc[(long["detector"] != "VERTEX") & ~stave_row_mask(long)]
    chips = chips.copy()
    good = ((chips["fit_status"] == "ok") & (chips["mu_err_mm"] > 0)
            & (chips["sigma_err_mm"] > 0))
    for column in ("mu_mm", "mu_err_mm", "sigma_mm", "sigma_err_mm"):
        chips.loc[~good, column] = np.nan
    indexed = chips.set_index(KEY_COLUMNS + ["run_index"])
    matrices = {}
    keys = None
    for name, column in (("mu", "mu_mm"), ("mu_err", "mu_err_mm"),
                         ("sigma", "sigma_mm"), ("sigma_err", "sigma_err_mm"),
                         ("entries", "entries")):
        wide = indexed[column].unstack("run_index").reindex(
            columns=range(n_runs))
        if keys is None:
            keys = wide.index.to_frame(index=False)
        matrices[name] = wide.to_numpy(float)
    return keys, matrices


def load_run_series(paths: list[Path], order: list[str] | None = None):
    """Read the per-run fit tables; return (long table, runs table)."""
    tables = [load_fit_table(path) for path in paths]
    labels = [table["label"].iloc[0] for table in tables]
    if len(set(labels)) != len(labels):
        sys.exit(f"ERROR: run labels must be distinct, got {labels}")
    if order is not None:
        missing = [label for label in labels if label not in order]
        if missing:
            sys.exit(f"ERROR: labels not in the --order file: {missing}")
        rank = [order.index(label) for label in labels]
        numbers = [run_number(label) for label in labels]
    else:
        numbers = [run_number(label) for label in labels]
        if any(number is None for number in numbers):
            sys.exit("ERROR: cannot read a run number from every label "
                     f"({labels}); give --order FILE (one label per line)")
        if len(set(numbers)) != len(numbers):
            sys.exit(f"ERROR: duplicate run numbers in {labels}; give --order")
        rank = numbers
    sequence = sorted(range(len(tables)), key=lambda i: rank[i])
    runs = pd.DataFrame({
        "run_index": range(len(tables)),
        "run": [labels[i] for i in sequence],
        "run_number": [numbers[i] for i in sequence],
        "csv": [paths[i].name for i in sequence],
        "sha256": [sha256sum(paths[i]) for i in sequence],
    })
    frames = []
    for run_index, i in enumerate(sequence):
        frames.append(tables[i].assign(run_index=run_index))
    return pd.concat(frames, ignore_index=True), runs


def run_number(label: str) -> int | None:
    digits = re.findall(r"\d+", str(label))
    return int(digits[-1]) if digits else None


def scan(long: pd.DataFrame, runs: pd.DataFrame, cfg: Config) -> dict:
    """The full scan; returns events, detections, series and summary data."""
    n = len(runs)
    if n < MIN_RUNS:
        sys.exit(f"ERROR: need at least {MIN_RUNS} runs, got {n}")
    keys, M = build_matrices(long, n)
    detector = keys["detector"].to_numpy()
    coord = keys["coord"].to_numpy()
    mu_shift = np.array([cfg.min_shift_mm[(d, c)]
                         for d, c in zip(detector, coord)])

    # 1-2. errors, outliers and bad runs on the raw series
    raw = {"mu": (M["mu"], M["mu_err"]), "sigma": (M["sigma"], M["sigma_err"])}
    first_pass = {name: find_outliers(X, total_error(X, E)[0],
                                      cfg.outlier_nsigma)[0]
                  for name, (X, E) in raw.items()}
    any_outlier = first_pass["mu"] | first_pass["sigma"]
    valid = np.isfinite(M["mu"])
    bad_runs = {}
    bad_mask = np.zeros_like(valid)
    for name in np.unique(detector):
        rows = detector == name
        with np.errstate(invalid="ignore", divide="ignore"):
            fraction = any_outlier[rows].sum(0) / valid[rows].sum(0)
        bad = np.flatnonzero(np.nan_to_num(fraction) > cfg.bad_run_fraction)
        bad_runs[str(name)] = {int(run): float(fraction[run]) for run in bad}
        bad_mask[np.ix_(rows, bad)] = True

    cleaned, outliers = {}, {}
    for name, (X, E) in raw.items():
        masked = np.where(bad_mask, np.nan, X)
        S, _ = total_error(masked, E)
        mask, delta, z = find_outliers(masked, S, cfg.outlier_nsigma)
        outliers[name] = (mask, delta, z, S)
        cleaned[name] = np.where(mask, np.nan, masked)

    # 3. common mode
    frames = []
    cm = None
    if cfg.common_mode != "none":
        cm_rows, cm_err_rows, cm_keys, CM, CME = common_mode(
            cleaned["mu"], M["mu_err"], keys, cfg)
        relative = cleaned["mu"] - cm_rows
        relative_err = np.sqrt(M["mu_err"] ** 2 + np.nan_to_num(
            cm_err_rows) ** 2)
        frames.append(Frame("relative", "mu", keys, relative, relative_err,
                            mu_shift))
        frames.append(Frame("raw", "mu", keys, cleaned["mu"], M["mu_err"],
                            mu_shift))
        cm_shift = np.array([cfg.min_shift_mm[(d, c)] for d, c in
                             zip(cm_keys["detector"], cm_keys["coord"])])
        frames.append(Frame("common_mode", "mu", cm_keys, CM, CME, cm_shift))
        cm = {"rows": cm_rows, "keys": cm_keys, "values": CM, "errors": CME}
    else:
        frames.append(Frame("raw", "mu", keys, cleaned["mu"], M["mu_err"],
                            mu_shift))
    sigma_ref = np.array([
        weighted_mean(*_first_valid(cleaned["sigma"][row], M["sigma_err"][row],
                                    cfg.baseline_runs))[0]
        if np.isfinite(cleaned["sigma"][row]).any() else np.nan
        for row in range(len(keys))])
    frames.append(Frame("raw", "sigma", keys, cleaned["sigma"],
                        M["sigma_err"], cfg.sigma_rel_threshold * sigma_ref))

    calibration = calibrate_frames(frames, n, cfg)
    results = {}
    for frame in frames:
        results[(frame.name, frame.quantity)] = scan_frame(frame, runs, cfg)

    events = assemble_events(frames, results, keys, outliers, bad_runs, runs,
                             cfg)
    detections = pd.concat([
        pd.DataFrame(found).assign(frame=frame.name, quantity=frame.quantity)
        .merge(frame.keys.reset_index(names="row"), on="row")
        for frame, (_, found, _) in zip(frames, results.values()) if found],
        ignore_index=True) if any(r[1] for r in results.values()) else \
        pd.DataFrame(columns=["row", "method", "run_index", "frame",
                              "quantity", *KEY_COLUMNS])
    return {"keys": keys, "matrices": M, "frames": frames,
            "results": results, "events": events, "detections": detections,
            "bad_runs": bad_runs, "outliers": outliers, "common_mode": cm,
            "calibration": calibration, "cleaned": cleaned}


def _first_valid(x, e, count):
    good = np.flatnonzero(np.isfinite(x) & np.isfinite(e) & (e > 0))[:count]
    return x[good], e[good]


def assemble_events(frames, results, keys, outliers, bad_runs, runs, cfg):
    """Events of every frame -> one table with dedup and grouping."""
    rows = []
    by_frame = {(frame.name, frame.quantity): frame for frame in frames}
    for (name, quantity), (events, _, _) in results.items():
        frame = by_frame[(name, quantity)]
        for event in events:
            key = frame.keys.iloc[event["row"]]
            rows.append({
                "class": ("layer_mode" if name == "common_mode"
                          else event["class"]),
                "detector": key["detector"], "layer": int(key["layer"]),
                "stave": int(key["stave"]), "chip": int(key["chip"]),
                "coord": key["coord"], "quantity": quantity, "frame": name,
                "component": key.get("component", ""),
                "run_index": event["run_index"],
                "end_run_index": event["end_run_index"],
                "delta_mm": event["delta"], "delta_err_mm": event["error"],
                "min_shift_mm": float(frame.min_shift[event["row"]]),
                "methods": event["methods"],
                "layer_mode_class": event["class"] if name == "common_mode"
                else "", "row": event["row"]})
    table = pd.DataFrame(rows, columns=[
        "class", "detector", "layer", "stave", "chip", "coord", "quantity",
        "frame", "component", "run_index", "end_run_index", "delta_mm",
        "delta_err_mm", "min_shift_mm", "methods", "layer_mode_class", "row"])
    # Layer-mode rows count the raw sensor events they absorb.
    table["n_members"] = np.where(table["class"] == "layer_mode", 0, 1)
    table["group_id"] = ""
    table["keep"] = True
    induced = 0

    if cfg.common_mode != "none":
        layer = table.loc[table["class"] == "layer_mode"]
        raw_mu = by_frame[("raw", "mu")]

        def layer_mode_near(event) -> bool:
            last = max(event["run_index"], event["end_run_index"])
            return bool(((layer["detector"] == event["detector"])
                         & (layer["layer"] == event["layer"])
                         & (layer["coord"] == event["coord"])
                         & (layer["run_index"] >= event["run_index"]
                            - cfg.window)
                         & (layer["run_index"] <= last + cfg.window)).any())

        # A relative-frame change is kept only if the sensor's raw series
        # changed too, unless a layer-mode change was reported there: a
        # partial-layer movement shifts the common mode below the minimum
        # shift and makes the unmoved sensors look moved after subtraction.
        for index, event in table.loc[(table["frame"] == "relative")
                                      ].iterrows():
            if layer_mode_near(event):
                continue
            change = raw_change(raw_mu, int(event["row"]), event,
                                max(cfg.baseline_runs, 5))
            if not abs(change) > event["min_shift_mm"]:
                table.at[index, "keep"] = False
                induced += 1

        # Raw-frame events: drop those the relative frame already has, and
        # absorb those explained by a layer-mode change.
        relative = table.loc[(table["frame"] == "relative") & table["keep"]]
        for index, event in table.loc[(table["frame"] == "raw")
                                      & (table["quantity"] == "mu")].iterrows():
            same = relative.loc[
                (relative[KEY_COLUMNS] == event[KEY_COLUMNS]).all(axis=1)]
            ends = np.where(same["end_run_index"] >= 0,
                            same["end_run_index"], same["run_index"])
            if ((same["run_index"] - cfg.window <= event["run_index"])
                    & (event["run_index"] <= ends + cfg.window)).any():
                table.at[index, "keep"] = False
                continue
            modes = layer.loc[(layer["detector"] == event["detector"])
                              & (layer["layer"] == event["layer"])
                              & (layer["coord"] == event["coord"])
                              & ((layer["run_index"] - event["run_index"]).abs()
                                 <= cfg.window)]
            if len(modes):
                table.at[index, "keep"] = False
                table.at[modes.index[0], "n_members"] += 1
    table = table.loc[table["keep"]].drop(columns=["keep", "row"]
                                          ).reset_index(drop=True)
    table["event_id"] = [f"E{i + 1}" for i in range(len(table))]

    # coherent_stave: at least half the chips of a stave step together.
    chips_per_stave = keys.groupby(["detector", "layer", "stave",
                                    "coord"]).size()
    extra = []
    steps = table.loc[(table["class"] == "step") & (table["quantity"] == "mu")]
    for (detector, layer, stave, coord), group in steps.groupby(
            ["detector", "layer", "stave", "coord"]):
        for members in _clusters(group, cfg.window):
            needed = math.ceil(chips_per_stave[(detector, layer, stave,
                                                coord)] / 2)
            if len(members) < needed:
                continue
            event_id = f"S{len(extra) + 1}"
            table.loc[members.index, "group_id"] = event_id
            extra.append(_group_row(event_id, "coherent_stave", members,
                                    detector, layer, stave, coord))
    staves = pd.DataFrame(extra)
    multi = []
    if len(staves):
        for (detector, layer, coord), group in staves.groupby(
                ["detector", "layer", "coord"]):
            for members in _clusters(group, cfg.window):
                if len(members) < cfg.multi_stave_min:
                    continue
                event_id = f"M{len(multi) + 1}"
                staves.loc[members.index, "group_id"] = event_id
                multi.append(_group_row(event_id, "multi_stave", members,
                                        detector, layer, -1, coord))
    table = pd.concat([table, staves, pd.DataFrame(multi)], ignore_index=True)

    # outlier runs and bad runs
    extra = []
    for quantity, (mask, delta, z, S) in outliers.items():
        for row, run in zip(*np.nonzero(mask)):
            key = keys.iloc[row]
            extra.append({"class": "outlier_run", "detector": key["detector"],
                          "layer": int(key["layer"]),
                          "stave": int(key["stave"]),
                          "chip": int(key["chip"]), "coord": key["coord"],
                          "quantity": quantity, "frame": "raw",
                          "run_index": int(run), "end_run_index": -1,
                          "delta_mm": float(delta[row, run]),
                          "delta_err_mm": float(abs(delta[row, run]
                                                    / z[row, run])),
                          "methods": "neighbours", "n_members": 1})
    for detector, runs_bad in bad_runs.items():
        for run, fraction in runs_bad.items():
            extra.append({"class": "bad_run", "detector": detector,
                          "layer": -1, "stave": -1, "chip": -1, "coord": "",
                          "quantity": "", "frame": "raw", "run_index": run,
                          "end_run_index": -1,
                          "methods": f"outlier fraction {fraction:.2f}",
                          "n_members": 1})
    table = pd.concat([table, pd.DataFrame(extra)], ignore_index=True)
    start = len([i for i in table["event_id"] if isinstance(i, str)
                 and i.startswith("E")])
    missing_id = table["event_id"].isna()
    table.loc[missing_id, "event_id"] = [f"E{start + i + 1}"
                                         for i in range(missing_id.sum())]
    for column, default in (("component", ""), ("group_id", ""),
                            ("methods", ""), ("layer_mode_class", "")):
        table[column] = table[column].fillna(default)
    table["n_methods"] = [len([m for m in str(methods).split(";")
                               if m in METHODS])
                          for methods in table["methods"]]
    with np.errstate(divide="ignore", invalid="ignore"):
        table["z"] = table["delta_mm"].abs() / table["delta_err_mm"]
    labels = runs["run"].tolist()
    table["run"] = [labels[i] if i >= 0 else "" for i in table["run_index"]]
    table["end_run"] = [labels[int(i)] if i >= 0 else ""
                        for i in table["end_run_index"].fillna(-1)]
    table["notes"] = ""
    table = table.sort_values(["run_index", "class", "detector", "layer",
                               "stave", "chip", "coord"]).reset_index(drop=True)
    table.attrs["common_mode_induced_dropped"] = induced
    return table


def raw_change(frame: Frame, row: int, event, n_side: int) -> float:
    """Raw-series change across an event: weighted mean of up to n_side
    valid runs after it minus that of up to n_side runs before it. For a
    drift, 'after' starts at its end; for a transient it is the runs
    inside it."""
    x, s = frame.X[row], frame.S[row]
    valid = np.flatnonzero(np.isfinite(x) & np.isfinite(s))
    start, end = int(event["run_index"]), int(event["end_run_index"])
    before = valid[valid < start][-n_side:]
    if event["class"] == "transient":
        after = valid[(valid >= start) & (valid < end)][:n_side]
    else:
        after = valid[valid >= (end if end >= 0 else start)][:n_side]
    if len(before) == 0 or len(after) == 0:
        return float("nan")
    return (weighted_mean(x[after], s[after])[0]
            - weighted_mean(x[before], s[before])[0])


def _clusters(group: pd.DataFrame, window: int):
    """Split events of one place into clusters by run (+-window) and sign."""
    for sign in (1, -1):
        part = group.loc[np.sign(group["delta_mm"]) == sign].sort_values(
            "run_index")
        current = []
        for index, event in part.iterrows():
            if current and event["run_index"] - part.at[current[-1],
                                                         "run_index"] > window:
                yield part.loc[current]
                current = []
            current.append(index)
        if current:
            yield part.loc[current]


def _group_row(event_id, name, members, detector, layer, stave, coord):
    return {"event_id": event_id, "class": name, "detector": detector,
            "layer": int(layer), "stave": int(stave), "chip": -1,
            "coord": coord, "quantity": "mu",
            "frame": ";".join(sorted(set(members["frame"]))),
            "run_index": int(members["run_index"].median()),
            "end_run_index": -1,
            "delta_mm": float(members["delta_mm"].median()),
            "delta_err_mm": float(members["delta_err_mm"].median()
                                  / math.sqrt(len(members))),
            "min_shift_mm": float(members["min_shift_mm"].iloc[0]),
            "methods": "", "n_members": int(len(members))}


# --------------------------------------------------------------- outputs

def annotate(events: pd.DataFrame, run_info: pd.DataFrame | None,
             runs: pd.DataFrame, window: int) -> None:
    """Attach the notes of run_info rows within window runs before an
    event (inclusive) to the event."""
    if run_info is None or "note" not in run_info.columns:
        return
    position = {label: i for i, label in enumerate(runs["run"])}
    by_number = {number: i for i, number in enumerate(runs["run_number"])}
    notes = {}
    for _, row in run_info.iterrows():
        note = str(row.get("note", "") or "").strip()
        if not note or note == "nan":
            continue
        label = str(row["run"])
        index = position.get(label)
        if index is None:
            index = by_number.get(run_number(label))
        if index is not None:
            notes.setdefault(index, []).append(note)
    events["notes"] = [
        "; ".join(note for i in range(run - window, run + 1)
                  for note in notes.get(i, []))
        for run in events["run_index"]]


def series_table(result: dict, runs: pd.DataFrame) -> pd.DataFrame:
    """Long table of every scanned series (one row per series and run)."""
    parts = []
    labels = runs["run"].to_numpy()
    for frame, (_, _, q) in zip(result["frames"], result["results"].values()):
        rows, n = frame.X.shape
        data = frame.keys.loc[np.repeat(np.arange(rows), n)].reset_index(
            drop=True)
        data["frame"] = frame.name
        data["quantity"] = frame.quantity
        data["run_index"] = np.tile(np.arange(n), rows)
        data["run"] = np.tile(labels, rows)
        data["value_mm"] = frame.X.ravel()
        data["fit_error_mm"] = frame.E.ravel()
        data["sigma_tot_mm"] = frame.S.ravel()
        data["bocpd_p"] = q.ravel()
        parts.append(data)
    table = pd.concat(parts, ignore_index=True)
    table["component"] = table.get("component", pd.Series(dtype=str))
    table["component"] = table["component"].fillna("")
    return table


def build_summary(result: dict, runs: pd.DataFrame, cfg: Config,
                  inputs: list[dict]) -> dict:
    events = result["events"]
    criteria = dataclasses.asdict(cfg)
    criteria["min_shift_mm"] = {f"{d}:{c}": v
                                for (d, c), v in cfg.min_shift_mm.items()}
    detections = result["detections"]
    return {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(
            timespec="seconds"),
        "tool": "run_stability.py",
        "runs": runs[["run_index", "run", "run_number"]].to_dict(
            orient="records"),
        "inputs": inputs,
        "criteria": criteria,
        "calibration": result["calibration"],
        "series_scanned": {f"{f.name}:{f.quantity}": int(len(f.keys))
                           for f in result["frames"]},
        "counts_by_class": events["class"].value_counts().sort_index()
        .to_dict(),
        "detections_by_method": (detections.groupby(["frame", "quantity",
                                                     "method"]).size()
                                 .rename("n").reset_index()
                                 .to_dict(orient="records")
                                 if len(detections) else []),
        "bad_runs": result["bad_runs"],
        "relative_changes_dropped_as_common_mode_induced": int(
            events.attrs.get("common_mode_induced_dropped", 0)),
        "not_scanned": "stave rows (chip = -1) and VERTEX rows",
    }


def format_delta(row) -> str:
    if pd.isna(row["delta_mm"]):
        return ""
    if row["quantity"] == "mu" and row["detector"] == "INTT" \
            and row["coord"] == "z":
        return f"{row['delta_mm']:+.3f} mm"
    return f"{1000.0 * row['delta_mm']:+.1f} um"


def build_report(result: dict, summary: dict, runs: pd.DataFrame) -> str:
    events = result["events"]
    first, last = runs["run"].iloc[0], runs["run"].iloc[-1]
    top = events.loc[events["group_id"] == ""]
    lines = [
        f"# Run-by-run stability: {first} to {last} ({len(runs)} runs)",
        f"Generated {summary['generated_utc']} by run_stability.py "
        "(deterministic, no AI).",
        "## Criteria",
        f"- Methods: CUSUM, PELT, BOCPD; a change needs "
        f"{summary['criteria']['min_methods']} of 3 within "
        f"+-{summary['criteria']['window']} runs, |delta| above the minimum "
        f"shift and z > {summary['criteria']['z_min']}.",
        f"- Thresholds: about {summary['criteria']['expected_false']} false "
        "alarm per method in the whole scan for Gaussian noise (Monte Carlo; "
        "see stability_summary JSON, 'calibration').",
        f"- Common mode: {summary['criteria']['common_mode']}.",
        "- Minimum shifts: " + ", ".join(
            f"{key} {1000.0 * value:g} um" for key, value in
            summary["criteria"]["min_shift_mm"].items()
            if not key.startswith("VERTEX")) + "; widths "
        f"{100 * summary['criteria']['sigma_rel_threshold']:g}% of the "
        "reference width.",
        "## Counts",
        "| Class | Events |", "|---|---|",
        *[f"| {name} | {count} |" for name, count in
          summary["counts_by_class"].items()],
        "",
        "Grouped chip events are listed under their stave/layer event "
        "(group_id in the CSV) and not repeated below.",
        f"## Events ({len(top)})",
    ]
    if top.empty:
        lines.append("No changes passed the criteria.")
    else:
        lines += ["| ID | Class | Det | Layer | Stave | Chip | Coord | Qty | "
                  "Frame | Run | End | Delta | z | Methods | Members | Notes |",
                  "|" + "---|" * 16]
        for _, row in top.iterrows():
            z_text = "" if pd.isna(row["z"]) else f"{row['z']:.1f}"
            lines.append(
                f"| {row['event_id']} | {row['class']} | {row['detector']} | "
                f"{row['layer']} | {row['stave']} | {row['chip']} | "
                f"{row['coord']}{'/' + row['component'] if row['component'] else ''}"
                f" | {row['quantity']} | {row['frame']} | {row['run']} | "
                f"{row['end_run']} | {format_delta(row)} | "
                f"{z_text} | "
                f"{row['methods']} | {row['n_members']} | {row['notes']} |")
    lines += [
        "## Caveats",
        "- A change of the residual mean is a sensor movement relative to "
        "the geometry used in reconstruction only if every run used the same "
        "alignment/geometry tag, field, track selection and distortion "
        "corrections; otherwise a condition change looks like a movement.",
        "- The sensor is part of the track fit, so the residual shift is "
        "smaller than the true displacement (biased residuals).",
        "- Weak modes (radial expansion, twist, telescope) and rigid "
        "movements of the whole tracker barely change residuals.",
        "- The thresholds are calibrated on Gaussian noise; check the false "
        "alarm rate on a stable run period before trusting the rates.",
        "- Not scanned: stave rows (chip = -1) and VERTEX rows.",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(result: dict, runs: pd.DataFrame, cfg: Config,
                  out_dir: Path, inputs: list[dict]) -> dict:
    tag = (f"{sanitize_for_filename(runs['run'].iloc[0])}_to_"
           f"{sanitize_for_filename(runs['run'].iloc[-1])}")
    summary = build_summary(result, runs, cfg, inputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "changepoints": out_dir / f"changepoints_{tag}.csv",
        "detections": out_dir / f"detections_{tag}.csv",
        "series": out_dir / f"series_{tag}.csv",
        "summary": out_dir / f"stability_summary_{tag}.json",
        "report": out_dir / f"stability_report_{tag}.md",
    }
    result["events"][EVENT_COLUMNS].to_csv(paths["changepoints"], index=False)
    result["detections"].to_csv(paths["detections"], index=False)
    series_table(result, runs).to_csv(paths["series"], index=False,
                                      float_format="%.8g")
    paths["summary"].write_text(json.dumps(summary, indent=2, default=str)
                                + "\n")
    paths["report"].write_text(build_report(result, summary, runs))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tables", nargs="+", type=Path,
                        help="residual_fits_<run>.csv, one per run")
    parser.add_argument("--out-dir", type=Path,
                        help="write the outputs here (default: a fresh "
                             "runs/stability_<first>_to_<last>)")
    parser.add_argument("--order", type=Path,
                        help="file with one run label per line, in run order "
                             "(default: the number in each label)")
    parser.add_argument("--run-info", type=Path,
                        help="CSV with columns run and note; notes are "
                             "attached to changes at or just after that run")
    parser.add_argument("--common-mode", choices=["offset", "harmonic",
                                                  "none"], default="offset")
    parser.add_argument("--baseline-runs", type=int, default=5)
    parser.add_argument("--expected-false", type=float, default=1.0)
    parser.add_argument("--z-min", type=float, default=5.0)
    parser.add_argument("--sigma-rel-threshold", type=float, default=0.20)
    parser.add_argument("--min-methods", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("--window", type=int, default=2)
    parser.add_argument("--min-shift", action="append", default=[],
                        metavar="DETECTOR:COORD=MM")
    parser.add_argument("--n-sims", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    cfg = Config(baseline_runs=args.baseline_runs,
                 expected_false=args.expected_false, z_min=args.z_min,
                 sigma_rel_threshold=args.sigma_rel_threshold,
                 min_shift_mm=parse_min_shift_overrides(args.min_shift),
                 common_mode=args.common_mode, min_methods=args.min_methods,
                 window=args.window, n_sims=args.n_sims, seed=args.seed)
    order = None
    if args.order:
        order = [line.strip() for line in args.order.read_text().splitlines()
                 if line.strip() and not line.startswith("#")]
    long, runs = load_run_series(args.tables, order)
    if args.out_dir is None:
        out_dir = RUNS_DIR / (f"stability_{sanitize_for_filename(runs['run'].iloc[0])}"
                              f"_to_{sanitize_for_filename(runs['run'].iloc[-1])}")
        fresh_run_dir(out_dir, args.tables)
    else:
        out_dir = args.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)

    result = scan(long, runs, cfg)
    run_info = pd.read_csv(args.run_info, dtype=str) if args.run_info else None
    annotate(result["events"], run_info, runs, cfg.window)
    inputs = runs[["run", "csv", "sha256"]].to_dict(orient="records")
    paths = write_outputs(result, runs, cfg, out_dir, inputs)
    counts = result["events"]["class"].value_counts().to_dict()
    print(f"{len(runs)} runs, "
          + ", ".join(f"{name} {count}" for name, count in sorted(
              counts.items())) + f" -> {paths['changepoints']}, "
          f"{paths['report']}")


if __name__ == "__main__":
    main()
