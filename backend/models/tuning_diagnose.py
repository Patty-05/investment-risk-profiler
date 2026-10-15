"""
Phase B, round 2 -- is the signal absent, or is the harness broken?

Round 1 ran twelve diverse configurations (raw vs cross-sectionally ranked
features, four model families, three regularisation strengths, deadbands,
recency weighting) and every single one landed between 0.496 and 0.508 AUC with
a negative accuracy edge. That is suspiciously flat. Two very different worlds
produce that picture:

  A. There is genuinely no exploitable signal in daily OHLCV technicals for
     20-day relative performance. Believable -- this is close to the efficient
     market null, and it is what the earlier walk-forward run already hinted at.
  B. Something in the pipeline is broken: rows misaligned against labels,
     a constant target, a scaler leaking, a fold that trains on nothing.

Publishing "no signal" while sitting in world B would be a serious error, so
this module is built to tell them apart before any conclusion is drawn.

  1. POSITIVE CONTROL. Add one feature that is a deliberately corrupted copy of
     the answer. If the harness cannot recover a planted signal, the harness is
     broken and every other number here is meaningless. Recovering it, and
     watching AUC fall smoothly as the noise rises, proves the measurement works
     and that a real edge WOULD have shown up.
  2. TRAIN vs TEST. If train AUC is high while test AUC is 0.50, the models are
     memorising noise -- the features carry no generalising information. If both
     are 0.50, the features carry no information at all, even in-sample.
  3. PER-FEATURE IC SCREEN. Rank all 26 technicals individually by
     cross-sectional information coefficient. This bypasses the models entirely.
     If not one feature clears a t-stat of 2, no model can build an edge.
  4. DECIMATION. 20-day labels sampled daily means ~19 of every 20 rows are
     redundant. Train on non-overlapping dates only.

Run:
    cd backend && python -m models.tuning_diagnose
"""

from __future__ import annotations

import torch_bootstrap  # noqa: F401  (Windows OpenMP load order)

import sys
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from models.panel import load_panel, cross_sectional_rank
from models.feature_engineer import FEATURE_NAMES
from models.walkforward import DEFAULT_FOLDS, time_split
from models.tuning import xgb_default, xgb_regularised, evaluate
from models.geo_alpha_test import information_coefficient


def positive_control(X, y, meta, folds, noise_levels=(0.0, 0.5, 0.8, 0.9, 0.95, 1.0)):
    """
    Plant a feature correlated with the label and check the harness finds it.

    At noise=p, that fraction of the planted column is replaced with random
    values, so p=0 is the label itself and p=1.0 is pure noise. AUC must fall
    monotonically from ~1.0 to ~0.5. The p=1.0 row is the true control: it is
    the same pipeline with a feature guaranteed to be worthless, so whatever it
    prints is this harness's honest reading of "no signal".
    """
    rng = np.random.default_rng(42)
    print("\n" + "-" * 78)
    print("1. POSITIVE CONTROL -- can this harness detect a signal that IS there?")
    print("-" * 78)
    print("{:<22}{:>12}{:>16}".format("planted feature", "test AUC", "verdict"))

    for p in noise_levels:
        planted = y.astype(float).copy()
        mask = rng.random(len(y)) < p
        planted[mask] = rng.random(mask.sum())
        planted = planted + rng.normal(0, 0.01, len(y))
        Xp = np.hstack([X, planted.reshape(-1, 1)])

        aucs = []
        for (_, tr_e, te_s, te_e) in folds:
            tr_m, te_m = time_split(meta, tr_e, te_s, te_e)
            if tr_m.sum() < 500 or te_m.sum() < 100:
                continue
            sc = RobustScaler()
            Xtr = sc.fit_transform(Xp[tr_m])
            Xte = sc.transform(Xp[te_m])
            m = xgb_default()
            m.fit(Xtr, y[tr_m], sample_weight=compute_sample_weight("balanced", y[tr_m]))
            aucs.append(roc_auc_score(y[te_m], m.predict_proba(Xte)[:, 1]))

        auc = float(np.mean(aucs))
        verdict = "detected" if auc > 0.55 else ("NULL (control)" if p >= 1.0 else "faint")
        print("{:<22}{:>12.4f}{:>16}".format(
            "{:.0f}% noise".format(p * 100), auc, verdict))

    print("\nIf the low-noise rows are near 1.0 and 100%-noise sits at ~0.50, the")
    print("harness measures correctly and a real edge would have been visible.")


def train_vs_test(X, y, meta, folds):
    """Are the models learning nothing, or learning noise?"""
    print("\n" + "-" * 78)
    print("2. TRAIN vs TEST AUC -- no information, or memorised noise?")
    print("-" * 78)
    print("{:<26}{:>12}{:>12}{:>12}".format("model", "train AUC", "test AUC", "gap"))

    for name, fn in (("default XGB", xgb_default), ("regularised XGB", xgb_regularised)):
        trs, tes = [], []
        for (_, tr_e, te_s, te_e) in folds:
            tr_m, te_m = time_split(meta, tr_e, te_s, te_e)
            if tr_m.sum() < 500 or te_m.sum() < 100:
                continue
            sc = RobustScaler()
            Xtr = sc.fit_transform(X[tr_m])
            Xte = sc.transform(X[te_m])
            m = fn()
            m.fit(Xtr, y[tr_m], sample_weight=compute_sample_weight("balanced", y[tr_m]))
            trs.append(roc_auc_score(y[tr_m], m.predict_proba(Xtr)[:, 1]))
            tes.append(roc_auc_score(y[te_m], m.predict_proba(Xte)[:, 1]))
        tr, te = float(np.mean(trs)), float(np.mean(tes))
        print("{:<26}{:>12.4f}{:>12.4f}{:>12.4f}".format(name, tr, te, tr - te))

    print("\nHigh train / 0.50 test = the features only support memorisation.")


def feature_ic_screen(X, meta):
    """Model-free: does any single technical feature carry cross-sectional signal?"""
    print("\n" + "-" * 78)
    print("3. PER-FEATURE IC SCREEN -- model-free, all 26 technicals")
    print("-" * 78)
    print("{:<22}{:>9}{:>9}{:>9}{:>8}{:>8}".format(
        "feature", "IC", "t-naive", "t-NW", "hit", "IR"))

    rows = []
    for i, nm in enumerate(FEATURE_NAMES):
        r = information_coefficient(X[:, i], meta)
        rows.append((nm, r))
    rows.sort(key=lambda kv: -abs(kv[1]["ic"] if np.isfinite(kv[1]["ic"]) else 0))

    for nm, r in rows:
        flag = "  <-- survives" if abs(r["t_nw"]) > 2 else ""
        print("{:<22}{:>9.4f}{:>9.2f}{:>9.2f}{:>8.3f}{:>8.3f}{}".format(
            nm, r["ic"], r["t_stat"], r["t_nw"], r["hit_rate"], r["ir"], flag))

    naive = [nm for nm, r in rows if abs(r["t_stat"]) > 2]
    strong = [nm for nm, r in rows if abs(r["t_nw"]) > 2]
    print("\n{} of {} features clear |t| > 2 on the NAIVE t-stat.".format(
        len(naive), len(rows)))
    print("{} of {} still clear it once Newey-West corrects for the 20-day label "
          "overlap.".format(len(strong), len(rows)))
    print("The second number is the real one. With 26 simultaneous tests, ~1")
    print("false positive at the 5% level is expected even under pure noise.")
    return rows


def decimation(X, y, meta, folds):
    """Non-overlapping training labels -- fewer rows, but real ones."""
    print("\n" + "-" * 78)
    print("4. DECIMATION -- training on non-overlapping labels")
    print("-" * 78)
    print("{:<34}{:>10}{:>10}{:>11}".format("configuration", "AUC", "sd", "edge"))

    for label, kw in (
        ("every date (overlapping, baseline)", {"rank": True}),
        ("every 5th date",                     {"rank": True, "decimate": 5}),
        ("every 20th date (independent)",      {"rank": True, "decimate": 20}),
    ):
        r = evaluate(X, y, meta, xgb_regularised, folds, **kw)
        if r:
            print("{:<34}{:>10.4f}{:>10.4f}{:>+11.4f}".format(
                label, r["auc"], r["auc_sd"], r["edge"]))


def main():
    print("=" * 78)
    print("PHASE B ROUND 2 -- DIAGNOSTICS")
    print("=" * 78)
    X, y, meta = load_panel()

    positive_control(X, y, meta, DEFAULT_FOLDS)
    train_vs_test(X, y, meta, DEFAULT_FOLDS)
    feature_ic_screen(X, meta)
    decimation(X, y, meta, DEFAULT_FOLDS)

    print("\n" + "=" * 78)
    print("Read these together: a passing positive control turns a flat 0.50")
    print("across every configuration from 'suspicious' into a real measurement.")
    print("=" * 78)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
