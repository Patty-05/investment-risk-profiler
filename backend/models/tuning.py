"""
Phase B -- can the technical model be pushed past 0.52 out-of-sample?

The honest baseline this project recorded is out-of-sample AUC 0.503-0.507 with a
negative accuracy edge: a coin flip. This module runs a disciplined ablation to
find out whether that is a modelling failure (fixable) or the actual information
content of daily OHLCV against a 20-day relative-return target (not fixable).

Every experiment is scored on the SAME purged, time-ordered folds as
models/walkforward.py. Nothing here is scored on training data.

THE HYPOTHESES, IN ORDER OF EXPECTED PAYOFF
-------------------------------------------
  1. CROSS-SECTIONAL RANKING. The label is relative to the Nifty but the features
     are absolute levels, so the model burns capacity learning the regime instead
     of the ranking. Converting each feature to its within-date percentile is the
     textbook fix and the single most likely reason the current model sits at
     0.50. See panel.cross_sectional_rank.
  2. TRAINING DEADBAND. Rows where the stock finished within a hair of the index
     are coin flips whose labels are essentially random, and they dominate the
     sample. Dropping the ambiguous middle FROM TRAINING ONLY (never from test)
     sharpens the decision boundary without touching the evaluation.
  3. REGULARISATION. 400 trees at depth 5 on a signal-to-noise ratio this low is
     an overfitting machine. Shallower trees, stronger shrinkage, more row/column
     subsampling.
  4. A LINEAR MODEL. On low-SNR financial panels, logistic regression on ranked
     features routinely beats gradient boosting, because there is not enough
     signal to justify learning interactions.
  5. RECENCY WEIGHTING. Market structure drifts; older rows may be actively
     misleading.
  6. FEATURE PRUNING. Several features are near-duplicates (momentum_5d /
     roc_10 / momentum_20d), and collinear noise costs more than it pays.

Run:
    cd backend && python -m models.tuning
    cd backend && python -m models.tuning --quick     (3 folds, fewer variants)
"""

from __future__ import annotations

import torch_bootstrap  # noqa: F401  (Windows OpenMP load order)

import sys
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from models.panel import load_panel, cross_sectional_rank
from models.feature_engineer import FEATURE_NAMES
from models.walkforward import DEFAULT_FOLDS, time_split

from xgboost import XGBClassifier

try:
    from lightgbm import LGBMClassifier
    LGBM = True
except ImportError:
    LGBM = False


# -- Model factories ---------------------------------------------------------
def xgb_default():
    return XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.5,
        min_child_weight=3, gamma=0.1, eval_metric="logloss",
        random_state=42, verbosity=0, n_jobs=-1)


def xgb_regularised():
    """Shallow, heavily shrunk, heavily subsampled -- built for low SNR."""
    return XGBClassifier(
        n_estimators=300, max_depth=3, learning_rate=0.02,
        subsample=0.6, colsample_bytree=0.5, reg_alpha=1.0, reg_lambda=10.0,
        min_child_weight=50, gamma=0.5, eval_metric="logloss",
        random_state=42, verbosity=0, n_jobs=-1)


def xgb_stump():
    """Depth-2 -- essentially an additive model with light pairwise interaction."""
    return XGBClassifier(
        n_estimators=250, max_depth=2, learning_rate=0.03,
        subsample=0.7, colsample_bytree=0.6, reg_alpha=1.0, reg_lambda=20.0,
        min_child_weight=100, eval_metric="logloss",
        random_state=42, verbosity=0, n_jobs=-1)


def logistic():
    return LogisticRegression(C=0.05, max_iter=2000, solver="lbfgs")


def lgbm_regularised():
    return LGBMClassifier(
        n_estimators=400, learning_rate=0.02, num_leaves=15, max_depth=4,
        subsample=0.7, subsample_freq=1, colsample_bytree=0.6,
        reg_alpha=1.0, reg_lambda=10.0, min_child_samples=200,
        random_state=42, n_jobs=1, verbosity=-1)


# -- One scored configuration ------------------------------------------------
def evaluate(X, y, meta, model_fn, folds, rank=False, deadband=0.0,
             recency_halflife=None, feature_idx=None, balance=True, decimate=0):
    """
    Mean out-of-sample AUC / accuracy / edge for one configuration.

    `deadband` drops training rows whose realised excess return is inside
    +/- deadband. `decimate` keeps only every Nth distinct training date.
    Both filter TRAINING ONLY -- test rows are never touched, so every variant
    is scored on the identical held-out set and the numbers stay comparable.

    WHY DECIMATE
    ------------
    The label is a 20-day forward return sampled daily, so consecutive rows for
    one stock share 19 of their 20 outcome days. 348k rows therefore contain
    closer to 17k genuinely independent observations. The model reads the
    redundancy as confirmation, fits the noise in it, and reports a confidence
    the data cannot support. Keeping every 20th date makes the training labels
    non-overlapping -- far fewer rows, but each one is real information.
    """
    Xw = cross_sectional_rank(X, meta, FEATURE_NAMES) if rank else X
    if feature_idx is not None:
        Xw = Xw[:, feature_idx]

    dates = pd.to_datetime(meta["date"])
    if decimate > 1:
        uniq = np.sort(dates.unique())
        keep_dates = set(pd.Series(uniq[::decimate]))
        decim_mask = dates.isin(keep_dates).values
    else:
        decim_mask = None

    aucs, accs, edges = [], [], []
    for (_, tr_e, te_s, te_e) in folds:
        tr_m, te_m = time_split(meta, tr_e, te_s, te_e)

        if deadband > 0:
            keep = np.abs(meta["excess_return"].values) >= deadband
            tr_m = tr_m & keep
        if decim_mask is not None:
            tr_m = tr_m & decim_mask

        Xtr_raw, ytr = Xw[tr_m], y[tr_m]
        Xte_raw, yte = Xw[te_m], y[te_m]
        if len(Xtr_raw) < 500 or len(Xte_raw) < 100 or len(np.unique(yte)) < 2:
            continue

        scaler = RobustScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)

        sw = compute_sample_weight("balanced", ytr) if balance else np.ones(len(ytr))
        if recency_halflife:
            age_days = (pd.Timestamp(te_s) - pd.to_datetime(meta.loc[tr_m, "date"])).dt.days.values
            sw = sw * np.power(0.5, age_days / float(recency_halflife))

        model = model_fn()
        try:
            model.fit(Xtr, ytr, sample_weight=sw)
        except TypeError:
            model.fit(Xtr, ytr)

        p = model.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        base = max(np.mean(yte), 1 - np.mean(yte))
        aucs.append(roc_auc_score(yte, p))
        accs.append(accuracy_score(yte, pred))
        edges.append(accuracy_score(yte, pred) - base)

    if not aucs:
        return None
    return {"auc": float(np.mean(aucs)), "acc": float(np.mean(accs)),
            "edge": float(np.mean(edges)), "auc_sd": float(np.std(aucs)),
            "n_folds": len(aucs), "auc_folds": [round(a, 4) for a in aucs]}


# -- Experiment grid ---------------------------------------------------------
def experiments(quick=False):
    """(label, model_fn, kwargs) triples. Ordered so each builds on the last."""
    grid = [
        ("1. baseline: raw feats, default XGB",   xgb_default,     {}),
        ("2. + cross-sectional rank",             xgb_default,     {"rank": True}),
        ("3. rank + regularised XGB",             xgb_regularised, {"rank": True}),
        ("4. rank + depth-2 XGB",                 xgb_stump,       {"rank": True}),
        ("5. rank + logistic",                    logistic,        {"rank": True}),
        ("6. rank + reg XGB + 2% deadband",       xgb_regularised, {"rank": True, "deadband": 0.02}),
        ("7. rank + reg XGB + 5% deadband",       xgb_regularised, {"rank": True, "deadband": 0.05}),
        ("8. rank + logistic + 5% deadband",      logistic,        {"rank": True, "deadband": 0.05}),
    ]
    if not quick:
        grid += [
            ("9. rank + reg XGB + recency 2y",    xgb_regularised, {"rank": True, "recency_halflife": 730}),
            ("10. raw feats + regularised XGB",   xgb_regularised, {}),
            ("11. rank + logistic, no balance",   logistic,        {"rank": True, "balance": False}),
        ]
        if LGBM:
            grid.append(("12. rank + regularised LGBM", lgbm_regularised, {"rank": True}))
    return grid


def main(quick=False):
    print("=" * 92)
    print("PHASE B -- TECHNICAL MODEL TUNING  (purged walk-forward, out-of-sample)")
    print("=" * 92)

    X, y, meta = load_panel()
    folds = DEFAULT_FOLDS[:3] if quick else DEFAULT_FOLDS

    base_rate = float(np.mean(y))
    print("\n[data] {:,} rows x {} features, {} tickers, {} -> {}".format(
        len(X), X.shape[1], meta["ticker"].nunique(),
        meta["date"].min().date(), meta["date"].max().date()))
    print("[data] label balance {:.3f} / {:.3f}\n".format(base_rate, 1 - base_rate))

    print("{:<40}{:>9}{:>8}{:>9}{:>10}   {}".format(
        "configuration", "AUC", "sd", "acc", "edge", "per-fold AUC"))
    print("-" * 92)

    results = []
    for label, model_fn, kw in experiments(quick):
        t0 = time.time()
        try:
            r = evaluate(X, y, meta, model_fn, folds, **kw)
        except Exception as e:
            print("{:<40}  FAILED: {}".format(label, str(e)[:40]))
            continue
        if not r:
            print("{:<40}  no folds produced results".format(label))
            continue
        results.append((label, r))
        print("{:<40}{:>9.4f}{:>8.4f}{:>9.4f}{:>+10.4f}   {}  ({:.0f}s)".format(
            label, r["auc"], r["auc_sd"], r["acc"], r["edge"],
            r["auc_folds"], time.time() - t0))

    print("-" * 92)
    if results:
        best = max(results, key=lambda kv: kv[1]["auc"])
        print("\nBEST: {}  ->  AUC {:.4f}  (edge {:+.4f})".format(
            best[0], best[1]["auc"], best[1]["edge"]))
        print("\nA mean AUC above 0.52 that holds in EVERY fold is a real result.")
        print("A mean above 0.52 driven by one fold is a lucky year, not a model.")
    return results


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main(quick="--quick" in sys.argv)
