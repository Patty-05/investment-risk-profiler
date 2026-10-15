"""
Does the geopolitical factor add incremental alpha over technicals?

Two independent tests, because they fail in different ways:

  1. INFORMATION COEFFICIENT (model-free). Each day, rank every stock by
     geo_signal and correlate that ranking with the realised 20-day excess
     return (Spearman). Average across days. This is the standard factor-research
     measure and it cannot be rescued by a lucky model or a lucky hyperparameter
     -- if the factor carries no cross-sectional information, IC is ~0. A daily
     IC around 0.02-0.03 with a t-stat above 2 is a genuinely useful equity
     factor; anything under ~0.01 is noise.

  2. INCREMENTAL AUC (model-based). Fit the same models on identical purged,
     time-ordered folds with and without the geo block, and difference the
     out-of-sample AUC. This catches signal that is real but only expressible in
     interaction with technicals, which the IC would miss.

Both are run on the SAME folds as models/walkforward.py, with the same embargo,
so the numbers are directly comparable to the technicals-only baseline already
recorded for this project (out-of-sample AUC 0.503-0.507).

Run:
    cd backend && python -m models.geo_alpha_test
"""

from __future__ import annotations

import torch_bootstrap  # noqa: F401  (Windows OpenMP load order)

import sys
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from models.panel import load_panel, attach_geo
from models.feature_engineer import FEATURE_NAMES
from models.walkforward import DEFAULT_FOLDS, time_split

try:
    from lightgbm import LGBMClassifier
    LGBM = True
except ImportError:
    LGBM = False
from xgboost import XGBClassifier


# -- Test 1: information coefficient -----------------------------------------
def newey_west_t(x, lags):
    """
    t-statistic for the mean of `x`, robust to autocorrelation up to `lags`.

    WHY THE NAIVE t IS WRONG HERE
    -----------------------------
    The IC series is one observation per day, but each day's label is a 20-day
    forward return. Today's IC and tomorrow's IC are computed against outcome
    windows sharing 19 of 20 days, so consecutive values are near-duplicates.
    The plain standard error assumes independence and therefore treats ~2,400
    daily ICs as 2,400 pieces of evidence when there are closer to 120. That
    inflates every t-stat by roughly sqrt(20), which is enough to turn noise
    into a wall of apparent significance.

    Newey-West widens the standard error using the series' own autocovariance
    (Bartlett kernel), which is the standard fix in this literature.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        return np.nan
    mu = x.mean()
    d = x - mu
    gamma0 = float(d @ d) / n
    var = gamma0
    for k in range(1, min(lags, n - 1) + 1):
        gk = float(d[k:] @ d[:-k]) / n
        var += 2.0 * (1.0 - k / (lags + 1.0)) * gk
    if var <= 0:
        return np.nan
    return float(mu / np.sqrt(var / n))


def information_coefficient(signal, meta, min_names=15, lags=20):
    """
    Mean daily cross-sectional Spearman IC of `signal` against realised excess
    return.

    Reports BOTH the naive t-stat and a Newey-West t-stat with `lags` = the
    label horizon. Only the latter should be believed; see newey_west_t.

    Days with fewer than `min_names` names, or with a constant signal (every
    name unexposed), carry no cross-sectional information and are dropped.
    """
    df = pd.DataFrame({
        "date": pd.to_datetime(meta["date"]).values,
        "sig": np.asarray(signal, dtype=float),
        "fwd": meta["excess_return"].values,
    }).dropna()

    ics = []
    for _, g in df.groupby("date"):
        if len(g) < min_names or g["sig"].nunique() < 3:
            continue
        rho = stats.spearmanr(g["sig"], g["fwd"]).correlation
        if np.isfinite(rho):
            ics.append(rho)

    ics = np.asarray(ics)
    if len(ics) < 30:
        return {"n_days": len(ics), "ic": np.nan, "t_stat": np.nan,
                "t_nw": np.nan, "hit_rate": np.nan, "ir": np.nan}

    ic = float(ics.mean())
    t = float(ic / (ics.std(ddof=1) / np.sqrt(len(ics))))
    return {
        "n_days": int(len(ics)),
        "ic": ic,
        "t_stat": t,
        "t_nw": newey_west_t(ics, lags),
        "hit_rate": float((ics > 0).mean()),
        "ir": float(ic / ics.std(ddof=1)) if ics.std(ddof=1) else np.nan,
    }


# -- Test 2: incremental AUC -------------------------------------------------
def _models():
    zoo = {
        "XGBoost": lambda: XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.5,
            min_child_weight=3, gamma=0.1, eval_metric="logloss",
            random_state=42, verbosity=0, n_jobs=-1),
    }
    if LGBM:
        zoo["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            min_child_samples=30, class_weight="balanced",
            random_state=42, n_jobs=1, verbosity=-1)
    return zoo


def run_variant(X, y, meta, folds, label, verbose=True):
    """Out-of-sample accuracy/AUC per model, averaged across folds."""
    per_model = {}
    for i, (_, tr_e, te_s, te_e) in enumerate(folds, 1):
        tr_m, te_m = time_split(meta, tr_e, te_s, te_e)
        Xtr_raw, ytr = X[tr_m], y[tr_m]
        Xte_raw, yte = X[te_m], y[te_m]
        if len(Xtr_raw) < 500 or len(Xte_raw) < 100 or len(np.unique(yte)) < 2:
            continue

        scaler = RobustScaler()
        Xtr = scaler.fit_transform(Xtr_raw)
        Xte = scaler.transform(Xte_raw)
        sw = compute_sample_weight("balanced", ytr)
        base = max(np.mean(yte), 1 - np.mean(yte))

        for name, factory in _models().items():
            try:
                m = factory()
                m.fit(Xtr, ytr, sample_weight=sw)
                p = m.predict_proba(Xte)[:, 1]
                per_model.setdefault(name, []).append({
                    "auc": roc_auc_score(yte, p),
                    "acc": accuracy_score(yte, (p >= 0.5).astype(int)),
                    "edge": accuracy_score(yte, (p >= 0.5).astype(int)) - base,
                })
            except Exception as e:
                print("  [warn] {} fold {} failed: {}".format(name, i, e))

    out = {n: {k: float(np.mean([r[k] for r in rs])) for k in ("auc", "acc", "edge")}
           for n, rs in per_model.items()}
    if verbose:
        for n, r in out.items():
            print("    {:<10} {:<22} AUC {:.4f}   acc {:.4f}   edge {:+.4f}".format(
                n, label, r["auc"], r["acc"], r["edge"]))
    return out


def main():
    print("=" * 78)
    print("GEO FACTOR -- INCREMENTAL ALPHA TEST")
    print("=" * 78)

    X, y, meta = load_panel()
    G, geo_names = attach_geo(meta)

    exposed = np.abs(G).sum(axis=1) > 0
    print("\n[scope] {:,} of {:,} rows carry any geo exposure ({:.1f}%)".format(
        exposed.sum(), len(G), 100.0 * exposed.mean()))

    # ---- Test 1: IC, on the exposed subset and on everything ----
    print("\n" + "-" * 78)
    print("TEST 1 -- CROSS-SECTIONAL INFORMATION COEFFICIENT (model-free)")
    print("-" * 78)
    print("{:<26}{:>8}{:>9}{:>9}{:>9}{:>8}{:>8}".format(
        "signal", "days", "IC", "t-naive", "t-NW", "hit", "IR"))
    for j, nm in enumerate(geo_names):
        r = information_coefficient(G[:, j], meta)
        print("{:<26}{:>8}{:>9.4f}{:>9.2f}{:>9.2f}{:>8.3f}{:>8.3f}".format(
            nm, r["n_days"], r["ic"], r["t_stat"], r["t_nw"], r["hit_rate"], r["ir"]))

    # Same, restricted to names the KB actually covers -- a factor should not be
    # penalised for the 58% of the universe it says nothing about.
    print("\n  (restricted to KB-covered rows only)")
    for j, nm in enumerate(geo_names):
        r = information_coefficient(G[exposed, j], meta[exposed].reset_index(drop=True))
        print("{:<26}{:>8}{:>9.4f}{:>9.2f}{:>9.2f}{:>8.3f}{:>8.3f}".format(
            nm, r["n_days"], r["ic"], r["t_stat"], r["t_nw"], r["hit_rate"], r["ir"]))

    # Benchmark: what does a technical factor score on the same measure?
    print("\n  (reference -- a known technical factor on the same test)")
    for ref in ["rs_20d", "momentum_20d", "rsi"]:
        if ref in FEATURE_NAMES:
            r = information_coefficient(X[:, FEATURE_NAMES.index(ref)], meta)
            print("{:<26}{:>8}{:>9.4f}{:>9.2f}{:>9.2f}{:>8.3f}{:>8.3f}".format(
                ref, r["n_days"], r["ic"], r["t_stat"], r["t_nw"], r["hit_rate"], r["ir"]))

    # ---- Test 2: incremental AUC ----
    print("\n" + "-" * 78)
    print("TEST 2 -- INCREMENTAL OUT-OF-SAMPLE AUC (purged walk-forward)")
    print("-" * 78)

    print("\n  [a] technicals only ({} features)".format(X.shape[1]))
    base = run_variant(X, y, meta, DEFAULT_FOLDS, "technicals")

    Xg = np.hstack([X, G])
    print("\n  [b] technicals + geo ({} features)".format(Xg.shape[1]))
    withgeo = run_variant(Xg, y, meta, DEFAULT_FOLDS, "technicals+geo")

    print("\n  [c] geo only ({} features)".format(G.shape[1]))
    geoonly = run_variant(G, y, meta, DEFAULT_FOLDS, "geo only")

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    print("{:<12}{:>12}{:>14}{:>12}{:>14}".format(
        "model", "tech AUC", "tech+geo AUC", "delta", "geo-only AUC"))
    print("-" * 78)
    for n in base:
        d = withgeo.get(n, {}).get("auc", np.nan) - base[n]["auc"]
        print("{:<12}{:>12.4f}{:>14.4f}{:>+12.4f}{:>14.4f}".format(
            n, base[n]["auc"], withgeo.get(n, {}).get("auc", np.nan), d,
            geoonly.get(n, {}).get("auc", np.nan)))
    print("-" * 78)
    print("t-NW is the one to read: it corrects for the 20-day overlap that makes")
    print("the naive t-stat roughly sqrt(20) too large.")
    print("A geo-only AUC near 0.500 means the block carries no standalone signal.")
    print("A delta within about +/-0.003 is fold noise, not an improvement.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
