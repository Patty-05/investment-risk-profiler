"""
Out-of-sample probability calibration for the 3-model ensemble.

WHY
---
The raw model probabilities are NOT trustworthy as stated. The RF is isotonic
calibrated, but on its own training data; out of sample the models memorise
noise (train AUC ~0.68 vs test AUC ~0.51 - see docs/ALPHA_RESEARCH_FINDINGS.md).
A raw "73% outperform" therefore does not mean 73% of such calls beat the index.

This script produces the honest mapping from raw model output to the empirical
frequency actually observed on unseen data:

  1. Purged, expanding-window walk-forward: train strictly on the past, predict
     strictly on the future (same protocol as models/walkforward.py).
  2. Pool every out-of-sample prediction from every fold.
  3. Fit a Platt (logistic) map  p_cal = sigmoid(a * logit(p_raw) + b)  per
     model and for the soft-vote ensemble. Platt (2 parameters) rather than
     isotonic: with weak signal isotonic fits steps to noise.
  4. Score the calibration NESTED (fit on folds < k, evaluate on fold k) so the
     reported Brier/ECE are not flattered by fitting and scoring on the same rows.

The output (`models/calibration.json`) is read by StockRiskModel.predict(), which
then reports calibrated probabilities plus the measured out-of-sample skill, so
the UI can state how much weight a prediction deserves.

Run:   cd backend && python -m models.calibrate        (several minutes)
"""

from __future__ import annotations

import torch_bootstrap  # noqa: F401  (must be first: Windows OpenMP load order)

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.preprocessing import RobustScaler
from sklearn.utils.class_weight import compute_sample_weight

from models.feature_engineer import FEATURE_NAMES
from models.panel import load_panel
from models.stock_model import StockRiskModel
from models.walkforward import _model_zoo, time_split

OUT_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
MODEL_KEYS = {"Calibrated RF": "rf", "XGBoost": "xgb", "Hybrid HGB": "hgb"}

# (train_end, test_start, test_end) - expanding window, each test year unseen.
FOLDS = [
    ("2021-12-31", "2022-01-01", "2022-12-31"),
    ("2022-12-31", "2023-01-01", "2023-12-31"),
    ("2023-12-31", "2024-01-01", "2024-12-31"),
    ("2024-12-31", "2025-01-01", "2026-12-31"),
]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def fit_platt(p, y):
    """(a, b) so that sigmoid(a*logit(p)+b) is calibrated against y."""
    lr = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    lr.fit(_logit(p).reshape(-1, 1), y)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def apply_platt(p, a, b):
    return _sigmoid(a * _logit(np.asarray(p, dtype=float)) + b)


def ece(p, y, bins=10):
    """Expected calibration error with equal-mass bins."""
    order = np.argsort(p)
    p, y = p[order], y[order]
    total = 0.0
    for chunk in np.array_split(np.arange(len(p)), bins):
        if len(chunk):
            total += len(chunk) / len(p) * abs(p[chunk].mean() - y[chunk].mean())
    return float(total)


def reliability(p, y, bins=8):
    order = np.argsort(p)
    p, y = p[order], y[order]
    rows = []
    for chunk in np.array_split(np.arange(len(p)), bins):
        if len(chunk):
            rows.append({"p_mean": round(float(p[chunk].mean()), 4),
                         "freq": round(float(y[chunk].mean()), 4),
                         "n": int(len(chunk))})
    return rows


def collect_oos():
    """Walk-forward out-of-sample raw probabilities for every model."""
    X, y, meta = load_panel()
    assert X.shape[1] == len(FEATURE_NAMES), (
        f"panel has {X.shape[1]} features, model expects {len(FEATURE_NAMES)} - rebuild the panel")
    meta = meta.copy()
    meta["date"] = pd.to_datetime(meta["date"])

    rows = []
    for i, (tr_end, te_start, te_end) in enumerate(FOLDS):
        t0 = time.time()
        tr, te = time_split(meta, tr_end, te_start, te_end)
        if tr.sum() < 5000 or te.sum() < 500:
            print(f"[skip] fold {i}: too few rows")
            continue
        scaler = RobustScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        rtr = StockRiskModel._rule_score_column(X[tr]).reshape(-1, 1)
        rte = StockRiskModel._rule_score_column(X[te]).reshape(-1, 1)
        Xtr_h, Xte_h = np.hstack([Xtr, rtr]), np.hstack([Xte, rte])
        ytr, yte = y[tr], y[te]
        sw = compute_sample_weight("balanced", ytr)

        out = {"fold": i, "date": meta.loc[te, "date"].values, "y": yte,
               "ticker": meta.loc[te, "ticker"].values}
        for name, factory in _model_zoo().items():
            if name not in MODEL_KEYS:
                continue  # LightGBM is a research-only model, not in the product
            m = factory()
            hyb = name == "Hybrid HGB"
            A, B = (Xtr_h, Xte_h) if hyb else (Xtr, Xte)
            if name in ("XGBoost", "Hybrid HGB"):
                m.fit(A, ytr, sample_weight=sw)
            else:
                m.fit(A, ytr)
            out[MODEL_KEYS[name]] = m.predict_proba(B)[:, 1]
        rows.append(out)
        auc = {k: roc_auc_score(yte, out[k]) for k in MODEL_KEYS.values() if k in out}
        print(f"[fold {i}] test {te_start[:4]}-{te_end[:4]}  train {tr.sum():,} / test {te.sum():,}  "
              f"AUC " + "  ".join(f"{k} {v:.4f}" for k, v in auc.items())
              + f"   ({time.time() - t0:.0f}s)", flush=True)

    df = pd.concat([pd.DataFrame(r) for r in rows], ignore_index=True)
    df["ens"] = df[[k for k in MODEL_KEYS.values()]].mean(axis=1)
    return df


def month_block_auc_ci(df, col, reps=300, seed=7):
    """Bootstrap AUC resampling whole months (rows within a month are dependent)."""
    rng = np.random.default_rng(seed)
    months = df["date"].dt.to_period("M")
    groups = {m: g.index.values for m, g in df.groupby(months)}
    keys = list(groups)
    y, p = df["y"].values, df[col].values
    aucs = []
    for _ in range(reps):
        pick = rng.choice(len(keys), size=len(keys), replace=True)
        idx = np.concatenate([groups[keys[j]] for j in pick])
        if len(np.unique(y[idx])) == 2:
            aucs.append(roc_auc_score(y[idx], p[idx]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def build_report(df):
    y = df["y"].values
    report = {"models": {}}
    folds = sorted(df["fold"].unique())

    for col in ["rf", "xgb", "hgb", "ens"]:
        p = df[col].values
        a, b = fit_platt(p, y)

        # Nested: calibrate fold k with a map fit only on folds < k.
        nested_p = np.full(len(df), np.nan)
        for k in folds[1:]:
            fit_mask = (df["fold"] < k).values
            te_mask = (df["fold"] == k).values
            ak, bk = fit_platt(p[fit_mask], y[fit_mask])
            nested_p[te_mask] = apply_platt(p[te_mask], ak, bk)
        ev = ~np.isnan(nested_p)
        base = float(np.mean(y[ev]))
        brier_const = brier_score_loss(y[ev], np.full(ev.sum(), base))

        report["models"][col] = {
            "a": round(a, 5), "b": round(b, 5),
            "auc": round(float(roc_auc_score(y, p)), 4),
            "auc_by_fold": [round(float(roc_auc_score(y[df['fold'] == k], p[df['fold'] == k])), 4)
                            for k in folds],
            "brier_raw": round(float(brier_score_loss(y[ev], p[ev])), 5),
            "brier_calibrated": round(float(brier_score_loss(y[ev], nested_p[ev])), 5),
            "brier_constant": round(float(brier_const), 5),
            "ece_raw": round(ece(p[ev], y[ev]), 4),
            "ece_calibrated": round(ece(nested_p[ev], y[ev]), 4),
            "raw_p_range": [round(float(np.percentile(p, 5)), 3), round(float(np.percentile(p, 95)), 3)],
        }

    ens = report["models"]["ens"]
    p_cal = apply_platt(df["ens"].values, ens["a"], ens["b"])
    lo, hi = month_block_auc_ci(df, "ens")
    ens["auc_ci95"] = [round(lo, 4), round(hi, 4)]
    ens["cal_p_range"] = [round(float(np.percentile(p_cal, 5)), 3), round(float(np.percentile(p_cal, 95)), 3)]
    ens["reliability_raw"] = reliability(df["ens"].values, y)
    ens["reliability_calibrated"] = reliability(p_cal, y)

    # Measured skill tier. The verdict is driven by the CI, not the point estimate:
    # an AUC of 0.52 with a CI straddling 0.50 is not evidence of skill.
    if lo > 0.55:
        tier, label = "moderate", "Moderate"
    elif lo > 0.51:
        tier, label = "weak", "Weak"
    else:
        tier, label = "none", "No demonstrated edge"

    report.update({
        "version": 1,
        "generated": time.strftime("%Y-%m-%d"),
        "horizon_days": 20,
        "n_oos": int(len(df)),
        "oos_period": [str(df["date"].min().date()), str(df["date"].max().date())],
        "n_folds": len(folds),
        "n_tickers": int(df["ticker"].nunique()),
        "skill": {"tier": tier, "label": label, "auc": ens["auc"], "auc_ci95": ens["auc_ci95"]},
        "base_rate": round(float(np.mean(y)), 4),
    })
    return report


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    df = collect_oos()
    print(f"\n[calibrate] pooled {len(df):,} OOS predictions from {df['ticker'].nunique()} tickers")
    report = build_report(df)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    e = report["models"]["ens"]
    print(f"[calibrate] ensemble OOS AUC {e['auc']} (95% CI {e['auc_ci95']})  -> skill: {report['skill']['label']}")
    print(f"[calibrate] Brier  raw {e['brier_raw']}  calibrated {e['brier_calibrated']}  constant {e['brier_constant']}")
    print(f"[calibrate] ECE    raw {e['ece_raw']}  calibrated {e['ece_calibrated']}")
    print(f"[calibrate] raw p 5-95%: {e['raw_p_range']}  ->  calibrated: {e['cal_p_range']}")
    print(f"[calibrate] wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
