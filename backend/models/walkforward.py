"""
Walk-forward (time-ordered) validation.

WHY THIS EXISTS
---------------
The models were previously scored with `StratifiedKFold`, which shuffles rows
at random. For a time series that is invalid twice over:

  1. LOOK-AHEAD LEAKAGE — a random split puts 2025 rows in the training set and
     2022 rows in the test set, so the model is effectively told the future.
  2. OVERLAPPING LABELS — our label is a 20-day forward return, so a row dated
     T and a row dated T+5 share 15 days of outcome. Neighbouring rows are
     near-duplicates; scattering them across folds leaks the answer.

Both inflate the score. This module replaces that with an honest protocol:

  * TIME-ORDERED SPLITS — train strictly on the past, test strictly on the
    future (expanding window).
  * PURGE / EMBARGO — drop the HORIZON trading days immediately before each
    test window from the training set, so no training label can "see into" the
    test period. (Purged cross-validation, after Lopez de Prado.)
  * POOLED, DATE-ALIGNED — every ticker's rows are pooled and split on the
    same calendar dates, so a fold is a genuine snapshot of the market.

Reported metrics are therefore out-of-sample and comparable to what live
trading would have produced.

Run:
    cd backend && python -m models.walkforward
"""

from __future__ import annotations

# MUST be first: fixes the Windows OpenMP load order for torch and lightgbm,
# both of which segfault if numpy/sklearn/xgboost get in ahead of them.
import torch_bootstrap  # noqa: F401

import sys
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.utils.class_weight import compute_sample_weight

from models.data_builder import DataBuilder
from models.feature_engineer import FEATURE_NAMES
from models.stock_model import TRAINING_TICKERS, StockRiskModel

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


HORIZON = DataBuilder.HORIZON


# ── Dataset assembly ───────────────────────────────────────────────────────

def build_pooled_dataset(tickers=None, period="10y", verbose=True):
    """Pool every ticker's (features, label, date) rows into one time-indexed set."""
    tickers = tickers or TRAINING_TICKERS
    X_all, y_all, meta_all = [], [], []
    ok = skipped = 0

    for tk in tickers:
        try:
            X, y, meta = DataBuilder(tk, period=period).build(with_meta=True)
            if X is None or len(X) == 0:
                skipped += 1
                continue
            X_all.append(X)
            y_all.append(y)
            meta_all.append(meta)
            ok += 1
            if verbose:
                print(f"  [ok]   {tk}: {len(X)} rows "
                      f"({meta['date'].min().date()} → {meta['date'].max().date()})")
        except Exception as e:
            skipped += 1
            if verbose:
                print(f"  [err]  {tk}: {e}")

    if not X_all:
        raise RuntimeError("No training data could be built.")

    X = np.vstack(X_all)
    y = np.concatenate(y_all)
    meta = pd.concat(meta_all, ignore_index=True)

    order = np.argsort(meta["date"].values, kind="stable")
    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)

    if verbose:
        print(f"\n[data] pooled {len(X):,} rows from {ok} tickers ({skipped} skipped)")
        print(f"[data] span {meta['date'].min().date()} → {meta['date'].max().date()}")
    return X, y, meta


# ── Splitting ──────────────────────────────────────────────────────────────

def time_split(meta, train_end, test_start, test_end, embargo_days=HORIZON):
    """
    Boolean masks for one purged walk-forward fold.

    The embargo removes the `embargo_days` trading sessions before `test_start`
    from training: those rows' 20-day labels resolve inside the test window, so
    keeping them would leak the future.
    """
    d = pd.to_datetime(meta["date"])
    train_cutoff = pd.Timestamp(train_end) - pd.Timedelta(days=int(embargo_days * 1.5))
    train_mask = d <= train_cutoff
    test_mask = (d >= pd.Timestamp(test_start)) & (d <= pd.Timestamp(test_end))
    return train_mask.values, test_mask.values


# ── Model zoo ──────────────────────────────────────────────────────────────

def _model_zoo(rule_col_train=None, rule_col_test=None):
    """Factories for each model. Kept in one place so folds stay identical."""
    zoo = {
        "Calibrated RF": lambda: CalibratedClassifierCV(
            RandomForestClassifier(
                n_estimators=300, max_depth=12, min_samples_split=6,
                min_samples_leaf=3, max_features="sqrt",
                class_weight="balanced", random_state=42, n_jobs=-1),
            method="isotonic", cv=3),
        "Hybrid HGB": lambda: HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_depth=6,
            l2_regularization=1.0, max_leaf_nodes=31,
            early_stopping=True, validation_fraction=0.1, random_state=42),
    }
    if XGB_AVAILABLE:
        zoo["XGBoost"] = lambda: XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.5,
            min_child_weight=3, gamma=0.1, eval_metric="logloss",
            random_state=42, verbosity=0, n_jobs=-1)
    if LGBM_AVAILABLE:
        # n_jobs=1: LightGBM segfaults ("access violation") on this Windows box
        # when its OpenMP runtime races with the one already loaded by
        # numpy/sklearn. Single-threaded is slower but stable.
        zoo["LightGBM"] = lambda: LGBMClassifier(
            n_estimators=500, learning_rate=0.05, num_leaves=31,
            max_depth=-1, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, min_child_samples=30,
            class_weight="balanced", random_state=42, n_jobs=1, verbosity=-1)
    return zoo


def evaluate_fold(X, y, meta, train_mask, test_mask, label="fold", verbose=True):
    """Fit every model on the train slice, score on the held-out future slice."""
    Xtr_raw, ytr = X[train_mask], y[train_mask]
    Xte_raw, yte = X[test_mask], y[test_mask]

    if len(Xtr_raw) < 500 or len(Xte_raw) < 100:
        if verbose:
            print(f"  [skip] {label}: too few rows (train {len(Xtr_raw)}, test {len(Xte_raw)})")
        return None
    if len(np.unique(yte)) < 2:
        if verbose:
            print(f"  [skip] {label}: test slice is single-class")
        return None

    scaler = RobustScaler()
    Xtr = scaler.fit_transform(Xtr_raw)
    Xte = scaler.transform(Xte_raw)

    # The hybrid model also takes the deterministic rule score as a feature.
    rule_tr = StockRiskModel._rule_score_column(Xtr_raw).reshape(-1, 1)
    rule_te = StockRiskModel._rule_score_column(Xte_raw).reshape(-1, 1)
    Xtr_h = np.hstack([Xtr, rule_tr])
    Xte_h = np.hstack([Xte, rule_te])

    baseline = max(np.mean(yte), 1 - np.mean(yte))
    results = {"_meta": {
        "label": label,
        "n_train": int(len(Xtr)), "n_test": int(len(Xte)),
        "baseline": float(baseline),
        "test_start": str(pd.to_datetime(meta.loc[test_mask, "date"]).min().date()),
        "test_end": str(pd.to_datetime(meta.loc[test_mask, "date"]).max().date()),
    }}

    sw = compute_sample_weight("balanced", ytr)
    for name, factory in _model_zoo().items():
        try:
            model = factory()
            hybrid = name == "Hybrid HGB"
            A, B = (Xtr_h, Xte_h) if hybrid else (Xtr, Xte)
            if name in ("XGBoost", "Hybrid HGB"):
                model.fit(A, ytr, sample_weight=sw)
            else:
                model.fit(A, ytr)
            proba = model.predict_proba(B)[:, 1]
            pred = (proba >= 0.5).astype(int)
            results[name] = {
                "accuracy": float(accuracy_score(yte, pred)),
                "roc_auc": float(roc_auc_score(yte, proba)),
                "edge": float(accuracy_score(yte, pred) - baseline),
            }
        except Exception as e:
            print(f"  [warn] {name} failed on {label}: {e}")

    if verbose:
        m = results["_meta"]
        print(f"\n  {label}  test {m['test_start']} → {m['test_end']}  "
              f"(train {m['n_train']:,} / test {m['n_test']:,})   baseline {m['baseline']:.3f}")
        for name in [k for k in results if k != "_meta"]:
            r = results[name]
            print(f"    {name:<16} acc {r['accuracy']:.3f}   AUC {r['roc_auc']:.3f}   "
                  f"edge {r['edge']:+.3f}")
    return results


# ── Protocol ───────────────────────────────────────────────────────────────

# Expanding-window folds. The final fold is the headline split the project
# committed to: train 2020-2023, test 2024-2025.
DEFAULT_FOLDS = [
    ("2020-01-01", "2022-12-31", "2023-01-01", "2023-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2024-12-31"),
    ("2020-01-01", "2024-12-31", "2025-01-01", "2025-12-31"),
    ("2020-01-01", "2023-12-31", "2024-01-01", "2025-12-31"),   # headline
]


def run(tickers=None, period="10y", folds=None, verbose=True):
    folds = folds or DEFAULT_FOLDS
    print("=" * 74)
    print("WALK-FORWARD VALIDATION — time-ordered, purged (embargo = "
          f"{HORIZON} sessions)")
    print("=" * 74)

    X, y, meta = build_pooled_dataset(tickers, period=period, verbose=verbose)

    all_results = []
    for i, (tr_s, tr_e, te_s, te_e) in enumerate(folds, 1):
        name = f"fold {i}: train ≤{tr_e[:4]} → test {te_s[:4]}-{te_e[:4]}"
        train_mask, test_mask = time_split(meta, tr_e, te_s, te_e)
        res = evaluate_fold(X, y, meta, train_mask, test_mask, label=name, verbose=verbose)
        if res:
            all_results.append(res)

    _summarise(all_results)
    return all_results


def _summarise(all_results):
    if not all_results:
        print("\nNo folds produced results.")
        return
    print("\n" + "=" * 74)
    print("SUMMARY — mean across folds (out-of-sample)")
    print("=" * 74)
    names = [k for k in all_results[0] if k != "_meta"]
    base = float(np.mean([r["_meta"]["baseline"] for r in all_results]))
    print(f"{'model':<18}{'acc':>8}{'AUC':>8}{'edge vs baseline':>20}")
    print("-" * 74)
    rows = []
    for n in names:
        accs = [r[n]["accuracy"] for r in all_results if n in r]
        aucs = [r[n]["roc_auc"] for r in all_results if n in r]
        if not accs:
            continue
        rows.append((n, float(np.mean(accs)), float(np.mean(aucs)),
                     float(np.mean(accs)) - base))
    for n, a, u, e in sorted(rows, key=lambda z: -z[2]):
        print(f"{n:<18}{a:>8.3f}{u:>8.3f}{e:>+20.3f}")
    print("-" * 74)
    print(f"{'(baseline)':<18}{base:>8.3f}")
    print("\nAUC > 0.5 and a positive edge mean genuine out-of-sample signal.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    run()
