"""
Phase B, round 3 -- feature-set surgery.

Round 2's IC screen turned up two things worth acting on.

  1. `vix_z` has exactly ONE unique value per date. It is the India VIX z-score,
     which is a property of the market, not of the stock, so every name shares it
     on any given day. Against a CROSS-SECTIONAL target -- does THIS stock beat
     the index -- a column that is constant across the cross-section cannot
     express a preference between two stocks. It can only tell the model what
     regime it is in, which is precisely the kind of thing that fits in-sample
     and evaporates out-of-sample. `market_trend` is nearly as bad at 2 distinct
     values per day.

     This is the same failure the project already hit once and fixed by deleting
     the absolute market series (mkt_ret_20d, vix_level, usdinr). `vix_z` was
     kept because it looked like a regime feature rather than a level. For a
     relative target the distinction does not matter: constant is constant.

  2. Seventeen features cleared |t| > 2 individually, led by atr_pct (IC 0.030)
     and dist_52w_low (0.022), with a consistent short-term REVERSAL pattern --
     momentum_5d, bollinger_position, macd_hist, stoch_k, roc_10 and rsi all
     carry NEGATIVE IC. If any of that is real rather than an artefact of
     overlapping labels, a model restricted to those features should do better
     than one that also has nine noise columns to overfit.

Both are tested here on the same purged walk-forward folds as everything else.

Run:
    cd backend && python -m models.tuning_features
"""

from __future__ import annotations

import torch_bootstrap  # noqa: F401  (Windows OpenMP load order)

import sys
import numpy as np

from models.panel import load_panel
from models.feature_engineer import FEATURE_NAMES
from models.walkforward import DEFAULT_FOLDS
from models.tuning import evaluate, xgb_regularised, xgb_default, logistic

# Zero (or near-zero) cross-sectional variation -- see module docstring.
MARKET_WIDE = ["vix_z", "market_trend"]

# Ranked by |IC| from the round-2 screen, strongest first.
IC_RANKED = [
    "atr_pct", "dist_52w_low", "volatility", "momentum_5d", "bollinger_position",
    "macd_hist", "momentum_20d", "rs_20d", "cci", "stoch_k", "roc_10",
    "downside_dev", "sharpe_ratio", "returns", "price_vs_sma50", "rsi",
    "return_skew",
]


def idx_of(names):
    return [FEATURE_NAMES.index(n) for n in names if n in FEATURE_NAMES]


def main():
    print("=" * 88)
    print("PHASE B ROUND 3 -- FEATURE-SET SURGERY (purged walk-forward)")
    print("=" * 88)
    X, y, meta = load_panel()

    all_idx = list(range(len(FEATURE_NAMES)))
    no_market = [i for i in all_idx if FEATURE_NAMES[i] not in MARKET_WIDE]

    variants = [
        ("all 26 features",                    all_idx,                    xgb_regularised, True),
        ("drop vix_z + market_trend (24)",     no_market,                  xgb_regularised, True),
        ("17 IC-significant only",             idx_of(IC_RANKED),          xgb_regularised, True),
        ("top-8 IC only",                      idx_of(IC_RANKED[:8]),      xgb_regularised, True),
        ("top-3 IC only",                      idx_of(IC_RANKED[:3]),      xgb_regularised, True),
        ("top-8 IC, logistic",                 idx_of(IC_RANKED[:8]),      logistic,        True),
        ("top-8 IC, default XGB",              idx_of(IC_RANKED[:8]),      xgb_default,     True),
        ("atr_pct alone",                      idx_of(["atr_pct"]),        xgb_regularised, True),
    ]

    print("\n{:<34}{:>6}{:>10}{:>9}{:>11}   {}".format(
        "feature set", "n", "AUC", "sd", "edge", "per-fold AUC"))
    print("-" * 88)

    results = []
    for label, idx, model_fn, rank in variants:
        r = evaluate(X, y, meta, model_fn, DEFAULT_FOLDS, rank=rank, feature_idx=idx)
        if not r:
            print("{:<34}  no result".format(label))
            continue
        results.append((label, r))
        print("{:<34}{:>6}{:>10.4f}{:>9.4f}{:>+11.4f}   {}".format(
            label, len(idx), r["auc"], r["auc_sd"], r["edge"], r["auc_folds"]))

    print("-" * 88)
    if results:
        best = max(results, key=lambda kv: kv[1]["auc"])
        print("\nBEST: {}  ->  AUC {:.4f}".format(best[0], best[1]["auc"]))
        print("\nReference points from round 2:")
        print("  pure-noise control feature ....... 0.5065")
        print("  full 26-feature model ............ 0.5072")
        print("  target the user asked about ...... 0.5200")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
