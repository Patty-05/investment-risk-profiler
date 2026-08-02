"""
Out-of-sample backtest / signal validation for the stock model.

Honest evaluation of whether the model's BUY/SELL/HOLD calls have real
predictive edge — the foundation for any claim of value.

Methodology (no look-ahead leakage):
  • TIME split, not ticker split: train a fresh model only on bars dated on or
    before `train_end`, then evaluate on bars strictly after it. The 5-day
    forward-return label means we also drop the final 5 bars of the train window
    so a label never peeks past `train_end`.
  • Features come from the SAME FeatureEngineer.build_frame used in production.
  • The headline metric is ECONOMIC edge, not accuracy: the mean 5-day forward
    return of BUY-tagged bars vs SELL-tagged bars vs the all-bar baseline, plus
    hit-rate (share of BUY bars that are actually positive) vs the base rate.

Run:  python -m models.backtest         (from the backend/ directory)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modules.data_fetcher import StockDataFetcher
from models.feature_engineer import FeatureEngineer, FEATURE_NAMES
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import accuracy_score
from sklearn.utils.class_weight import compute_sample_weight

try:
    from xgboost import XGBClassifier
    _XGB = True
except ImportError:
    _XGB = False

# Liquid NSE universe for the backtest (time-split keeps it honest even though
# these overlap the training universe — the test period is unseen).
BACKTEST_TICKERS = [
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS",
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "INDUSINDBK.NS", "BANKBARODA.NS",
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SBICARD.NS", "CHOLAFIN.NS", "HDFCLIFE.NS", "SBILIFE.NS",
    "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "ITC.NS", "TATACONSUM.NS", "DABUR.NS", "MARICO.NS",
    "MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS", "HEROMOTOCO.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "LUPIN.NS", "APOLLOHOSP.NS",
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS", "GAIL.NS", "TATAPOWER.NS",
    "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "JINDALSTEL.NS",
    "ULTRACEMCO.NS", "SHREECEM.NS", "GRASIM.NS",
    "LT.NS", "ADANIPORTS.NS", "ADANIENT.NS", "SIEMENS.NS", "HAVELLS.NS",
    "TITAN.NS", "ASIANPAINT.NS", "PIDILITIND.NS", "BHARTIARTL.NS", "DMART.NS", "TRENT.NS",
    "HAL.NS", "BEL.NS",
]

_FWD = 5           # forward-return horizon (trading days)
_UP, _DOWN = 0.03, -0.03


def _label(x):
    return 2 if x > _UP else (0 if x < _DOWN else 1)


def build_panel(tickers, period="5y"):
    """Stacked per-bar features + 5-day forward return + label, with a date column."""
    frames = []
    ok = 0
    for t in tickers:
        try:
            df = StockDataFetcher(t, period=period).get_historical_data()
        except Exception:
            df = None
        if df is None or len(df) < 120:
            continue
        fr = FeatureEngineer.build_frame(df)
        fr["fwd"] = df["Close"].shift(-_FWD) / df["Close"] - 1
        fr["label"] = fr["fwd"].apply(_label)
        fr = fr.reset_index()
        fr = fr.rename(columns={fr.columns[0]: "date"})
        fr["ticker"] = t
        fr = fr.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURE_NAMES + ["fwd", "label"])
        if len(fr):
            frames.append(fr)
            ok += 1
    panel = pd.concat(frames, ignore_index=True)
    panel["date"] = pd.to_datetime(panel["date"], utc=True)
    print(f"[backtest] panel: {len(panel)} bars from {ok} tickers "
          f"({panel['date'].min().date()} -> {panel['date'].max().date()})")
    return panel


def _fit(Xtr, ytr):
    scaler = RobustScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_leaf=3,
        class_weight="balanced", random_state=42, n_jobs=-1,
    ).fit(Xtr_s, ytr)
    xgb = None
    if _XGB:
        sw = compute_sample_weight("balanced", ytr)
        xgb = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.7, eval_metric="mlogloss",
            random_state=42, verbosity=0, n_jobs=-1,
        ).fit(Xtr_s, ytr, sample_weight=sw)
    return scaler, rf, xgb


def _report(name, test, pred_col):
    base = test["fwd"].mean()
    base_pos = (test["fwd"] > 0).mean()
    acc = accuracy_score(test["label"], test[pred_col])

    buy = test.loc[test[pred_col] == 2, "fwd"]
    sell = test.loc[test[pred_col] == 0, "fwd"]
    hold = test.loc[test[pred_col] == 1, "fwd"]

    def pct(x):
        return f"{x * 100:+.3f}%"

    print(f"\n===== {name} =====")
    print(f"  out-of-sample accuracy : {acc:.3f}")
    print(f"  baseline 5d fwd return : {pct(base)}   (all test bars; positive rate {base_pos*100:.1f}%)")
    print(f"  BUY-tagged  bars: n={len(buy):6d}  mean fwd {pct(buy.mean())}  "
          f"edge {pct(buy.mean()-base)}  hit {(buy>0).mean()*100:.1f}%")
    print(f"  SELL-tagged bars: n={len(sell):6d}  mean fwd {pct(sell.mean())}  "
          f"edge {pct(sell.mean()-base)}  hit {(sell>0).mean()*100:.1f}%")
    print(f"  HOLD-tagged bars: n={len(hold):6d}  mean fwd {pct(hold.mean())}")
    spread = buy.mean() - sell.mean()
    print(f"  BUY-minus-SELL spread  : {pct(spread)}   <- key economic-edge signal")


def run(train_end="2024-06-30"):
    panel = build_panel(BACKTEST_TICKERS)
    cutoff = pd.Timestamp(train_end, tz="UTC")

    # Drop the last _FWD bars before the cutoff so no train label peeks past it
    label_safe = cutoff - pd.Timedelta(days=_FWD * 2)
    train = panel[panel["date"] <= label_safe]
    test = panel[panel["date"] > cutoff]

    print(f"[backtest] train: {len(train)} bars (<= {label_safe.date()}), "
          f"test: {len(test)} bars (> {cutoff.date()})")
    dist = train["label"].value_counts().to_dict()
    print(f"[backtest] train label dist SELL/HOLD/BUY: "
          f"{dist.get(0,0)}/{dist.get(1,0)}/{dist.get(2,0)}")

    scaler, rf, xgb = _fit(train[FEATURE_NAMES].values, train["label"].values)
    Xte = scaler.transform(test[FEATURE_NAMES].values)

    test = test.copy()
    test["pred_rf"] = rf.predict(Xte)
    _report("Random Forest (out-of-sample)", test, "pred_rf")
    if xgb is not None:
        test["pred_xgb"] = xgb.predict(Xte)
        _report("XGBoost (out-of-sample)", test, "pred_xgb")

    return test


if __name__ == "__main__":
    run()
