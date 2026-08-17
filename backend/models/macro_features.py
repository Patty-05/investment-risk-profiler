"""
Macro / market-context features (new data sources).

Each per-stock model currently sees only that stock's own price in isolation.
This module adds the Indian market regime around it — Nifty 50 trend/momentum,
the India VIX fear gauge, and USD/INR — so the models can learn regime-dependent
behaviour (e.g. momentum tends to reverse when VIX spikes). Every series is from
yfinance and fully reconstructable historically, so train and serve stay aligned.

The frame is fetched ONCE per process (lazy singleton) and reused across all
tickers during training and every inference call.
"""

import numpy as np
import pandas as pd
import yfinance as yf

# yfinance symbols
_NIFTY   = "^NSEI"
_INDIAVIX = "^INDIAVIX"
_USDINR  = "INR=X"

# Macro feature columns contributed to the model (order matters — appended to
# FEATURE_NAMES in feature_engineer). `rs_20d` is computed in build_frame from
# the stock's own 20d momentum minus the market's, so it is NOT here.
MACRO_FEATURE_NAMES = [
    "mkt_ret_20d",    # Nifty 50 20-day momentum
    "mkt_trend",      # Nifty sma10 vs sma50 (regime direction)
    "mkt_vol_20d",    # Nifty 20-day realised volatility
    "vix_level",      # India VIX / 100  (~0.1–0.4)
    "vix_z",          # India VIX z-score (fear regime)
    "usdinr_ret_20d", # rupee 20-day move (matters for exporters/importers)
]


def _safe_hist(symbol, period="10y"):
    try:
        df = yf.Ticker(symbol).history(period=period)
        if df is not None and not df.empty:
            return df["Close"].copy()
    except Exception as e:
        print(f"[macro] fetch failed for {symbol}: {e}")
    return None


class _MacroData:
    def __init__(self):
        self._frame = None

    def frame(self):
        if self._frame is not None:
            return self._frame

        nifty  = _safe_hist(_NIFTY)
        vix    = _safe_hist(_INDIAVIX)
        usdinr = _safe_hist(_USDINR)

        # Build on the Nifty calendar; other series are reindexed onto it.
        if nifty is None:
            print("[macro] Nifty unavailable — macro features will be neutral.")
            self._frame = pd.DataFrame(columns=MACRO_FEATURE_NAMES)
            return self._frame

        nifty = nifty[~nifty.index.duplicated(keep="last")].sort_index()
        # Normalise index to tz-naive dates so it aligns with per-stock frames.
        nifty.index = pd.to_datetime(nifty.index).tz_localize(None).normalize()

        out = pd.DataFrame(index=nifty.index)
        nret = nifty.pct_change()
        sma10 = nifty.rolling(10).mean()
        sma50 = nifty.rolling(50).mean()
        out["mkt_ret_20d"] = nifty.pct_change(20)
        out["mkt_trend"] = np.sign(sma10 - sma50)
        out["mkt_vol_20d"] = nret.rolling(20).std()

        if vix is not None:
            vix = vix[~vix.index.duplicated(keep="last")].sort_index()
            vix.index = pd.to_datetime(vix.index).tz_localize(None).normalize()
            vix = vix.reindex(nifty.index, method="ffill")
            level = vix / 100.0
            out["vix_level"] = level
            out["vix_z"] = ((vix - vix.rolling(120).mean()) / (vix.rolling(120).std() + 1e-9)).clip(-4, 4)
        else:
            out["vix_level"] = 0.15
            out["vix_z"] = 0.0

        if usdinr is not None:
            usdinr = usdinr[~usdinr.index.duplicated(keep="last")].sort_index()
            usdinr.index = pd.to_datetime(usdinr.index).tz_localize(None).normalize()
            usdinr = usdinr.reindex(nifty.index, method="ffill")
            out["usdinr_ret_20d"] = usdinr.pct_change(20)
        else:
            out["usdinr_ret_20d"] = 0.0

        self._frame = out.replace([np.inf, -np.inf], np.nan)
        print(f"[macro] loaded macro frame: {len(self._frame)} rows, "
              f"cols={list(self._frame.columns)}")
        return self._frame


# Module-level singleton
_INSTANCE = _MacroData()


def get_macro_frame():
    return _INSTANCE.frame()


_NIFTY_CLOSE = {"series": None, "loaded": False}


def get_nifty_close():
    """
    Nifty 50 close series (tz-naive, date-normalised index), cached per process.

    This is the BENCHMARK for the model's label: the target is whether a stock
    beats this index over the forward horizon, not whether it rises in absolute
    terms. Returns None if the index can't be fetched.
    """
    if _NIFTY_CLOSE["loaded"]:
        return _NIFTY_CLOSE["series"]

    s = _safe_hist(_NIFTY)
    if s is not None:
        s = s[~s.index.duplicated(keep="last")].sort_index()
        s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    _NIFTY_CLOSE["series"] = s
    _NIFTY_CLOSE["loaded"] = True
    return s


def align_macro(index):
    """Return a macro feature frame reindexed (ffill) onto `index`.

    Missing columns/rows fall back to neutral values so callers always get a
    complete frame with MACRO_FEATURE_NAMES columns.
    """
    macro = get_macro_frame()
    neutral = {"mkt_ret_20d": 0.0, "mkt_trend": 0.0, "mkt_vol_20d": 0.01,
               "vix_level": 0.15, "vix_z": 0.0, "usdinr_ret_20d": 0.0}
    if macro is None or macro.empty:
        return pd.DataFrame({c: neutral[c] for c in MACRO_FEATURE_NAMES}, index=index)

    try:
        idx = pd.to_datetime(index).tz_localize(None).normalize()
    except (TypeError, ValueError):
        idx = pd.to_datetime(index).normalize()

    aligned = macro.reindex(idx, method="ffill")
    aligned.index = index  # restore the caller's original index
    for c in MACRO_FEATURE_NAMES:
        if c not in aligned:
            aligned[c] = neutral[c]
        aligned[c] = aligned[c].fillna(neutral[c])
    return aligned[MACRO_FEATURE_NAMES]
