"""
Feature Engineering — the SINGLE source of truth for ML features.

Both training (DataBuilder, over every historical bar) and inference
(StockRiskAnalyzer, on the latest bar) build features through `build_frame`
here, so the feature distribution the model is trained on exactly matches what
it is served. This removes the earlier train/serve mismatch where fundamentals
and sentiment were constant placeholders in training but real at inference, and
where a couple of technical features (market_trend, bollinger_position) were
computed with different formulas on each side.

Every feature is derived purely from OHLCV price/volume history, so it is
identical and reconstructable per bar. Fundamentals and sentiment are NOT model
inputs (they can't be reconstructed historically from free data) — they remain
separate factors in the overall score, and sentiment still feeds the rule-based
override as a heuristic.
"""

import numpy as np
import pandas as pd

# Canonical model feature order — used everywhere.
# Every feature is reconstructable per-bar from OHLCV history, so the training
# distribution matches inference exactly. Expanded (2026-08) from 12 → 24: the
# original core plus a wider technical/price-derived set (MACD, stochastic, CCI,
# trend strength, multi-window ROC/momentum, 52-week positioning, volatility
# regime, downside deviation, return skew, OBV trend). Fundamentals/sentiment
# are deliberately NOT here — yfinance can't reconstruct them historically, so
# they'd reintroduce a train/serve mismatch; they remain separate app factors.
FEATURE_NAMES = [
    # ── original core ──────────────────────────────────────────────────────
    "rsi",
    "returns",
    "momentum_5d",
    "momentum_20d",
    "volatility",
    "volume_change",
    "volume_ratio",
    "market_trend",
    "price_vs_sma50",
    "bollinger_position",
    "sharpe_ratio",
    "atr_pct",
    # ── expanded technical / price-derived set ─────────────────────────────
    "macd_hist",         # MACD histogram, normalised by price
    "stoch_k",           # stochastic %K (0-1) over 14
    "cci",               # commodity channel index (clipped)
    "trend_strength",    # (sma20 - sma50) / sma50
    "roc_10",            # 10-day rate of change
    "momentum_60d",      # ~quarter momentum
    "dist_52w_high",     # distance below the rolling 252-day high
    "dist_52w_low",      # distance above the rolling 252-day low
    "vol_regime",        # short (10d) vs long (60d) volatility ratio
    "downside_dev",      # 20-day downside deviation (negative returns only)
    "return_skew",       # 20-day return skew
    "obv_trend",         # normalised 10-day OBV momentum
]


def _rsi(close, window=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / (loss + 1e-9)
    return 100 - 100 / (1 + rs)


def _atr_pct(df, window=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean() / (close + 1e-9)


def _macd_hist(close, fast=12, slow=26, signal=9):
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return (macd - sig) / (close + 1e-9)


def _stoch_k(df, window=14):
    low_n = df["Low"].rolling(window).min()
    high_n = df["High"].rolling(window).max()
    return (df["Close"] - low_n) / ((high_n - low_n) + 1e-9)


def _cci(df, window=20):
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    sma_tp = tp.rolling(window).mean()
    mad = (tp - sma_tp).abs().rolling(window).mean()
    return ((tp - sma_tp) / (0.015 * mad + 1e-9)).clip(-300, 300) / 100.0


def _obv_trend(close, volume, window=10):
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    # Normalise the 10-day OBV change by recent turnover so it's scale-free.
    scale = (volume.rolling(20).mean() * window) + 1e-9
    return ((obv - obv.shift(window)) / scale).clip(-5, 5)


class FeatureEngineer:
    def get_feature_names(self):
        return list(FEATURE_NAMES)

    # ── Shared feature computation (used by train AND serve) ────────────────
    @staticmethod
    def build_frame(df):
        """
        Given an OHLCV DataFrame, return a DataFrame of the canonical features
        (one row per input bar, same index). NaNs remain for warm-up rows.
        """
        df = df.copy().replace([np.inf, -np.inf], np.nan)
        close = df["Close"]
        volume = df["Volume"]
        returns = close.pct_change()

        sma10 = close.rolling(10).mean()
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean()
        std20 = close.rolling(20).std()

        out = pd.DataFrame(index=df.index)
        # ── original core ──────────────────────────────────────────────────
        out["rsi"] = _rsi(close)
        out["returns"] = returns
        out["momentum_5d"] = close.pct_change(5)
        out["momentum_20d"] = close.pct_change(20)
        out["volatility"] = returns.rolling(20).std()
        out["volume_change"] = volume.pct_change()
        out["volume_ratio"] = volume / (volume.rolling(20).mean() + 1e-9)
        out["market_trend"] = np.sign(sma10 - sma50)
        out["price_vs_sma50"] = (close - sma50) / (sma50 + 1e-9)
        out["bollinger_position"] = ((close - sma20) / (2 * std20 + 1e-9)).clip(-1.5, 1.5)
        out["sharpe_ratio"] = (
            returns.rolling(20).mean() / (returns.rolling(20).std() + 1e-9)
        ).clip(-10, 10)
        out["atr_pct"] = _atr_pct(df)

        # ── expanded technical / price-derived set ─────────────────────────
        high_252 = close.rolling(252, min_periods=60).max()
        low_252 = close.rolling(252, min_periods=60).min()
        vol_short = returns.rolling(10).std()
        vol_long = returns.rolling(60).std()
        neg_ret = returns.where(returns < 0, 0.0)

        out["macd_hist"] = _macd_hist(close)
        out["stoch_k"] = _stoch_k(df)
        out["cci"] = _cci(df)
        out["trend_strength"] = (sma20 - sma50) / (sma50 + 1e-9)
        out["roc_10"] = close.pct_change(10)
        out["momentum_60d"] = close.pct_change(60)
        out["dist_52w_high"] = (close - high_252) / (high_252 + 1e-9)
        out["dist_52w_low"] = (close - low_252) / (low_252 + 1e-9)
        out["vol_regime"] = (vol_short / (vol_long + 1e-9)).clip(0, 5)
        out["downside_dev"] = neg_ret.rolling(20).std()
        out["return_skew"] = returns.rolling(20).skew()
        out["obv_trend"] = _obv_trend(close, volume)

        return out.replace([np.inf, -np.inf], np.nan)

    # ── Inference: latest bar → (vector, dict) ──────────────────────────────
    def build_latest(self, df):
        if df is None or len(df) == 0:
            return self._neutral()
        frame = self.build_frame(df).dropna()
        if len(frame) == 0:
            return self._neutral()
        row = frame.iloc[-1]
        feats = {n: float(row[n]) for n in FEATURE_NAMES}
        vec = np.array([[feats[n] for n in FEATURE_NAMES]], dtype=np.float32)
        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
        return vec, feats

    def _neutral(self):
        feats = {n: 0.0 for n in FEATURE_NAMES}
        return np.zeros((1, len(FEATURE_NAMES)), dtype=np.float32), feats
