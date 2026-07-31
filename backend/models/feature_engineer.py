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
FEATURE_NAMES = [
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
