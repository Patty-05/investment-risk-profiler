import numpy as np
import pandas as pd
from modules.data_fetcher import StockDataFetcher
from models.feature_engineer import FeatureEngineer, FEATURE_NAMES
from models.macro_features import get_nifty_close


class DataBuilder:
    """
    Builds (X, y) training data for one ticker from real OHLCV history.

    Features come from the SAME FeatureEngineer.build_frame used at inference,
    so training and serving see an identical feature distribution.

    LABEL — relative performance vs the Nifty 50 (binary):
        excess = stock 20-day forward return  -  Nifty 20-day forward return
        excess > 0  ->  OUTPERFORM (1)
        else        ->  UNDERPERFORM (0)

    Why relative and binary rather than absolute SELL/HOLD/BUY:
      * It answers the question an investor picking Indian equities actually
        asks — "will this beat the index?" — instead of "will it go up?", which
        is dominated by whatever the whole market does.
      * It strips out market-wide drift, so the model learns stock-specific
        edge rather than re-predicting the Nifty.
      * Two balanced classes give a clean ~50% baseline and genuinely
        meaningful confidence, instead of a HOLD-heavy 3-class problem where
        confidence was structurally capped and the majority class dominated.
    """

    HORIZON = 20   # trading days (~1 month)

    def __init__(self, ticker, period="5y"):
        self.ticker = ticker
        # 5y of daily bars per ticker → far more real labelled samples.
        self.fetcher = StockDataFetcher(ticker, period=period)

    def build(self, with_meta=False):
        """
        Returns (X, y) — or (X, y, meta) when `with_meta` is True, where meta is
        a DataFrame carrying the observation date and the realised excess return
        for each row. The dates are what make time-ordered (walk-forward)
        validation possible; the excess return is what a backtest needs to size
        realised performance rather than just count hits.
        """
        empty = (None, None, None) if with_meta else (None, None)

        df = self.fetcher.get_historical_data()
        if df is None or len(df) < 60:
            return empty

        # Same canonical features as inference
        frame = FeatureEngineer.build_frame(df)

        # Stock's own forward return over the horizon
        stock_fwd = df["Close"].shift(-self.HORIZON) / df["Close"] - 1

        # Benchmark's forward return over the same dates
        bench_fwd = self._benchmark_forward_return(df.index)
        if bench_fwd is None:
            return empty

        excess = stock_fwd - bench_fwd
        frame["excess_return"] = excess.values
        frame["label"] = (frame["excess_return"] > 0).astype(int)

        # Drop rows where the excess return is unknown (tail of the series).
        frame = frame.replace([np.inf, -np.inf], np.nan)
        frame = frame.dropna(subset=["excess_return"]).dropna()
        if len(frame) < 30:
            return empty

        X = frame[FEATURE_NAMES].clip(-1e3, 1e3).values.astype(float)
        y = frame["label"].astype(int).values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if not with_meta:
            return X, y

        try:
            dates = pd.to_datetime(frame.index).tz_localize(None).normalize()
        except (TypeError, ValueError):
            dates = pd.to_datetime(frame.index).normalize()
        meta = pd.DataFrame({
            "date": dates,
            "ticker": self.ticker,
            "excess_return": frame["excess_return"].values,
        })
        return X, y, meta

    def _benchmark_forward_return(self, index):
        """Nifty 50 forward return over HORIZON, aligned onto the stock's dates."""
        nifty = get_nifty_close()
        if nifty is None or len(nifty) == 0:
            print("[data] Nifty benchmark unavailable — cannot build relative labels.")
            return None

        try:
            idx = pd.to_datetime(index).tz_localize(None).normalize()
        except (TypeError, ValueError):
            idx = pd.to_datetime(index).normalize()

        # Forward return computed on the index's OWN calendar, then aligned to
        # the stock's trading days (ffill covers any holiday mismatch).
        nifty_fwd = nifty.shift(-self.HORIZON) / nifty - 1
        aligned = nifty_fwd.reindex(idx, method="ffill")
        aligned.index = index
        return aligned
