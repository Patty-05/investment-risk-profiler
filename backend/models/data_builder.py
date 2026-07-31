import numpy as np
from modules.data_fetcher import StockDataFetcher
from models.feature_engineer import FeatureEngineer, FEATURE_NAMES


class DataBuilder:
    """
    Builds (X, y) training data for one ticker from real OHLCV history.

    Features come from the SAME FeatureEngineer.build_frame used at inference,
    so training and serving see an identical feature distribution. The label is
    the 5-day forward return bucketed into SELL / HOLD / BUY.
    """

    def __init__(self, ticker, period="5y"):
        self.ticker = ticker
        # 5y of daily bars per ticker → far more real labelled samples.
        self.fetcher = StockDataFetcher(ticker, period=period)

    def build(self):
        df = self.fetcher.get_historical_data()
        if df is None or len(df) < 60:
            return None, None

        # Same canonical features as inference
        frame = FeatureEngineer.build_frame(df)

        # Label: 5-day forward return → SELL / HOLD / BUY
        frame["future_return"] = df["Close"].shift(-5) / df["Close"] - 1
        frame["label"] = frame["future_return"].apply(self.label_func)

        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 30:
            return None, None

        X = frame[FEATURE_NAMES].clip(-1e3, 1e3).values.astype(float)
        y = frame["label"].astype(int).values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    @staticmethod
    def label_func(x):
        if x > 0.03:
            return 2   # BUY
        elif x < -0.03:
            return 0   # SELL
        return 1       # HOLD
