import numpy as np
from modules.data_fetcher import StockDataFetcher
from models.feature_engineer import FeatureEngineer, FEATURE_NAMES


class DataBuilder:
    """
    Builds (X, y) training data for one ticker from real OHLCV history.

    Features come from the SAME FeatureEngineer.build_frame used at inference,
    so training and serving see an identical feature distribution. The label is
    the 20-day forward return bucketed into SELL / HOLD / BUY — a longer horizon
    than the old 5-day one, which carries real trend signal (5-day returns are
    close to noise) so the classes separate better and confidence rises.
    """

    # Forward-return horizon (trading days) and the SELL/BUY deadband. Wider
    # thresholds than the 5-day model because 20-day moves are ~2x larger.
    HORIZON = 20
    BUY_THRESHOLD = 0.05
    SELL_THRESHOLD = -0.05

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

        # Label: 20-day forward return → SELL / HOLD / BUY
        frame["future_return"] = df["Close"].shift(-self.HORIZON) / df["Close"] - 1
        frame["label"] = frame["future_return"].apply(self.label_func)

        frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < 30:
            return None, None

        X = frame[FEATURE_NAMES].clip(-1e3, 1e3).values.astype(float)
        y = frame["label"].astype(int).values

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X, y

    @classmethod
    def label_func(cls, x):
        if x > cls.BUY_THRESHOLD:
            return 2   # BUY
        elif x < cls.SELL_THRESHOLD:
            return 0   # SELL
        return 1       # HOLD
