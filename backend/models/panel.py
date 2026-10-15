"""
Cached pooled panel for research runs.

Rebuilding the training panel means ~150 yfinance downloads and several minutes.
Every experiment in this directory needs the same panel, so it is built once and
cached to `data/panel/` as npy + csv. Research iterations then cost seconds
instead of minutes, which is the difference between testing one idea and testing
twenty.

The cache stores the RAW per-row features, labels and metadata (date, ticker,
realised excess return). Anything a specific experiment wants to do on top --
cross-sectional normalisation, extra feature blocks, different folds -- happens
downstream, so the cache never bakes in an experiment's assumptions.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

_PANEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "panel")


def _paths(tag):
    return (os.path.join(_PANEL_DIR, "{}_X.npy".format(tag)),
            os.path.join(_PANEL_DIR, "{}_y.npy".format(tag)),
            os.path.join(_PANEL_DIR, "{}_meta.csv".format(tag)))


def load_panel(tag="base", period="10y", tickers=None, rebuild=False, verbose=True):
    """
    (X, y, meta) for the full training universe, date-sorted, cached to disk.

    meta columns: date, ticker, excess_return.
    """
    xp, yp, mp = _paths(tag)
    if os.path.exists(xp) and os.path.exists(yp) and os.path.exists(mp) and not rebuild:
        X = np.load(xp)
        y = np.load(yp)
        meta = pd.read_csv(mp, parse_dates=["date"])
        if verbose:
            print("[panel] loaded cache '{}': {:,} rows x {} feats, {} -> {}".format(
                tag, len(X), X.shape[1], meta["date"].min().date(), meta["date"].max().date()))
        return X, y, meta

    from models.walkforward import build_pooled_dataset
    X, y, meta = build_pooled_dataset(tickers, period=period, verbose=verbose)

    os.makedirs(_PANEL_DIR, exist_ok=True)
    np.save(xp, X)
    np.save(yp, y)
    meta.to_csv(mp, index=False)
    if verbose:
        print("[panel] cached '{}': {:,} rows x {} feats".format(tag, len(X), X.shape[1]))
    return X, y, meta


def attach_geo(meta, verbose=True):
    """
    Geo feature block aligned row-for-row with `meta`.

    Returns (G, names) where G is (n_rows, len(GEO_FEATURE_NAMES)). Tickers the
    knowledge base has no exposure for get zeros, which is the correct encoding:
    no tracked geopolitical channel touches them.
    """
    from models import geo_features as gf

    dates = pd.to_datetime(meta["date"])
    weights = gf.weight_table(dates.unique())

    G = np.zeros((len(meta), len(gf.GEO_FEATURE_NAMES)))
    covered = 0
    for tk, block in meta.groupby("ticker", sort=False):
        if not gf.ticker_theme_map(tk):
            continue
        covered += 1
        idx = pd.to_datetime(block["date"])
        series = gf.geo_series(tk, idx, weights=weights)
        G[block.index.values, :] = series[gf.GEO_FEATURE_NAMES].values

    if verbose:
        nz = int((np.abs(G).sum(axis=1) > 0).sum())
        print("[panel] geo attached: {} tickers with exposure, "
              "{:,}/{:,} rows non-zero ({:.1f}%)".format(
                  covered, nz, len(G), 100.0 * nz / max(len(G), 1)))
    return G, list(gf.GEO_FEATURE_NAMES)


def cross_sectional_rank(X, meta, feature_names, exclude=()):
    """
    Convert each feature to its cross-sectional percentile within its own date.

    WHY THIS MATTERS FOR THIS PROBLEM
    ---------------------------------
    The label is RELATIVE -- does this stock beat the Nifty over 20 days. That is
    a cross-sectional question, but the features are absolute levels. An RSI of
    65 means something completely different in a calm 2021 tape than in the March
    2020 crash, so a model trained on raw levels spends most of its capacity
    learning what regime it is in rather than which stock is better placed.

    Ranking within each date removes the common market component by construction,
    which is exactly the component the relative label already netted out. This is
    standard practice for cross-sectional equity models and is the most likely
    single reason the current out-of-sample AUC sits at ~0.50.

    No look-ahead: a rank uses only same-day data across names, never other dates.
    """
    df = pd.DataFrame(X, columns=feature_names)
    df["_date"] = pd.to_datetime(meta["date"]).values

    keep = [c for c in feature_names if c not in exclude]
    ranked = df.groupby("_date")[keep].rank(pct=True, method="average")

    out = df[feature_names].copy()
    out[keep] = ranked
    # A date with a single name has no cross-section; centre it instead of NaN.
    out = out.fillna(0.5)
    return out.values.astype(float)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    X, y, meta = load_panel()
    print("\nlabel balance:", np.bincount(y), " baseline",
          round(max(np.mean(y), 1 - np.mean(y)), 4))
    print("tickers:", meta["ticker"].nunique())
