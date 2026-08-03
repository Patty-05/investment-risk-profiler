"""
Peer alternatives — "if this one isn't performing, who in its sector is?"

Given the analysed NSE stock, find its sector cohort, rank the cohort by recent
price performance, and surface the better-performing peers as alternatives when
the analysed stock is lagging. Peer prices are batch-fetched in a single
yfinance call and cached per group to keep latency low. Educational comparison
of past performance — not a recommendation to buy.
"""

import time
import numpy as np
import pandas as pd
import yfinance as yf

# Sector cohorts (NSE). Mirrors the training universe groups so the peer set is
# curated and liquid. A stock is matched to its group by membership first, then
# by a coarse yfinance-sector fallback for names outside this list.
PEER_GROUPS = {
    "IT": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
           "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS"],
    "Banks": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
              "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS", "AUBANK.NS"],
    "Financials": ["BAJFINANCE.NS", "BAJAJFINSV.NS", "SBICARD.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS",
                   "MUTHOOTFIN.NS", "PFC.NS", "RECLTD.NS", "HDFCLIFE.NS", "SBILIFE.NS", "ICICIPRULI.NS", "HDFCAMC.NS"],
    "FMCG": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS",
             "GODREJCP.NS", "ITC.NS", "TATACONSUM.NS", "COLPAL.NS", "VBL.NS"],
    "Auto": ["MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
             "M&M.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "MOTHERSON.NS", "BOSCHLTD.NS", "MRF.NS"],
    "Pharma": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
               "TORNTPHARM.NS", "LUPIN.NS", "ZYDUSLIFE.NS", "ALKEM.NS", "BIOCON.NS", "APOLLOHOSP.NS", "MAXHEALTH.NS"],
    "Realty": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS", "BRIGADE.NS"],
    "Energy": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "BPCL.NS",
               "IOC.NS", "GAIL.NS", "OIL.NS", "PETRONET.NS", "ADANIGREEN.NS", "TATAPOWER.NS", "JSWENERGY.NS", "NHPC.NS", "IGL.NS"],
    "Metals": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "NMDC.NS",
               "JINDALSTEL.NS", "SAIL.NS", "NATIONALUM.NS", "HINDZINC.NS", "APLAPOLLO.NS"],
    "Cement": ["ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "DALBHARAT.NS", "GRASIM.NS"],
    "Infra": ["LT.NS", "ADANIPORTS.NS", "ADANIENT.NS", "SIEMENS.NS", "ABB.NS", "CUMMINSIND.NS",
              "HAVELLS.NS", "POLYCAB.NS", "DIXON.NS", "BEL.NS", "HAL.NS", "BDL.NS", "MAZDOCK.NS"],
    "Chemicals": ["SRF.NS", "DEEPAKNTR.NS", "PIIND.NS", "AARTIIND.NS", "NAVINFLUOR.NS",
                  "TATACHEM.NS", "UPL.NS", "COROMANDEL.NS"],
    "Consumer": ["ASIANPAINT.NS", "BERGEPAINT.NS", "TITAN.NS", "PIDILITIND.NS", "TRENT.NS",
                 "DMART.NS", "PAGEIND.NS", "JUBLFOOD.NS", "NYKAA.NS"],
    "Telecom": ["BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS", "ZOMATO.NS", "NAUKRI.NS",
                "IRCTC.NS", "INDIGO.NS", "INDHOTEL.NS"],
    "Textiles": ["KPRMILL.NS", "TRIDENT.NS", "WELSPUNLIV.NS"],
}

# Coarse fallback: yfinance sector string → our group (best-effort).
_YF_SECTOR_TO_GROUP = {
    "technology": "IT",
    "financial services": "Banks",
    "consumer defensive": "FMCG",
    "consumer cyclical": "Consumer",
    "healthcare": "Pharma",
    "energy": "Energy",
    "utilities": "Energy",
    "basic materials": "Metals",
    "industrials": "Infra",
    "real estate": "Realty",
    "communication services": "Telecom",
}

_TTL = 3600  # seconds
_cache = {}  # group -> (fetched_at, {ticker: {"return_6m":.., "return_1m":..}})


def _group_for(ticker, yf_sector=None):
    t = (ticker or "").upper().strip()
    for group, members in PEER_GROUPS.items():
        if t in members:
            return group
    if yf_sector:
        return _YF_SECTOR_TO_GROUP.get(str(yf_sector).strip().lower())
    return None


def _fetch_group_returns(group):
    """Batch-fetch 6-month closes for a group; return {ticker: {return_6m, return_1m}}."""
    now = time.time()
    cached = _cache.get(group)
    if cached and now - cached[0] < _TTL:
        return cached[1]

    tickers = PEER_GROUPS[group]
    out = {}
    try:
        data = yf.download(tickers, period="6mo", progress=False,
                           auto_adjust=True, threads=True)
        closes = data["Close"] if "Close" in data else data
        if isinstance(closes, pd.Series):          # single-ticker edge case
            closes = closes.to_frame()
        for tk in tickers:
            if tk not in closes.columns:
                continue
            s = closes[tk].dropna()
            if len(s) < 25:
                continue
            first, last = float(s.iloc[0]), float(s.iloc[-1])
            m1 = float(s.iloc[-21]) if len(s) > 21 else first
            if first > 0:
                out[tk] = {
                    "return_6m": round((last / first - 1) * 100, 2),
                    "return_1m": round((last / m1 - 1) * 100, 2) if m1 > 0 else None,
                }
    except Exception as e:
        print(f"[peers] batch fetch failed for {group}: {e}")

    _cache[group] = (now, out)
    return out


def get_peer_alternatives(ticker, yf_sector=None, self_return_6m=None, top_n=3):
    """
    Rank the sector cohort by 6-month return and, if the analysed stock lags,
    surface the better-performing peers. `self_return_6m` (percent) is passed in
    from the already-fetched history to avoid a redundant download.
    """
    t = (ticker or "").upper().strip()
    group = _group_for(t, yf_sector)
    if not group:
        return None

    returns = _fetch_group_returns(group)
    if not returns:
        return None

    # Ensure the analysed stock is represented (use passed-in return if we have it).
    if self_return_6m is not None:
        returns = {**returns, t: {**returns.get(t, {}), "return_6m": round(float(self_return_6m), 2)}}

    ranked = sorted(
        [{"ticker": k, **v} for k, v in returns.items() if v.get("return_6m") is not None],
        key=lambda x: x["return_6m"], reverse=True,
    )
    if not ranked:
        return None

    n = len(ranked)
    self_row = next((r for r in ranked if r["ticker"] == t), None)
    self_ret = self_row["return_6m"] if self_row else self_return_6m
    rank = next((i + 1 for i, r in enumerate(ranked) if r["ticker"] == t), None)
    median_ret = float(np.median([r["return_6m"] for r in ranked]))

    # "Not performing great" = below the cohort median (or bottom half by rank).
    underperforming = self_ret is not None and self_ret < median_ret

    # Alternatives = top peers that beat the analysed stock (exclude itself).
    alternatives = [
        {"ticker": r["ticker"].replace(".NS", ""), "ticker_full": r["ticker"],
         "return_6m": r["return_6m"], "return_1m": r.get("return_1m")}
        for r in ranked
        if r["ticker"] != t and (self_ret is None or r["return_6m"] > self_ret)
    ][:top_n]

    return {
        "sector_group": group,
        "self": {"ticker": t.replace(".NS", ""), "return_6m": self_ret, "rank": rank, "n": n},
        "cohort_median_return_6m": round(median_ret, 2),
        "underperforming": bool(underperforming),
        "alternatives": alternatives,
    }
