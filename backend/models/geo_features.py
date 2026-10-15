"""
Time-varying geopolitical exposure features.

THE PROBLEM WITH THE EXISTING GEO FACTOR
----------------------------------------
`modules/geopolitical_analysis.py` produces a per-ticker score from the War Room
knowledge base. It is a good explanatory factor for the UI, but it is a pure
function of the ticker: ONGC scores 90 today, scored 90 last year, and will
score 90 next year until somebody edits the JSON. Feeding that into a panel
model as a feature cannot produce alpha, because a per-ticker constant is a
ticker fixed effect. The model would simply learn "these 40 names outperform",
which is in-sample memorisation of whichever names happened to do well, and it
carries no information about WHEN to hold them.

For the factor to be testable it must move. This module makes it move, along two
axes, both anchored in real data rather than opinion:

  PRESSURE (daily)     -- how hard is the geopolitical channel actually pushing
                          right now? Measured from the traded price of the thing
                          the conflict disrupts: Brent for crude channels, NG for
                          gas/fertilizer, gold for safe-haven, USDINR for
                          risk-off, SOXX for semiconductors, REMX for rare
                          earths, and so on. A conflict nobody is pricing has
                          near-zero pressure.

  WEIGHT (annual)      -- how much does that channel actually matter to India?
                          Taken from UN Comtrade customs data via
                          `models/gov_data.py`: the commodity's share of India's
                          merchandise imports, times the share of that commodity
                          sourced from countries party to a live conflict. This
                          is why Russia-Ukraine weighs far more on India's crude
                          channel after 2022 (Russia ~35% of crude imports) than
                          before it (~2%).

  DIRECTION (static)   -- benefits or hurts, from the KB's theme -> sector ->
                          ticker maps.

    geo_signal(ticker, t) = SUM over themes
                              direction(ticker, theme)
                            * weight(theme, t)
                            * pressure(theme, t)

POINT-IN-TIME DISCIPLINE
------------------------
  * Pressure z-scores are causal: trailing rolling windows only, never centred,
    never full-sample.
  * Comtrade weights use `gov_data.as_of()`, which applies the publication lag,
    so a row dated 2023-04 sees the 2021 vintage, not 2023's.
  * A conflict contributes nothing before its `since` year.

KNOWN RESIDUAL LOOK-AHEAD (read before believing any result)
------------------------------------------------------------
The knowledge base itself was authored in 2026 with full hindsight. Its choice of
which conflicts to track, and which tickers to attach to which theme, embeds
knowledge that did not exist in 2017. There is no way to remove this without
re-authoring the KB from period sources, so this test is biased IN FAVOUR of the
geo factor. That asymmetry is useful: a positive result here is soft evidence and
needs more work, whereas a null result is strong evidence, because the factor
failed even with hindsight helping it.
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import pandas as pd

from models import gov_data

warnings.filterwarnings("ignore")

_KB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "geopolitics")

# Feature names this module contributes to the model.
GEO_FEATURE_NAMES = ["geo_signal", "geo_gross", "geo_crude_beta"]


# -- Theme -> tradeable driver ----------------------------------------------
# (yfinance symbol, sign, commodity key in gov_data.COMMODITIES)
#
# `sign` maps "driver goes up" onto "theme is active". Two cases are inverted
# on purpose: a semiconductor disruption shows up as semis SELLING OFF, and the
# China-plus-one tailwind shows up as Chinese equities UNDERPERFORMING.
THEME_DRIVERS = {
    "crude_oil_up":             ("BZ=F",     +1, "crude_oil"),
    "energy_importer_relief":   ("BZ=F",     -1, "crude_oil"),
    "natural_gas_up":           ("NG=F",     +1, "natural_gas"),
    "fertilizer_input_up":      ("NG=F",     +1, "fertilizers"),
    "edible_oil_up":            ("ZL=F",     +1, "palm_oil"),
    "wheat_export_window":      ("ZW=F",     +1, "wheat"),
    "coal_up":                  ("XME",      +1, "thermal_coal"),
    "gold_safe_haven":          ("GC=F",     +1, "gold"),
    "risk_off_rupee_weak":      ("USDINR=X", +1, None),
    "semiconductor_disruption": ("SOXX",     -1, "semiconductors"),
    "rare_earths_squeeze":      ("REMX",     +1, "rare_earths"),
    "defense_indigenization":   ("ITA",      +1, "defense"),
    "defense_exports":          ("ITA",      +1, "defense"),
    "china_plus_one":           ("FXI",      -1, "pharmaceuticals_api"),
    "shipping_freight_up":      ("BDRY",     +1, None),
    "textile_export_gain":      ("CT=F",     +1, "textiles"),
    # No liquid daily driver maps cleanly to generic generic-drug demand, so this
    # theme contributes direction but no pressure and drops out of the signal.
    "pharma_generic_demand":    (None,        0, "pharmaceuticals_api"),
}

DRIVER_SYMBOLS = sorted({v[0] for v in THEME_DRIVERS.values() if v[0]})

# Lookback for the momentum the pressure measures, and the trailing window the
# z-score is standardised over (~2 years of sessions).
_PRESSURE_LOOKBACK = 20
_ZSCORE_WINDOW = 504
_MIN_ZSCORE_OBS = 120

_cache = {}


def _as_naive_index(index):
    """
    Coerce anything date-like to a tz-naive, midnight-normalised DatetimeIndex.

    Callers hand this a Series (from a meta DataFrame column), a DatetimeIndex,
    or a raw datetime64 array, and only the DatetimeIndex form has .tz_localize.
    Normalising here keeps the alignment logic below from caring which it got.
    """
    idx = pd.DatetimeIndex(pd.to_datetime(np.asarray(index)))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx.normalize()


# -- Knowledge base ----------------------------------------------------------
def _load_kb(name):
    key = "kb_" + name
    if key not in _cache:
        with open(os.path.join(_KB_DIR, name + ".json"), "r", encoding="utf-8") as fh:
            _cache[key] = json.load(fh)
    return _cache[key]


def _conflict_since_year(conflict):
    """First year the conflict can contribute. Unparseable/ongoing -> always on."""
    raw = conflict.get("since")
    try:
        return int(str(raw)[:4])
    except (TypeError, ValueError):
        return 1900


def theme_exposures():
    """
    {theme_id: {ticker: +1 benefits / -1 hurts}} from the KB sector maps.

    A ticker appearing on both sides of one theme nets to zero, which is the
    honest answer for a conglomerate with offsetting businesses.
    """
    if "theme_exposures" in _cache:
        return _cache["theme_exposures"]

    themes = _load_kb("sector_map")["themes"]
    out = {}
    for theme_id, body in themes.items():
        acc = {}
        for side, sign in (("benefits", 1), ("hurts", -1)):
            for entry in body.get(side, []):
                for tk in entry.get("tickers", []):
                    acc[tk.upper()] = acc.get(tk.upper(), 0) + sign
        out[theme_id] = {k: np.sign(v) for k, v in acc.items() if v}
    _cache["theme_exposures"] = out
    return out


def _theme_conflicts():
    """
    {theme_id: [(conflict_id, since_year, parties, status), ...]}

    Reuses war_service's commodity -> theme mapping so the KB stays the single
    source of truth for which conflict lights up which channel.
    """
    if "theme_conflicts" in _cache:
        return _cache["theme_conflicts"]

    from services import war_service

    out = {}
    for c in _load_kb("conflicts")["conflicts"]:
        for theme_id in war_service._conflict_theme_ids(c):
            out.setdefault(theme_id, []).append({
                "id": c["id"],
                "since": _conflict_since_year(c),
                "parties": c.get("parties", []),
                "status": c.get("status"),
            })
    _cache["theme_conflicts"] = out
    return out


# -- Driver prices -> pressure ----------------------------------------------
def _driver_prices(start="2013-01-01"):
    if "driver_prices" in _cache:
        return _cache["driver_prices"]
    import yfinance as yf
    df = yf.download(DRIVER_SYMBOLS, start=start, progress=False, auto_adjust=True)
    px = df["Close"] if isinstance(df.columns, pd.MultiIndex) else df.to_frame()
    px.index = pd.to_datetime(px.index).tz_localize(None).normalize()
    _cache["driver_prices"] = px
    return px


def pressure_frame(start="2013-01-01"):
    """
    {theme_id: causal z-scored pressure series}, indexed by date.

    Pressure = trailing 20-session return of the driver, standardised against a
    trailing 2-year window of that same quantity, then signed so positive always
    means "this geopolitical channel is firing". Clipped to +/-3 so one crisis
    print cannot dominate the panel.
    """
    if "pressure" in _cache:
        return _cache["pressure"]

    px = _driver_prices(start)
    out = {}
    for theme_id, (sym, sign, _) in THEME_DRIVERS.items():
        if not sym or sym not in px.columns:
            continue
        s = px[sym].astype(float).ffill()
        ret = s.pct_change(_PRESSURE_LOOKBACK)
        mu = ret.rolling(_ZSCORE_WINDOW, min_periods=_MIN_ZSCORE_OBS).mean()
        sd = ret.rolling(_ZSCORE_WINDOW, min_periods=_MIN_ZSCORE_OBS).std()
        z = ((ret - mu) / sd.replace(0, np.nan)) * sign
        out[theme_id] = z.clip(-3, 3)

    frame = pd.DataFrame(out)
    _cache["pressure"] = frame
    return frame


# -- Comtrade -> weight ------------------------------------------------------
def _merch_imports(year):
    series = gov_data.fetch_worldbank("merch_imports_usd")
    for y in range(year, year - 5, -1):
        if series.get(y):
            return series[y]
    return None


def theme_weight(theme_id, date):
    """
    How much this channel matters to India, as of `date`, from customs data.

        weight = import_intensity * (0.5 + 0.5 * conflict_party_share)

    `import_intensity` is the commodity's share of India's merchandise imports
    (Comtrade commodity total / World Bank merchandise imports), rescaled so a
    typical channel sits near 1 rather than near 0.003.

    `conflict_party_share` is the fraction of that commodity India sources from
    countries party to a conflict that was already live at `date`. The 0.5 floor
    keeps a channel alive even when the supplier mix is not concentrated in a
    belligerent, since a global price shock still transmits.

    Themes with no commodity (risk-off, freight) get a flat weight of 1.
    """
    spec = THEME_DRIVERS.get(theme_id)
    if not spec:
        return 0.0
    commodity = spec[2]
    if commodity is None:
        return 1.0

    year = pd.Timestamp(date).year
    vintage, shares, total = gov_data.as_of(commodity, pd.Timestamp(date).date())
    if not shares or not total:
        return 1.0

    imports = _merch_imports(vintage)
    intensity = (total / imports) if imports else 0.0
    # Crude alone is ~20% of the import bill; most channels are well under 1%.
    # sqrt compresses that three-order-of-magnitude spread into a usable range.
    intensity = float(np.sqrt(max(intensity, 0.0)) * 4.0)

    live_parties = set()
    for conf in _theme_conflicts().get(theme_id, []):
        if conf["since"] <= year:
            live_parties.update(conf["parties"])
    party_share = float(sum(v for k, v in shares.items() if k in live_parties))

    return intensity * (0.5 + 0.5 * min(party_share, 1.0))


def weight_table(dates, themes=None):
    """Pre-compute theme weights per calendar year (they only move annually)."""
    themes = themes or list(THEME_DRIVERS)
    years = sorted({pd.Timestamp(d).year for d in dates})
    table = {}
    for th in themes:
        for y in years:
            table[(th, y)] = theme_weight(th, pd.Timestamp("{}-06-30".format(y)))
    return table


# -- The feature itself ------------------------------------------------------
def ticker_theme_map(ticker):
    """{theme_id: +1/-1} for one ticker, empty if it has no tracked exposure."""
    tk = (ticker or "").upper().strip()
    return {th: d[tk] for th, d in theme_exposures().items() if tk in d}


def geo_series(ticker, index, weights=None):
    """
    Per-date geo features for one ticker, aligned to `index`.

    Returns a DataFrame with GEO_FEATURE_NAMES columns:
      geo_signal     net directional geo push (positive = tailwind)
      geo_gross      total geo turbulence the name is exposed to, regardless of
                     sign; separates "buffeted from both sides" from "untouched"
      geo_crude_beta the crude channel alone, signed -- crude is India's dominant
                     exposure and deserves its own column rather than being
                     averaged away
    """
    idx = _as_naive_index(index)
    exposures = ticker_theme_map(ticker)
    empty = pd.DataFrame(0.0, index=idx, columns=GEO_FEATURE_NAMES)
    if not exposures:
        return empty

    press = pressure_frame()
    press = press.reindex(idx, method="ffill")

    conflicts = _theme_conflicts()
    years = idx.year

    signal = np.zeros(len(idx))
    gross = np.zeros(len(idx))
    crude = np.zeros(len(idx))
    # Total weight actually applied, used to turn the sum into a weighted MEAN.
    # Without this a name mapped to two correlated channels (IOC is on both
    # crude_oil_up and energy_importer_relief, which are the same Brent move
    # with opposite signs) gets roughly double the amplitude of a name on one
    # channel. In a cross-sectional ranking that difference is not information
    # about the stock, it is an artefact of how many rows the KB happens to
    # list it on, and it would drive the extremes of the ranking.
    wsum = np.zeros(len(idx))

    for theme_id, direction in exposures.items():
        if theme_id not in press.columns:
            continue
        p = press[theme_id].values.astype(float)
        p = np.nan_to_num(p, nan=0.0)

        # A conflict cannot contribute before it started.
        live = conflicts.get(theme_id, [])
        if live:
            first = min(c["since"] for c in live)
            p = np.where(years >= first, p, 0.0)

        if weights is None:
            w = np.array([theme_weight(theme_id, d) for d in idx])
        else:
            w = np.array([weights.get((theme_id, y), 0.0) for y in years])

        contrib = direction * w * p
        signal += contrib
        gross += np.abs(w * p)
        wsum += np.abs(w)
        if theme_id in ("crude_oil_up", "energy_importer_relief"):
            crude += contrib

    denom = np.where(wsum > 0, wsum, 1.0)
    return pd.DataFrame({
        "geo_signal": signal / denom,
        "geo_gross": gross,
        "geo_crude_beta": crude / denom,
    }, index=idx)


def coverage_report(tickers):
    """How many of `tickers` the KB actually has any exposure for."""
    covered = {t: ticker_theme_map(t) for t in tickers}
    hit = {t: v for t, v in covered.items() if v}
    return {
        "n_tickers": len(tickers),
        "n_with_exposure": len(hit),
        "pct": round(100.0 * len(hit) / max(len(tickers), 1), 1),
        "mean_themes_per_covered": round(
            float(np.mean([len(v) for v in hit.values()])) if hit else 0.0, 2),
    }


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    from models.stock_model import TRAINING_TICKERS

    print("=" * 70)
    print("GEO FEATURES -- self check")
    print("=" * 70)

    cov = coverage_report(TRAINING_TICKERS)
    print("\n[coverage] {n_with_exposure}/{n_tickers} training tickers "
          "({pct}%) have KB exposure, {mean_themes_per_covered} themes each"
          .format(**cov))

    press = pressure_frame()
    print("\n[pressure] {} themes, {} -> {}".format(
        press.shape[1], press.index.min().date(), press.index.max().date()))
    print(press.tail(1).T.round(2).to_string())

    print("\n[weights] crude_oil_up by year (Comtrade, point-in-time):")
    for y in range(2018, 2027):
        w = theme_weight("crude_oil_up", pd.Timestamp("{}-06-30".format(y)))
        print("   {}  {:.3f}".format(y, w))

    for tk in ["ONGC.NS", "IOC.NS", "HAL.NS", "ASIANPAINT.NS", "TCS.NS"]:
        m = ticker_theme_map(tk)
        if not m:
            print("\n[{}] no KB exposure".format(tk))
            continue
        idx = pd.date_range("2021-01-01", "2026-06-30", freq="B")
        g = geo_series(tk, idx)
        print("\n[{}] themes={} | geo_signal mean {:+.3f} sd {:.3f} "
              "min {:+.2f} max {:+.2f}".format(
                  tk, list(m.items())[:4], g["geo_signal"].mean(),
                  g["geo_signal"].std(), g["geo_signal"].min(), g["geo_signal"].max()))
