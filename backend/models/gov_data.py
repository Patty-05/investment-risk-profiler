"""
Official / government trade and macro data for the War Room knowledge base.

WHY THIS EXISTS
---------------
`data/geopolitics/trade_exposure.json` described India's import dependence in
hand-typed prose ("imports ~85-88% of crude needs", "API ~60-70% China"). Those
numbers were plausible but unsourced, undated, and static, which makes them
unusable for anything quantitative: you cannot backtest a factor whose weights
were guessed, and you cannot claim a conflict matters more this year than last
if the exposure number never moves.

This module replaces the prose with sourced, dated, machine-readable series:

  * UN COMTRADE (comtradeapi.un.org) -- the official customs statistics that
    national governments (here India, reporter code 699) report to the UN. Gives
    India's import/export value per HS commodity code per PARTNER COUNTRY per
    year. This is what tells us Russia went from ~2% of India's crude imports
    before 2022 to ~35% after: a real, dated, structural shift.
  * WORLD BANK Indicators API -- India macro context (fuel imports as a share of
    merchandise imports, energy import dependence).

Both are free and keyless. Everything is cached to data/gov/ as raw JSON so a
rebuild never re-hits the network, and every derived number can be traced back
to the exact API response it came from.

POINT-IN-TIME DISCIPLINE
------------------------
Comtrade annual data for year Y is not public until well into Y+1. `as_of()`
therefore applies a reporting lag so a backtest standing on date D only ever
sees vintages genuinely published by D. Skipping this is the single easiest way
to fabricate a backtest result.

SOURCES DELIBERATELY NOT USED
-----------------------------
  * PPAC (ppac.gov.in) publishes the Indian crude basket price, which is the
    India-correct crude benchmark, but only as signed PDFs behind a download
    shim. Parsing those is fragile and buys little: the basket tracks Brent at
    r > 0.99, and Brent is available daily and free. Brent is used instead.
  * FRED is unreachable from this network (connection reset / read timeout).
  * data.gov.in requires a registered API key.
"""

from __future__ import annotations

import os
import json
import time
import datetime as _dt

import requests

_GOV_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "gov")
_COMTRADE_DIR = os.path.join(_GOV_DIR, "comtrade")
_WB_DIR = os.path.join(_GOV_DIR, "worldbank")

_COMTRADE_URL = "https://comtradeapi.un.org/public/v1/preview/C/A/HS"
_WB_URL = "https://api.worldbank.org/v2/country/IND/indicator/{code}"

INDIA_REPORTER = 699

# The public preview endpoint allows one period per call and rate-limits hard
# (HTTP 429 after ~2 rapid calls), so every fetch is spaced and cached forever.
_REQUEST_SPACING_S = 7.0

# Annual customs data for year Y is not published until well into Y+1.
REPORTING_LAG_MONTHS = 12


# -- Commodity map -----------------------------------------------------------
# Keys match data/geopolitics/trade_exposure.json so the KB and the data line up.
# `flow` is the side India is exposed on: M = India imports it (a supply shock
# hurts), X = India exports it (a disruption elsewhere can help).
COMMODITIES = {
    "crude_oil":           {"hs": "2709", "flow": "M", "desc": "Petroleum oils, crude"},
    "natural_gas":         {"hs": "2711", "flow": "M", "desc": "Petroleum gases (incl. LNG)"},
    "thermal_coal":        {"hs": "2701", "flow": "M", "desc": "Coal; briquettes"},
    "wheat":               {"hs": "1001", "flow": "X", "desc": "Wheat and meslin"},
    "palm_oil":            {"hs": "1511", "flow": "M", "desc": "Palm oil"},
    "sunflower_oil":       {"hs": "1512", "flow": "M", "desc": "Sunflower/safflower oil"},
    "fertilizers":         {"hs": "31",   "flow": "M", "desc": "Fertilizers"},
    "potash":              {"hs": "3104", "flow": "M", "desc": "Potassic fertilizers"},
    "gold":                {"hs": "7108", "flow": "M", "desc": "Gold, unwrought"},
    "semiconductors":      {"hs": "8542", "flow": "M", "desc": "Electronic integrated circuits"},
    "rare_earths":         {"hs": "2846", "flow": "M", "desc": "Rare-earth compounds"},
    # HS 93 covers small arms and ammunition ONLY. Military aircraft sit in HS 88
    # and warships in HS 89, so this figure is roughly an order of magnitude
    # below India's real defence procurement and says nothing about the Russian
    # platform dependence that actually matters. Carried for partner MIX, not
    # for level; `SCOPE_CAVEATS` propagates the warning to anything that reads it.
    "defense":             {"hs": "93",   "flow": "M", "desc": "Arms and ammunition"},
    "pharmaceuticals_api": {"hs": "2933", "flow": "M", "desc": "Heterocyclic compounds (bulk API)"},
    "textiles":            {"hs": "62",   "flow": "X", "desc": "Apparel, not knitted"},
}

# UN M49 numeric codes for every country the KB tracks, plus the large energy
# suppliers that dominate India's crude and gas partner tables.
PARTNER_CODES = {
    "Russia": 643, "Ukraine": 804, "Israel": 376, "Iran": 364, "China": 156,
    "Pakistan": 586, "Taiwan": 490, "United States": 842, "Saudi Arabia": 682,
    "UAE": 784, "Indonesia": 360, "Malaysia": 458, "Venezuela": 862,
    "North Korea": 408, "South Korea": 410, "Philippines": 608,
    "Bangladesh": 50, "Myanmar": 104,
    # Not in the KB but materially present in India's import tables:
    "Iraq": 368, "Kuwait": 414, "Nigeria": 566, "Qatar": 634, "Angola": 24,
    "Australia": 36, "Singapore": 702, "Switzerland": 757, "Japan": 392,
    "Germany": 276, "France": 250, "South Africa": 710, "Brazil": 76,
    "Argentina": 32, "Mexico": 484, "Colombia": 170, "Oman": 512,
    # Added for the Geopolitical Analyzer's trade-partner tables (electronics,
    # machinery, chemicals, vehicles and pharma exports have a broader buyer set).
    "Hong Kong": 344, "Vietnam": 704, "Thailand": 764, "Italy": 380,
    "United Kingdom": 826, "Belgium": 56, "Netherlands": 528, "Canada": 124,
    "Egypt": 818, "Turkey": 792, "Sri Lanka": 144, "Nepal": 524, "Bhutan": 64,
    "Spain": 724, "Kenya": 404, "Tanzania": 834, "Ghana": 288, "Ivory Coast": 384,
    "Ethiopia": 231, "Mozambique": 508, "Peru": 604, "Chile": 152, "New Zealand": 554,
    "Ukraine ": 804,
}
PARTNER_CODES.pop("Ukraine ", None)
CODE_TO_COUNTRY = {v: k for k, v in PARTNER_CODES.items()}

_last_request_at = [0.0]

# In-process memo. Building a backtest weight table asks for the same few
# hundred (commodity, year) records thousands of times; without this every one
# of those is a fresh open()+json.loads of the same file.
_mem = {}


def _throttle():
    wait = _REQUEST_SPACING_S - (time.time() - _last_request_at[0])
    if wait > 0:
        time.sleep(wait)
    _last_request_at[0] = time.time()


# -- UN Comtrade -------------------------------------------------------------
def _comtrade_path(hs, year):
    return os.path.join(_COMTRADE_DIR, "hs{}_{}.json".format(hs, year))


def fetch_comtrade(hs, year, flow="M", force=False, cache_only=False):
    """
    One (commodity, year) partner table from UN Comtrade, cached to disk.

    Rows are filtered to the aggregate level (`partner2Code == 0`); Comtrade
    also returns second-partner breakdowns, and summing without this filter
    double-counts by roughly 3x.
    """
    key = (str(hs), int(year), flow)
    if key in _mem and not force:
        return _mem[key]
    # A cache_only miss is memoised under its OWN key: it records "not on disk",
    # not "does not exist". Sharing the key would let a serving-path miss
    # suppress a later real fetch in the same process.
    if cache_only and ("miss",) + key in _mem:
        return None

    path = _comtrade_path(hs, year)
    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as fh:
            rec = json.load(fh)
        _mem[key] = rec
        return rec

    # Serving paths pass cache_only=True: a cache miss there must return None
    # immediately rather than block a user request behind a 7s-throttled,
    # retrying network call (worst case ~40s while as_of walks back vintages).
    if cache_only:
        _mem[("miss",) + key] = True
        return None

    os.makedirs(_COMTRADE_DIR, exist_ok=True)
    params = dict(reporterCode=INDIA_REPORTER, period=int(year),
                  cmdCode=str(hs), flowCode=flow)
    payload = None
    for attempt in range(4):
        _throttle()
        try:
            r = requests.get(_COMTRADE_URL, params=params, timeout=45,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code == 429:
                time.sleep(20 * (attempt + 1))
                continue
            r.raise_for_status()
            payload = r.json()
            break
        except Exception as e:
            if attempt == 3:
                print("[gov] comtrade hs{} {} failed: {}".format(hs, year, e))
                return None
            time.sleep(10 * (attempt + 1))
    if payload is None:
        return None

    rows = [d for d in (payload.get("data") or []) if d.get("partner2Code") == 0]
    world = next((d["primaryValue"] for d in rows
                  if d.get("partnerCode") == 0 and d.get("primaryValue")), None)
    partners = {int(d["partnerCode"]): float(d["primaryValue"])
                for d in rows
                if d.get("partnerCode") and d.get("primaryValue")}

    record = {
        "source": "UN Comtrade (official customs statistics reported by India)",
        "url": _COMTRADE_URL,
        "reporter": INDIA_REPORTER,
        "hs": str(hs),
        "flow": flow,
        "year": int(year),
        "retrieved_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "total_usd": world,
        "partners_usd": partners,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=1)
    _mem[key] = record
    return record


def build_comtrade_cache(years=range(2015, 2025), commodities=None, verbose=True):
    """Populate the on-disk cache for every commodity/year. Slow (rate-limited)."""
    commodities = commodities or COMMODITIES
    done = failed = cached = 0
    for key, spec in commodities.items():
        for yr in years:
            if os.path.exists(_comtrade_path(spec["hs"], yr)):
                cached += 1
                continue
            rec = fetch_comtrade(spec["hs"], yr, flow=spec["flow"])
            if rec and rec.get("total_usd"):
                done += 1
                if verbose:
                    top = sorted(rec["partners_usd"].items(), key=lambda kv: -kv[1])[:1]
                    lead = CODE_TO_COUNTRY.get(top[0][0], top[0][0]) if top else "-"
                    share = (top[0][1] / rec["total_usd"] * 100) if top else 0.0
                    print("  [ok]  {:<22} {}  ${:7.1f}B  top={} {:.0f}%".format(
                        key, yr, rec["total_usd"] / 1e9, lead, share))
            else:
                failed += 1
                if verbose:
                    print("  [--]  {:<22} {}  no data".format(key, yr))
    if verbose:
        print("\n[gov] comtrade cache: {} fetched, {} already cached, {} empty".format(
            done, cached, failed))
    return done, cached, failed


def import_shares(commodity, year, cache_only=False):
    """{country_name: share_of_India_trade} for one commodity/year, or None."""
    spec = COMMODITIES.get(commodity)
    if not spec:
        return None
    rec = fetch_comtrade(spec["hs"], year, flow=spec["flow"], cache_only=cache_only)
    if not rec or not rec.get("total_usd"):
        return None
    total = rec["total_usd"]
    out = {}
    for code, val in rec["partners_usd"].items():
        name = CODE_TO_COUNTRY.get(int(code))
        if name:
            out[name] = val / total
    return out


def trade_total_usd(commodity, year, cache_only=False):
    spec = COMMODITIES.get(commodity)
    if not spec:
        return None
    rec = fetch_comtrade(spec["hs"], year, flow=spec["flow"], cache_only=cache_only)
    return rec.get("total_usd") if rec else None


def latest_published_year(date):
    """
    The most recent Comtrade reporting year that was actually published by
    `date`, given REPORTING_LAG_MONTHS. Standing on 2024-03-01 with a 12-month
    lag, the newest usable vintage is 2022.
    """
    if isinstance(date, str):
        d = _dt.date.fromisoformat(date[:10])
    elif isinstance(date, _dt.datetime):
        d = date.date()
    else:
        d = date
    # Year Y closes on 31 Dec Y; add the lag to get the publication date.
    y = d.year - 1
    cutoff_month = REPORTING_LAG_MONTHS % 12 or 12
    cutoff_year_offset = REPORTING_LAG_MONTHS // 12
    if (d.year, d.month) < (y + cutoff_year_offset + 1, cutoff_month):
        y -= 1
    return y


def as_of(commodity, date, cache_only=True):
    """
    Point-in-time view: the newest vintage published by `date`, walking back
    until a cached year exists. Returns (year, shares, total_usd).

    Defaults to cache_only because every caller is either serving a request or
    iterating a backtest panel; both want a fast miss over a slow fetch. Run
    `python -m models.gov_data` to populate the cache.
    """
    latest = latest_published_year(date)
    for yr in range(latest, latest - 6, -1):
        shares = import_shares(commodity, yr, cache_only=cache_only)
        if shares:
            return yr, shares, trade_total_usd(commodity, yr, cache_only=cache_only)
    return None, None, None


# -- World Bank --------------------------------------------------------------
WB_INDICATORS = {
    "fuel_import_share":  "TM.VAL.FUEL.ZS.UN",   # Fuel imports, % of merchandise imports
    "energy_import_dep":  "EG.IMP.CONS.ZS",      # Net energy imports, % of energy use
    "imports_pct_gdp":    "NE.IMP.GNFS.ZS",
    "merch_imports_usd":  "TM.VAL.MRCH.CD.WT",
    "merch_exports_usd":  "TX.VAL.MRCH.CD.WT",
}


def fetch_worldbank(name, force=False):
    """{year: value} for one World Bank India indicator, cached to disk."""
    code = WB_INDICATORS.get(name, name)
    memo_key = ("wb", code)
    if memo_key in _mem and not force:
        return _mem[memo_key]

    path = os.path.join(_WB_DIR, "{}.json".format(code))
    if os.path.exists(path) and not force:
        with open(path, "r", encoding="utf-8") as fh:
            series = {int(k): v for k, v in json.load(fh)["series"].items()}
        _mem[memo_key] = series
        return series

    os.makedirs(_WB_DIR, exist_ok=True)
    try:
        r = requests.get(_WB_URL.format(code=code),
                         params={"format": "json", "per_page": 200}, timeout=30,
                         headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        payload = r.json()
        series = {int(d["date"]): d["value"]
                  for d in (payload[1] or []) if d.get("value") is not None}
    except Exception as e:
        print("[gov] worldbank {} failed: {}".format(code, e))
        return {}

    with open(path, "w", encoding="utf-8") as fh:
        json.dump({
            "source": "World Bank Indicators API (national statistics / IMF / UN)",
            "indicator": code,
            "country": "IND",
            "retrieved_utc": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "series": series,
        }, fh, indent=1)
    _mem[memo_key] = series
    return series


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 70)
    print("GOVERNMENT DATA INGESTION -- UN Comtrade + World Bank")
    print("=" * 70)
    print("\n[worldbank]")
    for nm in WB_INDICATORS:
        s = fetch_worldbank(nm)
        recent = sorted(s.items())[-1] if s else ("-", "-")
        print("  {:<20} {:>3} yrs   latest {}: {}".format(nm, len(s), recent[0], recent[1]))
    print("\n[comtrade]  (rate-limited, ~7s per commodity-year)")
    build_comtrade_cache()
