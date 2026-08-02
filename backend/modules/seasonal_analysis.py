"""
Indian Equity Seasonal Analysis Module (market-relative + significance-tested).

Novel/refined approach vs. naive monthly-average seasonality:
  1. MARKET-RELATIVE (excess) returns — each sector's monthly return minus the
     Nifty benchmark. This strips out market beta so we measure true sector
     ROTATION (which sector beats the market in a month), not just "which goes
     up" (which is mostly the whole market rising, e.g. every sector peaking in
     April on new-FY optimism).
  2. STATISTICAL SIGNIFICANCE — a one-sample t-test on each sector-month's
     excess returns. Patterns are only surfaced as reliable when they clear a
     significance bar, and the ranking uses a t-statistic (signal-to-noise
     "seasonal edge") instead of an arbitrary composite.
  3. INDIAN SEASONAL-EVENT CALENDAR — each recommendation is tagged with the
     real India-specific driver (Budget, FY-end, monsoon, Diwali/Samvat festive
     demand, earnings seasons).

Educational analysis, not financial advice.
"""

import numpy as np
import pandas as pd
from datetime import datetime
import traceback

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("yfinance not installed - pip install yfinance")

try:
    from scipy import stats as _scipy_stats
    _SCIPY = True
except ImportError:
    _SCIPY = False


BENCHMARK_TICKER = "^NSEI"   # Nifty 50 — the market benchmark for excess returns

# ── Sector definitions ────────────────────────────────────────────────────────
SECTOR_CONFIG = {
    "IT": {"name": "Information Technology", "tickers": ["^CNXIT", "INFY.NS", "TCS.NS"],
           "color": "#3b82f6", "icon": "💻",
           "components": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"]},
    "Banking": {"name": "Banking & Finance", "tickers": ["^NSEBANK", "HDFCBANK.NS", "ICICIBANK.NS"],
                "color": "#10b981", "icon": "🏦",
                "components": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "BAJFINANCE.NS"]},
    "FMCG": {"name": "Fast Moving Consumer Goods", "tickers": ["^CNXFMCG", "HINDUNILVR.NS", "NESTLEIND.NS"],
             "color": "#f59e0b", "icon": "🛒",
             "components": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS"]},
    "Auto": {"name": "Automobile", "tickers": ["^CNXAUTO", "MARUTI.NS", "TATAMOTORS.NS"],
             "color": "#ef4444", "icon": "🚗",
             "components": ["MARUTI.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS", "TVSMOTOR.NS"]},
    "Pharma": {"name": "Pharmaceuticals", "tickers": ["^CNXPHARMA", "SUNPHARMA.NS", "DRREDDY.NS"],
               "color": "#8b5cf6", "icon": "💊",
               "components": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "TORNTPHARM.NS"]},
    "Realty": {"name": "Real Estate", "tickers": ["^CNXREALTY", "DLF.NS", "GODREJPROP.NS"],
               "color": "#ec4899", "icon": "🏗️",
               "components": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS", "BRIGADE.NS"]},
    "Hospitality": {"name": "Hospitality & Tourism", "tickers": ["INDHOTEL.NS", "LEMONTREE.NS", "CHALET.NS"],
                    "color": "#f97316", "icon": "🏨",
                    "components": ["INDHOTEL.NS", "LEMONTREE.NS", "CHALET.NS", "EIHOTEL.NS", "MHRIL.NS", "THOMASCOOK.NS"]},
    "Energy": {"name": "Energy & Power", "tickers": ["^CNXENERGY", "RELIANCE.NS", "ONGC.NS"],
               "color": "#eab308", "icon": "⚡",
               "components": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "BPCL.NS"]},
    "Metal": {"name": "Metals & Mining", "tickers": ["^CNXMETAL", "TATASTEEL.NS", "HINDALCO.NS"],
              "color": "#6b7280", "icon": "⛏️",
              "components": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "NMDC.NS", "SAIL.NS"]},
    "Infra": {"name": "Infrastructure", "tickers": ["^CNXINFRA", "LT.NS", "ADANIPORTS.NS"],
              "color": "#84cc16", "icon": "🏗️",
              "components": ["LT.NS", "ADANIPORTS.NS", "BHARTIARTL.NS", "ULTRACEMCO.NS", "GRASIM.NS", "SIEMENS.NS"]},
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── Indian seasonal-event calendar ────────────────────────────────────────────
MONTH_DRIVERS = {
    1:  "January effect; FII inflows resume, Q3 (Dec) earnings begin",
    2:  "Union Budget — policy-sensitive sectors swing on allocations",
    3:  "Financial year-end — window dressing, tax flows, advance-tax squeeze",
    4:  "New FY optimism + Q4 earnings — historically the market's strongest month",
    5:  "'Sell in May'; pre-monsoon and election-cycle caution",
    6:  "Monsoon onset — rural-demand sectors in focus",
    7:  "Q1 earnings season; monsoon progress drives agri/consumer demand",
    8:  "Monsoon peak; festive stocking begins",
    9:  "Quarter-end; pre-festive positioning",
    10: "Festive season + Diwali/Muhurat trading — consumption demand peaks",
    11: "Samvat new-year momentum; festive tailwind continues",
    12: "Year-end 'Santa' drift on thin holiday volumes",
}

SECTOR_MONTH_DRIVERS = {
    ("Auto", 10): "Festive vehicle demand peaks (Navratri → Diwali)",
    ("Auto", 11): "Post-Diwali dispatches and rural festive demand",
    ("Auto", 6):  "Monsoon sets the tone for rural two-wheeler/tractor demand",
    ("FMCG", 10): "Festive consumption and gifting demand",
    ("FMCG", 6):  "Monsoon revives rural FMCG demand",
    ("FMCG", 7):  "Monsoon-led rural demand recovery",
    ("Realty", 10): "Festive property launches and auspicious-buying bookings",
    ("IT", 1):    "Dec-quarter results + FY guidance; USD/INR seasonality",
    ("IT", 7):    "Q1 results and deal-pipeline commentary",
    ("Banking", 3): "FY-end credit push and advances growth",
    ("Banking", 4): "New-FY credit demand and clean-up rally",
    ("Metal", 4): "New-FY infra/capex demand; global restocking",
    ("Energy", 4): "Summer power-demand ramp-up",
    ("Energy", 5): "Peak summer electricity demand",
    ("Pharma", 7): "Monsoon-season acute-therapy demand",
    ("Hospitality", 10): "Festive and wedding-season travel demand",
    ("Hospitality", 12): "Year-end holiday travel peak",
    ("Hospitality", 4): "Summer-vacation travel season",
    ("Infra", 3): "FY-end government capex push",
}


def _sig_stars(p):
    if p is None:
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


class IndianSeasonalAnalyzer:
    def __init__(self, years: int = 10):
        self.years = years
        self.end_date = datetime.now()
        self.start_date = datetime(self.end_date.year - years, 1, 1)
        self._cache = {}
        self._bench_monthly = None  # cached benchmark monthly returns

    # ── Data fetching ──────────────────────────────────────────────────────
    def _fetch_ticker(self, ticker: str):
        if ticker in self._cache:
            return self._cache[ticker]
        if not _YF_AVAILABLE:
            return None
        try:
            df = yf.download(ticker, start=self.start_date.strftime("%Y-%m-%d"),
                             end=self.end_date.strftime("%Y-%m-%d"),
                             progress=False, auto_adjust=True)
            if df is None or len(df) < 50:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self._cache[ticker] = df
            return df
        except Exception as e:
            print(f"  Failed to fetch {ticker}: {e}")
            return None

    def _fetch_sector_data(self, sector_key: str):
        config = SECTOR_CONFIG[sector_key]
        for ticker in config["tickers"]:
            df = self._fetch_ticker(ticker)
            if df is not None and len(df) >= 50:
                print(f"  [ok] {sector_key}: {ticker} ({len(df)} rows)")
                return df
        # Fallback: equal-weight average of component closes
        frames = [self._fetch_ticker(c)["Close"] for c in config["components"][:4]
                  if self._fetch_ticker(c) is not None]
        if frames:
            combined = pd.concat(frames, axis=1).dropna(how="all")
            return pd.DataFrame({"Close": combined.mean(axis=1)}, index=combined.index)
        print(f"  [err] {sector_key}: no data")
        return None

    # ── Monthly return helpers ─────────────────────────────────────────────
    @staticmethod
    def _monthly_returns(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        monthly = df["Close"].resample("ME").last()
        r = monthly.pct_change().dropna() * 100
        return pd.DataFrame({"year": r.index.year, "month": r.index.month, "ret": r.values})

    def _benchmark_monthly(self) -> pd.DataFrame:
        if self._bench_monthly is not None:
            return self._bench_monthly
        df = self._fetch_ticker(BENCHMARK_TICKER)
        if df is None:
            # Fallback benchmark: a broad basket if the index fails
            df = self._fetch_sector_data("Banking")
        bm = self._monthly_returns(df).rename(columns={"ret": "bench"})
        self._bench_monthly = bm
        return bm

    # ── Aggregation with excess + significance ─────────────────────────────
    def _aggregate(self, sector_monthly: pd.DataFrame, bench_monthly: pd.DataFrame) -> dict:
        merged = sector_monthly.merge(bench_monthly, on=["year", "month"], how="inner")
        merged["excess"] = merged["ret"] - merged["bench"]

        avg, std, pos, samp, yr = {}, {}, {}, {}, {}
        ex_avg, ex_std, ex_win, t_by, p_by, sig_by = {}, {}, {}, {}, {}, {}

        for m in range(1, 13):
            raw = merged[merged["month"] == m]["ret"]
            exc = merged[merged["month"] == m]["excess"]
            n = len(raw)
            samp[m] = int(n)
            if n == 0:
                avg[m] = std[m] = ex_avg[m] = ex_std[m] = 0.0
                pos[m] = ex_win[m] = 50.0
                t_by[m] = 0.0; p_by[m] = 1.0; sig_by[m] = False; yr[m] = []
                continue
            avg[m] = round(float(raw.mean()), 2)
            std[m] = round(float(raw.std()), 2)
            pos[m] = round(float((raw > 0).mean() * 100), 1)
            yr[m] = [round(float(v), 2) for v in raw.values]
            ex_avg[m] = round(float(exc.mean()), 2)
            ex_std[m] = round(float(exc.std()), 2)
            ex_win[m] = round(float((exc > 0).mean() * 100), 1)   # % of years beating the market
            # one-sample t-test: is mean excess != 0?
            if _SCIPY and n >= 3 and exc.std() > 1e-9:
                try:
                    t, p = _scipy_stats.ttest_1samp(exc, 0.0)
                    t_by[m] = round(float(t), 2)
                    p_by[m] = round(float(p), 4)
                except Exception:
                    t_by[m], p_by[m] = 0.0, 1.0
            else:
                t_by[m], p_by[m] = 0.0, 1.0
            sig_by[m] = bool(p_by[m] < 0.10)

        return {
            "avg_by_month": avg, "std_by_month": std, "positive_rate": pos,
            "sample_count": samp, "year_by_month": yr,
            "excess_by_month": ex_avg, "excess_std_by_month": ex_std,
            "excess_win_rate": ex_win, "t_by_month": t_by, "p_by_month": p_by,
            "significant_by_month": sig_by,
        }

    @staticmethod
    def _aggregate_quarterly(sector_monthly, bench_monthly) -> dict:
        merged = sector_monthly.merge(bench_monthly, on=["year", "month"], how="inner")
        merged["excess"] = merged["ret"] - merged["bench"]
        qmap = {1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4,11:4,12:4}
        merged["q"] = merged["month"].map(qmap)
        out = {}
        for q in range(1, 5):
            sub = merged[merged["q"] == q]
            out[q] = {
                "avg": round(float(sub["ret"].mean()), 2) if len(sub) else 0.0,
                "std": round(float(sub["ret"].std()), 2) if len(sub) else 0.0,
                "positive_rate": round(float((sub["ret"] > 0).mean() * 100), 1) if len(sub) else 50.0,
                "excess": round(float(sub["excess"].mean()), 2) if len(sub) else 0.0,
            }
        return out

    def _find_best_periods(self, agg: dict) -> dict:
        # Rank months by the excess t-statistic (signal-to-noise of outperformance)
        edge = {m: agg["t_by_month"].get(m, 0.0) for m in range(1, 13)}
        ordered = sorted(range(1, 13), key=lambda m: edge[m], reverse=True)
        best, worst = ordered[:3], ordered[-3:]

        best_ws, best_start = -1e9, 1
        for start in range(1, 13):
            win = [(start - 1 + i) % 12 + 1 for i in range(3)]
            ws = sum(edge[m] for m in win)
            if ws > best_ws:
                best_ws, best_start = ws, start
        best_window = [(best_start - 1 + i) % 12 + 1 for i in range(3)]

        return {
            "best_months": [MONTHS[m - 1] for m in best],
            "worst_months": [MONTHS[m - 1] for m in worst],
            "best_window": [MONTHS[m - 1] for m in best_window],
            "edge_by_month": {MONTHS[m - 1]: round(edge[m], 2) for m in range(1, 13)},
        }

    # ── Current-month ranking (excess + significance) ──────────────────────
    def _current_month_rank(self, sector_stats: dict) -> list:
        cm = datetime.now().month
        rows = []
        for key, s in sector_stats.items():
            if "error" in s:
                continue
            mo = s["monthly"]
            ex = mo["excess_by_month"].get(cm, 0.0)
            t = mo["t_by_month"].get(cm, 0.0)
            p = mo["p_by_month"].get(cm, 1.0)
            rows.append({
                "sector": key, "name": SECTOR_CONFIG[key]["name"], "icon": SECTOR_CONFIG[key]["icon"],
                "avg_return": mo["avg_by_month"].get(cm, 0.0),
                "excess_return": ex,
                "excess_win_rate": mo["excess_win_rate"].get(cm, 50.0),
                "positive_rate": mo["positive_rate"].get(cm, 50.0),
                "std": mo["std_by_month"].get(cm, 0.0),
                "t_stat": t, "p_value": p, "significant": bool(p < 0.10),
                "sig_stars": _sig_stars(p),
                "edge_score": round(float(t), 3),   # signal-to-noise seasonal edge
            })
        rows.sort(key=lambda x: x["edge_score"], reverse=True)
        return rows

    def _driver_for(self, sector_key, month):
        return SECTOR_MONTH_DRIVERS.get((sector_key, month)) or MONTH_DRIVERS.get(month, "")

    def _confidence(self, p, n, win):
        if p < 0.05 and n >= 8:
            return "High"
        if p < 0.10 and n >= 6:
            return "Medium"
        return "Low"

    def _generate_top_picks(self, sector_stats, rankings) -> list:
        cm = datetime.now().month
        picks = []
        # Only sectors that beat the market this month (positive excess), ranked by edge
        candidates = [r for r in rankings if r["excess_return"] > 0]
        for r in candidates:
            key = r["sector"]
            n = sector_stats[key]["monthly"]["sample_count"].get(cm, 0)
            conf = self._confidence(r["p_value"], n, r["excess_win_rate"])
            driver = self._driver_for(key, cm)

            reason = []
            if r["significant"]:
                reason.append(f"statistically significant outperformance {r['sig_stars']} (p={r['p_value']})")
            reason.append(f"beats Nifty by {r['excess_return']:+.2f}%/mo in {MONTHS[cm-1]} on average")
            reason.append(f"outperformed the market in {r['excess_win_rate']:.0f}% of the last {n} years")
            reason_str = "; ".join(reason)

            picks.append({
                "rank": len(picks) + 1, "sector": key,
                "name": SECTOR_CONFIG[key]["name"], "icon": SECTOR_CONFIG[key]["icon"],
                "color": SECTOR_CONFIG[key]["color"],
                "avg_return": r["avg_return"],
                "excess_return": r["excess_return"],
                "excess_win_rate": r["excess_win_rate"],
                "positive_rate": r["positive_rate"],
                "t_stat": r["t_stat"], "p_value": r["p_value"],
                "significant": r["significant"], "sig_stars": r["sig_stars"],
                "confidence": conf,
                "best_months": sector_stats[key]["best_periods"]["best_months"],
                "driver": driver,
                "reason": reason_str,
                "components": SECTOR_CONFIG[key]["components"][:4],
            })
            if len(picks) == 3:
                break
        return picks

    # ── Heatmaps ───────────────────────────────────────────────────────────
    def _build_heatmap(self, sector_stats, field="avg_by_month") -> dict:
        sectors = [k for k in sector_stats if "error" not in sector_stats[k]]
        matrix = [[round(sector_stats[k]["monthly"][field].get(m, 0), 2) for m in range(1, 13)]
                  for k in sectors]
        return {
            "sectors": sectors,
            "sector_names": [SECTOR_CONFIG[s]["name"] for s in sectors],
            "sector_icons": [SECTOR_CONFIG[s]["icon"] for s in sectors],
            "months": MONTHS, "matrix": matrix,
        }

    # ── Main entry point ───────────────────────────────────────────────────
    def analyze(self) -> dict:
        print("Starting Indian seasonal analysis (market-relative + significance)...")
        bench = self._benchmark_monthly()
        sector_stats = {}

        for key in SECTOR_CONFIG:
            try:
                df = self._fetch_sector_data(key)
                if df is None:
                    sector_stats[key] = {"error": "No data available"}
                    continue
                sm = self._monthly_returns(df)
                agg = self._aggregate(sm, bench)
                quarterly = self._aggregate_quarterly(sm, bench)
                best = self._find_best_periods(agg)

                merged = sm.merge(bench, on=["year", "month"], how="inner")
                merged["excess"] = merged["ret"] - merged["bench"]
                yearly = merged.groupby("year")["ret"].sum().reset_index()

                sector_stats[key] = {
                    "name": SECTOR_CONFIG[key]["name"], "icon": SECTOR_CONFIG[key]["icon"],
                    "color": SECTOR_CONFIG[key]["color"],
                    "monthly": agg, "quarterly": quarterly, "best_periods": best,
                    "yearly": [{"year": int(r["year"]), "return": round(float(r["ret"]), 2)}
                               for _, r in yearly.iterrows()],
                    "overall_avg": round(float(sm["ret"].mean()), 2),
                    "overall_std": round(float(sm["ret"].std()), 2),
                    "overall_excess": round(float(merged["excess"].mean()), 2),
                }
            except Exception as e:
                print(f"  {key} failed: {traceback.format_exc()}")
                sector_stats[key] = {"error": str(e)}

        rankings = self._current_month_rank(sector_stats)
        top_picks = self._generate_top_picks(sector_stats, rankings)
        cm = datetime.now().month

        comparison = []
        for key, s in sector_stats.items():
            if "error" in s:
                continue
            comparison.append({
                "sector": key, "name": s["name"], "icon": s["icon"], "color": s["color"],
                "overall_avg": s["overall_avg"], "overall_std": s["overall_std"],
                "overall_excess": s["overall_excess"],
                "best_month": s["best_periods"]["best_months"][0] if s["best_periods"]["best_months"] else "N/A",
                "worst_month": s["best_periods"]["worst_months"][0] if s["best_periods"]["worst_months"] else "N/A",
                "best_window": " → ".join(s["best_periods"]["best_window"]),
            })
        comparison.sort(key=lambda x: x["overall_excess"], reverse=True)

        # ── Novel highlight: the seasonal patterns that are STATISTICALLY REAL ──
        # Across every sector × month, surface only the excess-return edges that
        # clear the significance bar — honest signal, not noise.
        sig_edges = []
        for key, s in sector_stats.items():
            if "error" in s:
                continue
            mo = s["monthly"]
            for m in range(1, 13):
                if mo["significant_by_month"].get(m):
                    ex = mo["excess_by_month"][m]
                    sig_edges.append({
                        "sector": key, "name": s["name"], "icon": s["icon"],
                        "month": MONTHS[m - 1], "month_num": m,
                        "excess": ex, "t_stat": mo["t_by_month"][m],
                        "p_value": mo["p_by_month"][m], "stars": _sig_stars(mo["p_by_month"][m]),
                        "direction": "outperform" if ex > 0 else "underperform",
                        "win_rate": mo["excess_win_rate"][m],
                        "driver": self._driver_for(key, m),
                    })
        sig_edges.sort(key=lambda x: abs(x["t_stat"]), reverse=True)

        return {
            "success": True,
            "generated_at": datetime.now().isoformat(),
            "current_month": MONTHS[cm - 1],
            "current_quarter": f"Q{(cm - 1) // 3 + 1}",
            "current_driver": MONTH_DRIVERS.get(cm, ""),
            "years_analyzed": self.years,
            "benchmark": "Nifty 50",
            "methodology": "Market-relative excess returns vs Nifty 50, t-tested for significance.",
            "sector_stats": sector_stats,
            "current_rankings": rankings,
            "top_picks": top_picks,
            "heatmap": self._build_heatmap(sector_stats, "avg_by_month"),
            "excess_heatmap": self._build_heatmap(sector_stats, "excess_by_month"),
            "significant_edges": sig_edges,
            "comparison": comparison,
        }
