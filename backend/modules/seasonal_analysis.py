"""
Indian Equity Seasonal Analysis Module
Analyses monthly/quarterly performance patterns of Indian sector indices
over 10 years (2014-2024) and recommends sectors based on historical seasonality.
"""

import numpy as np
import pandas as pd
from datetime import datetime, date
import traceback

try:
    import yfinance as yf
    _YF_AVAILABLE = True
except ImportError:
    _YF_AVAILABLE = False
    print("⚠️ yfinance not installed — pip install yfinance")


# ── Sector definitions ────────────────────────────────────────────────────────
# Primary ticker + fallback tickers for each sector
SECTOR_CONFIG = {
    "IT": {
        "name": "Information Technology",
        "tickers": ["^CNXIT", "INFY.NS", "TCS.NS"],
        "color": "#3b82f6",
        "components": ["TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS"],
        "icon": "💻"
    },
    "Banking": {
        "name": "Banking & Finance",
        "tickers": ["^NSEBANK", "HDFCBANK.NS", "ICICIBANK.NS"],
        "color": "#10b981",
        "components": ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS", "BAJFINANCE.NS"],
        "icon": "🏦"
    },
    "FMCG": {
        "name": "Fast Moving Consumer Goods",
        "tickers": ["^CNXFMCG", "HINDUNILVR.NS", "NESTLEIND.NS"],
        "color": "#f59e0b",
        "components": ["HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS", "GODREJCP.NS"],
        "icon": "🛒"
    },
    "Auto": {
        "name": "Automobile",
        "tickers": ["^CNXAUTO", "MARUTI.NS", "TATAMOTORS.NS"],
        "color": "#ef4444",
        "components": ["MARUTI.NS", "TATAMOTORS.NS", "M&M.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS"],
        "icon": "🚗"
    },
    "Pharma": {
        "name": "Pharmaceuticals",
        "tickers": ["^CNXPHARMA", "SUNPHARMA.NS", "DRREDDY.NS"],
        "color": "#8b5cf6",
        "components": ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS", "TORNTPHARM.NS"],
        "icon": "💊"
    },
    "Realty": {
        "name": "Real Estate",
        "tickers": ["^CNXREALTY", "DLF.NS", "GODREJPROP.NS"],
        "color": "#ec4899",
        "components": ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS", "SOBHA.NS"],
        "icon": "🏗️"
    },
    "Hospitality": {
        "name": "Hospitality & Tourism",
        "tickers": ["INDHOTEL.NS", "LEMONTRE.NS", "CHALET.NS"],
        "color": "#f97316",
        "components": ["INDHOTEL.NS", "LEMONTRE.NS", "CHALET.NS", "EIHOTEL.NS", "MAHINDRAHOLIDAYS.NS", "THOMASCOOK.NS"],
        "icon": "🏨"
    },
    "Energy": {
        "name": "Energy & Power",
        "tickers": ["^CNXENERGY", "RELIANCE.NS", "ONGC.NS"],
        "color": "#eab308",
        "components": ["RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS", "BPCL.NS"],
        "icon": "⚡"
    },
    "Metal": {
        "name": "Metals & Mining",
        "tickers": ["^CNXMETAL", "TATASTEEL.NS", "HINDALCO.NS"],
        "color": "#6b7280",
        "components": ["TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "NMDC.NS", "SAIL.NS"],
        "icon": "⛏️"
    },
    "Infra": {
        "name": "Infrastructure",
        "tickers": ["^CNXINFRA", "LT.NS", "ADANIPORTS.NS"],
        "color": "#84cc16",
        "components": ["LT.NS", "ADANIPORTS.NS", "BHARTIARTL.NS", "ULTRACEMCO.NS", "GRASIM.NS", "SIEMENS.NS"],
        "icon": "🏗️"
    }
}

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

QUARTERS = ["Q1 (Jan-Mar)", "Q2 (Apr-Jun)", "Q3 (Jul-Sep)", "Q4 (Oct-Dec)"]


class IndianSeasonalAnalyzer:
    """
    Fetches 10 years of Indian sector data and computes:
    - Monthly average returns per sector
    - Quarterly patterns
    - Best/worst months
    - Current month recommendation
    - Top 3 sector picks
    """

    def __init__(self, years: int = 10):
        self.years     = years
        self.end_date  = datetime.now()
        self.start_date = datetime(self.end_date.year - years, 1, 1)
        self._cache    = {}

    # ── Data fetching ──────────────────────────────────────────────────────

    def _fetch_ticker(self, ticker: str) -> pd.DataFrame | None:
        """Fetch OHLCV data for a single ticker with caching."""
        if ticker in self._cache:
            return self._cache[ticker]
        if not _YF_AVAILABLE:
            return None
        try:
            df = yf.download(
                ticker,
                start=self.start_date.strftime("%Y-%m-%d"),
                end=self.end_date.strftime("%Y-%m-%d"),
                progress=False,
                auto_adjust=True
            )
            if df is None or len(df) < 50:
                return None
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self._cache[ticker] = df
            return df
        except Exception as e:
            print(f"  ⚠️ Failed to fetch {ticker}: {e}")
            return None

    def _fetch_sector_data(self, sector_key: str) -> pd.DataFrame | None:
        """Try each ticker for a sector until one succeeds."""
        config  = SECTOR_CONFIG[sector_key]
        tickers = config["tickers"]

        for ticker in tickers:
            df = self._fetch_ticker(ticker)
            if df is not None and len(df) >= 50:
                print(f"  ✅ {sector_key}: loaded from {ticker} ({len(df)} rows)")
                return df

        # Fallback: average of component stocks
        print(f"  ⚠️ {sector_key}: index failed, trying component average...")
        frames = []
        for comp in config["components"][:4]:
            df = self._fetch_ticker(comp)
            if df is not None:
                frames.append(df["Close"])

        if frames:
            combined = pd.concat(frames, axis=1).dropna(how="all")
            avg_close = combined.mean(axis=1)
            result = pd.DataFrame({"Close": avg_close})
            result.index = combined.index
            print(f"  ✅ {sector_key}: built from {len(frames)} components")
            return result

        print(f"  ❌ {sector_key}: no data available")
        return None

    # ── Monthly return computation ─────────────────────────────────────────

    def _compute_monthly_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute monthly returns from daily OHLCV data.
        Returns DataFrame with columns: year, month, return_pct
        """
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        # Resample to monthly — use last close of each month
        monthly = df["Close"].resample("ME").last()
        monthly_returns = monthly.pct_change().dropna() * 100

        result = pd.DataFrame({
            "year":       monthly_returns.index.year,
            "month":      monthly_returns.index.month,
            "return_pct": monthly_returns.values
        })
        return result

    def _aggregate_monthly(self, monthly_df: pd.DataFrame) -> dict:
        """
        Aggregate monthly returns into:
        - avg_by_month: {1..12} -> avg return
        - std_by_month: {1..12} -> std dev
        - positive_rate: {1..12} -> % of years with positive return
        """
        avg_by_month      = {}
        std_by_month      = {}
        positive_rate     = {}
        sample_count      = {}
        year_by_month     = {}   # {month: [return per year]}

        for m in range(1, 13):
            subset = monthly_df[monthly_df["month"] == m]["return_pct"]
            if len(subset) == 0:
                avg_by_month[m]  = 0.0
                std_by_month[m]  = 0.0
                positive_rate[m] = 50.0
                sample_count[m]  = 0
                year_by_month[m] = []
            else:
                avg_by_month[m]  = round(float(subset.mean()), 2)
                std_by_month[m]  = round(float(subset.std()), 2)
                positive_rate[m] = round(float((subset > 0).mean() * 100), 1)
                sample_count[m]  = int(len(subset))
                year_by_month[m] = [round(float(v), 2) for v in subset.values]

        return {
            "avg_by_month":  avg_by_month,
            "std_by_month":  std_by_month,
            "positive_rate": positive_rate,
            "sample_count":  sample_count,
            "year_by_month": year_by_month
        }

    def _aggregate_quarterly(self, monthly_df: pd.DataFrame) -> dict:
        """Aggregate into Q1/Q2/Q3/Q4 average returns."""
        quarter_map = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2,
                       7: 3, 8: 3, 9: 3, 10: 4, 11: 4, 12: 4}
        monthly_df  = monthly_df.copy()
        monthly_df["quarter"] = monthly_df["month"].map(quarter_map)

        result = {}
        for q in range(1, 5):
            subset = monthly_df[monthly_df["quarter"] == q]["return_pct"]
            result[q] = {
                "avg":           round(float(subset.mean()), 2) if len(subset) > 0 else 0.0,
                "std":           round(float(subset.std()),  2) if len(subset) > 0 else 0.0,
                "positive_rate": round(float((subset > 0).mean() * 100), 1) if len(subset) > 0 else 50.0
            }
        return result

    # ── Best period detection ──────────────────────────────────────────────

    def _find_best_periods(self, agg: dict) -> dict:
        avg = agg["avg_by_month"]
        pos = agg["positive_rate"]

        # Composite score: 70% avg return + 30% positive rate normalised
        scores = {}
        for m in range(1, 13):
            scores[m] = avg[m] * 0.7 + (pos[m] - 50) * 0.06

        sorted_months = sorted(scores.keys(), key=lambda m: scores[m], reverse=True)

        best_months  = sorted_months[:3]
        worst_months = sorted_months[-3:]

        # Best consecutive 3-month window
        best_window_score = -999
        best_window_start = 1
        for start in range(1, 13):
            window = [(start - 1 + i) % 12 + 1 for i in range(3)]
            ws = sum(scores[m] for m in window)
            if ws > best_window_score:
                best_window_score = ws
                best_window_start = start

        best_window = [(best_window_start - 1 + i) % 12 + 1 for i in range(3)]

        return {
            "best_months":       [MONTHS[m - 1] for m in best_months],
            "worst_months":      [MONTHS[m - 1] for m in worst_months],
            "best_window":       [MONTHS[m - 1] for m in best_window],
            "best_window_score": round(best_window_score, 2),
            "scores":            {MONTHS[m - 1]: round(scores[m], 3) for m in range(1, 13)}
        }

    # ── Current month recommendation ───────────────────────────────────────

    def _current_month_rank(self, sector_stats: dict) -> list:
        """Rank sectors by their historical performance in the current month."""
        current_month = datetime.now().month
        rankings = []
        for sector_key, stats in sector_stats.items():
            if "error" in stats:
                continue
            avg  = stats["monthly"]["avg_by_month"].get(current_month, 0)
            pos  = stats["monthly"]["positive_rate"].get(current_month, 50)
            std  = stats["monthly"]["std_by_month"].get(current_month, 5)
            # Composite: return vs risk
            score = avg * 0.6 + (pos - 50) * 0.06 - std * 0.05
            rankings.append({
                "sector":   sector_key,
                "name":     SECTOR_CONFIG[sector_key]["name"],
                "icon":     SECTOR_CONFIG[sector_key]["icon"],
                "avg_return": avg,
                "positive_rate": pos,
                "std":       std,
                "score":     round(score, 3)
            })

        rankings.sort(key=lambda x: x["score"], reverse=True)
        return rankings

    # ── Top 3 picks ────────────────────────────────────────────────────────

    def _generate_top_picks(self, sector_stats: dict, current_rankings: list) -> list:
        """
        Generate top 3 sector picks combining:
        - Current month historical strength
        - Trend momentum (last 3 months vs long-term avg)
        - Consistency (positive rate)
        """
        current_month = datetime.now().month
        picks = []

        for rank_item in current_rankings[:5]:
            sector_key = rank_item["sector"]
            stats      = sector_stats[sector_key]

            # Recent momentum: avg of last 3 months vs overall avg
            prev_months = [(current_month - i - 1) % 12 + 1 for i in range(3)]
            recent_avg  = np.mean([stats["monthly"]["avg_by_month"].get(m, 0) for m in prev_months])
            overall_avg = np.mean(list(stats["monthly"]["avg_by_month"].values()))
            momentum    = recent_avg - overall_avg

            best_months = stats["best_periods"]["best_months"]
            is_best_now = MONTHS[current_month - 1] in best_months

            # Confidence: based on sample size and consistency
            sample = stats["monthly"]["sample_count"].get(current_month, 0)
            pos_rate = rank_item["positive_rate"]
            confidence = "High" if (sample >= 7 and pos_rate >= 65) else \
                         "Medium" if (sample >= 5 and pos_rate >= 55) else "Low"

            reason_parts = []
            if is_best_now:
                reason_parts.append(f"{MONTHS[current_month - 1]} is historically one of the best months")
            if momentum > 0.5:
                reason_parts.append("recent months showing positive momentum")
            if pos_rate >= 65:
                reason_parts.append(f"positive in {pos_rate}% of years for this month")
            if rank_item["avg_return"] > 1.5:
                reason_parts.append(f"avg monthly return of +{rank_item['avg_return']}%")

            reason = "; ".join(reason_parts) if reason_parts else "Historical seasonal strength"

            picks.append({
                "rank":          len(picks) + 1,
                "sector":        sector_key,
                "name":          SECTOR_CONFIG[sector_key]["name"],
                "icon":          SECTOR_CONFIG[sector_key]["icon"],
                "color":         SECTOR_CONFIG[sector_key]["color"],
                "avg_return":    rank_item["avg_return"],
                "positive_rate": pos_rate,
                "confidence":    confidence,
                "momentum":      round(float(momentum), 2),
                "best_months":   best_months,
                "reason":        reason,
                "components":    SECTOR_CONFIG[sector_key]["components"][:4]
            })

            if len(picks) == 3:
                break

        return picks

    # ── Heatmap data ───────────────────────────────────────────────────────

    def _build_heatmap(self, sector_stats: dict) -> dict:
        """
        Build sector × month matrix of avg returns for heatmap rendering.
        Returns: { sectors: [...], months: [...], matrix: [[...]] }
        """
        sectors = [k for k in sector_stats if "error" not in sector_stats[k]]
        matrix  = []
        for sector_key in sectors:
            row = []
            for m in range(1, 13):
                val = sector_stats[sector_key]["monthly"]["avg_by_month"].get(m, 0)
                row.append(round(val, 2))
            matrix.append(row)

        return {
            "sectors": sectors,
            "sector_names": [SECTOR_CONFIG[s]["name"] for s in sectors],
            "sector_icons": [SECTOR_CONFIG[s]["icon"]  for s in sectors],
            "months":  MONTHS,
            "matrix":  matrix
        }

    # ── Main entry point ───────────────────────────────────────────────────

    def analyze(self) -> dict:
        """
        Run full seasonal analysis across all sectors.
        Returns complete data structure for frontend consumption.
        """
        print("🌿 Starting Indian seasonal analysis...")
        sector_stats = {}

        for sector_key in SECTOR_CONFIG:
            print(f"  Fetching {sector_key}...")
            try:
                df = self._fetch_sector_data(sector_key)
                if df is None:
                    sector_stats[sector_key] = {"error": "No data available"}
                    continue

                monthly_df  = self._compute_monthly_returns(df)
                monthly_agg = self._aggregate_monthly(monthly_df)
                quarterly   = self._aggregate_quarterly(monthly_df)
                best_periods = self._find_best_periods(monthly_agg)

                # Year-over-year for sparkline
                yearly_returns = monthly_df.groupby("year")["return_pct"].sum().reset_index()
                yearly_returns = yearly_returns.rename(columns={"return_pct": "annual_return"})

                sector_stats[sector_key] = {
                    "name":          SECTOR_CONFIG[sector_key]["name"],
                    "icon":          SECTOR_CONFIG[sector_key]["icon"],
                    "color":         SECTOR_CONFIG[sector_key]["color"],
                    "monthly":       monthly_agg,
                    "quarterly":     quarterly,
                    "best_periods":  best_periods,
                    "yearly": [
                        {"year": int(r["year"]), "return": round(float(r["annual_return"]), 2)}
                        for _, r in yearly_returns.iterrows()
                    ],
                    "overall_avg":   round(float(monthly_df["return_pct"].mean()), 2),
                    "overall_std":   round(float(monthly_df["return_pct"].std()),  2),
                }

            except Exception as e:
                print(f"  ❌ {sector_key} failed: {traceback.format_exc()}")
                sector_stats[sector_key] = {"error": str(e)}

        current_rankings = self._current_month_rank(sector_stats)
        top_picks        = self._generate_top_picks(sector_stats, current_rankings)
        heatmap          = self._build_heatmap(sector_stats)

        # Cross-sector comparison summary
        comparison = []
        for sector_key, stats in sector_stats.items():
            if "error" in stats:
                continue
            comparison.append({
                "sector":      sector_key,
                "name":        stats["name"],
                "icon":        stats["icon"],
                "color":       stats["color"],
                "overall_avg": stats["overall_avg"],
                "overall_std": stats["overall_std"],
                "best_month":  stats["best_periods"]["best_months"][0] if stats["best_periods"]["best_months"] else "N/A",
                "worst_month": stats["best_periods"]["worst_months"][0] if stats["best_periods"]["worst_months"] else "N/A",
                "best_window": " → ".join(stats["best_periods"]["best_window"]),
            })
        comparison.sort(key=lambda x: x["overall_avg"], reverse=True)

        return {
            "success":          True,
            "generated_at":     datetime.now().isoformat(),
            "current_month":    MONTHS[datetime.now().month - 1],
            "current_quarter":  f"Q{(datetime.now().month - 1) // 3 + 1}",
            "years_analyzed":   self.years,
            "sector_stats":     sector_stats,
            "current_rankings": current_rankings,
            "top_picks":        top_picks,
            "heatmap":          heatmap,
            "comparison":       comparison,
        }