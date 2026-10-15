"""
Main Stock Analyzer
Orchestrates all analysis modules and generates comprehensive risk score
"""
import traceback
import pandas as pd

from modules.data_fetcher import StockDataFetcher
from modules.technical_analysis import TechnicalAnalyzer
from modules.fundamental_analysis import FundamentalAnalyzer
from modules.sentiment_analysis import SentimentAnalyzer
from modules.sentiment import EnhancedSentimentAnalyzer
from modules.volumetric_analysis import VolumetricAnalyzer
from modules.sector_comparison import SectorComparator
from modules.geopolitical_analysis import GeopoliticalAnalyzer
from modules.merits_analysis import build_merits
from modules.peer_analysis import get_peer_alternatives
from models.feature_engineer import FeatureEngineer
from models.stock_model import StockRiskModel
from services.gemini_service import generate_ai_insight
from models.portfolio_optimizer import PortfolioOptimizer

feature_engineer = FeatureEngineer()
stock_model = StockRiskModel()


# ✅ Clamp any sub-analyzer score to valid range
def _clamp(score, lo=0, hi=100):
    try:
        return max(lo, min(hi, float(score)))
    except (TypeError, ValueError):
        return 50.0  # safe default


def _num(v):
    try:
        f = float(v)
        return f if f == f and abs(f) != float("inf") else None
    except (TypeError, ValueError):
        return None


def _build_snapshot(info, hist):
    """Basic, glanceable company stats from yfinance info + price history."""
    info = info or {}
    close = hist["Close"].dropna() if hist is not None and "Close" in hist else None
    price = _num(info.get("currentPrice")) or _num(info.get("regularMarketPrice"))
    if price is None and close is not None and len(close):
        price = float(close.iloc[-1])
    prev = _num(info.get("previousClose")) or (float(close.iloc[-2]) if close is not None and len(close) > 1 else None)
    hi = _num(info.get("fiftyTwoWeekHigh")) or (float(close.iloc[-252:].max()) if close is not None and len(close) else None)
    lo = _num(info.get("fiftyTwoWeekLow")) or (float(close.iloc[-252:].min()) if close is not None and len(close) else None)

    def pct(v):
        v = _num(v)
        return round(v * 100, 1) if v is not None else None

    # dividend yield: derive from rate/price (yfinance's own field flips between % and fraction)
    div_rate = _num(info.get("dividendRate"))
    dy = round(div_rate / price * 100, 2) if div_rate and price else None

    def ret(days):
        if close is None or len(close) <= days:
            return None
        return round((float(close.iloc[-1]) / float(close.iloc[-1 - days]) - 1) * 100, 1)

    return {
        "name": info.get("longName") or info.get("shortName"),
        "sector": info.get("sector"), "industry": info.get("industry"),
        "price": round(price, 2) if price is not None else None,
        "day_change_pct": round((price / prev - 1) * 100, 2) if price and prev else None,
        "market_cap_cr": round(_num(info.get("marketCap")) / 1e7) if _num(info.get("marketCap")) else None,
        "pe": _num(info.get("trailingPE")), "forward_pe": _num(info.get("forwardPE")),
        "pb": _num(info.get("priceToBook")), "eps": _num(info.get("trailingEps")),
        "dividend_yield_pct": dy,
        "beta": _num(info.get("beta")),
        "week52_high": hi, "week52_low": lo,
        "range_position_pct": round((price - lo) / (hi - lo) * 100, 1) if price and hi and lo and hi > lo else None,
        "avg_volume": _num(info.get("averageVolume")),
        "roe_pct": pct(info.get("returnOnEquity")),
        "profit_margin_pct": pct(info.get("profitMargins")),
        "revenue_growth_pct": pct(info.get("revenueGrowth")),
        "debt_to_equity": _num(info.get("debtToEquity")),
        "return_1m_pct": ret(21), "return_6m_pct": ret(126), "return_1y_pct": ret(min(251, len(close) - 1)) if close is not None and len(close) >= 200 else None,
    }


class StockRiskAnalyzer:
    """Main class that performs comprehensive stock analysis"""

    def __init__(self, ticker, fetcher=None, risk_appetite=None, horizon=None):
        """
        Args:
            ticker:  Stock ticker symbol e.g. 'AAPL', 'RELIANCE.NS'
            fetcher: Optional pre-built StockDataFetcher (reuses its cache).
                     Pass the one already created in app.py to avoid a
                     duplicate network call.
            risk_appetite: Investor risk appetite ('Low'/'Medium'/'High') used
                     to personalise the portfolio allocation.
            horizon: Investment horizon in years (int), also used for the
                     portfolio allocation.
        """
        self.ticker = ticker
        # ✅ FIX: Accept an existing fetcher so app.py cache is reused
        self.data_fetcher = fetcher if fetcher is not None else StockDataFetcher(ticker)
        self.risk_appetite = risk_appetite
        self.horizon = horizon

    def analyze(self):

        """
        Perform comprehensive stock analysis.
        Sub-analyzer failures are isolated — partial results are returned
        rather than crashing the whole response.
        """
        try:
            print(f"Fetching data for {self.ticker}...")
            hist_data = self.data_fetcher.get_historical_data()
            current_info = self.data_fetcher.get_current_info()
            news = self.data_fetcher.get_news(limit=10)
            financials = self.data_fetcher.get_financials()

            if hist_data is None or len(hist_data) < 20:
                return {
                    'success': False,
                    'error': 'Insufficient historical data available'
                }

            if not isinstance(hist_data, pd.DataFrame):
                hist_data = pd.DataFrame(hist_data)

# ── Technical Analysis ─────────────────────────────────────────────────────
            print("Performing technical analysis...")

            technical_results, comprehensive_technical, technical_ml = self._run_safely(
                "technical",
                lambda: self._run_technical(hist_data)
            )

            # ── Fundamental Analysis ───────────────────────────────────────
            print("Performing fundamental analysis...")
            # ✅ FIX: financials may be None/partial — pass it as-is and let
            #         FundamentalAnalyzer handle missing data gracefully
            fundamental_results, comprehensive_fundamental = self._run_safely(
                "fundamental",
                lambda: self._run_fundamental(current_info, financials, news)
            )

            # ── Sentiment Analysis ─────────────────────────────────────────
            # Enhanced pipeline: hybrid news ingestion (RSS + scraping) →
            # FinBERT scoring → stock/sector/divergence aggregation. Falls back
            # to the keyword SentimentAnalyzer internally if FinBERT or news is
            # unavailable, so the output contract is always satisfied.
            # fetch_bodies=False: skip full-article scraping on the per-stock
            # path (the slow part). FinBERT still scores headline + summary,
            # which keeps the factor responsive; the standalone War Room LLM
            # path is unaffected. max_per_scope kept small for latency.
            print("Performing sentiment analysis...")
            sentiment_results, _ = self._run_safely(
                "sentiment",
                lambda: (
                    EnhancedSentimentAnalyzer(max_per_scope=6, fetch_bodies=False).analyze(
                        self.ticker, stock_info=current_info
                    ),
                    None,
                )
            )

            # ── Volumetric Analysis ────────────────────────────────────────
            print("Performing volumetric analysis...")
            volumetric_results, _ = self._run_safely(
                "volumetric",
                lambda: (VolumetricAnalyzer(hist_data).get_volume_score(), None)
            )

            # ── Sector Comparison ──────────────────────────────────────────
            print("Performing sector comparison...")
            sector_results, _ = self._run_safely(
                "sector",
                lambda: (SectorComparator(self.ticker, stock_info=current_info).get_sector_comparison(), None)
            )

            # ── Geopolitical Exposure (War Room phase 2) ───────────────────
            # Deterministic reverse-lookup against the conflict knowledge base;
            # no LLM call, so negligible latency.
            print("Performing geopolitical exposure analysis...")
            geopolitical_results, _ = self._run_safely(
                "geopolitical",
                lambda: (GeopoliticalAnalyzer(self.ticker).get_geopolitical_score(), None)
            )

            # ── Overall Score ──────────────────────────────────────────────
            overall_score = self._calculate_overall_score(
                technical_results,
                fundamental_results,
                sentiment_results,
                volumetric_results,
                sector_results,
                geopolitical_results
            )

            recommendations = self._generate_recommendations(
                overall_score,
                technical_results,
                fundamental_results,
                sentiment_results,
                volumetric_results
            )

            # =========================
            # 🔹 ML FEATURE ENGINEERING
            # =========================

            # Build the canonical technical features from the SAME function used
            # in training (feature_engineer.build_frame), on the latest bar — so
            # train and serve see an identical feature distribution.
            features_array, feature_dict = feature_engineer.build_latest(hist_data)

            # Real sentiment is not a model input (no historical source), but it
            # still informs the rule-based override as a heuristic.
            if isinstance(sentiment_results, dict):
                feature_dict = {
                    **feature_dict,
                    "sentiment_score": sentiment_results.get("score", 50),
                }
            print("ML FEATURES:", feature_dict)

            # =========================
            # 🔹 ML PREDICTION
            # =========================

            ml_result = stock_model.predict(features_array, feature_dict=feature_dict)

            feature_importance = stock_model.get_feature_importance(
                feature_engineer.get_feature_names()
            )

            # ✅ NEW: XGBoost feature importance
            xgb_feature_importance = stock_model.get_xgb_feature_importance(
                feature_engineer.get_feature_names()
            )

            # Hybrid (rule-aware HGB) permutation importance — includes rule_signal
            hgb_feature_importance = stock_model.get_hgb_feature_importance()

            # =========================
            # 🔹 EXPLAINABILITY
            # =========================

            def explain(features):
                explanation = []

                if features.get("volatility", 0) > 0.5:
                    explanation.append("High volatility increases risk")

                if features.get("rsi", 50) > 70:
                    explanation.append("Stock is overbought")

                if features.get("sharpe_ratio", 0) > 1:
                    explanation.append("Strong risk-adjusted returns")

                if features.get("market_trend", 0) == 1:
                    explanation.append("Market trend is bullish")

                return explanation

            # =========================
            # 🔹 FINAL RESPONSE OBJECT
            # =========================

            results = {
                'success': True,
                'ticker': self.ticker,
                'overall_score': overall_score,

                'technical_analysis': technical_results,
                'comprehensive_technical': comprehensive_technical,

                'fundamental_analysis': fundamental_results,
                'comprehensive_fundamental': comprehensive_fundamental,

                'sentiment_analysis': sentiment_results,
                'volumetric_analysis': volumetric_results,
                'sector_comparison': sector_results,
                'geopolitical_analysis': geopolitical_results,

                'recommendations': recommendations,
                'risk_level': self._get_risk_level(overall_score['score']),

                # ── Calibrated RF (existing) ──────────────────────────────
                'ml_prediction':    ml_result['prediction'],
                'ml_confidence':    ml_result['confidence'],
                'ml_probabilities': ml_result['probabilities'],

                'features_used': feature_dict,

                'comparison': {
                    'rule_score': overall_score['score'],
                    'ml_signal':  ml_result['prediction']
                },

                # P(beats the Nifty over the next ~20 trading days), soft-vote
                # ensemble — the headline number for the relative-performance model.
                'outperform_probability': (ml_result.get('ensemble') or {}).get('outperform_probability'),

                'explanation': explain(feature_dict),

                # ── Tri-model intelligence ────────────────────────────────
                'xgb_prediction':       ml_result.get('xgb_prediction'),
                'xgb_confidence':       ml_result.get('xgb_confidence'),
                'hgb_prediction':       ml_result.get('hgb_prediction'),
                'hgb_confidence':       ml_result.get('hgb_confidence'),
                'model_disagreement':   ml_result.get('model_disagreement', False),
                'consensus':            ml_result.get('consensus'),
                'ensemble':             ml_result.get('ensemble'),
                'model_calibration':    ml_result.get('calibration'),
                'rule_signal':          ml_result.get('rule_signal'),

                # Per-model key drivers (global feature importance)
                'feature_importance':     feature_importance,
                'xgb_feature_importance': xgb_feature_importance,
                'hgb_feature_importance': hgb_feature_importance,
            }

            # =========================
            # 🔹 PORTFOLIO (before Gemini, so the insight can reference it)
            # =========================
            optimizer = PortfolioOptimizer()
            portfolio = optimizer.optimize(
                results,
                risk_appetite=self.risk_appetite,
                horizon=self.horizon,
            )
            results["portfolio"] = portfolio

            # Key stats for the snapshot strip (and the valuation merits)
            results["snapshot"] = _build_snapshot(current_info, hist_data)

            # Price history for the in-app chart (the embedded TradingView widget
            # refuses NSE symbols, so we draw our own from the data we already have).
            try:
                close = hist_data["Close"].dropna().iloc[-252:]
                step = max(1, len(close) // 130)
                sampled = close.iloc[::step]
                if sampled.index[-1] != close.index[-1]:
                    sampled = pd.concat([sampled, close.iloc[[-1]]])
                results["price_history"] = [
                    {"d": str(i)[:10], "c": round(float(v), 2)} for i, v in sampled.items()
                ]
            except Exception as e:
                print(f"[price_history] failed: {e}")
                results["price_history"] = []

            # =========================
            # 🔹 MERITS / DEMERITS — why-to-buy vs why-not (deterministic)
            # =========================
            results["merits_analysis"] = build_merits(results)

            # =========================
            # 🔹 PEER ALTERNATIVES — better-performing names in the same sector
            # =========================
            try:
                close = hist_data["Close"].dropna()
                self_6m = None
                if len(close) > 25:
                    window = close.iloc[-126:] if len(close) >= 126 else close
                    first = float(window.iloc[0])
                    if first > 0:
                        self_6m = (float(window.iloc[-1]) / first - 1) * 100
                yf_sector = (sector_results or {}).get("sector")
                results["peer_analysis"] = get_peer_alternatives(
                    self.ticker, yf_sector=yf_sector, self_return_6m=self_6m)
            except Exception as e:
                print(f"[peers] failed: {e}")
                results["peer_analysis"] = None

            # =========================
            # 🔹 GEMINI AI — India-aware reasoning layer over the full analysis
            # =========================
            results["gemini_insight"] = generate_ai_insight(results)

            # =========================
            # 🔹 RETURN
            # =========================

            return results

        except Exception as e:
            # ✅ FIX: Log full traceback so root cause is visible in server logs
            print(f"❌ analyze() failed for {self.ticker}:\n{traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }

    # ── Private helpers ────────────────────────────────────────────────────

    def _run_safely(self, component_name, func):
        try:
            result = func()
            # Return result directly — callers handle their own unpacking
            return result
        except Exception as e:
            print(f"⚠️ {component_name} failed:\n{traceback.format_exc()}")
            # Return shape matches what each caller expects
            if component_name == "technical":
                return {'score': 50, 'error': f'{component_name} unavailable'}, None, None
            return {'score': 50, 'error': f'{component_name} unavailable'}, None

    def _get_default_result(self, name):
        """Safe fallback for any failed sub-analyzer."""
        return {'score': 50, 'error': f'{name} analysis unavailable'}, None, None

    def _run_technical(self, hist_data):
        ta = TechnicalAnalyzer(hist_data)

        score = ta.get_technical_score()
        full_analysis = ta.get_comprehensive_analysis()

        # ✅ NEW → ML-safe raw features
        ml_features = ta.get_ml_safe_features()

        return score, full_analysis, ml_features

    def _run_fundamental(self, current_info, financials, news):
        fa = FundamentalAnalyzer(current_info, financials, news)
        return fa.get_fundamental_score(), fa.get_comprehensive_fundamental_analysis()

    def _calculate_overall_score(self, tech, fund, sent, vol, sector, geo=None):
        """Calculate weighted overall score from all analyses."""

        # ✅ FIX: If sector stub always returns 50, redistribute its weight
        #         so it doesn't silently anchor the score toward the midpoint.
        sector_score_raw = sector.get('score', 50)
        sector_is_stub = sector.get('error') is not None  # set by _run_safely on failure

        if sector_is_stub:
            # Drop sector weight and redistribute to remaining analyzers
            weights = {
                'technical':   0.294,   # 0.25 / 0.85
                'fundamental': 0.353,   # 0.30 / 0.85
                'sentiment':   0.176,   # 0.15 / 0.85
                'volumetric':  0.176,   # 0.15 / 0.85
                'sector':      0.0
            }
        else:
            weights = {
                'technical':   0.25,
                'fundamental': 0.30,
                'sentiment':   0.15,
                'volumetric':  0.15,
                'sector':      0.15
            }

        # ── Geopolitical factor: only contributes when the stock is actually
        #    exposed to a tracked conflict, so unexposed (e.g. US) stocks are
        #    unaffected. When exposed, it takes a slice and the others scale down.
        geo_exposed = bool(geo and geo.get('exposed'))
        geo_score = _clamp(geo.get('score', 50)) if geo else 50.0
        if geo_exposed:
            # Evidence-scaled: the geo factor has no demonstrated predictive
            # power (IC ~ 0), so it carries 3-10% depending on how strong the
            # exposure is, rather than a flat 12%.
            geo_weight = 0.03 + 0.07 * float(geo.get('strength', 0.5) or 0.0)
            for k in weights:
                weights[k] *= (1 - geo_weight)
            weights['geopolitical'] = geo_weight
        else:
            weights['geopolitical'] = 0.0

        # ✅ FIX: Clamp each sub-score to [0, 100] before weighting
        tech_score   = _clamp(tech.get('score', 50))
        fund_score   = _clamp(fund.get('score', 50))
        sent_score   = _clamp(sent.get('score', 50))
        vol_score    = _clamp(vol.get('score', 50))
        sector_score = _clamp(sector_score_raw)

        overall = (
            tech_score   * weights['technical'] +
            fund_score   * weights['fundamental'] +
            sent_score   * weights['sentiment'] +
            vol_score    * weights['volumetric'] +
            sector_score * weights['sector'] +
            geo_score    * weights['geopolitical']
        )

        return {
            'score': round(_clamp(overall), 2),
            'breakdown': {
                'technical':    round(tech_score, 2),
                'fundamental':  round(fund_score, 2),
                'sentiment':    round(sent_score, 2),
                'volumetric':   round(vol_score, 2),
                'sector':       round(sector_score, 2),
                'geopolitical': round(geo_score, 2)
            },
            'weights_used': weights  # useful for debugging
        }

    def _generate_recommendations(self, overall, tech, fund, sent, vol):
        """Generate investment recommendations based on analysis."""
        recommendations = []
        overall_score = overall['score']

        # Overall rating
        if overall_score >= 75:
            rating = ('High', 'Strong Buy — positive indicators across all analyses')
        elif overall_score >= 65:
            rating = ('High', 'Buy — good overall fundamentals with positive trends')
        elif overall_score >= 55:
            rating = ('Medium', 'Hold — mixed signals, wait for clearer trend')
        elif overall_score >= 40:
            rating = ('Medium', 'Caution — multiple negative indicators, consider reducing exposure')
        else:
            rating = ('High', 'Avoid/Sell — strong negative indicators across multiple analyses')

        recommendations.append({'priority': rating[0], 'type': 'Rating', 'message': rating[1]})

        # Technical flag
        if tech.get('score', 50) < 30:
            recommendations.append({
                'priority': 'High',
                'type': 'Technical',
                'message': 'Poor technical indicators — stock may be in a downtrend'
            })

        # Fundamental flag
        if fund.get('score', 50) < 40:
            recommendations.append({
                'priority': 'High',
                'type': 'Fundamental',
                'message': 'Weak fundamentals — review financial health metrics'
            })

        # Sentiment flag
        if sent.get('score', 50) < 40:
            recommendations.append({
                'priority': 'Medium',
                'type': 'Sentiment',
                'message': 'Negative news sentiment — monitor news flow closely'
            })

        # ✅ FIX: High volume score = volatility/risk, not automatic positive signal
        vol_score = vol.get('score', 50)
        if vol_score > 75:
            recommendations.append({
                'priority': 'Medium',
                'type': 'Volume',
                'message': 'Elevated volume activity — may indicate volatility; verify trend direction'
            })
        elif vol_score < 30:
            recommendations.append({
                'priority': 'Low',
                'type': 'Volume',
                'message': 'Low volume — limited liquidity, wider spreads possible'
            })

        return recommendations

    def _get_risk_level(self, score):
        if score >= 75:
            return 'Low Risk / Strong Buy'
        elif score >= 65:
            return 'Low-Medium Risk / Buy'
        elif score >= 55:
            return 'Medium Risk / Hold'
        elif score >= 40:
            return 'Medium-High Risk / Caution'
        elif score >= 25:
            return 'High Risk / Avoid'
        else:
            return 'Very High Risk / Sell'
