"""
EnhancedSentimentAnalyzer — public entry point for the sentiment package.

Ties the three phases together:
    ingest (news_ingestor) → score (finbert_scorer) → aggregate (aggregator)

Behaviour:
    • Per-ticker in-memory cache with a short TTL (news sentiment goes stale
      fast, and FinBERT scoring is the slow part we want to avoid repeating).
    • Graceful degradation: if FinBERT/torch is unavailable, or ingestion
      returns nothing, fall back to the legacy keyword SentimentAnalyzer so the
      overall score still gets a sentiment input.
    • Output always satisfies the legacy contract consumed by
      StockRiskAnalyzer._calculate_overall_score and FeatureEngineer.
"""

from __future__ import annotations

import time

from .news_ingestor import NewsIngestor
from .finbert_scorer import FinBertScorer
from . import aggregator


_CACHE: dict[str, dict] = {}
_CACHE_TTL_SECONDS = 30 * 60  # 30 minutes


class EnhancedSentimentAnalyzer:
    def __init__(self, ttl_seconds: int = _CACHE_TTL_SECONDS,
                 max_per_scope: int = 12, fetch_bodies: bool = True):
        self.ttl = ttl_seconds
        self.max_per_scope = max_per_scope
        self.fetch_bodies = fetch_bodies

    def _cache_get(self, ticker: str):
        entry = _CACHE.get(ticker)
        if entry and (time.time() - entry["_ts"] < self.ttl):
            cached = dict(entry["data"])
            cached["cached"] = True
            return cached
        return None

    def _cache_put(self, ticker: str, data: dict):
        _CACHE[ticker] = {"_ts": time.time(), "data": data}

    def analyze(self, ticker: str, stock_info: dict | None = None) -> dict:
        ticker = (ticker or "").upper().strip()

        cached = self._cache_get(ticker)
        if cached is not None:
            return cached

        try:
            # ① Ingest
            ing = NewsIngestor(
                ticker, stock_info=stock_info,
                max_per_scope=self.max_per_scope,
                fetch_bodies=self.fetch_bodies,
            )
            bundle = ing.fetch()
            stock_articles = bundle["stock_articles"]
            sector_articles = bundle["sector_articles"]

            if not stock_articles and not sector_articles:
                return self._legacy_fallback(ticker, reason="no news found")

            # ② Score
            scorer = FinBertScorer()
            if not scorer.is_available:
                return self._legacy_fallback(
                    ticker, reason="FinBERT unavailable",
                    prefetched=stock_articles,
                )

            scored_stock = scorer.score_articles(stock_articles)
            scored_sector = scorer.score_articles(sector_articles)

            # ③ Aggregate
            result = aggregator.aggregate(scored_stock, scored_sector)
            result["ticker"] = ticker
            result["company_name"] = bundle.get("company_name")
            result["sector_name"] = bundle.get("sector")
            result["cached"] = False

            self._cache_put(ticker, result)
            return result

        except Exception as e:
            print(f"[sentiment] EnhancedSentimentAnalyzer failed for {ticker}: {e}")
            return self._legacy_fallback(ticker, reason=str(e))

    # ── Fallback to the original keyword analyzer ────────────────────────────

    def _legacy_fallback(self, ticker: str, reason: str,
                         prefetched: list | None = None) -> dict:
        """
        Use the existing keyword SentimentAnalyzer so the app still gets a
        sentiment score when the enhanced path can't run.
        """
        print(f"[sentiment] falling back to keyword engine ({reason})")
        try:
            from modules.sentiment_analysis import SentimentAnalyzer
            # Convert any prefetched Article objects into the dict shape the
            # legacy analyzer expects; otherwise pass empty (it handles that).
            legacy_articles = []
            for a in (prefetched or []):
                legacy_articles.append({
                    "title": a.title,
                    "summary": a.summary,
                    "publisher": a.source,
                })
            base = SentimentAnalyzer().get_sentiment_score(legacy_articles)
        except Exception as e:
            base = {"score": 50, "sentiment": "Neutral", "confidence": "Low",
                    "sample_size": 0, "article_scores": [], "keywords": {}}
            print(f"[sentiment] legacy sentiment also failed: {e}")

        # Wrap in the enhanced contract (neutral sector/divergence)
        fb = aggregator.neutral_fallback(reason=reason)
        fb.update({
            "score": base.get("score", 50),
            "sentiment": base.get("sentiment", "Neutral"),
            "confidence": base.get("confidence", "Low"),
            "sample_size": base.get("sample_size", 0),
            "article_scores": base.get("article_scores", []),
            "keywords": base.get("keywords", {}),
            "engine": "keyword-fallback",
            "ticker": ticker,
            "cached": False,
        })
        return fb
