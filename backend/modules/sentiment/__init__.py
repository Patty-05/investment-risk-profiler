"""
Enhanced sentiment analysis package.

Three-stage pipeline:
    ① news_ingestor  — hybrid fetch (RSS + scraping fallback), relevancy filter,
                       stock + sector scopes
    ② finbert_scorer — finance-tuned transformer scoring (Phase 2)
    ③ aggregator     — stock score, sector score, divergence signal (Phase 3)

The package keeps the 0–100 `score` output contract used by the rest of the app
so it is a drop-in deepening of the existing keyword-based SentimentAnalyzer.
"""

from .news_ingestor import NewsIngestor, Article
from .analyzer import EnhancedSentimentAnalyzer

__all__ = ["NewsIngestor", "Article", "EnhancedSentimentAnalyzer"]
