"""
Aggregator — Phase 3 of the enhanced sentiment pipeline.

Combines per-article FinBERT scores into the signals the app consumes:

    • stock sentiment   (0–100) — recency- and relevance-weighted
    • sector sentiment  (0–100) — same, over the sector-scoped articles
    • divergence        — stock minus sector; the "is this stock moving with or
                          against its sector?" signal (the automobile example)
    • reasoning         — the articles that most drove each side, in plain text

Output keeps the legacy keys used by StockRiskAnalyzer / FeatureEngineer
(`score`, `sentiment`, `confidence`, `sample_size`, `article_scores`,
`keywords`) so this is a drop-in replacement, and adds the new sector/divergence
fields on top.
"""

from __future__ import annotations

import time
import math

from modules.sentiment.news_ingestor import source_weight


# Recency half-life: an article this many days old carries half the weight of a
# brand-new one. News sentiment decays fast, so keep it short.
_RECENCY_HALFLIFE_DAYS = 3.0
_SECONDS_PER_DAY = 86400.0


def _recency_weight(published_ts: float, now: float) -> float:
    if not published_ts:
        return 0.5  # unknown date → middling weight
    age_days = max(0.0, (now - published_ts) / _SECONDS_PER_DAY)
    return 0.5 ** (age_days / _RECENCY_HALFLIFE_DAYS)


def _weighted_stats(scored: list[dict], now: float) -> tuple[float, float, float]:
    """
    Return (weighted_score, total_weight, agreement) for a list of scored
    articles. `agreement` is 1 − normalised dispersion of the directional
    votes (how much the articles agree on direction).
    """
    if not scored:
        return 50.0, 0.0, 0.0

    num = 0.0
    den = 0.0
    directions = []
    for s in scored:
        # Weight combines recency, topical relevance (~[0.15,1]) and publisher
        # credibility, so a Moneycontrol/ET markets report outweighs an
        # unattributed aggregator repost of the same headline.
        cred = source_weight(s.get("url") or s.get("source") or "")
        w = (_recency_weight(s.get("published_ts", 0.0), now)
             * (0.5 + s.get("relevance", 0.0))
             * cred)
        num += s["score"] * w
        den += w
        directions.append(1 if s["score"] > 55 else (-1 if s["score"] < 45 else 0))

    weighted = (num / den) if den else 50.0

    # Agreement: 1 - stdev(direction)/max_stdev. All-same → 1.0
    if len(directions) > 1:
        mean = sum(directions) / len(directions)
        var = sum((d - mean) ** 2 for d in directions) / len(directions)
        agreement = max(0.0, 1.0 - math.sqrt(var))
    else:
        agreement = 0.5

    return round(weighted, 2), round(den, 3), round(agreement, 3)


def _label(score: float) -> str:
    if score >= 70:   return "Very Positive"
    if score >= 58:   return "Positive"
    if score >= 42:   return "Neutral"
    if score >= 30:   return "Negative"
    return "Very Negative"


def _confidence(sample_size: int, agreement: float, avg_conf: float) -> str:
    """Blend sample size, directional agreement, and model confidence."""
    if sample_size == 0:
        return "None"
    score = (
        min(1.0, sample_size / 6.0) * 0.4 +
        agreement * 0.35 +
        avg_conf * 0.25
    )
    if score >= 0.66:
        return "High"
    if score >= 0.4:
        return "Medium"
    return "Low"


def _divergence_note(stock_score: float, sector_score: float) -> tuple[str, str]:
    """Return (label, human explanation) for the stock-vs-sector gap."""
    gap = stock_score - sector_score
    if abs(gap) < 6:
        return "In line", "Stock sentiment is moving in line with its sector."
    if gap >= 6:
        return "Outperforming", (
            "Stock sentiment is notably more positive than its sector - "
            "company-specific optimism beyond the sector trend."
        )
    return "Underperforming", (
        "Stock sentiment is notably weaker than its sector - company-specific "
        "concerns despite the broader sector mood."
    )


def _top_drivers(scored: list[dict], positive: bool, k: int = 3) -> list[dict]:
    key = (lambda s: s["score"]) if positive else (lambda s: -s["score"])
    ranked = sorted(scored, key=key, reverse=True)
    picked = []
    for s in ranked[:k]:
        # only include if it actually leans the requested way
        if positive and s["score"] <= 55:
            continue
        if not positive and s["score"] >= 45:
            continue
        picked.append({
            "title": s["title"][:120],
            "source": s["source"],
            "score": s["score"],
            "label": s["label"],
        })
    return picked


def aggregate(scored_stock: list[dict], scored_sector: list[dict],
              keywords: dict | None = None) -> dict:
    """
    Build the final sentiment result from scored stock and sector articles.
    """
    now = time.time()

    stock_score, _, stock_agree = _weighted_stats(scored_stock, now)
    sector_score, _, sector_agree = _weighted_stats(scored_sector, now)

    stock_n = len(scored_stock)
    sector_n = len(scored_sector)

    stock_avg_conf = (
        sum(s.get("confidence", 0.0) for s in scored_stock) / stock_n
        if stock_n else 0.0
    )

    div_label, div_note = _divergence_note(stock_score, sector_score)

    # Per-article breakdown for the UI (cap size)
    article_scores = [
        {
            "title": s["title"][:100],
            "source": s["source"],
            "score": s["score"],
            "sentiment": s["label"],
            "scope": s["scope"],
        }
        for s in sorted(scored_stock, key=lambda x: x.get("published_ts", 0), reverse=True)[:5]
    ]

    return {
        # ── legacy contract (drop-in) ────────────────────────────────────────
        "score": stock_score,
        "sentiment": _label(stock_score),
        "confidence": _confidence(stock_n, stock_agree, stock_avg_conf),
        "sample_size": stock_n,
        "article_scores": article_scores,
        "keywords": keywords or {},

        # ── new: sector + divergence ─────────────────────────────────────────
        "sector_sentiment": {
            "score": sector_score,
            "label": _label(sector_score),
            "sample_size": sector_n,
            "agreement": sector_agree,
        },
        "divergence": {
            "value": round(stock_score - sector_score, 2),
            "label": div_label,
            "note": div_note,
        },
        "reasoning": {
            "positive_drivers": _top_drivers(scored_stock, positive=True),
            "negative_drivers": _top_drivers(scored_stock, positive=False),
            "summary": _build_summary(
                stock_score, _label(stock_score), sector_score,
                _label(sector_score), div_label, stock_n
            ),
        },
        "engine": "finbert",
    }


def _build_summary(stock_score, stock_label, sector_score, sector_label,
                   div_label, n) -> str:
    if n == 0:
        return "No relevant news found to assess sentiment."
    return (
        f"Across {n} recent articles, stock sentiment is {stock_label.lower()} "
        f"({stock_score:.0f}/100) while the sector reads {sector_label.lower()} "
        f"({sector_score:.0f}/100) - {div_label.lower()}."
    )


def neutral_fallback(reason: str = "unavailable") -> dict:
    """Safe neutral result when scoring cannot run."""
    return {
        "score": 50.0, "sentiment": "Neutral", "confidence": "Low",
        "sample_size": 0, "article_scores": [], "keywords": {},
        "sector_sentiment": {"score": 50.0, "label": "Neutral",
                             "sample_size": 0, "agreement": 0.0},
        "divergence": {"value": 0.0, "label": "In line",
                       "note": "Insufficient data for sector comparison."},
        "reasoning": {"positive_drivers": [], "negative_drivers": [],
                      "summary": f"Sentiment analysis {reason}."},
        "engine": "fallback",
    }
