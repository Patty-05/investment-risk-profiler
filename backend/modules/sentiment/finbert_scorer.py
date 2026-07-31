"""
FinBERT Scorer — Phase 2 of the enhanced sentiment pipeline.

Wraps the finance-tuned transformer ``ProsusAI/finbert`` to score article text
as positive / negative / neutral with calibrated probabilities, and maps that
onto the 0–100 scale the rest of the app already uses.

Design notes:
    • The model + tokenizer are expensive to load (~400MB, several seconds) so
      they are loaded once, lazily, and cached at module level.
    • FinBERT's input limit is 512 tokens. Long article bodies are split into
      word-chunks and scored independently, then combined as a length-weighted
      average of class probabilities.
    • 0–100 mapping:  score = 50 + (P(pos) − P(neg)) * 50
      → all-positive ≈ 100, all-negative ≈ 0, purely neutral ≈ 50.
    • torch/transformers are imported lazily so the rest of the app (and Phase 1
      ingestion) keep working even when the heavy deps are not installed.
"""

from __future__ import annotations

import re

_MODEL_NAME = "ProsusAI/finbert"
_MAX_WORDS_PER_CHUNK = 350      # keeps each chunk comfortably under 512 tokens
_MAX_CHUNKS = 6                 # cap work per article (long bodies get truncated)
_MAX_CHARS_PER_ARTICLE = 6000   # hard cap before chunking, for latency

# Module-level singletons (populated on first use)
_pipeline = None
_load_error = None


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]


def _chunk_text(text: str) -> list[str]:
    """Split text into <=_MAX_WORDS_PER_CHUNK word chunks, sentence-aware."""
    text = text[:_MAX_CHARS_PER_ARTICLE]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for sentence in _split_sentences(text):
        words = sentence.split()
        if count + len(words) > _MAX_WORDS_PER_CHUNK and current:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.extend(words)
        count += len(words)
        if len(chunks) >= _MAX_CHUNKS:
            break
    if current and len(chunks) < _MAX_CHUNKS:
        chunks.append(" ".join(current))
    return chunks or ([text] if text.strip() else [])


def _load_pipeline():
    """Lazily construct and cache the FinBERT pipeline. Returns None on failure."""
    global _pipeline, _load_error
    if _pipeline is not None or _load_error is not None:
        return _pipeline
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            pipeline,
        )
        tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(_MODEL_NAME)
        _pipeline = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
            truncation=True,
            max_length=512,
            top_k=None,          # return scores for all classes
        )
        print("[sentiment] FinBERT pipeline loaded")
    except Exception as e:  # torch/transformers missing, or download failed
        _load_error = str(e)
        print(f"[sentiment] FinBERT unavailable ({e}); scorer will report is_available=False")
    return _pipeline


class FinBertScorer:
    """Scores text and Article objects with FinBERT."""

    def __init__(self):
        self._pipe = _load_pipeline()

    @property
    def is_available(self) -> bool:
        return self._pipe is not None

    # ── Core scoring ─────────────────────────────────────────────────────────

    def _probs_for_chunk(self, chunk: str) -> dict:
        """Return {'positive':p, 'negative':p, 'neutral':p} for one chunk."""
        raw = self._pipe(chunk)
        # transformers may return [[{...}, ...]] or [{...}, ...] depending on ver
        scores = raw[0] if raw and isinstance(raw[0], list) else raw
        out = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for item in scores:
            out[item["label"].lower()] = float(item["score"])
        return out

    def score_text(self, text: str) -> dict:
        """
        Score a text string.

        Returns dict:
            score       0–100
            label       Positive/Negative/Neutral (+ Very variants)
            probs       {positive, negative, neutral} length-weighted average
            confidence  max class prob (0–1)
            chunks      how many chunks were scored
        """
        text = (text or "").strip()
        if not self.is_available or not text:
            return {
                "score": 50.0, "label": "Neutral",
                "probs": {"positive": 0.0, "negative": 0.0, "neutral": 1.0},
                "confidence": 0.0, "chunks": 0,
            }

        chunks = _chunk_text(text)
        agg = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        total_w = 0.0
        for chunk in chunks:
            w = max(1, len(chunk.split()))
            probs = self._probs_for_chunk(chunk)
            for k in agg:
                agg[k] += probs.get(k, 0.0) * w
            total_w += w

        if total_w > 0:
            for k in agg:
                agg[k] /= total_w

        score = 50.0 + (agg["positive"] - agg["negative"]) * 50.0
        score = round(max(0.0, min(100.0, score)), 2)
        confidence = round(max(agg.values()), 3)

        return {
            "score": score,
            "label": self._label(score),
            "probs": {k: round(v, 3) for k, v in agg.items()},
            "confidence": confidence,
            "chunks": len(chunks),
        }

    @staticmethod
    def _label(score: float) -> str:
        if score >= 70:   return "Very Positive"
        if score >= 58:   return "Positive"
        if score >= 42:   return "Neutral"
        if score >= 30:   return "Negative"
        return "Very Negative"

    # ── Article batch scoring ────────────────────────────────────────────────

    def score_articles(self, articles: list) -> list[dict]:
        """
        Score a list of Article objects. Returns a list of per-article result
        dicts (does not mutate the Article), each carrying the source relevance,
        recency timestamp, and title for downstream aggregation/reasoning.
        """
        results = []
        for a in articles:
            try:
                scored = self.score_text(a.text_for_scoring())
            except Exception as e:
                print(f"[sentiment] FinBERT scoring failed for '{a.title[:40]}': {e}")
                continue
            results.append({
                "title": a.title,
                "source": a.source,
                "url": a.url,
                "scope": a.scope,
                "relevance": a.relevance,
                "published_at": a.published_at,
                "published_ts": a._published_ts,
                "has_body": bool(a.body),
                **scored,
            })
        return results
