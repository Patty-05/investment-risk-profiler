"""
Sentiment Analysis Module
Analyzes news sentiment for the stock
"""
import re


# ✅ FIX: Expanded stopword list for keyword extraction
_KEYWORD_STOPWORDS = {
    'The', 'In', 'On', 'At', 'For', 'With', 'And', 'Or', 'But', 'To',
    'Of', 'By', 'As', 'An', 'Is', 'It', 'Its', 'Be', 'Was', 'Are',
    'Has', 'Had', 'Have', 'Not', 'From', 'After', 'This', 'That'
}

# ✅ FIX: Negation window — if one of these appears before a sentiment word,
#         flip its contribution
_NEGATIONS = {'not', 'no', 'never', "n't", 'neither', 'nor', 'without'}


class SentimentAnalyzer:
    """Analyzes sentiment from news articles using keyword scoring."""

    def __init__(self):
        # ✅ FIX: Removed generic short words ('up', 'down', 'high', 'low')
        #         that substring-match inside unrelated words.
        #         Words are now matched as whole tokens via regex (see _tokenize).
        self.positive_words = {
            'profit', 'gain', 'growth', 'strong', 'positive', 'rise', 'surge',
            'outperform', 'bullish', 'buy', 'upgrade', 'beat', 'exceed', 'success',
            'breakthrough', 'record', 'momentum', 'rally', 'optimism', 'jump',
            'soar', 'rebound', 'expansion', 'recovery', 'dividend', 'earnings',
            'revenue', 'acquisition', 'partnership', 'launch', 'innovation'
        }

        self.negative_words = {
            'loss', 'fall', 'decline', 'weak', 'negative', 'drop', 'plunge',
            'underperform', 'bearish', 'sell', 'downgrade', 'miss', 'disappoint',
            'concern', 'risk', 'slump', 'crash', 'uncertainty', 'pessimism',
            'worry', 'lawsuit', 'fraud', 'default', 'debt', 'layoff', 'recall',
            'investigation', 'penalty', 'fine', 'bankruptcy', 'warning'
        }

        # ✅ Boost/penalty words — kept SEPARATE from main lists to avoid double-counting
        self._strong_positive = {'surge', 'soar', 'jump', 'rally', 'rebound', 'breakthrough'}
        self._strong_negative  = {'crash', 'plunge', 'collapse', 'slump', 'bankruptcy', 'default'}

    # ── Text helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text):
        """Return a list of lowercase word tokens."""
        return re.findall(r"\b[a-z']+\b", text.lower())

    def _article_text(self, article):
        """
        ✅ FIX: Build text from keys that actually exist in the cleaned news dict
        from data_fetcher ('title', 'publisher').  'summary' kept for backwards
        compatibility in case it's present.
        """
        return ' '.join(filter(None, [
            article.get('title', ''),
            article.get('summary', ''),      # may be empty — that's fine
            article.get('publisher', ''),
        ]))

    # ── Scoring ─────────────────────────────────────────────────────────────

    def _calculate_text_sentiment(self, text):
        """
        Score a text string 0–100.

        Fixes applied:
        - Whole-word matching via tokenisation (no more substring hits)
        - Negation window: a negation within 2 tokens flips the word's sign
        - Proportional scoring (±5 per net word) instead of ±10 jumps
        - Strong-word bonus applied only once, after main scoring
        """
        if not text:
            return 50

        tokens = self._tokenize(text)
        if not tokens:
            return 50

        positive_count = 0
        negative_count = 0

        for i, token in enumerate(tokens):
            # Check for negation in a 2-token window before this word
            window_start = max(0, i - 2)
            negated = any(t in _NEGATIONS for t in tokens[window_start:i])

            if token in self.positive_words:
                if negated:
                    negative_count += 1   # "not profitable" → negative
                else:
                    positive_count += 1

            elif token in self.negative_words:
                if negated:
                    positive_count += 1   # "no loss" → positive
                else:
                    negative_count += 1

        net = positive_count - negative_count

        # ✅ FIX: ±5 per net word — much less explosive than ±10
        score = 50 + net * 5

        # Strong-word one-time bonus (only if not already counted above to avoid double)
        token_set = set(tokens)
        if token_set & self._strong_positive - self.positive_words:
            score += 8
        if token_set & self._strong_negative - self.negative_words:
            score -= 8

        return int(min(100, max(0, score)))

    def _get_sentiment_label(self, score):
        if score >= 70:   return 'Very Positive'
        elif score >= 60: return 'Positive'
        elif score >= 40: return 'Neutral'
        elif score >= 30: return 'Negative'
        else:             return 'Very Negative'

    # ── Public methods ───────────────────────────────────────────────────────

    def analyze_news_sentiment(self, news_articles):
        """
        Analyse sentiment from a list of news article dicts.

        Returns:
            dict with score (0–100), label, confidence, and per-article breakdown
        """
        if not news_articles:
            return {
                'score': 50, 'sentiment': 'Neutral',
                'confidence': 'Low', 'sample_size': 0,
                'article_scores': []
            }

        article_scores = []
        for article in news_articles:
            try:
                text  = self._article_text(article)
                score = self._calculate_text_sentiment(text)
                # ✅ Improve: include title so scores are debuggable
                article_scores.append({
                    'title': article.get('title', 'Unknown')[:80],
                    'score': score,
                    'sentiment': self._get_sentiment_label(score)
                })
            except Exception as e:
                # ✅ FIX: Don't let one bad article crash the whole pass
                print(f"⚠️ Sentiment scoring failed for article: {e}")
                continue

        if not article_scores:
            return {
                'score': 50, 'sentiment': 'Neutral',
                'confidence': 'Low', 'sample_size': 0,
                'article_scores': []
            }

        avg_score = sum(a['score'] for a in article_scores) / len(article_scores)
        n = len(article_scores)
        confidence = 'High' if n >= 5 else ('Medium' if n >= 3 else 'Low')

        return {
            'score': round(avg_score, 2),
            'sentiment': self._get_sentiment_label(avg_score),
            'confidence': confidence,
            'sample_size': n,
            'article_scores': article_scores[:5]   # top 5 for response size
        }

    def analyze_keywords(self, news_articles):
        """
        Extract top capitalised keywords from news titles.
        ✅ FIX: reads correct field names; expanded stopword filter.
        """
        keywords = {}
        for article in news_articles:
            try:
                # Only use title — 'summary' is not guaranteed to exist
                text = article.get('title', '')
                words = re.findall(r'\b[A-Z][a-z]{2,}\b', text)   # min 3 chars
                for word in words:
                    if word not in _KEYWORD_STOPWORDS:
                        keywords[word] = keywords.get(word, 0) + 1
            except Exception:
                continue

        sorted_kw = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_kw[:10])

    def get_sentiment_score(self, news_articles):
        """
        Main entry point called by StockRiskAnalyzer.
        ✅ FIX: wrapped in try/except so a total failure returns a safe neutral score.
        """
        try:
            sentiment = self.analyze_news_sentiment(news_articles)
            keywords  = self.analyze_keywords(news_articles or [])
            return {**sentiment, 'keywords': keywords}
        except Exception as e:
            print(f"❌ Sentiment analysis failed: {e}")
            return {
                'score': 50, 'sentiment': 'Neutral',
                'confidence': 'Low', 'sample_size': 0,
                'article_scores': [], 'keywords': {},
                'error': str(e)
            }