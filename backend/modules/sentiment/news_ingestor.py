"""
News Ingestor — Phase 1 of the enhanced sentiment pipeline.

Hybrid sourcing:
    • RSS base layer   — Google News RSS (stock + sector queries) and Yahoo's
                         per-ticker feed. Free, no API key, covers both scopes.
    • Scraping fallback— trafilatura pulls the full article body from the link
                         when a feed only provides a headline/summary.

On top of sourcing it does two things that matter for signal quality:
    • Relevancy filter — resolves the company's real name / sector / industry
                         from yfinance info, then scores each article so that
                         "Reliance the person" ≠ "RELIANCE the stock", and the
                         sector bucket does not fill with unrelated noise.
    • Scope tagging    — every article is tagged 'stock' or 'sector' so the
                         aggregator (Phase 3) can compute a divergence signal.

No heavy dependencies here (feedparser + trafilatura). FinBERT/torch arrive in
Phase 2. The output is a plain dict of normalised Article objects, ready to be
scored.
"""

from __future__ import annotations

import re
import json
import time
import html
import urllib.parse
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

try:
    import feedparser
except ImportError:  # pragma: no cover - dependency guard
    feedparser = None

try:
    import trafilatura
except ImportError:  # pragma: no cover - dependency guard
    trafilatura = None

try:
    import requests
except ImportError:  # pragma: no cover - dependency guard
    requests = None

_UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
_GNEWS_ARTICLE_MARK = "news.google.com/rss/articles/"


# ── Tuning knobs ────────────────────────────────────────────────────────────
_DEFAULT_MAX_PER_SCOPE = 12      # cap articles kept per scope after ranking
_BODY_FETCH_TOP_K = 6            # only scrape full bodies for the top-K by relevance
_BODY_FETCH_TIMEOUT = 8          # seconds per article body fetch
_RECENCY_WINDOW = "7d"           # Google News recency filter
_MIN_RELEVANCE = 0.15            # drop articles below this relevance score

# Generic words that must not count as "the company" when matching relevance.
_NAME_STOPWORDS = {
    "inc", "inc.", "corp", "corp.", "corporation", "company", "co", "co.",
    "ltd", "ltd.", "limited", "plc", "group", "holdings", "the", "and",
    "&", "class", "common", "stock", "shares", "sa", "ag", "nv",
}


@dataclass
class Article:
    """Normalised news article shared across the sentiment pipeline."""
    title: str
    summary: str = ""
    body: str = ""
    source: str = ""
    url: str = ""
    published_at: str | None = None          # ISO-8601 string
    scope: str = "stock"                      # 'stock' | 'sector'
    relevance: float = 0.0                    # 0–1
    _published_ts: float = field(default=0.0, repr=False)

    def text_for_scoring(self) -> str:
        """Prefer the full body; fall back to summary/title."""
        return (self.body or self.summary or self.title or "").strip()

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("_published_ts", None)
        return d


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _parse_date(entry) -> tuple[str | None, float]:
    """Return (iso_string, epoch_ts) from a feedparser entry, best-effort."""
    # feedparser exposes a parsed struct_time when it can
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if st:
            try:
                ts = time.mktime(st)
                return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(), ts
            except (OverflowError, ValueError):
                pass
    # fall back to raw RFC-822 string
    raw = entry.get("published") or entry.get("updated")
    if raw:
        try:
            dt = parsedate_to_datetime(raw)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.isoformat(), dt.timestamp()
        except (TypeError, ValueError):
            pass
    return None, 0.0


def _clean_html(text: str) -> str:
    """Strip tags/entities from RSS summary blurbs."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_google_news_url(url: str, timeout: int = 10) -> str:
    """
    Resolve a Google News RSS redirect stub to the real publisher URL.

    Google News RSS <link> values look like
        https://news.google.com/rss/articles/CBMi...<base64>...
    which are *not* HTTP redirects — the real URL is obtained via Google's
    internal `batchexecute` endpoint using a per-article signature + timestamp
    embedded in the article page. This is best-effort: any failure returns the
    original URL so the caller can still fall back to headline/summary text.
    """
    if requests is None or _GNEWS_ARTICLE_MARK not in url:
        return url
    try:
        art_id = url.split("/articles/")[-1].split("?")[0]
        page = requests.get(url, timeout=timeout, headers=_UA)
        sig = re.search(r'data-n-a-sg="([^"]+)"', page.text)
        ts = re.search(r'data-n-a-ts="([^"]+)"', page.text)
        if not (sig and ts):
            return url

        inner = json.dumps([
            "garturlreq",
            [["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1,
              None, None, None, None, None, 0, 1],
             "X", "X", 1, [1, 1, 1], 1, 1, None, 0, 0, None, 0],
            art_id, int(ts.group(1)), sig.group(1),
        ])
        freq = json.dumps([[["Fbv4je", inner, None, "generic"]]])
        resp = requests.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute",
            data="f.req=" + urllib.parse.quote(freq),
            headers={**_UA, "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
            timeout=timeout,
        )
        for line in resp.text.split("\n"):
            if "garturlres" in line:
                payload = json.loads(json.loads(line)[0][2])
                real = payload[1]
                if isinstance(real, str) and real.startswith("http"):
                    return real
    except Exception as e:
        print(f"[sentiment] Google News URL decode failed for {url[:50]}: {e}")
    return url


class NewsIngestor:
    """
    Fetch and normalise news for a ticker across stock and sector scopes.

    Usage:
        ing = NewsIngestor("AAPL", stock_info=info)
        result = ing.fetch()
        # result = {
        #     'sector': 'Technology',
        #     'stock_articles':  [Article, ...],
        #     'sector_articles': [Article, ...],
        # }
    """

    def __init__(
        self,
        ticker: str,
        stock_info: dict | None = None,
        max_per_scope: int = _DEFAULT_MAX_PER_SCOPE,
        fetch_bodies: bool = True,
    ):
        self.ticker = (ticker or "").upper().strip()
        self.info = stock_info or {}
        self.max_per_scope = max_per_scope
        self.fetch_bodies = fetch_bodies

        # Resolve identity terms used for relevancy scoring / query building
        self.company_name = self._resolve_company_name()
        self.sector = (self.info.get("sector") or "").strip()
        self.industry = (self.info.get("industry") or "").strip()

        # Relevance term sets (lowercased tokens)
        self._stock_terms = self._build_stock_terms()
        self._sector_terms = self._build_sector_terms()

    # ── Identity resolution ─────────────────────────────────────────────────

    def _resolve_company_name(self) -> str:
        for key in ("longName", "shortName", "displayName"):
            name = self.info.get(key)
            if name:
                return str(name).strip()
        return self.ticker

    def _significant_name_tokens(self) -> list[str]:
        """Company-name tokens with generic corporate suffixes removed."""
        toks = [t for t in _tokenize(self.company_name) if t not in _NAME_STOPWORDS]
        # keep the ticker itself as a strong term too
        if self.ticker:
            toks.append(self.ticker.lower())
        return list(dict.fromkeys(toks))  # dedupe, preserve order

    def _build_stock_terms(self) -> set[str]:
        return set(self._significant_name_tokens())

    def _build_sector_terms(self) -> set[str]:
        terms: set[str] = set()
        for source in (self.sector, self.industry):
            terms.update(_tokenize(source))
        terms.discard("")
        return terms

    # ── Query construction ──────────────────────────────────────────────────

    def _google_news_rss(self, query: str) -> str:
        q = urllib.parse.quote_plus(f"{query} when:{_RECENCY_WINDOW}")
        return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"

    def _yahoo_ticker_rss(self) -> str:
        return (
            "https://feeds.finance.yahoo.com/rss/2.0/headline"
            f"?s={urllib.parse.quote(self.ticker)}&region=US&lang=en-US"
        )

    def _stock_feed_urls(self) -> list[str]:
        urls = [self._yahoo_ticker_rss()]
        # Prefer the real company name for the news query; fall back to ticker
        name_q = self.company_name if self.company_name != self.ticker else self.ticker
        urls.append(self._google_news_rss(f'"{name_q}" stock'))
        return urls

    def _sector_feed_urls(self) -> list[str]:
        if not (self.sector or self.industry):
            return []
        label = self.industry or self.sector
        return [self._google_news_rss(f"{label} sector")]

    # ── Relevancy scoring ───────────────────────────────────────────────────

    def _relevance(self, text: str, terms: set[str]) -> float:
        """Fraction of distinct relevance terms that appear in the text (0–1)."""
        if not terms:
            return 0.0
        toks = set(_tokenize(text))
        if not toks:
            return 0.0
        hits = len(terms & toks)
        return hits / len(terms)

    # ── Feed parsing ────────────────────────────────────────────────────────

    def _parse_feed(self, url: str, scope: str, terms: set[str]) -> list[Article]:
        if feedparser is None:
            print("[sentiment] feedparser not installed - skipping RSS ingestion")
            return []
        try:
            parsed = feedparser.parse(url)
        except Exception as e:  # network / parse failure is non-fatal
            print(f"[sentiment] RSS fetch failed ({scope}) for {url[:60]}: {e}")
            return []

        articles: list[Article] = []
        for entry in parsed.get("entries", []):
            title = _clean_html(entry.get("title", ""))
            summary = _clean_html(entry.get("summary", "") or entry.get("description", ""))
            if not title:
                continue

            iso, ts = _parse_date(entry)
            source = ""
            src = entry.get("source")
            if isinstance(src, dict):
                source = src.get("title", "") or ""
            source = source or parsed.feed.get("title", "")

            rel = self._relevance(f"{title} {summary}", terms)
            articles.append(
                Article(
                    title=title,
                    summary=summary,
                    source=source,
                    url=entry.get("link", ""),
                    published_at=iso,
                    scope=scope,
                    relevance=round(rel, 3),
                    _published_ts=ts,
                )
            )
        return articles

    # ── Body scraping fallback ──────────────────────────────────────────────

    def _fetch_body(self, url: str) -> str:
        if not url or trafilatura is None:
            return ""
        try:
            # Google News links are redirect stubs — resolve to the real
            # publisher URL first, otherwise trafilatura extracts nothing.
            if _GNEWS_ARTICLE_MARK in url:
                url = _decode_google_news_url(url, timeout=_BODY_FETCH_TIMEOUT)
                if _GNEWS_ARTICLE_MARK in url:
                    return ""  # decode failed — no scrapable body

            downloaded = trafilatura.fetch_url(url)
            if not downloaded:
                return ""
            extracted = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                no_fallback=False,
            )
            return (extracted or "").strip()
        except Exception as e:
            print(f"[sentiment] Body fetch failed for {url[:60]}: {e}")
            return ""

    # ── Dedupe + rank ───────────────────────────────────────────────────────

    @staticmethod
    def _dedupe(articles: list[Article]) -> list[Article]:
        seen: set[str] = set()
        out: list[Article] = []
        for a in articles:
            key = re.sub(r"\W+", "", a.title.lower())[:80]
            if key and key not in seen:
                seen.add(key)
                out.append(a)
        return out

    def _rank_and_trim(self, articles: list[Article]) -> list[Article]:
        # Rank by relevance first, then recency
        articles.sort(key=lambda a: (a.relevance, a._published_ts), reverse=True)
        kept = [a for a in articles if a.relevance >= _MIN_RELEVANCE]
        # If the relevance filter is too aggressive (sparse metadata), keep top few
        if not kept and articles:
            kept = articles[: max(3, self.max_per_scope // 2)]
        return kept[: self.max_per_scope]

    # ── Orchestration ───────────────────────────────────────────────────────

    def _collect_scope(self, urls: list[str], scope: str, terms: set[str]) -> list[Article]:
        collected: list[Article] = []
        for url in urls:
            collected.extend(self._parse_feed(url, scope, terms))
        collected = self._dedupe(collected)
        collected = self._rank_and_trim(collected)

        if self.fetch_bodies:
            for a in collected[:_BODY_FETCH_TOP_K]:
                body = self._fetch_body(a.url)
                if body:
                    a.body = body
        return collected

    def fetch(self) -> dict:
        """Run the full ingestion and return normalised articles per scope."""
        stock_articles = self._collect_scope(
            self._stock_feed_urls(), "stock", self._stock_terms
        )
        sector_articles = self._collect_scope(
            self._sector_feed_urls(), "sector", self._sector_terms
        )

        return {
            "ticker": self.ticker,
            "company_name": self.company_name,
            "sector": self.sector,
            "industry": self.industry,
            "stock_articles": stock_articles,
            "sector_articles": sector_articles,
            "counts": {
                "stock": len(stock_articles),
                "sector": len(sector_articles),
            },
        }
