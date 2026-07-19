"""
Sector Comparison Module
Compares stock performance with sector peers and market index
"""
import pandas as pd
import numpy as np
import yfinance as yf


# ✅ Sector ETF map for US stocks (SPDR sector ETFs)
_US_SECTOR_ETFS = {
    'Technology':             'XLK',
    'Financial Services':     'XLF',
    'Healthcare':             'XLV',
    'Consumer Cyclical':      'XLY',
    'Consumer Defensive':     'XLP',
    'Industrials':            'XLI',
    'Energy':                 'XLE',
    'Utilities':              'XLU',
    'Real Estate':            'XLRE',
    'Basic Materials':        'XLB',
    'Communication Services': 'XLC',
}

# ✅ Sector ETF map for Indian stocks (NSE-listed)
_IN_SECTOR_ETFS = {
    'Technology':         '^NSEIT',
    'Financial Services': '^NSEBANK',
    'Healthcare':         '^CNXPHARMA',
    'Energy':             '^CNXENERGY',
    'Consumer Cyclical':  '^CNXAUTO',
    'Basic Materials':    '^CNXMETAL',
}

# ✅ Market index by exchange suffix
_MARKET_INDEX = {
    'NS':  '^NSEI',    # NSE (India)
    'BO':  '^BSESN',   # BSE (India)
    'L':   '^FTSE',    # London
    'TO':  '^GSPTSE',  # Toronto
    'HK':  '^HSI',     # Hong Kong
}
_DEFAULT_MARKET_INDEX = '^GSPC'   # S&P 500 for everything else


def _get_exchange_suffix(ticker: str) -> str:
    """Extract the exchange suffix from a ticker (e.g. 'RELIANCE.NS' → 'NS')."""
    parts = ticker.upper().split('.')
    return parts[-1] if len(parts) > 1 else ''


def _calc_period_return(ticker_obj, period='3mo') -> float | None:
    """
    Return the percentage price change over `period` for a yfinance Ticker object.
    Returns None on failure instead of 0 so callers can distinguish missing data.
    """
    try:
        hist = ticker_obj.history(period=period)
        if hist is None or len(hist) < 2:
            return None
        start = hist['Close'].iloc[0]
        end   = hist['Close'].iloc[-1]
        if start == 0:
            return None
        return float((end - start) / start * 100)
    except Exception as e:
        print(f"⚠️ _calc_period_return failed: {e}")
        return None


class SectorComparator:
    """Compares stock performance with sector and market index."""

    def __init__(self, ticker, sector=None, stock_info=None):
        """
        Args:
            ticker:     Stock ticker symbol.
            sector:     Sector name (optional — fetched from info if not supplied).
            stock_info: Pre-fetched info dict from StockDataFetcher (avoids duplicate API call).
        """
        self.ticker      = ticker.upper().strip()
        self.sector      = sector
        self._suffix     = _get_exchange_suffix(self.ticker)
        self.stock       = yf.Ticker(self.ticker)
        # ✅ FIX: Accept pre-fetched info so we don't repeat the API call made in StockRiskAnalyzer
        self._stock_info = stock_info or {}

    def get_sector_comparison(self):
        """Compare stock vs sector ETF and market index. Returns a 0–100 score."""
        try:
            # Use cached info if available, otherwise fetch once
            if not self._stock_info:
                self._stock_info = self.stock.info or {}

            if not self._stock_info:
                # ✅ FIX: Return score=50 (neutral stub) not 0, with error flag set
                #         so stock_analyzer._run_safely() can detect it
                return {'score': 50, 'error': 'Unable to fetch stock information'}

            self.sector = self._stock_info.get('sector') or self.sector or 'Unknown'

            stock_perf  = _calc_period_return(self.stock)
            sector_perf = self._get_sector_performance()
            market_perf = self._get_market_performance()

            # If stock data itself is missing we can't score anything meaningful
            if stock_perf is None:
                return {'score': 50, 'error': 'Insufficient stock price data'}

            vs_sector_score = self._calculate_relative_score(
                stock_perf, sector_perf if sector_perf is not None else stock_perf
            )
            vs_market_score = self._calculate_relative_score(
                stock_perf, market_perf if market_perf is not None else stock_perf
            )

            # If sector data was a stub, only use market score
            if sector_perf is None:
                overall_score = vs_market_score
            else:
                overall_score = (vs_sector_score + vs_market_score) / 2

            return {
                'score': round(overall_score, 2),
                'vs_sector': {
                    'score':           vs_sector_score,
                    'stock_perf':      round(stock_perf, 2),
                    'sector_perf':     round(sector_perf, 2) if sector_perf is not None else None,
                    'outperformance':  round(stock_perf - (sector_perf or stock_perf), 2),
                    'sector_ticker':   self._get_sector_etf()
                },
                'vs_market': {
                    'score':          vs_market_score,
                    'market_perf':    round(market_perf, 2) if market_perf is not None else None,
                    'outperformance': round(stock_perf - (market_perf or stock_perf), 2),
                    'market_index':   self._get_market_index_ticker()
                },
                'sector': self.sector
            }

        except Exception as e:
            print(f"❌ SectorComparator.get_sector_comparison() failed: {e}")
            return {'score': 50, 'error': str(e)}

    # ── Helpers ─────────────────────────────────────────────────────────────

    def _get_sector_etf(self) -> str | None:
        """
        ✅ FIX: Choose sector ETF based on exchange suffix.
        Indian tickers (.NS/.BO) use NSE indices; US tickers use SPDR ETFs.
        """
        if self._suffix in ('NS', 'BO'):
            return _IN_SECTOR_ETFS.get(self.sector)
        return _US_SECTOR_ETFS.get(self.sector)

    def _get_market_index_ticker(self) -> str:
        """✅ FIX: Return the appropriate market index for the stock's exchange."""
        return _MARKET_INDEX.get(self._suffix, _DEFAULT_MARKET_INDEX)

    def _get_sector_performance(self) -> float | None:
        """
        ✅ FIX: Fetch the actual sector ETF/index return instead of hardcoding 5.0.
        Returns None if no ETF mapping exists for this sector/exchange.
        """
        etf = self._get_sector_etf()
        if not etf:
            print(f"⚠️ No sector ETF mapped for sector='{self.sector}' suffix='{self._suffix}' — skipping sector score")
            return None
        return _calc_period_return(yf.Ticker(etf))

    def _get_market_performance(self) -> float | None:
        """✅ FIX: Use the exchange-appropriate market index."""
        idx = self._get_market_index_ticker()
        return _calc_period_return(yf.Ticker(idx))

    def _get_stock_performance(self, period='3mo') -> float:
        """Wrapper kept for backwards compatibility with compare_with_rivals()."""
        result = _calc_period_return(self.stock, period)
        return result if result is not None else 0.0

    def _calculate_relative_score(self, stock_perf: float, benchmark_perf: float) -> float:
        """Score 0–100 based on outperformance vs a benchmark."""
        outperf = stock_perf - benchmark_perf
        if outperf > 20:   return 100
        elif outperf > 15: return 95
        elif outperf > 10: return 85
        elif outperf > 5:  return 75
        elif outperf > 0:  return 60
        elif outperf > -5: return 50
        elif outperf > -10: return 40
        elif outperf > -15: return 25
        elif outperf > -20: return 15
        else:               return 5

    # ── Rivals ───────────────────────────────────────────────────────────────

    def get_rivals_list(self) -> list:
        """
        ✅ FIX: Warns caller this is not implemented instead of silently returning [].
        """
        print(f"⚠️ get_rivals_list() not implemented for {self.ticker} — returning []")
        return []

    def compare_with_rivals(self, rival_tickers: list) -> dict:
        """Compare stock performance with specific rival tickers."""
        stock_perf = self._get_stock_performance()
        results = []

        for ticker in rival_tickers:
            try:
                rival_obj  = yf.Ticker(ticker)
                rival_perf = _calc_period_return(rival_obj)
                if rival_perf is None:
                    print(f"⚠️ Could not fetch performance for rival {ticker}")
                    continue
                results.append({
                    'ticker':      ticker,
                    'performance': round(rival_perf, 2),
                    'vs_stock':    round(stock_perf - rival_perf, 2)
                })
            except Exception as e:
                # ✅ FIX: Log instead of bare pass
                print(f"⚠️ Rival comparison failed for {ticker}: {e}")

        avg_perf       = sum(r['performance'] for r in results) / len(results) if results else 0.0
        avg_comparison = stock_perf - avg_perf

        return {
            'stock_performance':        round(stock_perf, 2),
            'rivals':                   results,
            'average_rival_performance': round(avg_perf, 2),
            'average_outperformance':   round(avg_comparison, 2)
        }

    def _get_stock_performance_per_ticker(self, stock_obj) -> float:
        result = _calc_period_return(stock_obj)
        return result if result is not None else 0.0