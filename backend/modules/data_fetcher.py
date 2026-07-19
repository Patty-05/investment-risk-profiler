"""
Data Fetcher Module for Stock Data
Enhanced with validation, fallback, and error handling
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import time

# Periods where < 30 rows is expected and valid — don't trigger fallback
SHORT_PERIODS = {'1d', '5d', '1wk', '1mo'}


class StockDataFetcher:
    """Fetches stock data from Yahoo Finance API"""

    def __init__(self, ticker, period='1y'):
        self.ticker = ticker.upper().strip()
        self.period = period
        self.stock = yf.Ticker(self.ticker)
        self._cached_hist = None  # ✅ NEW: cache to avoid double fetches

    def get_historical_data(self):
        """Get historical price data with validation + fallback"""

        # ✅ NEW: Return cached result if already fetched
        if self._cached_hist is not None:
            return self._cached_hist

        try:
            hist = self.stock.history(period=self.period)

            # ✅ FIX: Only trigger fallback for long periods where < 30 rows means bad data
            if (hist is None or hist.empty or
                    (len(hist) < 30 and self.period not in SHORT_PERIODS)):

                print(f"⚠️ Insufficient data for period '{self.period}', trying fallback...")

                # ✅ FIX: Fallback to a longer version of self.period, not hardcoded "2y"
                fallback_period = '2y' if self.period in ('6mo', '1y') else '1y'
                hist = self.stock.history(period=fallback_period)

            if hist is None or hist.empty:
                print(f"❌ No historical data available for {self.ticker}")
                return None

            self._cached_hist = hist  # cache it
            return hist

        except Exception as e:
            print(f"❌ Error fetching historical data for {self.ticker}: {e}")
            return None

    def get_current_info(self):
        """Get current stock information"""
        try:
            info = self.stock.info

            # ✅ FIX: yfinance returns {} for invalid tickers — check for meaningful key
            if not info or 'regularMarketPrice' not in info and 'currentPrice' not in info:
                print(f"⚠️ No valid stock info returned for {self.ticker}")
                return None

            return info

        except Exception as e:
            print(f"❌ Error fetching current info for {self.ticker}: {e}")
            return None

    def get_financials(self):
        """Get financial statements"""
        try:
            income_stmt = self.stock.income_stmt
            balance_sheet = self.stock.balance_sheet
            cashflow = self.stock.cashflow

            # ✅ FIX: Return partial data with flags so caller knows what's missing
            result = {}
            result['income_stmt'] = income_stmt if income_stmt is not None and not income_stmt.empty else None
            result['balance_sheet'] = balance_sheet if balance_sheet is not None and not balance_sheet.empty else None
            result['cashflow'] = cashflow if cashflow is not None and not cashflow.empty else None
            result['partial'] = any(v is None for v in result.values())

            return result

        except Exception as e:
            print(f"❌ Error fetching financials for {self.ticker}: {e}")
            # ✅ FIX: Return structured empty result instead of None
            return {
                'income_stmt': None,
                'balance_sheet': None,
                'cashflow': None,
                'partial': True,
                'error': str(e)
            }

    def get_news(self, limit=10):
        """Get recent news about the stock, cleaned for JSON safety"""
        try:
            news = self.stock.news

            if not news:
                return []

            cleaned = []
            for item in news[:limit]:
                # ✅ FIX: Convert epoch timestamp to ISO string, drop unsafe fields
                cleaned.append({
                    'title': item.get('title', ''),
                    'publisher': item.get('publisher', ''),
                    'link': item.get('link', ''),
                    'published_at': datetime.fromtimestamp(
                        item['providerPublishTime']
                    ).isoformat() if item.get('providerPublishTime') else None,
                    'type': item.get('type', ''),
                    'thumbnail': (
                        item.get('thumbnail', {})
                        .get('resolutions', [{}])[0]
                        .get('url', None)
                    ) if item.get('thumbnail') else None
                })

            return cleaned

        except Exception as e:
            print(f"❌ Error fetching news for {self.ticker}: {e}")
            return []

    def get_recommendations(self):
        """Get analyst recommendations"""
        try:
            rec = self.stock.recommendations

            if rec is None or rec.empty:
                return None

            return rec

        except Exception as e:
            print(f"❌ Error fetching recommendations for {self.ticker}: {e}")
            return None

    def validate_ticker(self):
        """
        Validate if ticker is real and has tradeable data.
        Reuses cached history if already fetched.
        """
        try:
            # ✅ FIX: Reuse cache — avoids a redundant API call when app.py
            # calls validate_ticker() then immediately calls get_historical_data()
            if self._cached_hist is not None:
                return not self._cached_hist.empty

            hist = self.stock.history(period='5d')

            if hist is None or hist.empty:
                return False

            # ✅ FIX: Also check info has a real symbol (catches delisted/junk tickers)
            info = self.stock.info
            if not info or info.get('quoteType') is None:
                return False

            return True

        except Exception as e:
            # ✅ FIX: Log the actual error instead of swallowing silently
            print(f"⚠️ Ticker validation failed for {self.ticker}: {e}")
            return False

    def get_sector_data(self, sector):
        """
        Get peer tickers for sector comparison.
        ✅ FIX: Warns caller this is not implemented instead of silently returning []
        """
        print(f"⚠️ get_sector_data() not implemented — returning empty list for sector: {sector}")
        return []