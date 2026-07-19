"""
Volumetric Analysis Module
Analyzes volume patterns and trends
"""
import pandas as pd
import numpy as np


class VolumetricAnalyzer:
    """Analyzes volume patterns and price-volume relationships"""

    def __init__(self, df):
        self.df = df
        self.data = df.copy()
        self._has_volume = self._check_volume()
        self._calculate_indicators()

    # ── Guards ─────────────────────────────────────────────────────────────

    def _check_volume(self):
        """
        ✅ NEW: Verify that a usable Volume column exists before any calculation.
        Returns False if Volume is missing, all-zero, or all-NaN.
        """
        if 'Volume' not in self.data.columns:
            print("⚠️ VolumetricAnalyzer: 'Volume' column missing — scores will be neutral")
            return False
        if self.data['Volume'].isna().all() or (self.data['Volume'] == 0).all():
            print("⚠️ VolumetricAnalyzer: 'Volume' column is all NaN/zero")
            return False
        return True

    # ── Indicator calculations (vectorised) ────────────────────────────────

    def _calculate_indicators(self):
        if not self._has_volume:
            return
        self._calculate_obv()
        self._calculate_volume_sma()
        self._calculate_price_volume_trend()
        self._calculate_acc_distribution()

    def _calculate_obv(self):
        """
        ✅ FIX: Fully vectorised — eliminates O(n) Python loop and .loc[] writes.
        Uses np.sign to assign +volume / -volume / 0, then cumsum().
        """
        direction = np.sign(self.data['Close'].diff().fillna(0))
        self.data['OBV'] = (direction * self.data['Volume']).cumsum()

    def _calculate_volume_sma(self):
        self.data['Volume_SMA_20'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_SMA_50'] = self.data['Volume'].rolling(window=50).mean()

    def _calculate_price_volume_trend(self):
        """
        ✅ FIX: Vectorised + division-by-zero guard (replace 0 prev-close with NaN).
        PVT = cumulative sum of (pct_change * volume).
        """
        prev_close = self.data['Close'].shift(1).replace(0, np.nan)
        pct_change = (self.data['Close'] - prev_close) / prev_close
        self.data['PVT'] = (pct_change.fillna(0) * self.data['Volume']).cumsum()

    def _calculate_acc_distribution(self):
        """
        ✅ FIX: Vectorised + division-by-zero guard for High == Low candles.
        CLV = ((Close-Low) - (High-Close)) / (High-Low)
        """
        hl_range = (self.data['High'] - self.data['Low']).replace(0, np.nan)
        clv = ((self.data['Close'] - self.data['Low']) -
               (self.data['High'] - self.data['Close'])) / hl_range
        clv = clv.fillna(0)
        self.data['AD'] = (clv * self.data['Volume']).cumsum()

    # ── Scoring ────────────────────────────────────────────────────────────

    def get_volume_score(self):
        """Calculate overall volume score (0–100)."""
        # ✅ Return neutral score immediately if volume data is unavailable
        if not self._has_volume:
            return {
                'score': 50,
                'signal': 'Neutral',
                'details': {'error': 'Volume data unavailable'},
            }

        latest = self.data.iloc[-1]
        recent = self.data.tail(5)

        vol_trend_score    = self._analyze_volume_trend(recent)
        obv_score          = self._analyze_obv(latest)
        price_vol_score    = self._analyze_price_volume_confirmation(recent)
        pattern_score      = self._analyze_volume_patterns(recent)

        total = (vol_trend_score['score'] + obv_score['score'] +
                 price_vol_score['score'] + pattern_score['score'])

        return {
            'score': round(min(100, max(0, total)), 2),
            'signal': self._get_volume_signal(total),
            'details': {
                'volume_trend':   vol_trend_score,
                'obv':            obv_score,
                'price_volume':   price_vol_score,
                'volume_patterns': pattern_score,
            }
        }

    def _analyze_volume_trend(self, recent):
        """Analyse volume trend over the recent window (0–35 pts)."""
        if len(recent) < 2:
            return {'score': 10, 'status': 'Insufficient data'}

        latest_vol = recent['Volume'].iloc[-1]
        avg_vol    = recent['Volume'].mean()

        # ✅ FIX: Guard against zero avg_vol (e.g. all-zero volume window)
        if avg_vol == 0:
            return {'score': 10, 'status': 'No volume data', 'change_pct': 0}

        pct = (latest_vol - avg_vol) / avg_vol * 100

        if pct > 50:
            score, status = 35, 'High Volume'
        elif pct > 20:
            score, status = 28, 'Above Average Volume'
        elif pct > 0:
            score, status = 20, 'Average Volume'
        elif pct > -20:
            score, status = 15, 'Below Average Volume'
        else:
            score, status = 8,  'Low Volume'

        return {'score': score, 'status': status, 'change_pct': round(pct, 2)}

    def _analyze_obv(self, latest):
        """Analyse On-Balance Volume trend (0–25 pts)."""
        if len(self.data) < 20 or 'OBV' not in self.data.columns:
            return {'score': 12, 'status': 'Insufficient data'}

        recent_obv = self.data['OBV'].tail(5).mean()
        prev_obv   = self.data['OBV'].tail(20).head(10).mean()

        if prev_obv == 0:
            obv_trend = 0.0
        else:
            obv_trend = ((recent_obv - prev_obv) / abs(prev_obv)) * 100

        if obv_trend > 5:
            score, status = 25, 'Accumulation'
        elif obv_trend > 0:
            score, status = 20, 'Slight Accumulation'
        elif obv_trend > -5:
            score, status = 15, 'Slight Distribution'
        else:
            score, status = 10, 'Distribution'

        return {'score': score, 'status': status, 'trend_pct': round(obv_trend, 2)}

    def _analyze_price_volume_confirmation(self, recent):
        """Check price/volume confirmation (0–25 pts)."""
        if len(recent) < 2:
            return {'score': 12, 'status': 'Insufficient data'}

        open_price = recent['Close'].iloc[0]
        avg_vol    = recent['Volume'].mean()

        # ✅ FIX: Guard both denominators
        price_change  = ((recent['Close'].iloc[-1] - open_price) / open_price * 100
                         if open_price != 0 else 0.0)
        volume_change = ((recent['Volume'].iloc[-1] - avg_vol) / avg_vol * 100
                         if avg_vol != 0 else 0.0)

        if price_change > 0 and volume_change > 0:
            score, status = 25, 'Bullish Confirmation'
        elif price_change < 0 and volume_change > 0:
            score, status = 10, 'Bearish Confirmation'
        elif abs(price_change) > 2 and volume_change < -20:
            score, status = 12, 'Weak Volume'
        else:
            score, status = 18, 'Neutral'

        return {
            'score': score, 'status': status,
            'price_change': round(price_change, 2),
            'volume_change': round(volume_change, 2)
        }

    def _analyze_volume_patterns(self, recent):
        """Detect bullish volume patterns (0–15 pts)."""
        if len(recent) < 5:
            return {'score': 8, 'status': 'Insufficient data'}

        up_days_with_vol = sum(
            1 for i in range(1, len(recent))
            if recent['Close'].iloc[i] > recent['Close'].iloc[i - 1]
            and recent['Volume'].iloc[i] > recent['Volume'].iloc[i - 1]
        )

        avg_vol    = recent['Volume'].mean()
        vol_spikes = int((recent['Volume'] > avg_vol * 1.5).sum()) if avg_vol > 0 else 0

        score = 8
        if up_days_with_vol >= 3:
            score += 4
        if vol_spikes >= 2:
            score += 3

        return {'score': min(15, score), 'status': 'Volume Pattern Analysis',
                'up_days_with_rising_volume': up_days_with_vol,
                'volume_spikes': vol_spikes}

    def _get_volume_signal(self, score):
        if score >= 75:   return 'Very Bullish'
        elif score >= 60: return 'Bullish'
        elif score >= 40: return 'Neutral'
        elif score >= 25: return 'Bearish'
        else:             return 'Very Bearish'