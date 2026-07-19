"""
Technical Analysis Module
Implements various technical indicators and trading strategies for stock analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime
from scipy.signal import argrelextrema
import warnings
warnings.filterwarnings('ignore')


def _safe_call(name, fn):
    """Run fn() and return None on any exception, logging the error."""
    try:
        return fn()
    except Exception as e:
        print(f"⚠️ {name} failed: {e}")
        return None


def _to_date_str(idx_val):
    """Safely convert a DataFrame index value to an ISO date string."""
    try:
        if hasattr(idx_val, 'strftime'):
            return idx_val.strftime('%Y-%m-%d')
        return str(idx_val)
    except Exception:
        return str(idx_val)


class TechnicalAnalyzer:
    """Analyzes stock based on technical indicators"""

    def __init__(self, df):
        self.df = df
        self.data = df.copy()
        self._calculate_indicators()

    def _calculate_indicators(self):
        self._calculate_moving_averages()
        self._calculate_rsi()
        self._calculate_macd()
        self._calculate_bollinger_bands()
        self._calculate_support_resistance()
        self._calculate_stochastics()
        self._calculate_parabolic_sar()
        self._calculate_supertrend()
        self._calculate_vwap()
        self._calculate_heiken_ashi()

    # ── Core indicators ────────────────────────────────────────────────────

    def _calculate_moving_averages(self):
        self.data['SMA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['EMA_12'] = self.data['Close'].ewm(span=12, adjust=False).mean()
        self.data['EMA_26'] = self.data['Close'].ewm(span=26, adjust=False).mean()

    def _calculate_rsi(self, period=14):
        """✅ FIX: Guard against division by zero when loss == 0."""
        delta = self.data['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        # Replace 0 loss with a tiny epsilon to avoid inf/NaN
        loss = loss.replace(0, 1e-10)
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))

    def _calculate_macd(self):
        self.data['MACD'] = self.data['EMA_12'] - self.data['EMA_26']
        self.data['MACD_Signal'] = self.data['MACD'].ewm(span=9, adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']

    def _calculate_bollinger_bands(self, period=20, std_dev=2):
        self.data['BB_Middle'] = self.data['Close'].rolling(window=period).mean()
        rolling_std = self.data['Close'].rolling(window=period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * std_dev)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * std_dev)

    def _calculate_support_resistance(self):
        window = 20
        self.data['Support'] = self.data['Low'].rolling(window=window).min()
        self.data['Resistance'] = self.data['High'].rolling(window=window).max()

    def _calculate_stochastics(self, k_period=14, d_period=3):
        """✅ FIX: Guard against division by zero on flat price periods."""
        if len(self.data) < k_period:
            return

        low_min = self.data['Low'].rolling(window=k_period).min()
        high_max = self.data['High'].rolling(window=k_period).max()
        denom = (high_max - low_min).replace(0, np.nan)  # NaN where range is 0

        self.data['Stoch_K'] = 100 * ((self.data['Close'] - low_min) / denom)
        self.data['Stoch_D'] = self.data['Stoch_K'].rolling(window=d_period).mean()
        self.data['Stoch_Oversold'] = (self.data['Stoch_K'] < 20) & (self.data['Stoch_D'] < 20)
        self.data['Stoch_Overbought'] = (self.data['Stoch_K'] > 80) & (self.data['Stoch_D'] > 80)

    def _calculate_vwap(self):
        if 'Volume' not in self.data.columns:
            return
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        pv = typical_price * self.data['Volume']
        cum_vol = self.data['Volume'].cumsum().replace(0, np.nan)
        self.data['VWAP'] = pv.cumsum() / cum_vol

    def _calculate_heiken_ashi(self):
        """✅ FIX: Vectorised ha_open — no slow Python loop, no chained-iloc warnings."""
        if len(self.data) < 2:
            return

        ha_close = (self.data['Open'] + self.data['High'] +
                    self.data['Low'] + self.data['Close']) / 4

        # Vectorised: shift and average (first row seeded from OHLC average)
        seed = (self.data['Open'].iloc[0] + self.data['Close'].iloc[0]) / 2
        ha_open_vals = [seed]
        for i in range(1, len(self.data)):
            ha_open_vals.append((ha_open_vals[-1] + ha_close.iloc[i - 1]) / 2)
        ha_open = pd.Series(ha_open_vals, index=self.data.index, dtype=float)

        ha_high = pd.concat([self.data['High'], ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([self.data['Low'], ha_open, ha_close], axis=1).min(axis=1)

        self.data['HA_Open'] = ha_open
        self.data['HA_High'] = ha_high
        self.data['HA_Low'] = ha_low
        self.data['HA_Close'] = ha_close
        self.data['HA_Body'] = (ha_close - ha_open).abs()
        self.data['HA_Trend'] = (ha_close > ha_open).map({True: 'Bullish', False: 'Bearish'})

    # ── Scoring ────────────────────────────────────────────────────────────

    def get_technical_score(self):
        """
        ✅ FIX: max_score is now a fixed constant (100 points total) so the
        scaling is always correct regardless of which RSI branch fires.
        NaN indicators are handled before comparison.
        """
        latest = self.data.iloc[-1]
        prev = self.data.iloc[-2] if len(self.data) > 1 else latest

        score = 0

        # ── RSI (25 pts) ──────────────────────────────────────────────────
        rsi = latest.get('RSI', np.nan)
        if pd.isna(rsi):
            score += 12  # neutral when unavailable
        elif 30 < rsi < 70:
            score += 25  # healthy range
        elif rsi > 70:
            score += 8   # overbought
        else:
            score += 8   # oversold

        # ── Moving Average Trend (25 pts) ─────────────────────────────────
        close = latest['Close']
        sma20 = latest.get('SMA_20', np.nan)
        sma50 = latest.get('SMA_50', np.nan)
        if pd.isna(sma20) or pd.isna(sma50):
            score += 12
        elif close > sma20 > sma50:
            score += 25  # strong bullish
        elif close < sma20 < sma50:
            score += 5   # strong bearish
        else:
            score += 15  # mixed

        # ── MACD (25 pts) ─────────────────────────────────────────────────
        macd = latest.get('MACD', np.nan)
        macd_sig = latest.get('MACD_Signal', np.nan)
        prev_macd = prev.get('MACD', np.nan)
        prev_sig = prev.get('MACD_Signal', np.nan)
        if pd.isna(macd) or pd.isna(macd_sig):
            score += 12
        elif macd > macd_sig and (pd.isna(prev_macd) or prev_macd <= prev_sig):
            score += 25  # bullish crossover
        elif macd < macd_sig and (pd.isna(prev_macd) or prev_macd >= prev_sig):
            score += 5   # bearish crossover
        else:
            score += 15

        # ── Bollinger Bands (25 pts) ──────────────────────────────────────
        bb_upper = latest.get('BB_Upper', np.nan)
        bb_lower = latest.get('BB_Lower', np.nan)
        if pd.isna(bb_upper) or pd.isna(bb_lower):
            score += 12
        elif bb_lower <= close <= bb_upper:
            score += 25  # within bands — normal
        else:
            score += 5   # outside bands — extreme

        # ── Fixed denominator of 100 ──────────────────────────────────────
        final_score = min(100, max(0, score))

        rsi_status = 'Unavailable'
        if not pd.isna(rsi):
            rsi_status = 'Overbought' if rsi > 70 else ('Oversold' if rsi < 30 else 'Neutral')

        return {
            'score': round(final_score, 2),
            'rsi': round(rsi, 2) if not pd.isna(rsi) else None,
            'signal': self._get_technical_signal(latest),
            'details': {
                'rsi_status': rsi_status,
                'trend': self._get_trend(latest),
                'macd_signal': 'Bullish' if (not pd.isna(macd) and not pd.isna(macd_sig) and macd > macd_sig) else 'Bearish'
            }
        }

    def _get_technical_signal(self, latest):
        bullish = 0
        if latest['Close'] > latest.get('SMA_20', float('inf')):
            bullish += 1
        macd = latest.get('MACD', np.nan)
        macd_sig = latest.get('MACD_Signal', np.nan)
        if not pd.isna(macd) and not pd.isna(macd_sig) and macd > macd_sig:
            bullish += 1
        rsi = latest.get('RSI', np.nan)
        if not pd.isna(rsi) and 30 < rsi < 70:
            bullish += 1
        return 'Bullish' if bullish >= 2 else ('Neutral' if bullish == 1 else 'Bearish')

    def _get_trend(self, latest):
        close = latest['Close']
        sma20 = latest.get('SMA_20', np.nan)
        sma50 = latest.get('SMA_50', np.nan)
        if pd.isna(sma20) or pd.isna(sma50):
            return 'Insufficient Data'
        if close > sma20 > sma50:
            return 'Strong Uptrend'
        elif close < sma20 < sma50:
            return 'Strong Downtrend'
        elif close > sma50:
            return 'Uptrend'
        elif close < sma50:
            return 'Downtrend'
        return 'Sideways'

    def get_latest_data(self):
        return self.data.iloc[-1].to_dict()

    # ── Trading Strategies ─────────────────────────────────────────────────

    def calculate_fibonacci_retracements(self, lookback_period=50):
        if len(self.data) < lookback_period:
            return None
        recent_data = self.data.tail(lookback_period)
        swing_high = recent_data['High'].max()
        swing_low = recent_data['Low'].min()
        price_range = swing_high - swing_low
        if price_range == 0:
            return None
        current_price = self.data['Close'].iloc[-1]
        fib_levels = {
            '0.0':   swing_high,
            '0.236': swing_high - (0.236 * price_range),
            '0.382': swing_high - (0.382 * price_range),
            '0.5':   swing_high - (0.5 * price_range),
            '0.618': swing_high - (0.618 * price_range),
            '1.0':   swing_low
        }
        nearest_level = min(fib_levels, key=lambda k: abs(current_price - fib_levels[k]))
        reversal_zone = abs(current_price - fib_levels['0.382']) / price_range < 0.05
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'fibonacci_levels': fib_levels,
            'current_price': current_price,
            'nearest_level': nearest_level,
            'reversal_zone_382': reversal_zone,
            'signal': 'Potential Reversal' if reversal_zone else 'Normal'
        }

    def detect_breakout_patterns(self, consolidation_period=20, breakout_threshold=0.02):
        """✅ FIX: Vectorised — avoids O(n²) rolling window loop."""
        if len(self.data) < consolidation_period + 5:
            return None

        roll_high = self.data['High'].rolling(window=consolidation_period).max().shift(1)
        roll_low = self.data['Low'].rolling(window=consolidation_period).min().shift(1)
        prev_close = self.data['Close'].shift(1)
        pct_change = (self.data['Close'] - prev_close) / prev_close.replace(0, np.nan)

        up_mask = (self.data['Close'] > roll_high) & (pct_change.abs() > breakout_threshold)
        dn_mask = (self.data['Close'] < roll_low) & (pct_change.abs() > breakout_threshold)

        signals = []
        for i in self.data[up_mask].index:
            signals.append({
                'date': _to_date_str(i),
                'type': 'Upward Breakout',
                'price': float(self.data.loc[i, 'Close']),
                'resistance_breached': float(roll_high.loc[i]),
                'volume': float(self.data.loc[i, 'Volume']) if 'Volume' in self.data.columns else 0
            })
        for i in self.data[dn_mask].index:
            signals.append({
                'date': _to_date_str(i),
                'type': 'Downward Breakout',
                'price': float(self.data.loc[i, 'Close']),
                'support_breached': float(roll_low.loc[i]),
                'volume': float(self.data.loc[i, 'Volume']) if 'Volume' in self.data.columns else 0
            })

        signals.sort(key=lambda x: x['date'])
        return {
            'breakout_signals': signals[-5:],
            'latest_signal': signals[-1] if signals else None,
            'total_breakouts': len(signals)
        }

    def detect_reversal_patterns(self, lookback=10):
        if len(self.data) < lookback + 5:
            return None
        reversals = []
        for i in range(lookback, len(self.data) - 1):
            window = self.data.iloc[i - lookback:i]
            current = self.data.iloc[i]
            next_candle = self.data.iloc[i + 1]
            window_trend = (window['Close'].iloc[-1] - window['Close'].iloc[0]) / (window['Close'].iloc[0] or 1)
            current_change = (next_candle['Close'] - current['Close']) / (current['Close'] or 1)
            if window_trend < -0.02 and current_change > 0.01:
                reversals.append({'date': _to_date_str(self.data.index[i]), 'type': 'Bullish Reversal',
                                   'price': float(next_candle['Close']), 'strength': abs(window_trend)})
            elif window_trend > 0.02 and current_change < -0.01:
                reversals.append({'date': _to_date_str(self.data.index[i]), 'type': 'Bearish Reversal',
                                   'price': float(next_candle['Close']), 'strength': abs(window_trend)})
        return {
            'reversal_signals': reversals[-5:],
            'latest_reversal': reversals[-1] if reversals else None,
            'total_reversals': len(reversals)
        }

    def detect_fair_value_gap(self, lookback=50):
        if len(self.data) < 3:
            return None
        self.data['Candle_Body'] = (self.data['Close'] - self.data['Open']).abs()
        self.data['Is_Green'] = self.data['Close'] > self.data['Open']
        green_candles = self.data[self.data['Is_Green']].nlargest(5, 'Candle_Body')
        if green_candles.empty:
            return None
        fvg_zones = []
        for idx in green_candles.index:
            pos = self.data.index.get_loc(idx)
            if 0 < pos < len(self.data) - 1:
                predecessor = self.data.iloc[pos - 1]
                successor = self.data.iloc[pos + 1]
                fvg_high = float(predecessor['High'])
                fvg_low = float(successor['Low'])
                if fvg_high > fvg_low:
                    current_price = float(self.data['Close'].iloc[-1])
                    fvg_zones.append({
                        'date': _to_date_str(idx),
                        'zone_high': fvg_high,
                        'zone_low': fvg_low,
                        'current_price_in_zone': fvg_low <= current_price <= fvg_high,
                        'acts_as_support': current_price > fvg_high,
                        'acts_as_resistance': current_price < fvg_low
                    })
        latest_fvg = fvg_zones[-1] if fvg_zones else None
        return {
            'fvg_zones': fvg_zones,
            'latest_fvg': latest_fvg,
            'signal': ('Potential Support' if latest_fvg and latest_fvg['acts_as_support']
                       else 'Potential Resistance' if latest_fvg and latest_fvg['acts_as_resistance']
                       else None)
        }

    def detect_candlestick_patterns(self):
        if len(self.data) < 2:
            return None
        patterns = []
        for i in range(1, len(self.data)):
            prev = self.data.iloc[i - 1]
            curr = self.data.iloc[i]
            prev_body = abs(prev['Close'] - prev['Open'])
            curr_body = abs(curr['Close'] - curr['Open'])
            curr_upper_shadow = curr['High'] - max(curr['Open'], curr['Close'])
            curr_lower_shadow = min(curr['Open'], curr['Close']) - curr['Low']
            if (prev['Close'] < prev['Open'] and curr['Close'] > curr['Open'] and
                    curr['Open'] < prev['Close'] and curr['Close'] > prev['Open']):
                patterns.append({'date': _to_date_str(self.data.index[i]),
                                  'pattern': 'Bullish Engulfing', 'significance': 'Bullish', 'strength': 'High'})
            elif (prev['Close'] > prev['Open'] and curr['Close'] < curr['Open'] and
                  curr['Open'] > prev['Close'] and curr['Close'] < prev['Open']):
                patterns.append({'date': _to_date_str(self.data.index[i]),
                                  'pattern': 'Bearish Engulfing', 'significance': 'Bearish', 'strength': 'High'})
            elif (prev['Close'] < prev['Open'] and curr_body > 0 and
                  curr_lower_shadow > curr_body * 2 and curr_upper_shadow < curr_body * 0.5):
                patterns.append({'date': _to_date_str(self.data.index[i]),
                                  'pattern': 'Hammer', 'significance': 'Bullish Reversal', 'strength': 'Medium'})
            elif (prev['Close'] > prev['Open'] and curr_body > 0 and
                  curr_upper_shadow > curr_body * 2 and curr_lower_shadow < curr_body * 0.5):
                patterns.append({'date': _to_date_str(self.data.index[i]),
                                  'pattern': 'Shooting Star', 'significance': 'Bearish Reversal', 'strength': 'Medium'})
            elif curr_body < (curr['High'] - curr['Low']) * 0.1:
                patterns.append({'date': _to_date_str(self.data.index[i]),
                                  'pattern': 'Doji', 'significance': 'Neutral/Indecision', 'strength': 'Low'})
        return {
            'patterns': patterns[-10:],
            'latest_pattern': patterns[-1] if patterns else None,
            'total_patterns': len(patterns)
        }

    def get_heiken_ashi_trend(self):
        if 'HA_Trend' not in self.data.columns:
            return None
        latest = self.data.iloc[-1]
        recent_ha = self.data.tail(5)
        bullish_count = (recent_ha['HA_Trend'] == 'Bullish').sum()
        avg_body = recent_ha['HA_Body'].mean()
        median_body = self.data['HA_Body'].median()
        trend_strength = 'Strong' if (median_body > 0 and avg_body > median_body * 1.5) else 'Moderate'
        return {
            'current_trend': latest['HA_Trend'],
            'trend_strength': trend_strength,
            'bullish_percentage': float(bullish_count / len(recent_ha) * 100),
            'avg_candle_size': float(avg_body),
            'max_candle_size': float(recent_ha['HA_Body'].max())
        }

    def analyze_moon_phases(self):
        """
        ✅ FIX: Moon phases have no evidence-based link to market returns.
        Marked as experimental/entertainment only so users aren't misled.
        """
        try:
            from astral import moon
            current_date = datetime.now()
            moon_phase = moon.phase(current_date)
            if moon_phase < 0.1 or moon_phase > 0.9:
                phase_name, sentiment = 'New Moon', 'Neutral'
            elif 0.4 < moon_phase < 0.6:
                phase_name, sentiment = 'Full Moon', 'Neutral'
            elif 0.1 <= moon_phase <= 0.4:
                phase_name, sentiment = 'Waxing Moon', 'Neutral'
            else:
                phase_name, sentiment = 'Waning Moon', 'Neutral'
            return {
                'current_phase': phase_name,
                'moon_phase_value': moon_phase,
                'sentiment': sentiment,
                'disclaimer': 'No evidence-based link between moon phases and market returns. Shown for reference only.',
                'days_to_new_moon': abs(moon_phase) * 29.5,
                'days_to_full_moon': abs(0.5 - moon_phase) * 29.5
            }
        except ImportError:
            return {'error': 'astral library not installed', 'note': 'pip install astral'}

    def detect_harmonic_patterns(self):
        if len(self.data) < 50:
            return None
        highs = argrelextrema(self.data['High'].values, np.greater, order=5)[0]
        lows = argrelextrema(self.data['Low'].values, np.less, order=5)[0]
        if len(highs) < 2 or len(lows) < 2:
            return None
        return {
            'swing_highs': len(highs),
            'swing_lows': len(lows),
            'pattern_detected': 'Complex pattern analysis requires additional swing point validation',
            'note': 'Harmonic patterns require manual validation of Fibonacci ratios'
        }

    def calculate_trend_lines(self, lookback=50):
        if len(self.data) < lookback:
            return None
        recent_data = self.data.tail(lookback)
        highs = argrelextrema(recent_data['High'].values, np.greater, order=3)[0]
        lows = argrelextrema(recent_data['Low'].values, np.less, order=3)[0]
        if len(highs) < 2 or len(lows) < 2:
            return None
        high_prices = [recent_data.iloc[i]['High'] for i in highs]
        low_prices = [recent_data.iloc[i]['Low'] for i in lows]
        low_slope = (low_prices[-1] - low_prices[0]) / max(lows[-1] - lows[0], 1)
        high_slope = (high_prices[-1] - high_prices[0]) / max(highs[-1] - highs[0], 1)
        current_price = float(self.data['Close'].iloc[-1])
        support_level = float(min(low_prices))
        resistance_level = float(max(high_prices))
        return {
            'trend': 'Bullish' if low_slope > 0 else ('Bearish' if high_slope < 0 else 'Sideways'),
            'support_level': support_level,
            'resistance_level': resistance_level,
            'distance_to_support': (current_price - support_level) / support_level * 100,
            'distance_to_resistance': (resistance_level - current_price) / current_price * 100,
            'trend_strength': float(abs(low_slope) if low_slope > 0 else abs(high_slope))
        }

    def calculate_gann_angles(self, lookback=50):
        if len(self.data) < lookback:
            return None
        recent_data = self.data.tail(lookback)
        swing_high = float(recent_data['High'].max())
        swing_low = float(recent_data['Low'].min())
        price_range = swing_high - swing_low
        gann_45_slope = price_range / max(lookback, 1)
        current_price = float(self.data['Close'].iloc[-1])
        gann_45_price = swing_low + (gann_45_slope * lookback)
        return {
            'swing_high': swing_high,
            'swing_low': swing_low,
            'gann_45_slope': gann_45_slope,
            'current_price_vs_45_line': current_price - gann_45_price,
            'above_45_line': current_price > gann_45_price,
            'signal': 'Bullish' if current_price > gann_45_price else 'Bearish'
        }

    def _calculate_parabolic_sar(self, af_start=0.02, af_increment=0.02, af_max=0.2):
        if len(self.data) < 2:
            return
        sar = pd.Series(index=self.data.index, dtype=float)
        ep = pd.Series(index=self.data.index, dtype=float)
        af = pd.Series(index=self.data.index, dtype=float)
        trend = pd.Series(index=self.data.index, dtype=int)
        sar.iloc[0] = self.data['Low'].iloc[0]
        ep.iloc[0] = self.data['High'].iloc[0]
        af.iloc[0] = af_start
        trend.iloc[0] = 1
        for i in range(1, len(self.data)):
            prev_sar = sar.iloc[i - 1]
            prev_ep = ep.iloc[i - 1]
            prev_af = af.iloc[i - 1]
            prev_trend = trend.iloc[i - 1]
            new_sar = prev_sar + prev_af * (prev_ep - prev_sar)
            if prev_trend == 1:
                new_sar = min(new_sar, self.data['Low'].iloc[i - 1], self.data['Low'].iloc[i])
                if self.data['High'].iloc[i] > prev_ep:
                    ep.iloc[i] = self.data['High'].iloc[i]
                    af.iloc[i] = min(prev_af + af_increment, af_max)
                else:
                    ep.iloc[i] = prev_ep
                    af.iloc[i] = prev_af
                if self.data['Low'].iloc[i] < new_sar:
                    trend.iloc[i] = -1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = self.data['Low'].iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = 1
                    sar.iloc[i] = new_sar
            else:
                new_sar = max(new_sar, self.data['High'].iloc[i - 1], self.data['High'].iloc[i])
                if self.data['Low'].iloc[i] < prev_ep:
                    ep.iloc[i] = self.data['Low'].iloc[i]
                    af.iloc[i] = min(prev_af + af_increment, af_max)
                else:
                    ep.iloc[i] = prev_ep
                    af.iloc[i] = prev_af
                if self.data['High'].iloc[i] > new_sar:
                    trend.iloc[i] = 1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = self.data['High'].iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = -1
                    sar.iloc[i] = new_sar
        self.data['PSAR'] = sar
        self.data['PSAR_Trend'] = trend.map({1: 'Bullish', -1: 'Bearish'})

    def _calculate_supertrend(self, period=10, multiplier=3.0):
        if len(self.data) < period:
            return
        high_low = self.data['High'] - self.data['Low']
        high_close = (self.data['High'] - self.data['Close'].shift()).abs()
        low_close = (self.data['Low'] - self.data['Close'].shift()).abs()
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        hl_avg = (self.data['High'] + self.data['Low']) / 2
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        supertrend = pd.Series(index=self.data.index, dtype=float)
        trend = pd.Series(index=self.data.index, dtype=object)
        supertrend.iloc[0] = upper_band.iloc[0]
        trend.iloc[0] = 'Bearish'
        for i in range(1, len(self.data)):
            if self.data['Close'].iloc[i] <= supertrend.iloc[i - 1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                trend.iloc[i] = 'Bearish'
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                trend.iloc[i] = 'Bullish'
            if trend.iloc[i] == 'Bullish' and supertrend.iloc[i] < supertrend.iloc[i - 1]:
                supertrend.iloc[i] = supertrend.iloc[i - 1]
            elif trend.iloc[i] == 'Bearish' and supertrend.iloc[i] > supertrend.iloc[i - 1]:
                supertrend.iloc[i] = supertrend.iloc[i - 1]
        self.data['Supertrend'] = supertrend
        self.data['Supertrend_Signal'] = trend

    def get_stochastics_signal(self):
        if 'Stoch_K' not in self.data.columns:
            return None
        latest = self.data.iloc[-1]
        if pd.isna(latest.get('Stoch_K')) or pd.isna(latest.get('Stoch_D')):
            return None
        signal = 'Neutral'
        if latest['Stoch_Oversold']:
            signal = 'Bullish Reversal (Oversold)'
        elif latest['Stoch_Overbought']:
            signal = 'Bearish Reversal (Overbought)'
        elif latest['Stoch_K'] > latest['Stoch_D']:
            signal = 'Bullish (K above D)'
        elif latest['Stoch_K'] < latest['Stoch_D']:
            signal = 'Bearish (K below D)'
        return {
            'k_percent': round(float(latest['Stoch_K']), 2),
            'd_percent': round(float(latest['Stoch_D']), 2),
            'signal': signal,
            'oversold': bool(latest['Stoch_Oversold']),
            'overbought': bool(latest['Stoch_Overbought'])
        }

    def detect_divergence(self, lookback=30):
        """✅ FIX: price_lows/rsi_lows now defined before use in all branches."""
        if len(self.data) < lookback:
            return None
        recent_data = self.data.tail(lookback)
        divergences = []

        price_highs = recent_data['Close'].rolling(window=5).max()
        price_lows = recent_data['Close'].rolling(window=5).min()  # ✅ defined early
        rsi_highs = recent_data['RSI'].rolling(window=5).max()
        rsi_lows = recent_data['RSI'].rolling(window=5).min()      # ✅ defined early

        if len(price_highs) >= 2:
            if price_highs.iloc[-1] > price_highs.iloc[-2] and rsi_highs.iloc[-1] < rsi_highs.iloc[-2]:
                divergences.append({'type': 'Bearish RSI Divergence', 'indicator': 'RSI',
                                     'signal': 'Potential Reversal Down'})
            if price_lows.iloc[-1] < price_lows.iloc[-2] and rsi_lows.iloc[-1] > rsi_lows.iloc[-2]:
                divergences.append({'type': 'Bullish RSI Divergence', 'indicator': 'RSI',
                                     'signal': 'Potential Reversal Up'})

        if 'MACD' in recent_data.columns:
            macd_highs = recent_data['MACD'].rolling(window=5).max()
            macd_lows = recent_data['MACD'].rolling(window=5).min()
            if len(macd_highs) >= 2:
                if price_highs.iloc[-1] > price_highs.iloc[-2] and macd_highs.iloc[-1] < macd_highs.iloc[-2]:
                    divergences.append({'type': 'Bearish MACD Divergence', 'indicator': 'MACD',
                                         'signal': 'Potential Reversal Down'})
                if price_lows.iloc[-1] < price_lows.iloc[-2] and macd_lows.iloc[-1] > macd_lows.iloc[-2]:
                    divergences.append({'type': 'Bullish MACD Divergence', 'indicator': 'MACD',
                                         'signal': 'Potential Reversal Up'})

        return {
            'divergences': divergences,
            'latest_divergence': divergences[-1] if divergences else None,
            'total_divergences': len(divergences)
        }

    def calculate_volume_profile(self, bins=20):
        """✅ FIX: POC price lookup was broken — now uses bin midpoints properly."""
        if 'Volume' not in self.data.columns:
            return None
        typical_price = (self.data['High'] + self.data['Low'] + self.data['Close']) / 3
        price_min = typical_price.min()
        price_max = typical_price.max()
        if price_min == price_max:
            return None

        bin_edges = np.linspace(price_min, price_max, bins + 1)
        bin_labels = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]  # midpoints

        bin_indices = np.digitize(typical_price, bin_edges[:-1]) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        volume_by_bin = {}
        for idx, b in enumerate(bin_indices):
            price_mid = bin_labels[b]
            volume_by_bin[price_mid] = volume_by_bin.get(price_mid, 0) + float(self.data['Volume'].iloc[idx])

        if not volume_by_bin:
            return None

        poc_price = max(volume_by_bin, key=volume_by_bin.get)  # ✅ direct midpoint key

        return {
            'volume_profile': {round(k, 2): round(v) for k, v in volume_by_bin.items()},
            'point_of_control': round(poc_price, 2),
            'highest_volume_price': round(poc_price, 2)
        }

    def detect_supply_demand_zones(self, lookback=100, zone_threshold=0.02):
        if len(self.data) < lookback:
            return None
        recent_data = self.data.tail(lookback)
        demand_zones, supply_zones = [], []
        for i in range(5, len(recent_data) - 5):
            window_before = recent_data.iloc[i - 5:i]
            window_after = recent_data.iloc[i:i + 5]
            avg_before = window_before['Close'].mean()
            avg_after = window_after['Close'].mean()
            if avg_before == 0:
                continue
            price_change_pct = (avg_after - avg_before) / avg_before
            if price_change_pct > zone_threshold:
                demand_zones.append({
                    'date': _to_date_str(recent_data.index[i]),
                    'zone_low': float(window_before['Low'].min()),
                    'zone_high': float(window_before['High'].max()),
                    'strength': float(price_change_pct)
                })
            elif price_change_pct < -zone_threshold:
                supply_zones.append({
                    'date': _to_date_str(recent_data.index[i]),
                    'zone_low': float(window_after['Low'].min()),
                    'zone_high': float(window_before['High'].max()),
                    'strength': float(abs(price_change_pct))
                })
        current_price = float(self.data['Close'].iloc[-1])
        nearest_demand = next((z for z in demand_zones if z['zone_low'] <= current_price <= z['zone_high']), None)
        nearest_supply = next((z for z in supply_zones if z['zone_low'] <= current_price <= z['zone_high']), None)
        return {
            'demand_zones': demand_zones[-10:],
            'supply_zones': supply_zones[-10:],
            'current_in_demand_zone': nearest_demand is not None,
            'current_in_supply_zone': nearest_supply is not None,
            'nearest_demand_zone': nearest_demand,
            'nearest_supply_zone': nearest_supply
        }

    def get_comprehensive_analysis(self):
        """✅ FIX: Each sub-call wrapped in _safe_call — one failure won't crash the whole method."""
        latest = self.data.iloc[-1]
        analysis = {
            'basic_indicators':    _safe_call('basic_indicators', self.get_technical_score),
            'fibonacci':           _safe_call('fibonacci', self.calculate_fibonacci_retracements),
            'breakouts':           _safe_call('breakouts', self.detect_breakout_patterns),
            'reversals':           _safe_call('reversals', self.detect_reversal_patterns),
            'fair_value_gap':      _safe_call('fair_value_gap', self.detect_fair_value_gap),
            'candlestick_patterns':_safe_call('candlestick_patterns', self.detect_candlestick_patterns),
            'heiken_ashi':         _safe_call('heiken_ashi', self.get_heiken_ashi_trend),
            'moon_phases':         _safe_call('moon_phases', self.analyze_moon_phases),
            'harmonic_patterns':   _safe_call('harmonic_patterns', self.detect_harmonic_patterns),
            'trend_lines':         _safe_call('trend_lines', self.calculate_trend_lines),
            'gann_angles':         _safe_call('gann_angles', self.calculate_gann_angles),
            'stochastics':         _safe_call('stochastics', self.get_stochastics_signal),
            'divergence':          _safe_call('divergence', self.detect_divergence),
            'volume_profile':      _safe_call('volume_profile', self.calculate_volume_profile),
            'supply_demand':       _safe_call('supply_demand', self.detect_supply_demand_zones),
        }

        def _safe_float(val, default=0):
            try:
                return float(val) if not pd.isna(val) else default
            except Exception:
                return default

        analysis['momentum_indicators'] = {
            'macd': {
                'value': round(_safe_float(latest.get('MACD')), 4),
                'signal': round(_safe_float(latest.get('MACD_Signal')), 4),
                'trend': 'Bullish' if _safe_float(latest.get('MACD')) > _safe_float(latest.get('MACD_Signal')) else 'Bearish',
                'crossover_below_zero': _safe_float(latest.get('MACD')) < 0 and _safe_float(latest.get('MACD_Signal')) < 0,
                'crossover_above_zero': _safe_float(latest.get('MACD')) > 0 and _safe_float(latest.get('MACD_Signal')) > 0
            },
            'moving_averages': {
                'price_vs_sma20': 'Above' if latest['Close'] > _safe_float(latest.get('SMA_20'), float('inf')) else 'Below',
                'price_vs_sma50': 'Above' if latest['Close'] > _safe_float(latest.get('SMA_50'), float('inf')) else 'Below',
                'sma20_vs_sma50': 'Above' if _safe_float(latest.get('SMA_20')) > _safe_float(latest.get('SMA_50')) else 'Below',
                'trend': 'Bullish' if (latest['Close'] > _safe_float(latest.get('SMA_20')) > _safe_float(latest.get('SMA_50'))) else 'Bearish'
            },
            'parabolic_sar': {
                'value': round(_safe_float(latest.get('PSAR')), 2),
                'trend': latest.get('PSAR_Trend', 'Neutral'),
                'signal': 'Bullish' if latest['Close'] > _safe_float(latest.get('PSAR')) else 'Bearish'
            },
            'supertrend': {
                'value': round(_safe_float(latest.get('Supertrend')), 2),
                'signal': latest.get('Supertrend_Signal', 'Neutral'),
                'price_vs_supertrend': 'Above' if latest['Close'] > _safe_float(latest.get('Supertrend')) else 'Below'
            }
        }

        if 'VWAP' in self.data.columns:
            analysis['volume_indicators'] = {
                'vwap': {
                    'value': round(_safe_float(latest.get('VWAP')), 2),
                    'price_vs_vwap': 'Above' if latest['Close'] > _safe_float(latest.get('VWAP')) else 'Below',
                    'signal': 'Bullish' if latest['Close'] > _safe_float(latest.get('VWAP')) else 'Bearish'
                }
            }

        return analysis
    
    def get_ml_safe_features(self):
        """
        Returns clean, ML-safe features (NO NaN / NO inf)
        """

        latest = self.data.iloc[-1]

        def safe(val, default=0.0):
            try:
                val = float(val)
                if np.isnan(val) or np.isinf(val):
                    return default
                return val
            except:
                return default

        # 🔹 BASIC FEATURES
        returns = self.data['Close'].pct_change().iloc[-1] if len(self.data) > 1 else 0
        volume_change = self.data['Volume'].pct_change().iloc[-1] if 'Volume' in self.data else 0

        # ── In get_ml_safe_features(), replace sma_200 line ───────────────────────

        # ✅ FIX: SMA_200 isn't calculated in this module — calculate it on the fly
        sma_200_val = self.data['Close'].rolling(window=200).mean().iloc[-1]

        return {
            "rsi": safe(latest.get("RSI"), 50),
            "volatility": safe(self.data['Close'].pct_change().std()),
            "returns": safe(returns),
            "volume_change": safe(volume_change),
            "price": safe(latest.get("Close")),
            "bollinger_upper": safe(latest.get("BB_Upper")),
            "bollinger_lower": safe(latest.get("BB_Lower")),
            "sma_50":  safe(latest.get("SMA_50")),
            "sma_200": safe(sma_200_val),   # ✅ real 200-day MA, not SMA_50 fallback
            "returns_series": (
                self.data['Close']
                .pct_change()
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                .tolist()
            )
        }