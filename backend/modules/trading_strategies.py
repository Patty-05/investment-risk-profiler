"""
Trading Strategies Module
Implements rule-based trading systems that can be backtested.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ✅ Top-level import so missing deps fail at startup, not mid-backtest
try:
    from sklearn.linear_model import LinearRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    print("⚠️ sklearn not installed — MeanReversionStrategy will be disabled. pip install scikit-learn")

try:
    from statsmodels.tsa.stattools import coint as _coint
    _STATSMODELS_AVAILABLE = True
except ImportError:
    _STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels not installed — cointegration test disabled. pip install statsmodels")


def _rsi_safe(close: pd.Series, period: int = 14) -> pd.Series:
    """
    ✅ FIX: Shared RSI helper with division-by-zero guard (loss=0 → epsilon).
    Used by both TrendFollowingStrategy and TechnicalIndicatorStrategy.
    """
    delta = close.diff()
    gain  = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    loss  = loss.replace(0, 1e-10)
    return 100 - (100 / (1 + gain / loss))


def _propagate_position(signals: pd.DataFrame) -> pd.DataFrame:
    """
    ✅ FIX: Vectorised position propagation — replaces the O(n) .loc[] loop
    in generate_signals() for both TrendFollowing and TechnicalIndicator.
    """
    signals = signals.copy()
    # Replace 0-signals with NaN so ffill carries the last non-zero signal
    signals['position'] = signals['signal'].replace(0, np.nan).ffill().fillna(0).astype(int)
    return signals


# ── Base ────────────────────────────────────────────────────────────────────

class TradingStrategy:
    def __init__(self, data: pd.DataFrame, parameters: Dict = None):
        self.data       = data.copy()
        self.parameters = parameters or {}
        self.signals    = pd.DataFrame(index=data.index)
        self.positions  = []
        self.trades     = []

    def generate_signals(self):
        raise NotImplementedError

    def calculate_position_size(self, signal: str, entry_price: float, account_value: float = 100000):
        raise NotImplementedError


# ── Trend Following ─────────────────────────────────────────────────────────

class TrendFollowingStrategy(TradingStrategy):

    def __init__(self, data: pd.DataFrame, parameters: Dict = None):
        default_params = {
            'short_ma': 50, 'long_ma': 200,
            'adx_period': 14, 'adx_threshold': 25,
            'lookback_months': 12, 'top_n': 12,
            'rebalance_freq': 'monthly'
        }
        default_params.update(parameters or {})
        super().__init__(data, default_params)
        self._calculate_indicators()

    def _calculate_indicators(self):
        self.data['SMA_Short'] = self.data['Close'].rolling(window=self.parameters['short_ma']).mean()
        self.data['SMA_Long']  = self.data['Close'].rolling(window=self.parameters['long_ma']).mean()
        self._calculate_adx()
        if len(self.data) >= 252:
            self.data['Momentum_12M'] = self.data['Close'].pct_change(252)

    def _calculate_adx(self):
        period = self.parameters['adx_period']
        high_low    = self.data['High'] - self.data['Low']
        high_close  = (self.data['High'] - self.data['Close'].shift()).abs()
        low_close   = (self.data['Low']  - self.data['Close'].shift()).abs()
        tr  = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean().replace(0, np.nan)

        plus_dm  = np.where(
            (self.data['High'] - self.data['High'].shift()) > (self.data['Low'].shift() - self.data['Low']),
            np.maximum(self.data['High'] - self.data['High'].shift(), 0), 0)
        minus_dm = np.where(
            (self.data['Low'].shift() - self.data['Low']) > (self.data['High'] - self.data['High'].shift()),
            np.maximum(self.data['Low'].shift() - self.data['Low'], 0), 0)

        plus_di  = 100 * (pd.Series(plus_dm,  index=self.data.index).rolling(window=period).mean() / atr)
        minus_di = 100 * (pd.Series(minus_dm, index=self.data.index).rolling(window=period).mean() / atr)

        denom = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / denom
        self.data['ADX']  = dx.rolling(window=period).mean()
        self.data['+DI']  = plus_di
        self.data['-DI']  = minus_di

    def generate_signals(self):
        self.signals['signal']   = 0
        self.signals['position'] = 0

        golden_cross = (
            (self.data['SMA_Short'] > self.data['SMA_Long']) &
            (self.data['SMA_Short'].shift(1) <= self.data['SMA_Long'].shift(1))
        )
        death_cross = (
            (self.data['SMA_Short'] < self.data['SMA_Long']) &
            (self.data['SMA_Short'].shift(1) >= self.data['SMA_Long'].shift(1))
        )
        strong_trend = self.data['ADX'] > self.parameters['adx_threshold']
        uptrend      = self.data['+DI'] > self.data['-DI']

        self.signals.loc[golden_cross & strong_trend &  uptrend, 'signal'] =  1
        self.signals.loc[death_cross  & strong_trend & ~uptrend, 'signal'] = -1

        # ✅ FIX: Vectorised position propagation
        self.signals = _propagate_position(self.signals)
        return self.signals

    def calculate_position_size(self, signal: str, entry_price: float,
                                 account_value: float = 100000, risk_pct: float = 0.02):
        if entry_price <= 0:
            return 0
        risk_amount = account_value * risk_pct
        return max(0, int(risk_amount / entry_price))

    def momentum_portfolio_signals(self, universe_data: Dict[str, pd.DataFrame]) -> Dict:
        lookback_days = self.parameters['lookback_months'] * 21
        returns = {}
        for ticker, df in universe_data.items():
            if len(df) >= lookback_days:
                start = df['Close'].iloc[-lookback_days]
                if start and start != 0:
                    returns[ticker] = (df['Close'].iloc[-1] / start - 1) * 100
        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        top_n = sorted_returns[:self.parameters['top_n']]
        return {
            ticker: {'signal': 'BUY', 'return_pct': ret, 'rank': i + 1}
            for i, (ticker, ret) in enumerate(top_n)
        }


# ── Mean Reversion ──────────────────────────────────────────────────────────

class MeanReversionStrategy(TradingStrategy):

    def __init__(self, data: pd.DataFrame, pair_data: Optional[pd.DataFrame] = None,
                 parameters: Dict = None):
        default_params = {
            'entry_threshold': 2.0, 'exit_threshold': 1.0,
            'stop_threshold': 3.0,  'lookback_period': 252,
            'cointegration_pvalue': 0.05
        }
        default_params.update(parameters or {})
        super().__init__(data, default_params)
        self.pair_data = pair_data
        if pair_data is not None:
            self._calculate_pair_metrics()

    def _calculate_pair_metrics(self):
        if not _SKLEARN_AVAILABLE:
            # ✅ FIX: Fail loudly at setup rather than silently at signal time
            raise ImportError("scikit-learn is required for MeanReversionStrategy. pip install scikit-learn")

        if self.pair_data is None or len(self.pair_data) == 0:
            return

        aligned = pd.merge(
            self.data[['Close']].rename(columns={'Close': 'Stock_Y'}),
            self.pair_data[['Close']].rename(columns={'Close': 'Stock_X'}),
            left_index=True, right_index=True, how='inner'
        )
        if len(aligned) < self.parameters['lookback_period']:
            return

        aligned['Ratio'] = aligned['Stock_Y'] / aligned['Stock_X'].replace(0, np.nan)

        X = aligned['Stock_X'].values.reshape(-1, 1)
        y = aligned['Stock_Y'].values

        reg = LinearRegression()
        reg.fit(X, y)

        aligned['Predicted_Y'] = reg.predict(X)
        aligned['Residual']    = aligned['Stock_Y'] - aligned['Predicted_Y']
        aligned['Residual_Mean'] = aligned['Residual'].rolling(
            window=self.parameters['lookback_period']).mean()
        aligned['Residual_Std']  = aligned['Residual'].rolling(
            window=self.parameters['lookback_period']).std()

        # ✅ FIX: Replace 0 std with NaN so Z_Score is NaN, not inf
        std_safe = aligned['Residual_Std'].replace(0, np.nan)
        aligned['Z_Score'] = (aligned['Residual'] - aligned['Residual_Mean']) / std_safe

        self.beta             = reg.coef_[0]
        self.intercept        = reg.intercept_
        self.aligned_data     = aligned
        self.regression_model = reg

    def generate_signals(self):
        self.signals['signal']   = 0
        self.signals['position'] = 0

        if not hasattr(self, 'aligned_data'):
            return self.signals

        z  = self.aligned_data['Z_Score'].reindex(self.signals.index)
        zs = z.shift(1)

        entry  = self.parameters['entry_threshold']
        ex     = self.parameters['exit_threshold']

        long_entry  = z <= -entry
        short_entry = z >=  entry
        long_exit   = (z >= -ex) & (zs < -ex)
        short_exit  = (z <=  ex) & (zs >  ex)

        self.signals.loc[long_entry,  'signal'] =  1
        self.signals.loc[short_entry, 'signal'] = -1
        self.signals.loc[long_exit | short_exit, 'signal'] = 0

        return self.signals

    def calculate_position_size(self, signal: str, entry_price: float,
                                 account_value: float = 100000, pair_price: float = None):
        if pair_price is None or not hasattr(self, 'beta') or entry_price <= 0:
            return {'stock_y_shares': 0, 'stock_x_shares': 0}
        capital_per_stock = account_value / 2
        return {
            'stock_y_shares': max(0, int(capital_per_stock / entry_price)),
            'stock_x_shares': max(0, int((capital_per_stock * self.beta) / pair_price)),
            'beta': self.beta,
            'hedge_ratio': f'1:{self.beta:.2f}'
        }

    def test_cointegration(self):
        if not _STATSMODELS_AVAILABLE:
            return {'cointegrated': False, 'pvalue': None,
                    'error': 'statsmodels not installed — pip install statsmodels'}

        if not hasattr(self, 'aligned_data') or len(self.aligned_data) < self.parameters['lookback_period']:
            return {'cointegrated': False, 'pvalue': None}

        try:
            result = _coint(
                self.aligned_data['Stock_Y'].tail(self.parameters['lookback_period']),
                self.aligned_data['Stock_X'].tail(self.parameters['lookback_period'])
            )
            pvalue = result[1]
            is_coint = pvalue < self.parameters['cointegration_pvalue']
            return {
                'cointegrated': is_coint, 'pvalue': pvalue,
                'interpretation': 'Cointegrated — suitable for pairs trading' if is_coint else 'Not cointegrated'
            }
        except Exception as e:
            # ✅ FIX: Log real error instead of bare except
            print(f"⚠️ Cointegration test failed: {e}")
            return {'cointegrated': False, 'pvalue': None, 'error': str(e)}


# ── Technical Indicator Strategy ────────────────────────────────────────────

class TechnicalIndicatorStrategy(TradingStrategy):

    def __init__(self, data: pd.DataFrame, parameters: Dict = None):
        default_params = {
            'rsi_period': 14, 'rsi_oversold': 30, 'rsi_overbought': 70,
            'macd_fast': 12, 'macd_slow': 26, 'macd_signal': 9,
            'bb_period': 20, 'bb_std': 2, 'atr_period': 14,
            'use_rsi': True, 'use_macd': True, 'use_bollinger': True
        }
        default_params.update(parameters or {})
        super().__init__(data, default_params)
        self._calculate_indicators()

    def _calculate_indicators(self):
        if self.parameters['use_rsi']:
            # ✅ FIX: Use shared safe RSI helper
            self.data['RSI'] = _rsi_safe(self.data['Close'], self.parameters['rsi_period'])
        if self.parameters['use_macd']:
            self._calculate_macd()
        if self.parameters['use_bollinger']:
            self._calculate_bollinger_bands()
        self._calculate_atr()

    def _calculate_macd(self):
        fast, slow, sig = (self.parameters['macd_fast'],
                           self.parameters['macd_slow'],
                           self.parameters['macd_signal'])
        self.data['EMA_Fast']       = self.data['Close'].ewm(span=fast, adjust=False).mean()
        self.data['EMA_Slow']       = self.data['Close'].ewm(span=slow, adjust=False).mean()
        self.data['MACD']           = self.data['EMA_Fast'] - self.data['EMA_Slow']
        self.data['MACD_Signal']    = self.data['MACD'].ewm(span=sig, adjust=False).mean()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']

    def _calculate_bollinger_bands(self):
        period, std_mult = self.parameters['bb_period'], self.parameters['bb_std']
        self.data['BB_Middle'] = self.data['Close'].rolling(window=period).mean()
        rolling_std            = self.data['Close'].rolling(window=period).std()
        self.data['BB_Upper']  = self.data['BB_Middle'] + rolling_std * std_mult
        self.data['BB_Lower']  = self.data['BB_Middle'] - rolling_std * std_mult
        self.data['BB_Width']  = self.data['BB_Upper']  - self.data['BB_Lower']

    def _calculate_atr(self):
        period     = self.parameters['atr_period']
        high_low   = self.data['High'] - self.data['Low']
        high_close = (self.data['High'] - self.data['Close'].shift()).abs()
        low_close  = (self.data['Low']  - self.data['Close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        self.data['ATR'] = tr.rolling(window=period).mean()

    def generate_signals(self):
        self.signals['signal']   = 0
        self.signals['position'] = 0

        buy_conditions  = []
        sell_conditions = []

        if self.parameters['use_rsi'] and 'RSI' in self.data.columns:
            rsi = self.data['RSI']
            buy_conditions.append(
                (rsi > self.parameters['rsi_oversold']) &
                (rsi.shift(1) <= self.parameters['rsi_oversold'])
            )
            sell_conditions.append(
                (rsi < self.parameters['rsi_overbought']) &
                (rsi.shift(1) >= self.parameters['rsi_overbought'])
            )

        if self.parameters['use_macd'] and 'MACD' in self.data.columns:
            macd, sig = self.data['MACD'], self.data['MACD_Signal']
            buy_conditions.append((macd > sig) & (macd.shift(1) <= sig.shift(1)))
            sell_conditions.append((macd < sig) & (macd.shift(1) >= sig.shift(1)))

        if self.parameters['use_bollinger'] and 'BB_Lower' in self.data.columns:
            buy_conditions.append(self.data['Close'] <= self.data['BB_Lower'])
            sell_conditions.append(self.data['Close'] >= self.data['BB_Upper'])

        if buy_conditions:
            self.signals.loc[pd.concat(buy_conditions,  axis=1).any(axis=1), 'signal'] =  1
        if sell_conditions:
            self.signals.loc[pd.concat(sell_conditions, axis=1).any(axis=1), 'signal'] = -1

        # ✅ FIX: Vectorised position propagation
        self.signals = _propagate_position(self.signals)
        return self.signals

    def calculate_position_size(self, signal: str, entry_price: float,
                                 account_value: float = 100000,
                                 risk_pct: float = 0.02, atr_multiplier: float = 2.0):
        if entry_price <= 0:
            return 0
        if 'ATR' not in self.data.columns:
            return max(0, int(account_value * risk_pct / entry_price))

        latest_atr    = self.data['ATR'].iloc[-1]
        stop_distance = latest_atr * atr_multiplier
        if stop_distance <= 0:
            return max(0, int(account_value * risk_pct / entry_price))

        risk_amount = account_value * risk_pct
        shares      = max(0, int(risk_amount / stop_distance))
        return {
            'shares':        shares,
            'stop_distance': stop_distance,
            'stop_price':    (entry_price - stop_distance if signal == 'BUY'
                              else entry_price + stop_distance)
        }


# ── Calendar Spread ─────────────────────────────────────────────────────────

class CalendarSpreadStrategy(TradingStrategy):

    def __init__(self, near_month: pd.DataFrame, next_month: pd.DataFrame,
                 parameters: Dict = None):
        default_params = {'entry_threshold': 1.0, 'lookback_period': 60}
        default_params.update(parameters or {})
        super().__init__(near_month, default_params)
        self.near_month = near_month.copy()
        self.next_month = next_month.copy()
        self._calculate_spread_metrics()

    def _calculate_spread_metrics(self):
        aligned = pd.merge(
            self.near_month[['Close']].rename(columns={'Close': 'Near'}),
            self.next_month[['Close']].rename(columns={'Close': 'Next'}),
            left_index=True, right_index=True, how='inner'
        )
        aligned['Spread']      = aligned['Near'] - aligned['Next']
        aligned['Spread_Mean'] = aligned['Spread'].rolling(
            window=self.parameters['lookback_period']).mean()
        aligned['Spread_Std']  = aligned['Spread'].rolling(
            window=self.parameters['lookback_period']).std()

        # ✅ FIX: Guard zero std (same as MeanReversion)
        std_safe = aligned['Spread_Std'].replace(0, np.nan)
        aligned['Z_Score'] = (aligned['Spread'] - aligned['Spread_Mean']) / std_safe
        self.spread_data = aligned

    def generate_signals(self):
        self.signals['signal']   = 0
        self.signals['position'] = 0
        if not hasattr(self, 'spread_data'):
            return self.signals

        threshold   = self.parameters['entry_threshold']
        z  = self.spread_data['Z_Score'].reindex(self.signals.index)
        zs = z.shift(1)

        self.signals.loc[z <= -threshold, 'signal'] =  1
        self.signals.loc[z >=  threshold, 'signal'] = -1
        self.signals.loc[((z >= 0) & (zs < 0)) | ((z <= 0) & (zs > 0)), 'signal'] = 0
        return self.signals

    def calculate_position_size(self, signal: str, entry_price: float,
                                 account_value: float = 100000):
        if entry_price <= 0:
            return 0
        return max(0, int(account_value / 2 / entry_price))


# ── Risk Manager ─────────────────────────────────────────────────────────────

class RiskManager:

    @staticmethod
    def calculate_stop_loss(entry_price: float, signal_type: str, stop_type: str = 'atr',
                            atr: float = None, atr_multiplier: float = 2.0,
                            percent_stop: float = 0.02, support_resistance: float = None):
        if stop_type == 'atr' and atr:
            d = atr * atr_multiplier
            return entry_price - d if signal_type == 'BUY' else entry_price + d
        elif stop_type == 'percent':
            d = entry_price * percent_stop
            return entry_price - d if signal_type == 'BUY' else entry_price + d
        elif stop_type == 'support_resistance' and support_resistance:
            return support_resistance
        return None

    @staticmethod
    def calculate_trailing_stop(current_price: float, highest_price: float,
                                 lowest_price: float, signal_type: str,
                                 atr: float, atr_multiplier: float = 2.0):
        d = atr * atr_multiplier
        return highest_price - d if signal_type == 'BUY' else lowest_price + d

    @staticmethod
    def calculate_risk_reward_ratio(entry_price: float, stop_price: float, target_price: float):
        risk   = abs(entry_price - stop_price)
        reward = abs(target_price - entry_price)
        if reward == 0:
            return None
        rr = risk / reward
        return {
            'risk_reward_ratio': rr,
            'risk': risk, 'reward': reward,
            'interpretation': 'Good (Reward > Risk)' if rr < 1 else 'Poor (Risk > Reward)'
        }


# ── Backtester ───────────────────────────────────────────────────────────────

class Backtester:

    def __init__(self, strategy: TradingStrategy, initial_capital: float = 100000,
                 commission: float = 0.001, slippage: float = 0.0005,
                 risk_free_rate: float = 0.02):
        self.strategy        = strategy
        self.initial_capital = initial_capital
        self.commission      = commission
        self.slippage        = slippage
        # ✅ Improve: risk_free_rate is now a constructor param
        self.risk_free_rate  = risk_free_rate
        self.results         = None

    def run_backtest(self):
        signals  = self.strategy.generate_signals()
        capital  = self.initial_capital
        position = 0
        entry_price = 0.0
        open_trade  = None
        closed_trades = []
        equity_curve  = []

        for idx in signals.index:
            current_price = float(self.strategy.data.loc[idx, 'Close'])
            signal        = int(signals.loc[idx, 'signal'])

            # ── Entry ────────────────────────────────────────────────────
            if signal != 0 and position == 0 and capital > 0:
                position    = signal
                entry_price = current_price * (1 + self.slippage if signal > 0 else 1 - self.slippage)
                # ✅ FIX: Guard capital <= 0 before dividing
                shares      = max(1, int(capital / entry_price)) if entry_price > 0 else 0
                if shares == 0:
                    position = 0
                    continue
                open_trade = {
                    'date':   idx,
                    'action': 'BUY' if signal > 0 else 'SELL',
                    'price':  entry_price,
                    'shares': shares
                }

            # ── Exit (opposite signal) ───────────────────────────────────
            elif signal != 0 and position != 0 and signal == -position and open_trade:
                exit_price = current_price * (1 - self.slippage if position > 0 else 1 + self.slippage)
                pnl        = (exit_price - entry_price) * position * open_trade['shares']
                cost       = (entry_price + exit_price) * open_trade['shares'] * self.commission
                net_pnl    = pnl - cost
                capital   += net_pnl

                closed_trades.append({
                    **open_trade,
                    'exit_date':  idx,
                    'exit_price': exit_price,
                    'pnl':        net_pnl,
                    'return_pct': (net_pnl / (entry_price * open_trade['shares'])) * 100
                })
                position   = 0
                open_trade = None

            equity_curve.append({'date': idx, 'equity': capital, 'position': position})

        # ✅ FIX: Force-close any open position at end of data
        if open_trade and position != 0:
            last_price = float(self.strategy.data['Close'].iloc[-1])
            exit_price = last_price * (1 - self.slippage if position > 0 else 1 + self.slippage)
            pnl        = (exit_price - entry_price) * position * open_trade['shares']
            cost       = (entry_price + exit_price) * open_trade['shares'] * self.commission
            net_pnl    = pnl - cost
            capital   += net_pnl
            closed_trades.append({
                **open_trade,
                'exit_date':  self.strategy.data.index[-1],
                'exit_price': exit_price,
                'pnl':        net_pnl,
                'return_pct': (net_pnl / (entry_price * open_trade['shares'])) * 100,
                'note':       'Force-closed at end of data'
            })

        equity_df = pd.DataFrame(equity_curve)
        returns   = equity_df['equity'].pct_change().dropna() if len(equity_df) > 1 else pd.Series(dtype=float)

        winning = [t for t in closed_trades if t.get('pnl', 0) > 0]
        losing  = [t for t in closed_trades if t.get('pnl', 0) <= 0]
        n_closed = len(closed_trades)

        self.results = {
            'trades':        closed_trades,
            'equity_curve':  equity_curve,
            'final_capital': capital,
            'total_return':  ((capital - self.initial_capital) / self.initial_capital) * 100,
            'total_trades':  n_closed,
            'winning_trades': len(winning),
            'losing_trades':  len(losing),
            'win_rate':      (len(winning) / n_closed * 100) if n_closed > 0 else 0,
            'avg_win':       float(np.mean([t['pnl'] for t in winning])) if winning else 0,
            'avg_loss':      float(np.mean([t['pnl'] for t in losing]))  if losing  else 0,
            'max_drawdown':  self._calculate_max_drawdown(equity_df),
            'sharpe_ratio':  self._calculate_sharpe_ratio(returns) if len(returns) > 0 else 0
        }
        return self.results

    def _calculate_max_drawdown(self, equity_df: pd.DataFrame) -> float:
        if equity_df.empty:
            return 0.0
        peak     = equity_df['equity'].cummax()
        drawdown = (equity_df['equity'] - peak) / peak.replace(0, np.nan)
        return float(drawdown.min() * 100)

    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        if len(returns) == 0 or returns.std() == 0:
            return 0.0
        annual_return = returns.mean() * 252
        annual_std    = returns.std() * np.sqrt(252)
        if annual_std == 0:
            return 0.0
        return float((annual_return - self.risk_free_rate) / annual_std)


# ── Strategy Selector ────────────────────────────────────────────────────────

class StrategySelector:

    @staticmethod
    def create_strategy(strategy_type: str, data: pd.DataFrame,
                        pair_data: pd.DataFrame = None,
                        parameters: Dict = None) -> TradingStrategy:
        t = strategy_type.lower()
        if t == 'trend_following':
            return TrendFollowingStrategy(data, parameters)
        elif t in ('mean_reversion', 'pairs'):
            if pair_data is None:
                raise ValueError("pair_data required for mean_reversion strategy")
            return MeanReversionStrategy(data, pair_data, parameters)
        elif t == 'technical':
            return TechnicalIndicatorStrategy(data, parameters)
        elif t == 'calendar_spread':
            if pair_data is None:
                raise ValueError("pair_data (next month contract) required for calendar_spread")
            return CalendarSpreadStrategy(data, pair_data, parameters)
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")

    @staticmethod
    def get_available_strategies() -> Dict:
        return {
            'trend_following': {
                'name': 'Trend Following (Momentum)',
                'description': 'Follows price trends using moving averages and ADX',
                'required_data': 'Single stock price data',
                'parameters': {'short_ma': 50, 'long_ma': 200, 'adx_threshold': 25}
            },
            'mean_reversion': {
                'name': 'Mean Reversion (Pairs Trading)',
                'description': 'Trades price deviations from mean using pairs',
                'required_data': 'Two cointegrated stocks',
                'parameters': {'entry_threshold': 2.0, 'exit_threshold': 1.0, 'stop_threshold': 3.0},
                'requires': ['scikit-learn', 'statsmodels']
            },
            'technical': {
                'name': 'Technical Indicators',
                'description': 'Combines RSI, MACD, and Bollinger Bands',
                'required_data': 'Single stock price data',
                'parameters': {'rsi_period': 14, 'macd_fast': 12, 'macd_slow': 26}
            },
            'calendar_spread': {
                'name': 'Calendar Spread',
                'description': 'Trades futures contract spreads',
                'required_data': 'Near and next month contracts',
                'parameters': {'entry_threshold': 1.0}
            }
        }