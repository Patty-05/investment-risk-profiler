/**
 * strategyService.js
 * Handles /api/trading-strategies and /api/backtest-strategy endpoints.
 */
import api from '../api';

/**
 * Fetch the list of available trading strategies from the backend.
 *
 * @returns {Promise<object>} Map of strategy_type → metadata
 */
export const getAvailableStrategies = async () => {
  const { data } = await api.get('/api/trading-strategies');

  if (!data.success) {
    throw { message: data.error || 'Failed to fetch strategies', status: 400 };
  }

  return data.strategies;
};

/**
 * Run a backtest for a given strategy.
 *
 * @param {object} params
 * @param {string} params.ticker           — Primary ticker e.g. 'AAPL'
 * @param {string} params.strategy_type    — 'trend_following' | 'mean_reversion' | 'technical' | 'calendar_spread'
 * @param {object} [params.parameters]     — Strategy-specific overrides
 * @param {number} [params.initial_capital] — Starting capital (default 100000)
 * @param {string} [params.pair_ticker]    — Required for mean_reversion / calendar_spread
 *
 * @returns {Promise<object>}
 *
 * Response shape (on success):
 * {
 *   success: true,
 *   strategy_type, ticker,
 *   backtest_results: {
 *     initial_capital, final_capital,
 *     total_return_pct, total_trades,
 *     winning_trades, losing_trades, win_rate_pct,
 *     avg_win, avg_loss, max_drawdown_pct, sharpe_ratio
 *   },
 *   trades: [...] // up to 20 most recent
 * }
 */
export const backtestStrategy = async ({
  ticker,
  strategy_type,
  parameters = {},
  initial_capital = 100000,
  pair_ticker = null,
}) => {
  if (!ticker)         throw { message: 'Ticker is required',         status: 400 };
  if (!strategy_type)  throw { message: 'Strategy type is required',  status: 400 };

  const PAIR_STRATEGIES = ['mean_reversion', 'pairs', 'calendar_spread'];
  if (PAIR_STRATEGIES.includes(strategy_type) && !pair_ticker) {
    throw {
      message: `A pair ticker is required for the ${strategy_type} strategy`,
      status: 400,
    };
  }

  if (Number(initial_capital) <= 0) {
    throw { message: 'Initial capital must be a positive number', status: 400 };
  }

  const { data } = await api.post('/api/backtest-strategy', {
    ticker:          ticker.toUpperCase().trim(),
    strategy_type,
    parameters,
    initial_capital: Number(initial_capital),
    pair_ticker:     pair_ticker ? pair_ticker.toUpperCase().trim() : null,
  });

  if (!data.success) {
    throw { message: data.error || 'Backtest failed', status: 400 };
  }

  return data;
};
