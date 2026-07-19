/**
 * signalService.js
 * Handles /api/strategy-signals endpoint — live signal generation.
 */
import api from '../api';

/**
 * Get the current trading signal for a strategy on a ticker.
 *
 * @param {object} params
 * @param {string} params.ticker
 * @param {string} params.strategy_type
 * @param {object} [params.parameters]
 * @param {string} [params.pair_ticker]
 * @param {number} [params.account_value]
 *
 * @returns {Promise<object>}
 *
 * Response shape (on success):
 * {
 *   success: true,
 *   ticker, strategy_type,
 *   current_signal: {
 *     signal: -1 | 0 | 1,
 *     signal_text: 'BUY' | 'SELL' | 'HOLD',
 *     position: number,
 *     current_price: number,
 *     date: string
 *   },
 *   position_size: object | null,
 *   recommendation: 'Buy' | 'Sell' | 'Hold'
 * }
 */
export const getStrategySignals = async ({
  ticker,
  strategy_type,
  parameters = {},
  pair_ticker = null,
  account_value = 100000,
}) => {
  if (!ticker)        throw { message: 'Ticker is required',        status: 400 };
  if (!strategy_type) throw { message: 'Strategy type is required', status: 400 };

  const { data } = await api.post('/api/strategy-signals', {
    ticker:        ticker.toUpperCase().trim(),
    strategy_type,
    parameters,
    pair_ticker:   pair_ticker ? pair_ticker.toUpperCase().trim() : null,
    account_value: Number(account_value),
  });

  if (!data.success) {
    throw { message: data.error || 'Failed to get signals', status: 400 };
  }

  return data;
};
