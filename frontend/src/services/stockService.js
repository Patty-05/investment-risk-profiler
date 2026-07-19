/**
 * stockService.js
 * Handles /api/analyze-stock endpoint.
 */
import api from '../api';

/**
 * Analyse a stock ticker comprehensively.
 *
 * @param {string} ticker  — e.g. 'AAPL' or 'RELIANCE.NS'
 * @returns {Promise<object>} Full analysis result from the backend
 *
 * Response shape (on success):
 * {
 *   success: true,
 *   ticker,
 *   overall_score: { score, breakdown },
 *   technical_analysis, comprehensive_technical,
 *   fundamental_analysis, comprehensive_fundamental,
 *   sentiment_analysis,
 *   volumetric_analysis,
 *   sector_comparison,
 *   recommendations,
 *   risk_level
 * }
 */
export const analyzeStock = async (ticker) => {
  if (!ticker || typeof ticker !== 'string') {
    throw { message: 'A valid ticker symbol is required', status: 400 };
  }

  const { data } = await api.post('/api/analyze-stock', {
    ticker: ticker.toUpperCase().trim(),
  });

  if (!data.success) {
    return {
      success: false,
      error: data.error || 'Stock analysis failed'
    };
  }

  return data;
};