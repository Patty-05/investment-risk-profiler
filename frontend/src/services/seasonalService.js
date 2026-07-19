/**
 * seasonalService.js
 * Handles /api/seasonal-analysis endpoint.
 * Place in: frontend/src/services/seasonalService.js
 */
import api from '../api';

/**
 * Fetch Indian sector seasonal analysis.
 * @param {boolean} refresh - Force fresh fetch (bypass cache)
 */
export const getSeasonalAnalysis = async (refresh = false) => {
  const { data } = await api.get(`/api/seasonal-analysis${refresh ? '?refresh=true' : ''}`);
  if (!data.success) {
    throw new Error(data.error || 'Seasonal analysis failed');
  }
  return data;
};