/**
 * warService.js
 * War Room — India-specific geopolitical conflict analysis.
 * Endpoints: GET /api/conflicts (watchlist), POST /api/war-analysis.
 */
import api from '../api';

/** Fetch the curated conflict watchlist. */
export const getConflicts = async () => {
  const { data } = await api.get('/api/conflicts');
  if (!data.success) {
    return { success: false, error: data.error || 'Failed to load conflicts' };
  }
  return data;
};

/**
 * Analyse a conflict's India impact.
 * @param {{conflictId?: string, scenarioText?: string}} args
 */
export const analyzeWar = async ({ conflictId, scenarioText, useAi = false } = {}) => {
  const { data } = await api.post('/api/war-analysis', {
    conflict_id: conflictId ?? null,
    scenario_text: scenarioText ?? null,
    use_ai: !!useAi,
  });
  return data;
};


/** India trade by sector and partner nation (official customs data). */
export const getIndiaTrade = async () => {
  const { data } = await api.get('/api/india-trade');
  return data;
};
