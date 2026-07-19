/**
 * profileService.js
 * Handles /api/profile endpoint — investor risk profiling.
 */
import api from '../api';

/**
 * @typedef {Object} InvestorProfile
 * @property {string} goal          — 'Wealth Building' | 'Retirement' | 'Education'
 * @property {number} horizon       — Investment horizon in years (≥1)
 * @property {number} income        — Annual income in INR
 * @property {string} riskAppetite  — 'High' | 'Medium' | 'Low'
 */

/**
 * Submit investor profile and receive risk scoring + recommendations.
 *
 * @param {InvestorProfile} profile
 * @returns {Promise<object>}
 *
 * Response shape (on success):
 * {
 *   success: true,
 *   risk_score: number,       // 0-12
 *   risk_level: string,       // 'Low' | 'Medium' | 'High'
 *   recommendations: string[]
 * }
 */
export const profileInvestor = async (profile) => {
  const { goal, horizon, income, riskAppetite } = profile;

  // Client-side validation — mirrors backend rules
  if (!goal)         throw { message: 'Investment goal is required', status: 400 };
  if (!riskAppetite) throw { message: 'Risk appetite is required',   status: 400 };

  const horizonNum = Number(horizon);
  const incomeNum  = Number(income);

  if (!Number.isFinite(horizonNum) || horizonNum < 1)
    throw { message: 'Investment horizon must be at least 1 year', status: 400 };
  if (!Number.isFinite(incomeNum) || incomeNum < 0)
    throw { message: 'Income must be a non-negative number', status: 400 };

  const { data } = await api.post('/api/profile', {
    goal,
    horizon: horizonNum,
    income:  incomeNum,
    riskAppetite,
  });

  if (!data.success) {
    throw { message: data.error || 'Profile submission failed', status: 400 };
  }

  return data;
};

/** Valid options — useful for populating form dropdowns */
export const PROFILE_OPTIONS = {
  goals:         ['Wealth Building', 'Retirement', 'Education'],
  riskAppetites: ['Low', 'Medium', 'High'],
};