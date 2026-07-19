/**
 * healthService.js
 * Pings /api/health to check if the Flask backend is reachable.
 * Useful to show a connection status banner in the UI.
 */
import api from '../api';

/**
 * @returns {Promise<boolean>} true if backend is up
 */
export const checkHealth = async () => {
  try {
    const { data } = await api.get('/api/health');
    return data.status === 'healthy';
  } catch {
    return false;
  }
};
