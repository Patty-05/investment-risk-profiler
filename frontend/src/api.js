import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.error ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred';
    const status = error.response?.status || 0;
    if (process.env.NODE_ENV === 'development') {
      console.error(`[API Error] ${status}: ${message}`, error.config?.url);
    }
    return Promise.reject({ message, status, original: error });
  }
);

export default api;