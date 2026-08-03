/**
 * useApi.js — Generic async hook
 *
 * Wraps any service call and exposes { data, loading, error, execute }.
 * Components never manage their own loading/error state manually.
 *
 * Usage:
 *   const { data, loading, error, execute } = useApi(analyzeStock);
 *   // trigger:
 *   execute('AAPL');
 */
import { useState, useCallback, useRef, useEffect } from 'react';

/**
 * @template T
 * @param {(...args: any[]) => Promise<T>} serviceFunction  — Any service fn
 * @param {object} [options]
 * @param {T}      [options.initialData]  — Data before first call
 * @param {boolean} [options.immediate]  — Call immediately (no args)
 * @returns {{ data: T|null, loading: boolean, error: string|null, execute: Function, reset: Function }}
 */
const useApi = (serviceFunction, { initialData = null, immediate = false } = {}) => {
  const [data,    setData]    = useState(initialData);
  const [loading, setLoading] = useState(immediate);
  const [error,   setError]   = useState(null);

  // Prevent state updates on unmounted components. Must be driven by the mount
  // lifecycle: in React 18 StrictMode the initial mount is unmount+remount, and
  // without this effect the ref could get stuck `false`, silently dropping a
  // successful response (200 received but nothing renders).
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  const execute = useCallback(
    async (...args) => {
      setLoading(true);
      setError(null);

      try {
        const result = await serviceFunction(...args);
        if (mountedRef.current) {
          setData(result);
          setError(null);
        }
        return result;
      } catch (err) {
        const message = err?.message || 'Something went wrong';
        if (mountedRef.current) {
          setError(message);
          setData(null);
        }
        return null;   // re-throw so callers can chain .catch() if needed
      } finally {
        if (mountedRef.current) setLoading(false);
      }
    },
    [serviceFunction]
  );

  const reset = useCallback(() => {
    setData(initialData);
    setError(null);
    setLoading(false);
  }, [initialData]);

  // Cleanup on unmount
  const cleanup = useCallback(() => {
    mountedRef.current = false;
  }, []);

  return { data, loading, error, execute, reset, cleanup };
};

export default useApi;
