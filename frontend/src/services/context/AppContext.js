/**
 * AppContext.js — Global app state via Context API
 *
 * Stores the last successful results from each feature so navigating
 * between pages doesn't lose data. Keeps state management simple —
 * no Redux needed at this scale.
 *
 * Usage in any component:
 *   import { useAppContext } from '../context/AppContext';
 *   const { stockResult, setStockResult } = useAppContext();
 */
import React, { createContext, useContext, useState, useCallback } from 'react';
import { checkHealth } from '../healthService';

const AppContext = createContext(null);

export const AppProvider = ({ children }) => {
  // ── Per-feature cached results ────────────────────────────────────────
  const [stockResult,    setStockResult]    = useState(null);
  const [profileResult,  setProfileResult]  = useState(null);
  const [backtestResult, setBacktestResult] = useState(null);
  const [signalResult,   setSignalResult]   = useState(null);

  // ── Backend health ────────────────────────────────────────────────────
  const [backendOnline,  setBackendOnline]  = useState(null); // null = not checked yet

  const pingBackend = useCallback(async () => {
    const ok = await checkHealth();
    setBackendOnline(ok);
    return ok;
  }, []);

  // ── Clear everything (e.g. on ticker change) ─────────────────────────
  const clearAll = useCallback(() => {
    setStockResult(null);
    setProfileResult(null);
    setBacktestResult(null);
    setSignalResult(null);
  }, []);

  const value = {
    // Stock analysis
    stockResult,
    setStockResult,

    // Investor profile
    profileResult,
    setProfileResult,

    // Backtest
    backtestResult,
    setBacktestResult,

    // Signals
    signalResult,
    setSignalResult,

    // Health
    backendOnline,
    pingBackend,

    // Utilities
    clearAll,
  };

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
};

/** Hook — throws if used outside AppProvider */
export const useAppContext = () => {
  const ctx = useContext(AppContext);
  if (!ctx) {
    throw new Error('useAppContext must be used inside <AppProvider>');
  }
  return ctx;
};

export default AppContext;