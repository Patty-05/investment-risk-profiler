// src/services/index.js

export { analyzeStock }                             from './stockService';
export { profileInvestor, PROFILE_OPTIONS }         from './profileService';
export { getAvailableStrategies, backtestStrategy } from './strategyService';
export { getStrategySignals }                       from './signalService';
export { getConflicts, analyzeWar, getIndiaTrade }                from './warService';
export { checkHealth }                              from './healthService';
export { default as useApi }                        from './useApi';
export { AppProvider, useAppContext }               from './context/AppContext';