import React, { useState, useEffect } from 'react';
import './App.css';
import SeasonalAnalysis from './SeasonalAnalysis';


import { analyzeStock, profileInvestor, PROFILE_OPTIONS,
         getAvailableStrategies, backtestStrategy, getStrategySignals,
         useApi, useAppContext, AppProvider } from './services';

export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}

function AppShell() {
  const [activeTab, setActiveTab] = useState('stock');
  const { backendOnline, pingBackend } = useAppContext();
  useEffect(() => { pingBackend(); }, [pingBackend]);

  return (
    <div className="container">
      <h1 style={{
        textAlign: "center",
        background: "linear-gradient(90deg, #00ffcc, #00ff88)",
        WebkitBackgroundClip: "text",
        color: "transparent"
      }}>
        Investment Risk Profiler
      </h1>

      {backendOnline === false && (
        <div className="error" style={{ marginBottom: 16 }}>
          ⚠️ Cannot reach the backend (localhost:5000). Make sure Flask is running.
        </div>
      )}

      <div className="tabs">
        {[
          { key: 'stock',      label: 'Stock Analysis'    },
          { key: 'investor',   label: 'Investor Profiling' },
          { key: 'strategies', label: 'Trading Strategies' },
          { key: 'seasonal', label: '🌿 Seasonal' }
        ].map(({ key, label }) => (
          <button key={key} className={activeTab === key ? 'tab active' : 'tab'} onClick={() => setActiveTab(key)}>
            {label}
          </button>
        ))}
      </div>

      {activeTab === 'stock'      && <StockAnalysis />}
      {activeTab === 'investor'   && <InvestorProfiling />}
      {activeTab === 'strategies' && <TradingStrategies />}
      {activeTab === 'seasonal' && <SeasonalAnalysis />}
    </div>
  );
}


// ── Stock Analysis ────────────────────────────────────────────────────────
function StockAnalysis() {
  const [ticker, setTicker] = useState('');
  const { data, loading, error, execute } = useApi(analyzeStock);
  const result = data?.data;
  const { setStockResult } = useAppContext();
  const [autoRefresh, setAutoRefresh] = useState(true);

  const handleSubmit = async (e) => {
    e.preventDefault();
    try { const res = await execute(ticker); setStockResult(res); } catch (_) {}
  };

  useEffect(() => {
    if (!ticker || !autoRefresh) return;
    const interval = setInterval(() => { execute(ticker); }, 300000);
    return () => clearInterval(interval);
  }, [ticker, autoRefresh]);

  const getScoreClass = (score) => {
    if (score >= 75) return 'score-excellent';
    if (score >= 65) return 'score-good';
    if (score >= 45) return 'score-fair';
    if (score >= 25) return 'score-poor';
    return 'score-very-poor';
  };

  if (loading) {
    return (
      <div className="analysis-card skeleton" style={{ marginTop: 24, padding: 40, textAlign: 'center' }}>
        Loading analysis…
      </div>
    );
  }

  return (
    <div>
      <p>Analyse stock risk based on technical, fundamental, sentiment, volumetric, and sector analysis.</p>
      {autoRefresh && <p style={{ color: "#00ffcc" }}>● Live updates every 5 minutes</p>}

      <div className="form-container">
        <button className="btn" onClick={() => setAutoRefresh(!autoRefresh)} style={{ marginBottom: 16 }}>
          {autoRefresh ? "⏸ Stop Live" : "▶ Start Live"}
        </button>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="ticker">Stock Ticker Symbol:</label>
            <input type="text" id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value)}
              placeholder="e.g., AAPL, MSFT, RELIANCE.NS" required />
            <small>US stocks: AAPL · Indian stocks: RELIANCE.NS</small>
          </div>
          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Analysing…' : 'Analyse Stock'}
          </button>
        </form>
        {error && <div className="error">{error}</div>}
      </div>

      {result?.success && (
        <div className="result-container">
          <h2>Stock Analysis Results for {result?.ticker}</h2>

          {/* TradingView Chart */}
          <div style={{ marginBottom: "20px" }}>
            <iframe
              src={`https://s.tradingview.com/widgetembed/?symbol=${result.ticker}&interval=D&theme=dark`}
              width="100%" height="400" frameBorder="0" title="Price chart"
            />
          </div>

          {/* ── Technical Indicator Visuals ── */}
          {result.technical_analysis && (
            <div className="analysis-card">
              <h4>📈 Technical Indicators</h4>

              {/* RSI — ✅ FIX: read from rsi_value, fall back to details.rsi */}
              {(() => {
                const rsiVal = result.technical_analysis?.rsi_value
                  ?? result.technical_analysis?.details?.rsi
                  ?? result.technical_analysis?.rsi
                  ?? null;

                const rsiStatus = result.technical_analysis?.details?.rsi_status
                  ?? result.technical_analysis?.rsi_status
                  ?? null;

                const rsiColor = rsiVal > 70 ? "#ff4d4d" : rsiVal < 30 ? "#00ff88" : "#ffd166";
                const rsiLabel = rsiVal > 70 ? "Overbought ⚠️" : rsiVal < 30 ? "Oversold 🟢" : "Neutral";

                return (
                  <div style={{ marginBottom: "15px" }}>
                    <p>RSI: {rsiVal != null ? rsiVal.toFixed(1) : (rsiStatus ?? 'N/A')}</p>
                    <div style={{ height: "8px", background: "#222", borderRadius: "5px" }}>
                      <div style={{
                        width: `${rsiVal != null ? Math.min(rsiVal, 100) : 50}%`,
                        height: "100%", borderRadius: "5px", background: rsiColor
                      }} />
                    </div>
                    <small>{rsiVal != null ? rsiLabel : (rsiStatus ?? 'N/A')}</small>
                  </div>
                );
              })()}

              {/* MACD */}
              <div>
                <p>MACD: {result.technical_analysis?.details?.macd?.toFixed(3) ?? result.technical_analysis?.macd?.toFixed(3) ?? 'N/A'}</p>
                <small>
                  {(result.technical_analysis?.details?.macd ?? result.technical_analysis?.macd ?? 0) > 0
                    ? "Bullish Momentum 📈" : "Bearish Momentum 📉"}
                </small>
              </div>
            </div>
          )}

          {/* Overall Score — ✅ FIX: force white text so it's always visible */}
          <div className={`overall-score pulse ${getScoreClass(result.overall_score.score)}`}>
            <div className="score-display">
              <h3>Overall Risk Score</h3>
              <div className="score-value" style={{ color: '#fff' }}>
                {result.overall_score.score}/100
              </div>
              <div className="risk-level" style={{ color: 'rgba(255,255,255,0.85)' }}>
                {result.risk_level}
              </div>
            </div>
          </div>

          {/* Analysis Grid */}
          <div className="analysis-grid">
            <AnalysisCard title="Technical Analysis"
              score={result.technical_analysis?.score} signal={result.technical_analysis?.signal}
              items={[
                `RSI: ${result.technical_analysis?.details?.rsi_status ?? result.technical_analysis?.rsi_status ?? 'N/A'}`,
                `Trend: ${result.technical_analysis?.details?.trend ?? result.technical_analysis?.trend ?? 'N/A'}`,
              ]} />
            <AnalysisCard title="Fundamental Analysis"
              score={result.fundamental_analysis?.score} signal={result.fundamental_analysis?.signal}
              items={[
                `P/E: ${result.fundamental_analysis?.ratios?.valuation?.pe_ratio?.value?.toFixed(1) ?? 'N/A'}`,
                `ROE: ${result.fundamental_analysis?.ratios?.profitability?.roe?.value?.toFixed(1) ?? 'N/A'}%`,
              ]} />
            <AnalysisCard title="Sentiment Analysis"
              score={result.sentiment_analysis?.score} signal={result.sentiment_analysis?.sentiment}
              items={[`Confidence: ${result.sentiment_analysis?.confidence ?? 'N/A'}`]} />
            <AnalysisCard title="Volumetric Analysis"
              score={result.volumetric_analysis?.score} signal={result.volumetric_analysis?.signal}
              items={[`Volume Trend: ${result.volumetric_analysis?.details?.volume_trend?.status ?? 'N/A'}`]} />
            <AnalysisCard title="Sector Comparison"
              score={result.sector_comparison?.score}
              signal={`${result.sector_comparison?.vs_market?.outperformance ?? 0}% vs Market`}
              items={[`Sector: ${result.sector_comparison?.sector ?? 'N/A'}`]} />
          </div>

          {/* ══════════════════════════════════════════════════════════
              🤖 DUAL-MODEL AI SECTION — improved layout
              ══════════════════════════════════════════════════════════ */}
          {result.ml_prediction && (
            <div className="ai-section">
              <div className="ai-section-title">🤖 Dual-Model AI Intelligence</div>

              {/* Model signal cards row */}
              <div className="ai-signals-row">

                {/* Calibrated RF */}
                <div className="ai-signal-card ai-signal-card--rf">
                  <div className="ai-signal-model-label">Calibrated RF</div>
                  <div className={`ml-signal ${result.ml_prediction?.toLowerCase()}`}>
                    {result.ml_prediction}
                  </div>
                  <div className="ai-signal-conf">
                    Confidence: {((result.ml_confidence || 0) * 100).toFixed(1)}%
                  </div>
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: `${(result.ml_confidence || 0) * 100}%` }} />
                  </div>
                </div>

                {/* XGBoost */}
                {result.xgb_prediction && (
                  <div className="ai-signal-card ai-signal-card--xgb">
                    <div className="ai-signal-model-label">XGBoost</div>
                    <div className={`ml-signal ${result.xgb_prediction?.toLowerCase()}`}>
                      {result.xgb_prediction}
                    </div>
                    <div className="ai-signal-conf">
                      Confidence: {((result.xgb_confidence || 0) * 100).toFixed(1)}%
                    </div>
                    <div className="confidence-bar">
                      <div className="confidence-fill confidence-fill--teal"
                        style={{ width: `${(result.xgb_confidence || 0) * 100}%` }} />
                    </div>
                  </div>
                )}

                {/* Rule vs ML */}
                {result.comparison && (
                  <div className="ai-signal-card ai-signal-card--rule">
                    <div className="ai-signal-model-label">Rule vs ML</div>
                    <div className="rule-vs-row">
                      <span className="rule-vs-key">Rule Score</span>
                      <span className="rule-vs-val">{result.comparison.rule_score}</span>
                    </div>
                    <div className="rule-vs-row">
                      <span className="rule-vs-key">ML Signal</span>
                      <span className={`rule-vs-val ml-signal-sm ${result.comparison.ml_signal?.toLowerCase()}`}>
                        {result.comparison.ml_signal}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              {/* Probability Distribution */}
              {result.xgb_probabilities && (
                <>
                  <div className="xgb-section-heading">XGBoost · Probability Distribution</div>
                  <XGBProbabilityPanel probabilities={result.xgb_probabilities} />
                </>
              )}

              {/* Model Disagreement */}
              {result.xgb_prediction && (
                <>
                  <div className="xgb-section-heading">Dual-Model Consensus</div>
                  <ModelDisagreementPanel
                    rfSignal={result.ml_prediction}
                    xgbSignal={result.xgb_prediction}
                    disagreement={result.model_disagreement}
                  />
                </>
              )}

              {/* ✅ Side-by-side Key Drivers */}
              {(result.feature_importance?.length > 0 || result.xgb_feature_importance?.length > 0) && (
                <>
                  <div className="xgb-section-heading">Key Drivers · Both Models</div>
                  <div className="drivers-grid">
                    {result.feature_importance?.length > 0 && (
                      <div className="drivers-card">
                        <div className="drivers-card-header drivers-card-header--rf">
                          <span className="drivers-dot drivers-dot--rf" />
                          Calibrated RF
                        </div>
                        <FeatureImportanceBars
                          data={result.feature_importance}
                          fillClass="drivers-bar-fill--rf"
                        />
                      </div>
                    )}
                    {result.xgb_feature_importance?.length > 0 && (
                      <div className="drivers-card">
                        <div className="drivers-card-header drivers-card-header--xgb">
                          <span className="drivers-dot drivers-dot--xgb" />
                          XGBoost
                        </div>
                        <FeatureImportanceBars
                          data={result.xgb_feature_importance}
                          fillClass="drivers-bar-fill--xgb"
                        />
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          )}

          {/* Explanation */}
          {result.explanation?.length > 0 && (
            <div className="recommendations" style={{ marginTop: "10px" }}>
              <h4>🧠 Why this prediction?</h4>
              <ul>
                {result.explanation.map((exp, i) => <li key={i}>{exp}</li>)}
              </ul>
            </div>
          )}

          {/* Gemini */}
          {result.gemini_insight && (
            <div className="analysis-card" style={{
              marginTop: "10px",
              background: "linear-gradient(135deg, #0f2027, #203a43, #2c5364)"
            }}>
              <h4>🤖 AI Investment Insight</h4>
              <p style={{ lineHeight: "1.6", fontSize: "14px", opacity: 0.9 }}>{result.gemini_insight}</p>
            </div>
          )}

          {result?.portfolio?.allocation && <PortfolioCard portfolio={result.portfolio} />}

          <div className="recommendations">
            <h3>Investment Recommendations:</h3>
            {result.recommendations?.map((rec, i) => (
              <div key={i} className="recommendation-item">
                <span className={`priority-${rec.priority.toLowerCase()}`}>{rec.priority}</span>
                <span className="rec-type">{rec.type}</span>
                <span className="rec-message">{rec.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


// ── XGBoost Probability Panel ─────────────────────────────────────────────
function XGBProbabilityPanel({ probabilities }) {
  if (!probabilities) return null;
  const entries = [
    { label: 'BUY',  value: probabilities.BUY  || 0, cls: 'buy',  icon: '▲' },
    { label: 'HOLD', value: probabilities.HOLD || 0, cls: 'hold', icon: '◆' },
    { label: 'SELL', value: probabilities.SELL || 0, cls: 'sell', icon: '▼' },
  ].sort((a, b) => b.value - a.value);

  return (
    <div className="prob-panel">
      <div className="prob-panel-header">
        <h4>Probability Distribution</h4>
        <span className={`prob-dominant-chip ${entries[0].cls}`}>{entries[0].label} dominant</span>
      </div>
      <div className="prob-bars">
        {entries.map(({ label, value, cls, icon }) => (
          <div key={label} className="prob-row">
            <div className="prob-label-group">
              <span className="prob-icon" style={{
                color: cls === 'buy' ? 'var(--green)' : cls === 'hold' ? 'var(--amber)' : 'var(--red)'
              }}>{icon}</span>
              <span className="prob-name">{label}</span>
            </div>
            <div className="prob-track">
              <div className={`prob-fill ${cls}`} style={{ width: `${(value * 100).toFixed(1)}%` }} />
            </div>
            <span className={`prob-pct ${cls}`}>{(value * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <p className="prob-note">XGBoost softmax probabilities across SELL / HOLD / BUY classes</p>
    </div>
  );
}


// ── Model Disagreement Panel ──────────────────────────────────────────────
function ModelDisagreementPanel({ rfSignal, xgbSignal, disagreement }) {
  if (!rfSignal || !xgbSignal) return null;
  return (
    <div className={`disagree-panel ${disagreement ? 'warn' : 'ok'}`}>
      <div className="disagree-header">
        <span className="disagree-icon">{disagreement ? '⚠️' : '✅'}</span>
        <h4>{disagreement ? 'Models Disagree — Exercise Caution' : 'Models Agree'}</h4>
      </div>
      <div className="disagree-signals">
        <div className="disagree-model">
          <span className="disagree-model-name">Calibrated RF</span>
          <span className={`disagree-signal-chip ${rfSignal.toLowerCase()}`}>{rfSignal}</span>
        </div>
        <div className="disagree-vs">vs</div>
        <div className="disagree-model">
          <span className="disagree-model-name">XGBoost</span>
          <span className={`disagree-signal-chip ${xgbSignal.toLowerCase()}`}>{xgbSignal}</span>
        </div>
      </div>
      <p className={`disagree-msg ${disagreement ? 'warn' : 'ok'}`}>
        {disagreement
          ? 'The two models reached different conclusions. Consider waiting for stronger confirmation before acting.'
          : 'Both models independently reached the same conclusion, increasing overall signal confidence.'}
      </p>
    </div>
  );
}


// ── Feature Importance Bars (shared) ─────────────────────────────────────
function FeatureImportanceBars({ data, fillClass }) {
  if (!data || data.length === 0) return null;
  const top = [...data].sort((a, b) => b.importance - a.importance).slice(0, 6);
  const max = top[0]?.importance || 1;
  return (
    <div className="fi-bars">
      {top.map((f, i) => (
        <div key={i} className="fi-row">
          <div className="fi-label">
            <span className="fi-name">{f.feature}</span>
            <span className="fi-pct">{(f.importance * 100).toFixed(1)}%</span>
          </div>
          <div className="fi-track">
            <div
              className={`fi-fill ${fillClass}`}
              style={{ width: `${(f.importance / max) * 100}%` }}
            />
          </div>
        </div>
      ))}
    </div>
  );
}


// ── Shared helpers ────────────────────────────────────────────────────────
function AnalysisCard({ title, score, signal, items = [] }) {
  return (
    <div className="analysis-card">
      <h4>{title}</h4>
      <div className="score-badge">{score != null ? Number(score).toFixed(1) : 'N/A'}</div>
      <p>{signal}</p>
      <ul>{items.map((item, i) => <li key={i}>{item}</li>)}</ul>
    </div>
  );
}

function PortfolioCard({ portfolio }) {
  if (!portfolio) return null;
  return (
    <div className="analysis-card">
      <h4>💼 Portfolio Allocation</h4>
      {portfolio.allocation.map((item, i) => (
        <div key={i} style={{ marginBottom: "10px" }}>
          <div style={{ display: "flex", justifyContent: "space-between" }}>
            <span>{item.asset}</span><strong>{item.weight}%</strong>
          </div>
          <div style={{ height: "6px", background: "#222", borderRadius: "5px" }}>
            <div style={{ width: `${item.weight}%`, height: "100%", background: "#00ffcc", transition: "width 1s ease" }} />
          </div>
        </div>
      ))}
      <p>Diversification Score: {portfolio.diversification_score}</p>
    </div>
  );
}


// ── Investor Profiling ────────────────────────────────────────────────────
function InvestorProfiling() {
  const [formData, setFormData] = useState({ goal: 'Wealth Building', horizon: 10, income: '', riskAppetite: 'Medium' });
  const [validationErr, setValidationErr] = useState('');
  const { data: result, loading, error, execute } = useApi(profileInvestor);
  const { setProfileResult } = useAppContext();

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
    setValidationErr('');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!formData.income || Number(formData.income) <= 0) { setValidationErr('Please enter a valid annual income.'); return; }
    try { const res = await execute(formData); setProfileResult(res); } catch (_) {}
  };

  const getRiskClass = (level) =>
    ({ low: 'risk-low', medium: 'risk-medium', high: 'risk-high' }[level?.toLowerCase()] ?? '');

  return (
    <div>
      <p>Assess your investment risk profile based on your financial goals and preferences.</p>
      <div className="form-container">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="goal">Investment Goal:</label>
            <select id="goal" name="goal" value={formData.goal} onChange={handleChange} required>
              {PROFILE_OPTIONS.goals.map(g => <option key={g} value={g}>{g}</option>)}
            </select>
          </div>
          <div className="form-group">
            <label htmlFor="horizon">Investment Horizon: {formData.horizon} years</label>
            <input type="range" id="horizon" name="horizon" min="1" max="30" value={formData.horizon} onChange={handleChange} />
          </div>
          <div className="form-group">
            <label htmlFor="income">Annual Income (₹):</label>
            <input type="number" id="income" name="income" value={formData.income} onChange={handleChange} placeholder="Enter your annual income" min="1" required />
          </div>
          <div className="form-group">
            <label>Risk Appetite:</label>
            <div className="radio-group">
              {PROFILE_OPTIONS.riskAppetites.map(appetite => (
                <label key={appetite}>
                  <input type="radio" name="riskAppetite" value={appetite} checked={formData.riskAppetite === appetite} onChange={handleChange} />
                  {appetite}
                </label>
              ))}
            </div>
          </div>
          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Calculating…' : 'Get Risk Profile'}
          </button>
        </form>
        {validationErr && <div className="error">{validationErr}</div>}
        {error && <div className="error">{error}</div>}
      </div>
      {result?.success && (
        <div className="result-container">
          <h2>Your Risk Profile Results</h2>
          <div className={`risk-score ${getRiskClass(result.risk_level)}`}>Risk Score: {result.risk_score}/12</div>
          <div className="risk-score">Risk Level: <span className={getRiskClass(result.risk_level)}>{result.risk_level}</span></div>
          <div className="recommendations">
            <h3>Investment Recommendations:</h3>
            <ul>{result.recommendations?.map((rec, i) => <li key={i}>{rec}</li>)}</ul>
          </div>
        </div>
      )}
    </div>
  );
}


// ── Trading Strategies ────────────────────────────────────────────────────
const PAIR_STRATEGIES = ['mean_reversion', 'pairs', 'calendar_spread'];

function TradingStrategies() {
  const [ticker, setTicker] = useState('');
  const [pairTicker, setPairTicker] = useState('');
  const [strategyType, setStrategyType] = useState('trend_following');
  const [action, setAction] = useState('signals');
  const [initialCapital, setInitialCapital] = useState(100000);

  const strategiesApi = useApi(getAvailableStrategies);
  const signalsApi    = useApi(getStrategySignals);
  const backtestApi   = useApi(backtestStrategy);
  const { setSignalResult, setBacktestResult } = useAppContext();
  const activeApi = action === 'signals' ? signalsApi : backtestApi;

  useEffect(() => { strategiesApi.execute(); }, []); // eslint-disable-line

  const needsPair = PAIR_STRATEGIES.includes(strategyType);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const payload = { ticker, strategy_type: strategyType, pair_ticker: needsPair ? pairTicker : null, initial_capital: Number(initialCapital) };
    try {
      if (action === 'signals') { const res = await signalsApi.execute(payload); setSignalResult(res); }
      else                      { const res = await backtestApi.execute(payload); setBacktestResult(res); }
    } catch (_) {}
  };

  const getSignalClass = (sig) => ({ BUY: 'signal-buy', SELL: 'signal-sell' }[sig] ?? 'signal-hold');
  const sr = signalsApi.data;
  const br = backtestApi.data;

  return (
    <div>
      <p>Backtest and get trading signals from various quantitative trading strategies.</p>
      <div className="form-container">
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label>Action Type:</label>
            <select value={action} onChange={(e) => setAction(e.target.value)}>
              <option value="signals">Get Current Signals</option>
              <option value="backtest">Backtest Strategy</option>
            </select>
          </div>
          <div className="form-group">
            <label>Trading Strategy:</label>
            <select value={strategyType} onChange={(e) => setStrategyType(e.target.value)}>
              <option value="trend_following">Trend Following (Momentum)</option>
              <option value="mean_reversion">Mean Reversion (Pairs Trading)</option>
              <option value="technical">Technical Indicators</option>
              <option value="calendar_spread">Calendar Spread</option>
            </select>
            {strategiesApi.loading && <small>Loading strategy info…</small>}
            {strategiesApi.data && <small>{strategiesApi.data[strategyType]?.description}</small>}
          </div>
          <div className="form-group">
            <label>Stock Ticker:</label>
            <input type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="e.g., AAPL, MSFT" required />
          </div>
          {needsPair && (
            <div className="form-group">
              <label>Pair / Contract Ticker:</label>
              <input type="text" value={pairTicker} onChange={(e) => setPairTicker(e.target.value)} placeholder="e.g., MSFT for pairs" required />
            </div>
          )}
          {action === 'backtest' && (
            <div className="form-group">
              <label>Initial Capital ($):</label>
              <input type="number" value={initialCapital} onChange={(e) => setInitialCapital(e.target.value)} min="1000" step="1000" />
            </div>
          )}
          <button type="submit" className="btn" disabled={activeApi.loading}>
            {activeApi.loading ? 'Processing…' : action === 'signals' ? 'Get Signals' : 'Run Backtest'}
          </button>
        </form>
        {activeApi.error && <div className="error">{activeApi.error}</div>}
      </div>

      {sr?.success && action === 'signals' && (
        <div className="result-container">
          <h2>Trading Signals for {sr.ticker}</h2>
          <div className={`strategy-signal ${getSignalClass(sr.current_signal.signal_text)}`}>
            <div className="signal-display">
              <h3>Current Signal</h3>
              <div className="signal-value">{sr.current_signal.signal_text}</div>
              <div className="signal-details">
                <p>Price: ${Number(sr.current_signal.current_price).toFixed(2)}</p>
                <p>Position: {sr.current_signal.position}</p>
                <p>Date: {sr.current_signal.date}</p>
              </div>
            </div>
          </div>
          {sr.position_size && (
            <div className="position-recommendation">
              <h3>Position Size Recommendation</h3>
              {typeof sr.position_size === 'number' ? (
                <p>Recommended Shares: {sr.position_size}</p>
              ) : (
                <><p>Hedge Ratio: {sr.position_size.hedge_ratio}</p>
                  <p>Stock Y Shares: {sr.position_size.stock_y_shares}</p>
                  <p>Stock X Shares: {sr.position_size.stock_x_shares}</p></>
              )}
            </div>
          )}
          <div className="recommendation">
            <p><strong>Recommendation:</strong> {sr.recommendation}</p>
          </div>
        </div>
      )}

      {br?.success && action === 'backtest' && br.backtest_results && (
        <div className="result-container">
          <h2>Backtest Results for {br.ticker}</h2>
          <div className="backtest-summary">
            <h3>Performance Summary</h3>
            <div className="backtest-grid">
              <div className="backtest-card">
                <h4>Returns</h4>
                <div className="metric-value">{br.backtest_results.total_return_pct}%</div>
                <p>Initial: ${Number(br.backtest_results.initial_capital).toLocaleString()}</p>
                <p>Final: ${Number(br.backtest_results.final_capital).toLocaleString()}</p>
              </div>
              <div className="backtest-card">
                <h4>Trade Statistics</h4>
                <div className="metric-value">{br.backtest_results.total_trades}</div>
                <p>Total Trades</p>
                <p>Win Rate: {br.backtest_results.win_rate_pct}%</p>
              </div>
              <div className="backtest-card">
                <h4>Risk Metrics</h4>
                <div className="metric-value">{br.backtest_results.max_drawdown_pct}%</div>
                <p>Max Drawdown</p>
                <p>Sharpe: {br.backtest_results.sharpe_ratio}</p>
              </div>
              <div className="backtest-card">
                <h4>Trade Analysis</h4>
                <p>Wins: {br.backtest_results.winning_trades}</p>
                <p>Losses: {br.backtest_results.losing_trades}</p>
                <p>Avg Win: ${Number(br.backtest_results.avg_win).toFixed(2)}</p>
                <p>Avg Loss: ${Number(br.backtest_results.avg_loss).toFixed(2)}</p>
              </div>
            </div>
          </div>
          {br.trades?.length > 0 && (
            <div className="trades-list">
              <h3>Recent Trades</h3>
              <div className="table-container">
                <table>
                  <thead>
                    <tr><th>Date</th><th>Action</th><th>Price</th><th>Exit Date</th><th>Exit Price</th><th>P&amp;L</th><th>Return %</th></tr>
                  </thead>
                  <tbody>
                    {br.trades.map((trade, i) => {
                      const pnl    = trade.pnl       != null ? Number(trade.pnl).toFixed(2) : null;
                      const retPct = trade.return_pct != null ? Number(trade.return_pct).toFixed(2) : null;
                      const exitP  = trade.exit_price != null ? Number(trade.exit_price).toFixed(2) : null;
                      return (
                        <tr key={i}>
                          <td>{trade.date ? new Date(trade.date).toLocaleDateString() : '-'}</td>
                          <td>{trade.action}</td>
                          <td>${Number(trade.price).toFixed(2)}</td>
                          <td>{trade.exit_date ? new Date(trade.exit_date).toLocaleDateString() : '-'}</td>
                          <td>{exitP ? `$${exitP}` : '-'}</td>
                          <td className={pnl != null ? (pnl >= 0 ? 'profit' : 'loss') : ''}>{pnl != null ? `$${pnl}` : '-'}</td>
                          <td className={retPct != null ? (retPct >= 0 ? 'profit' : 'loss') : ''}>{retPct != null ? `${retPct}%` : '-'}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
