import React, { useState, useEffect } from 'react';
import './App.css';
import SeasonalAnalysis from './SeasonalAnalysis';

import { analyzeStock,
         getAvailableStrategies, backtestStrategy, getStrategySignals,
         getConflicts, analyzeWar,
         useApi, useAppContext, AppProvider } from './services';

export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}

function AppShell() {
  const [view, setView] = useState('analysis'); // 'analysis' | 'war' | 'seasonal'
  const [handoff, setHandoff] = useState(null);  // { ticker, ts } from War Room → Analysis
  const { backendOnline, pingBackend } = useAppContext();
  useEffect(() => { pingBackend(); }, [pingBackend]);

  // Phase 3: jump from a War Room report straight into a ticker's full analysis
  const analyseTicker = (ticker) => {
    if (!ticker) return;
    setHandoff({ ticker: ticker.toUpperCase().trim(), ts: Date.now() });
    setView('analysis');
  };

  return (
    <div className="container">
      <header className="app-header">
        <div className="brand">
          <span className="brand-mark">◈</span>
          <div>
            <h1>Investment Risk Profiler</h1>
            <p className="brand-sub">Multi-factor equity analysis · ML signals · personalised allocation</p>
          </div>
        </div>
        <nav className="app-nav">
          <button className={view === 'analysis' ? 'nav-link active' : 'nav-link'} onClick={() => setView('analysis')}>
            Analysis
          </button>
          <button className={view === 'war' ? 'nav-link active' : 'nav-link'} onClick={() => setView('war')}>
            War Room
          </button>
          <button className={view === 'seasonal' ? 'nav-link active' : 'nav-link'} onClick={() => setView('seasonal')}>
            Seasonal
          </button>
          <span className={`status-dot ${backendOnline === false ? 'offline' : 'online'}`}
            title={backendOnline === false ? 'Backend offline' : 'Backend online'} />
        </nav>
      </header>

      {backendOnline === false && (
        <div className="error" style={{ marginBottom: 16 }}>
          Cannot reach the backend (localhost:5000). Make sure Flask is running.
        </div>
      )}

      {view === 'analysis' && <StockAnalysis handoff={handoff} />}
      {view === 'war' && <WarRoom onAnalyseTicker={analyseTicker} />}
      {view === 'seasonal' && <SeasonalAnalysis />}
    </div>
  );
}


// ── Tone helper: classify a status string as good / bad / neutral ──────────
const GOOD_WORDS = ['bullish', 'positive', 'strong', 'uptrend', 'oversold', 'good', 'buy',
                    'healthy', 'rising', 'accumulation', 'outperform', 'high', 'very positive'];
const BAD_WORDS  = ['bearish', 'negative', 'weak', 'downtrend', 'overbought', 'poor', 'sell',
                    'declining', 'distribution', 'underperform', 'very negative', 'low'];
function toneOf(text) {
  const t = String(text || '').toLowerCase();
  if (GOOD_WORDS.some(w => t.includes(w))) return 'good';
  if (BAD_WORDS.some(w => t.includes(w)))  return 'bad';
  return 'neutral';
}
function toneOfScore(s) {
  if (s == null) return 'neutral';
  if (s >= 60) return 'good';
  if (s < 45)  return 'bad';
  return 'neutral';
}
const titleize = (s) => String(s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());


// ── Stock Analysis (unified page) ──────────────────────────────────────────
function StockAnalysis({ handoff }) {
  const [ticker, setTicker] = useState('');
  const [riskAppetite, setRiskAppetite] = useState('Medium');
  const [horizon, setHorizon] = useState(10);
  const { data, loading, error, execute } = useApi(analyzeStock);
  const result = data?.data;
  const { setStockResult } = useAppContext();

  const runAnalysis = async (sym, profile) => {
    try {
      const res = await execute(sym, profile);
      setStockResult(res);
    } catch (_) {}
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    runAnalysis(ticker, { riskAppetite, horizon });
  };

  // Phase 3: a ticker handed off from the War Room auto-fills and auto-runs.
  useEffect(() => {
    if (!handoff?.ticker) return;
    setTicker(handoff.ticker);
    runAnalysis(handoff.ticker, { riskAppetite, horizon });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }, [handoff?.ts]); // eslint-disable-line react-hooks/exhaustive-deps

  const getScoreClass = (score) => {
    if (score >= 75) return 'score-excellent';
    if (score >= 65) return 'score-good';
    if (score >= 45) return 'score-fair';
    if (score >= 25) return 'score-poor';
    return 'score-very-poor';
  };

  return (
    <div>
      {/* ── Control panel ── */}
      <div className="form-container">
        <form onSubmit={handleSubmit} className="analyze-form">
          <div className="form-group form-group--ticker">
            <label htmlFor="ticker">Stock Ticker</label>
            <input type="text" id="ticker" value={ticker} onChange={(e) => setTicker(e.target.value)}
              placeholder="e.g. AAPL, MSFT, RELIANCE.NS" required />
            <small>US: AAPL · India: RELIANCE.NS</small>
          </div>

          <div className="form-group">
            <label>Risk Appetite</label>
            <div className="segmented">
              {['Low', 'Medium', 'High'].map(opt => (
                <button type="button" key={opt}
                  className={riskAppetite === opt ? 'seg active' : 'seg'}
                  onClick={() => setRiskAppetite(opt)}>
                  {opt}
                </button>
              ))}
            </div>
            <small>Shapes your portfolio allocation</small>
          </div>

          <div className="form-group">
            <label htmlFor="horizon">Horizon · {horizon} {horizon === 1 ? 'year' : 'years'}</label>
            <input type="range" id="horizon" min="1" max="30" value={horizon}
              onChange={(e) => setHorizon(Number(e.target.value))} />
            <small>Longer horizons tolerate more equity</small>
          </div>

          <button type="submit" className="btn btn--primary" disabled={loading}>
            {loading ? 'Analysing…' : 'Analyse Stock'}
          </button>
        </form>
        {error && <div className="error">{error}</div>}
      </div>

      {loading && (
        <div className="result-container skeleton-block">
          <div className="sk-line sk-line--lg" />
          <div className="sk-grid">
            {Array.from({ length: 5 }).map((_, i) => <div key={i} className="sk-card" />)}
          </div>
        </div>
      )}

      {!loading && result?.success && (
        <StockResult result={result} getScoreClass={getScoreClass}
          profile={{ riskAppetite, horizon }} />
      )}
    </div>
  );
}


function StockResult({ result, getScoreClass, profile }) {
  const t  = result.technical_analysis || {};
  const f  = result.fundamental_analysis || {};
  const se = result.sentiment_analysis || {};
  const v  = result.volumetric_analysis || {};
  const sc = result.sector_comparison || {};
  const g  = result.geopolitical_analysis || {};

  // ── Build explainability bullets per analysis from backend detail data ──
  const techExplain = [
    { text: `RSI: ${t.details?.rsi_status ?? t.rsi_status ?? 'N/A'}`, tone: toneOf(t.details?.rsi_status ?? t.rsi_status) },
    { text: `MACD: ${t.details?.macd_signal ?? 'N/A'}`, tone: toneOf(t.details?.macd_signal) },
    { text: `Trend: ${t.details?.trend ?? t.trend ?? 'N/A'}`, tone: toneOf(t.details?.trend ?? t.trend) },
  ];

  const fundExplain = Object.entries(f.details || {})
    .filter(([, val]) => val && typeof val === 'object' && 'score' in val)
    .map(([cat, val]) => ({
      text: `${titleize(cat)}: ${val.score}/${val.max_score ?? '—'}`,
      tone: toneOfScore(val.max_score ? (val.score / val.max_score) * 100 : null),
    }));

  const sentExplain = [
    ...(se.reasoning?.summary ? [{ text: se.reasoning.summary, tone: toneOfScore(se.score) }] : []),
    ...(se.reasoning?.positive_drivers || []).slice(0, 2).map(d => ({ text: `▲ ${d.title}`, tone: 'good' })),
    ...(se.reasoning?.negative_drivers || []).slice(0, 2).map(d => ({ text: `▼ ${d.title}`, tone: 'bad' })),
  ];

  const volExplain = Object.entries(v.details || {})
    .filter(([, val]) => val && typeof val === 'object')
    .map(([cat, val]) => ({ text: `${titleize(cat)}: ${val.status ?? 'N/A'}`, tone: toneOf(val.status) }));

  const sectorExplain = [
    { text: `vs Market: ${sc.vs_market?.outperformance ?? 0}%`, tone: toneOfScore((sc.vs_market?.outperformance ?? 0) >= 0 ? 60 : 40) },
    { text: `vs Sector: ${sc.vs_sector?.outperformance ?? 0}%`, tone: toneOfScore((sc.vs_sector?.outperformance ?? 0) >= 0 ? 60 : 40) },
    { text: `Sector index: ${sc.vs_market?.market_index ?? 'N/A'}`, tone: 'neutral' },
  ];

  const geoExplain = (g.drivers || []).map(d => ({
    text: `${d.conflict} → ${d.sector}${d.rationale ? ` — ${d.rationale}` : ''}`,
    tone: d.direction === 'benefit' ? 'good' : 'bad',
  }));

  return (
    <div className="result-container">
      <div className="result-head">
        <div>
          <h2>{result.ticker}</h2>
          <p className="result-sub">Comprehensive multi-factor analysis</p>
        </div>
        <span className={`pill pill--${getScoreClass(result.overall_score.score).replace('score-', '')}`}>
          {result.risk_level}
        </span>
      </div>

      {/* Hero: overall score */}
      <div className={`overall-score ${getScoreClass(result.overall_score.score)}`}>
        <div className="score-display">
          <h3>Overall Score</h3>
          <div className="score-value">{result.overall_score.score}<span className="score-max">/100</span></div>
          <div className="risk-level">{result.risk_level}</div>
        </div>
      </div>

      {/* Price chart */}
      <div className="chart-wrap">
        <iframe
          src={`https://s.tradingview.com/widgetembed/?symbol=${result.ticker}&interval=D&theme=dark`}
          width="100%" height="380" frameBorder="0" title="Price chart"
        />
      </div>

      {/* Analysis grid with explainability */}
      <div className="section-label">Factor Breakdown</div>
      <div className="analysis-grid">
        <AnalysisCard title="Technical" score={t.score} signal={t.signal} explain={techExplain} />
        <AnalysisCard title="Fundamental" score={f.score} signal={f.signal} explain={fundExplain} />
        <AnalysisCard title="Sentiment" score={se.score} signal={se.sentiment}
          items={[
            `${se.sample_size ?? 0} articles · ${se.confidence ?? 'N/A'} conf`,
            `Sector mood: ${se.sector_sentiment?.score ?? 'N/A'} (${se.sector_sentiment?.label ?? 'N/A'})`,
            `Divergence: ${se.divergence?.label ?? 'N/A'} (${(se.divergence?.value ?? 0) >= 0 ? '+' : ''}${se.divergence?.value ?? 0})`,
          ]}
          explain={sentExplain} />
        <AnalysisCard title="Volumetric" score={v.score} signal={v.signal} explain={volExplain} />
        <AnalysisCard title="Sector" score={sc.score} signal={`${sc.vs_market?.outperformance ?? 0}% vs Market`}
          items={[`Sector: ${sc.sector ?? 'N/A'}`]} explain={sectorExplain} />
        <AnalysisCard title="Geopolitical" score={g.score} signal={g.signal}
          items={[g.exposed ? g.summary : 'No exposure to tracked conflicts']}
          explain={geoExplain} />
      </div>

      {/* Dual-model AI section */}
      {result.ml_prediction && <AISection result={result} />}

      {/* Explanation */}
      {result.explanation?.length > 0 && (
        <div className="callout">
          <h4>Why this prediction?</h4>
          <ul className="callout-list">
            {result.explanation.map((exp, i) => <li key={i}>{exp}</li>)}
          </ul>
        </div>
      )}

      {/* Gemini insight */}
      {result.gemini_insight && !String(result.gemini_insight).toLowerCase().includes('unavailable') && (
        <div className="callout callout--ai">
          <h4>AI Investment Insight</h4>
          <p>{result.gemini_insight}</p>
        </div>
      )}

      {/* Portfolio allocation (profile-driven) */}
      {result?.portfolio?.allocation && <PortfolioCard portfolio={result.portfolio} profile={profile} />}

      {/* Recommendations */}
      {result.recommendations?.length > 0 && (
        <div className="recommendations">
          <div className="section-label">Recommendations</div>
          {result.recommendations.map((rec, i) => (
            <div key={i} className="recommendation-item">
              <span className={`priority-${rec.priority.toLowerCase()}`}>{rec.priority}</span>
              <span className="rec-type">{rec.type}</span>
              <span className="rec-message">{rec.message}</span>
            </div>
          ))}
        </div>
      )}

      {/* Nested Trading Strategies */}
      <TradingStrategies defaultTicker={result.ticker} />
    </div>
  );
}


// ── Analysis card with expandable explainability ───────────────────────────
function AnalysisCard({ title, score, signal, items = [], explain = [] }) {
  const [open, setOpen] = useState(false);
  const tone = toneOfScore(score);
  return (
    <div className={`analysis-card ac--${tone}`}>
      <div className="ac-head">
        <h4>{title}</h4>
        <span className={`ac-score ac-score--${tone}`}>{score != null ? Number(score).toFixed(0) : 'N/A'}</span>
      </div>
      {signal && <p className="ac-signal">{signal}</p>}
      {items.length > 0 && <ul className="ac-items">{items.map((it, i) => <li key={i}>{it}</li>)}</ul>}
      {explain.length > 0 && (
        <div className="ac-explain-wrap">
          <button type="button" className="ac-why" onClick={() => setOpen(o => !o)} aria-expanded={open}>
            <span>Why this score?</span>
            <span className={`ac-chev ${open ? 'up' : ''}`}>⌄</span>
          </button>
          {open && (
            <ul className="ac-explain">
              {explain.map((e, i) => (
                <li key={i} className={`ex-${e.tone || 'neutral'}`}>
                  <span className="ex-dot" />{e.text}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}


function PortfolioCard({ portfolio, profile }) {
  if (!portfolio) return null;
  const basis = portfolio.profile_basis;
  const colorFor = (type) => ({
    Equity: 'var(--accent)', Debt: 'var(--blue)', Commodity: 'var(--amber)', Cash: 'var(--text-secondary)',
  }[type] || 'var(--accent)');

  return (
    <div className="portfolio-card">
      <div className="portfolio-head">
        <div className="section-label" style={{ margin: 0 }}>Portfolio Allocation</div>
        {profile && (
          <span className="portfolio-profile">
            {profile.riskAppetite} risk · {profile.horizon}y horizon
          </span>
        )}
      </div>

      {/* Stacked allocation bar */}
      <div className="alloc-bar">
        {portfolio.allocation.map((item, i) => (
          <div key={i} className="alloc-seg"
            style={{ width: `${item.weight}%`, background: colorFor(item.type) }}
            title={`${item.asset}: ${item.weight}%`} />
        ))}
      </div>

      <div className="alloc-legend">
        {portfolio.allocation.map((item, i) => (
          <div key={i} className="alloc-row">
            <span className="alloc-key">
              <span className="alloc-swatch" style={{ background: colorFor(item.type) }} />
              {item.asset}
            </span>
            <strong>{item.weight}%</strong>
          </div>
        ))}
      </div>

      {basis?.reasons?.length > 0 && (
        <div className="portfolio-basis">
          <span className="basis-label">How your profile shaped this:</span>
          <ul>{basis.reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </div>
      )}
      {portfolio.suggestions?.length > 0 && (
        <ul className="portfolio-suggestions">
          {portfolio.suggestions.map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      )}
      <p className="diversification">Diversification score: <strong>{portfolio.diversification_score}</strong></p>
    </div>
  );
}


// ── Dual-model AI section (extracted, unchanged logic) ─────────────────────
function AISection({ result }) {
  return (
    <div className="ai-section">
      <div className="ai-section-title">Dual-Model AI Intelligence</div>
      <div className="ai-signals-row">
        <div className="ai-signal-card ai-signal-card--rf">
          <div className="ai-signal-model-label">Calibrated RF</div>
          <div className={`ml-signal ${result.ml_prediction?.toLowerCase()}`}>{result.ml_prediction}</div>
          <div className="ai-signal-conf">Confidence: {((result.ml_confidence || 0) * 100).toFixed(1)}%</div>
          <div className="confidence-bar">
            <div className="confidence-fill" style={{ width: `${(result.ml_confidence || 0) * 100}%` }} />
          </div>
        </div>
        {result.xgb_prediction && (
          <div className="ai-signal-card ai-signal-card--xgb">
            <div className="ai-signal-model-label">XGBoost</div>
            <div className={`ml-signal ${result.xgb_prediction?.toLowerCase()}`}>{result.xgb_prediction}</div>
            <div className="ai-signal-conf">Confidence: {((result.xgb_confidence || 0) * 100).toFixed(1)}%</div>
            <div className="confidence-bar">
              <div className="confidence-fill confidence-fill--teal" style={{ width: `${(result.xgb_confidence || 0) * 100}%` }} />
            </div>
          </div>
        )}
        {result.comparison && (
          <div className="ai-signal-card ai-signal-card--rule">
            <div className="ai-signal-model-label">Rule vs ML</div>
            <div className="rule-vs-row"><span className="rule-vs-key">Rule Score</span><span className="rule-vs-val">{result.comparison.rule_score}</span></div>
            <div className="rule-vs-row"><span className="rule-vs-key">ML Signal</span>
              <span className={`rule-vs-val ml-signal-sm ${result.comparison.ml_signal?.toLowerCase()}`}>{result.comparison.ml_signal}</span>
            </div>
          </div>
        )}
      </div>

      {result.xgb_probabilities && (
        <>
          <div className="xgb-section-heading">XGBoost · Probability Distribution</div>
          <XGBProbabilityPanel probabilities={result.xgb_probabilities} />
        </>
      )}

      {result.xgb_prediction && (
        <>
          <div className="xgb-section-heading">Dual-Model Consensus</div>
          <ModelDisagreementPanel rfSignal={result.ml_prediction} xgbSignal={result.xgb_prediction} disagreement={result.model_disagreement} />
        </>
      )}

      {(result.feature_importance?.length > 0 || result.xgb_feature_importance?.length > 0) && (
        <>
          <div className="xgb-section-heading">Key Drivers · Both Models</div>
          <div className="drivers-grid">
            {result.feature_importance?.length > 0 && (
              <div className="drivers-card">
                <div className="drivers-card-header drivers-card-header--rf"><span className="drivers-dot drivers-dot--rf" />Calibrated RF</div>
                <FeatureImportanceBars data={result.feature_importance} fillClass="drivers-bar-fill--rf" />
              </div>
            )}
            {result.xgb_feature_importance?.length > 0 && (
              <div className="drivers-card">
                <div className="drivers-card-header drivers-card-header--xgb"><span className="drivers-dot drivers-dot--xgb" />XGBoost</div>
                <FeatureImportanceBars data={result.xgb_feature_importance} fillClass="drivers-bar-fill--xgb" />
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}


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
              <span className="prob-icon" style={{ color: cls === 'buy' ? 'var(--green)' : cls === 'hold' ? 'var(--amber)' : 'var(--red)' }}>{icon}</span>
              <span className="prob-name">{label}</span>
            </div>
            <div className="prob-track"><div className={`prob-fill ${cls}`} style={{ width: `${(value * 100).toFixed(1)}%` }} /></div>
            <span className={`prob-pct ${cls}`}>{(value * 100).toFixed(1)}%</span>
          </div>
        ))}
      </div>
      <p className="prob-note">XGBoost softmax probabilities across SELL / HOLD / BUY classes</p>
    </div>
  );
}


function ModelDisagreementPanel({ rfSignal, xgbSignal, disagreement }) {
  if (!rfSignal || !xgbSignal) return null;
  return (
    <div className={`disagree-panel ${disagreement ? 'warn' : 'ok'}`}>
      <div className="disagree-header">
        <span className="disagree-icon">{disagreement ? '△' : '●'}</span>
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


function FeatureImportanceBars({ data, fillClass }) {
  if (!data || data.length === 0) return null;
  const top = [...data].sort((a, b) => b.importance - a.importance).slice(0, 6);
  const max = top[0]?.importance || 1;
  return (
    <div className="fi-bars">
      {top.map((f, i) => (
        <div key={i} className="fi-row">
          <div className="fi-label"><span className="fi-name">{f.feature}</span><span className="fi-pct">{(f.importance * 100).toFixed(1)}%</span></div>
          <div className="fi-track"><div className={`fi-fill ${fillClass}`} style={{ width: `${(f.importance / max) * 100}%` }} /></div>
        </div>
      ))}
    </div>
  );
}


// ── War Room (India-specific geopolitical analysis) ────────────────────────
function WarRoom({ onAnalyseTicker }) {
  const conflictsApi = useApi(getConflicts);
  const analysisApi = useApi(analyzeWar);
  const [scenario, setScenario] = useState('');
  const [activeId, setActiveId] = useState(null);

  useEffect(() => { conflictsApi.execute(); }, []); // eslint-disable-line

  const conflicts = conflictsApi.data?.conflicts || [];
  const res = analysisApi.data;

  const runConflict = (id) => { setActiveId(id); setScenario(''); analysisApi.execute({ conflictId: id }); };
  const runScenario = (e) => {
    e.preventDefault();
    if (!scenario.trim()) return;
    setActiveId(null);
    analysisApi.execute({ scenarioText: scenario.trim() });
  };

  const biasClass = (b) => ({ 'risk-on': 'good', 'risk-off': 'bad', mixed: 'neutral' }[b] || 'neutral');

  return (
    <div>
      <div className="war-intro">
        <p>India-specific geopolitical risk analysis. Pick a tracked conflict or describe a scenario — see how it ripples through Indian sectors and stocks via trade exposure and bilateral ties.</p>
        <span className="war-disclaimer">Educational analysis · not financial advice</span>
      </div>

      <div className="section-label">Conflict Watchlist</div>
      {conflictsApi.loading && <p>Loading watchlist…</p>}
      {conflictsApi.error && <div className="error">{conflictsApi.error}</div>}
      <div className="war-grid">
        {conflicts.map(c => (
          <button key={c.id} type="button"
            className={`war-card ${activeId === c.id ? 'active' : ''}`}
            onClick={() => runConflict(c.id)}>
            <div className="war-card-head">
              <span className="war-card-name">{c.name}</span>
              <span className={`war-status war-status--${c.status}`}>{titleize(c.status)}</span>
            </div>
            <div className="war-parties">{(c.parties || []).join('  vs  ')}</div>
            <p className="war-card-sum">{c.summary}</p>
            <span className="war-relevance">India relevance: {titleize(c.india_relevance)}</span>
          </button>
        ))}
      </div>

      <div className="section-label">Or describe a scenario</div>
      <form className="war-scenario" onSubmit={runScenario}>
        <input type="text" value={scenario} onChange={(e) => setScenario(e.target.value)}
          placeholder="e.g. What if the Israel–Iran conflict closes the Strait of Hormuz?" />
        <button type="submit" className="btn btn--primary" disabled={analysisApi.loading}>
          {analysisApi.loading ? 'Analysing…' : 'Analyse'}
        </button>
      </form>

      {analysisApi.error && <div className="error">{analysisApi.error}</div>}
      {analysisApi.loading && (
        <div className="result-container"><p>Tracing transmission channels into Indian markets…</p></div>
      )}
      {!analysisApi.loading && res?.success && (
        <WarReport res={res} biasClass={biasClass} onAnalyseTicker={onAnalyseTicker} />
      )}
    </div>
  );
}


function WarReport({ res, biasClass, onAnalyseTicker }) {
  const report = res.report;
  const magDots = (m) => '●'.repeat(Math.max(0, Math.min(5, m || 0))) + '○'.repeat(5 - Math.max(0, Math.min(5, m || 0)));

  if (!report) {
    return (
      <div className="result-container">
        <h2>{res.conflict?.name}</h2>
        <p className="war-summary">{res.raw_text}</p>
      </div>
    );
  }

  return (
    <div className="result-container">
      <div className="result-head">
        <div>
          <h2>{res.conflict?.name}</h2>
          <p className="result-sub">{(res.conflict?.parties || []).join('  vs  ')}</p>
        </div>
        <span className={`pill pill--${biasClass(report.net_bias)}`}>
          {report.net_bias} · {report.confidence} confidence
        </span>
      </div>

      <p className="war-summary">{report.india_summary}</p>

      <div className="war-section-head">
        <div className="section-label" style={{ margin: 0 }}>Transmission Channels</div>
        <span className="war-tip">Tap a ticker to run its full analysis →</span>
      </div>
      {report.transmission_channels?.map((ch, i) => (
        <div key={i} className="war-channel">
          <div className="war-channel-head">
            <span className="war-driver">{ch.driver}</span>
            <span className={`war-dir war-dir--${ch.direction}`}>{ch.direction === 'up' ? '↑' : '↓'} {ch.direction}</span>
          </div>
          <div className="war-affected">
            {ch.affected?.map((a, j) => (
              <div key={j} className={`war-impact war-impact--${a.impact}`}>
                <div className="war-impact-top">
                  <span className="war-sector">{a.sector}</span>
                  <span className="war-mag" title={`Magnitude ${a.magnitude}/5`}>{magDots(a.magnitude)}</span>
                </div>
                <p className="war-rationale">{a.rationale}</p>
                {a.tickers?.length > 0 && (
                  <div className="war-tickers">
                    {a.tickers.map((t, k) => (
                      <button key={k} type="button" className="war-ticker war-ticker--btn"
                        onClick={() => onAnalyseTicker && onAnalyseTicker(t)}
                        title={`Analyse ${t}`}>
                        {t} <span className="war-ticker-go">→</span>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      ))}

      {report.watch_items?.length > 0 && (
        <div className="callout">
          <h4>Watch items — what would confirm or escalate this</h4>
          <ul className="callout-list">{report.watch_items.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}

      {report.caveats?.length > 0 && (
        <div className="callout callout--ai">
          <h4>Caveats</h4>
          <ul className="callout-list">{report.caveats.map((w, i) => <li key={i}>{w}</li>)}</ul>
        </div>
      )}

      {res.grounding && (
        <p className="war-grounding">
          Grounded on — parties: {(res.grounding.parties_profiled || []).join(', ') || '—'} ·
          commodities: {(res.grounding.commodities || []).join(', ') || '—'}
        </p>
      )}
      <p className="war-disclaimer-foot">{res.disclaimer || 'Educational geopolitical-risk analysis, not financial advice.'}</p>
    </div>
  );
}


// ── Trading Strategies (nested under Stock Analysis) ───────────────────────
const PAIR_STRATEGIES = ['mean_reversion', 'pairs', 'calendar_spread'];

function TradingStrategies({ defaultTicker = '' }) {
  const [open, setOpen] = useState(false);
  const [ticker, setTicker] = useState(defaultTicker);
  const [pairTicker, setPairTicker] = useState('');
  const [strategyType, setStrategyType] = useState('trend_following');
  const [action, setAction] = useState('signals');
  const [initialCapital, setInitialCapital] = useState(100000);

  const strategiesApi = useApi(getAvailableStrategies);
  const signalsApi    = useApi(getStrategySignals);
  const backtestApi   = useApi(backtestStrategy);
  const { setSignalResult, setBacktestResult } = useAppContext();
  const activeApi = action === 'signals' ? signalsApi : backtestApi;

  useEffect(() => { if (open) strategiesApi.execute(); }, [open]); // eslint-disable-line
  useEffect(() => { setTicker(defaultTicker); }, [defaultTicker]);

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
    <div className="strategies-section">
      <button type="button" className="strategies-toggle" onClick={() => setOpen(o => !o)} aria-expanded={open}>
        <span className="section-label" style={{ margin: 0 }}>Trading Strategies</span>
        <span className="strategies-hint">{open ? 'Hide' : 'Backtest & signals'} <span className={`ac-chev ${open ? 'up' : ''}`}>⌄</span></span>
      </button>

      {open && (
        <div className="strategies-body">
          <form onSubmit={handleSubmit} className="strategies-form">
            <div className="form-group">
              <label>Action</label>
              <select value={action} onChange={(e) => setAction(e.target.value)}>
                <option value="signals">Get Current Signals</option>
                <option value="backtest">Backtest Strategy</option>
              </select>
            </div>
            <div className="form-group">
              <label>Strategy</label>
              <select value={strategyType} onChange={(e) => setStrategyType(e.target.value)}>
                <option value="trend_following">Trend Following (Momentum)</option>
                <option value="mean_reversion">Mean Reversion (Pairs)</option>
                <option value="technical">Technical Indicators</option>
                <option value="calendar_spread">Calendar Spread</option>
              </select>
              {strategiesApi.data && <small>{strategiesApi.data[strategyType]?.description}</small>}
            </div>
            <div className="form-group">
              <label>Ticker</label>
              <input type="text" value={ticker} onChange={(e) => setTicker(e.target.value)} placeholder="e.g. AAPL" required />
            </div>
            {needsPair && (
              <div className="form-group">
                <label>Pair / Contract</label>
                <input type="text" value={pairTicker} onChange={(e) => setPairTicker(e.target.value)} placeholder="e.g. MSFT" required />
              </div>
            )}
            {action === 'backtest' && (
              <div className="form-group">
                <label>Initial Capital ($)</label>
                <input type="number" value={initialCapital} onChange={(e) => setInitialCapital(e.target.value)} min="1000" step="1000" />
              </div>
            )}
            <button type="submit" className="btn btn--primary" disabled={activeApi.loading}>
              {activeApi.loading ? 'Processing…' : action === 'signals' ? 'Get Signals' : 'Run Backtest'}
            </button>
          </form>
          {activeApi.error && <div className="error">{activeApi.error}</div>}

          {sr?.success && action === 'signals' && (
            <div className="strategy-result">
              <h3>Signal · {sr.ticker}</h3>
              <div className={`strategy-signal ${getSignalClass(sr.current_signal.signal_text)}`}>
                <div className="signal-display">
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
                  <h4>Position Size</h4>
                  {typeof sr.position_size === 'number'
                    ? <p>Recommended Shares: {sr.position_size}</p>
                    : <><p>Hedge Ratio: {sr.position_size.hedge_ratio}</p><p>Stock Y Shares: {sr.position_size.stock_y_shares}</p><p>Stock X Shares: {sr.position_size.stock_x_shares}</p></>}
                </div>
              )}
              <div className="recommendation"><p><strong>Recommendation:</strong> {sr.recommendation}</p></div>
            </div>
          )}

          {br?.success && action === 'backtest' && br.backtest_results && (
            <div className="strategy-result">
              <h3>Backtest · {br.ticker}</h3>
              <div className="backtest-grid">
                <div className="backtest-card">
                  <h4>Returns</h4>
                  <div className="metric-value">{br.backtest_results.total_return_pct}%</div>
                  <p>Initial: ${Number(br.backtest_results.initial_capital).toLocaleString()}</p>
                  <p>Final: ${Number(br.backtest_results.final_capital).toLocaleString()}</p>
                </div>
                <div className="backtest-card">
                  <h4>Trades</h4>
                  <div className="metric-value">{br.backtest_results.total_trades}</div>
                  <p>Win Rate: {br.backtest_results.win_rate_pct}%</p>
                </div>
                <div className="backtest-card">
                  <h4>Risk</h4>
                  <div className="metric-value">{br.backtest_results.max_drawdown_pct}%</div>
                  <p>Max Drawdown</p>
                  <p>Sharpe: {br.backtest_results.sharpe_ratio}</p>
                </div>
                <div className="backtest-card">
                  <h4>Breakdown</h4>
                  <p>Wins: {br.backtest_results.winning_trades}</p>
                  <p>Losses: {br.backtest_results.losing_trades}</p>
                  <p>Avg Win: ${Number(br.backtest_results.avg_win).toFixed(2)}</p>
                  <p>Avg Loss: ${Number(br.backtest_results.avg_loss).toFixed(2)}</p>
                </div>
              </div>
              {br.trades?.length > 0 && (
                <div className="table-container">
                  <table>
                    <thead><tr><th>Date</th><th>Action</th><th>Price</th><th>Exit Date</th><th>Exit Price</th><th>P&amp;L</th><th>Return %</th></tr></thead>
                    <tbody>
                      {br.trades.map((trade, i) => {
                        const pnl    = trade.pnl        != null ? Number(trade.pnl).toFixed(2) : null;
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
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}
