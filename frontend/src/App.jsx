import React, { useState, useEffect } from 'react';
import './App.css';
import SeasonalAnalysis from './SeasonalAnalysis';

import { analyzeStock,
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

// TradingView needs an exchange-prefixed symbol (NSE:INFY), not the yfinance
// "INFY.NS" form — that mismatch is why the chart rendered blank.
const tvSymbol = (t) => {
  const s = String(t || '').toUpperCase().trim();
  if (s.endsWith('.NS')) return 'NSE:' + s.slice(0, -3);
  if (s.endsWith('.BO')) return 'BSE:' + s.slice(0, -3);
  return s;
};


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
              placeholder="e.g. RELIANCE, TCS, HDFCBANK" required />
            <small>NSE stocks — the “.NS” is added automatically (BSE: use .BO)</small>
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
          profile={{ riskAppetite, horizon }}
          onAnalysePeer={(t) => runAnalysis(t, { riskAppetite, horizon })} />
      )}
    </div>
  );
}


function StockResult({ result, getScoreClass, profile, onAnalysePeer }) {
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
          src={`https://s.tradingview.com/widgetembed/?symbol=${encodeURIComponent(tvSymbol(result.ticker))}&interval=D&theme=dark&hidevolume=0&hidelegend=0&studies=[]`}
          width="100%" height="380" frameBorder="0" title="Price chart"
        />
        <a className="chart-fallback" href={`https://in.tradingview.com/chart/?symbol=${encodeURIComponent(tvSymbol(result.ticker))}`}
           target="_blank" rel="noopener noreferrer">Chart not loading? Open {tvSymbol(result.ticker)} on TradingView ↗</a>
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

      {/* Merits / demerits — why buy vs why not */}
      {result.merits_analysis && <MeritsCard merits={result.merits_analysis} />}

      {/* Better-performing sector peers as alternatives */}
      {result.peer_analysis && <PeerAlternatives peers={result.peer_analysis} onAnalysePeer={onAnalysePeer} />}

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

      {basis?.guidance?.length > 0 && (
        <div className="portfolio-basis">
          <span className="basis-label">How your profile should be shaped:</span>
          <ul>{basis.guidance.map((r, i) => <li key={i}>{r}</li>)}</ul>
        </div>
      )}
      {portfolio.suggestions?.length > 0 && (
        <ul className="portfolio-suggestions">
          {portfolio.suggestions.map((s, i) => <li key={i}>{s}</li>)}
        </ul>
      )}
      <p className="diversification">
        Diversification score: <strong>{portfolio.diversification_score}</strong>
        {portfolio.diversification_label && <span className="diversification-label"> · {portfolio.diversification_label}</span>}
      </p>
    </div>
  );
}


// ── 3-Model AI Intelligence ────────────────────────────────────────────────
function AISection({ result }) {
  const consensus = result.consensus;
  const models = (consensus?.models || []).filter(m => m.signal);

  const cardMod = {
    'Calibrated RF': 'ai-signal-card--rf',
    'XGBoost': 'ai-signal-card--xgb',
    'Hybrid (rule-aware)': 'ai-signal-card--hgb',
  };
  const fillMod = {
    'Calibrated RF': '',
    'XGBoost': 'confidence-fill--teal',
    'Hybrid (rule-aware)': 'confidence-fill--violet',
  };

  const drivers = [
    { name: 'Calibrated RF',       data: result.feature_importance,     fill: 'drivers-bar-fill--rf',  dot: 'drivers-dot--rf',  head: 'drivers-card-header--rf' },
    { name: 'XGBoost',             data: result.xgb_feature_importance, fill: 'drivers-bar-fill--xgb', dot: 'drivers-dot--xgb', head: 'drivers-card-header--xgb' },
    { name: 'Hybrid (rule-aware)', data: result.hgb_feature_importance, fill: 'drivers-bar-fill--hgb', dot: 'drivers-dot--hgb', head: 'drivers-card-header--hgb' },
  ].filter(d => d.data?.length > 0);

  return (
    <div className="ai-section">
      <div className="ai-section-title">3-Model AI Intelligence</div>

      <div className="ai-signals-row">
        {models.length > 0 ? models.map((m, i) => (
          <div key={i} className={`ai-signal-card ${cardMod[m.name] || 'ai-signal-card--rf'}`}>
            <div className="ai-signal-model-label">{m.name}</div>
            <div className={`ml-signal ${String(m.signal).toLowerCase()}`}>{m.signal}</div>
            <div className="ai-signal-conf">Confidence: {((m.confidence || 0) * 100).toFixed(1)}%</div>
            <div className="confidence-bar">
              <div className={`confidence-fill ${fillMod[m.name] || ''}`} style={{ width: `${(m.confidence || 0) * 100}%` }} />
            </div>
          </div>
        )) : (
          <div className="ai-signal-card ai-signal-card--rf">
            <div className="ai-signal-model-label">Calibrated RF</div>
            <div className={`ml-signal ${result.ml_prediction?.toLowerCase()}`}>{result.ml_prediction}</div>
            <div className="ai-signal-conf">Confidence: {((result.ml_confidence || 0) * 100).toFixed(1)}%</div>
          </div>
        )}
      </div>

      {consensus && <ConsensusPanel consensus={consensus} />}

      {drivers.length > 0 && (
        <>
          <div className="xgb-section-heading">Key Drivers · All Models</div>
          <div className="drivers-grid">
            {drivers.map((d, i) => (
              <div key={i} className="drivers-card">
                <div className={`drivers-card-header ${d.head}`}><span className={`drivers-dot ${d.dot}`} />{d.name}</div>
                <FeatureImportanceBars data={d.data} fillClass={d.fill} />
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}


function ConsensusPanel({ consensus }) {
  const models = (consensus.models || []).filter(m => m.signal);
  if (models.length === 0) return null;
  const agree = consensus.agree;
  const rule = consensus.rule_signal;
  return (
    <>
      <div className="xgb-section-heading">3-Model Consensus</div>
      <div className={`disagree-panel ${agree ? 'ok' : 'warn'}`}>
        <div className="disagree-header">
          <span className="disagree-icon">{agree ? '●' : '△'}</span>
          <h4>{agree
            ? 'All three models agree'
            : `Split — ${consensus.agreement} agree · majority ${consensus.majority}`}</h4>
        </div>
        <div className="disagree-signals">
          {models.map((m, i) => (
            <React.Fragment key={i}>
              {i > 0 && <div className="disagree-vs">vs</div>}
              <div className="disagree-model">
                <span className="disagree-model-name">{m.name}</span>
                <span className={`disagree-signal-chip ${String(m.signal).toLowerCase()}`}>{m.signal}</span>
              </div>
            </React.Fragment>
          ))}
        </div>
        {rule && (
          <p className="consensus-rule">
            Rule-engine cross-check: <strong className={String(rule).toLowerCase()}>{rule}</strong>
          </p>
        )}
        <p className={`disagree-msg ${agree ? 'ok' : 'warn'}`}>
          {agree
            ? 'All three models independently reached the same conclusion, strengthening the signal.'
            : 'The models diverge — treat the majority as tentative and wait for stronger confirmation before acting.'}
        </p>
      </div>
    </>
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


// ── Merits / Demerits — why buy vs why not ─────────────────────────────────
function MeritsCard({ merits }) {
  const pros = merits?.merits || [];
  const cons = merits?.demerits || [];
  if (pros.length === 0 && cons.length === 0) return null;
  return (
    <div className="merits-card">
      <div className="section-label" style={{ margin: '0 0 12px' }}>Why Buy · Why Not</div>
      <div className="merits-grid">
        <div className="merits-col merits-col--pro">
          <div className="merits-col-head merits-col-head--pro"><span className="merits-icon">▲</span> Why to buy</div>
          <ul className="merits-list">
            {pros.map((m, i) => (
              <li key={i}><span className="merits-title">{m.title}</span><span className="merits-detail">{m.detail}</span></li>
            ))}
          </ul>
        </div>
        <div className="merits-col merits-col--con">
          <div className="merits-col-head merits-col-head--con"><span className="merits-icon">▼</span> Why not to buy</div>
          <ul className="merits-list">
            {cons.map((m, i) => (
              <li key={i}><span className="merits-title">{m.title}</span><span className="merits-detail">{m.detail}</span></li>
            ))}
          </ul>
        </div>
      </div>
      <p className="merits-foot">Framed around what investors typically look for — educational, not financial advice.</p>
    </div>
  );
}


// ── Better-performing sector peers ─────────────────────────────────────────
function PeerAlternatives({ peers, onAnalysePeer }) {
  const alts = peers?.alternatives || [];
  const self = peers?.self || {};
  const showAlts = peers.underperforming && alts.length > 0;
  const fmt = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v}%`);
  return (
    <div className="peers-card">
      <div className="section-label" style={{ margin: '0 0 10px' }}>Sector Peers · {peers.sector_group}</div>
      <p className="peers-standing">
        <strong>{self.ticker}</strong> ranks <strong>#{self.rank}</strong> of {self.n} in {peers.sector_group} by 6-month return
        {self.return_6m != null && <> ({fmt(self.return_6m)})</>} · cohort median {fmt(peers.cohort_median_return_6m)}.
      </p>
      {showAlts ? (
        <>
          <p className="peers-hint">Lagging its sector — stronger performers you could compare:</p>
          <div className="peers-grid">
            {alts.map((a, i) => (
              <button key={i} type="button" className="peer-chip"
                onClick={() => onAnalysePeer && onAnalysePeer(a.ticker)} title={`Analyse ${a.ticker}`}>
                <span className="peer-tk">{a.ticker}</span>
                <span className={`peer-ret ${a.return_6m >= 0 ? 'up' : 'down'}`}>{fmt(a.return_6m)} · 6M</span>
                <span className="peer-go">Analyse →</span>
              </button>
            ))}
          </div>
        </>
      ) : (
        <p className="peers-hint">At or above its sector median — no stronger same-sector alternative flagged.</p>
      )}
      <p className="merits-foot">Past 6-month price performance, sector cohort only — not a recommendation to buy.</p>
    </div>
  );
}
