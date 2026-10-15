import React, { useState, useEffect } from 'react';
import './App.css';
import SeasonalAnalysis from './SeasonalAnalysis';
import GeoAnalyzer from './GeoAnalyzer';

import { analyzeStock, useApi, useAppContext, AppProvider } from './services';

export default function App() {
  return (
    <AppProvider>
      <AppShell />
    </AppProvider>
  );
}

function useTheme() {
  const [theme, setTheme] = useState(() => {
    try { return localStorage.getItem('theme') || 'dark'; } catch (_) { return 'dark'; }
  });
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('theme', theme); } catch (_) {}
  }, [theme]);
  return [theme, () => setTheme(t => (t === 'dark' ? 'light' : 'dark'))];
}

function AppShell() {
  const [theme, toggleTheme] = useTheme();
  const [view, setView] = useState('analysis'); // 'analysis' | 'war' | 'seasonal'
  const [handoff, setHandoff] = useState(null);  // { ticker, ts } from the Geopolitical Analyzer → Analysis
  const { backendOnline, pingBackend } = useAppContext();
  useEffect(() => { pingBackend(); }, [pingBackend]);

  // Phase 3: jump from a Geopolitical Analyzer report straight into a ticker's full analysis
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
            <p className="brand-sub">Multi-factor NSE equity analysis · calibrated ML · personalised allocation</p>
          </div>
        </div>
        <nav className="app-nav">
          <button className={view === 'analysis' ? 'nav-link active' : 'nav-link'} onClick={() => setView('analysis')}>
            Analysis
          </button>
          <button className={view === 'war' ? 'nav-link active' : 'nav-link'} onClick={() => setView('war')}>
            Geopolitical Analyzer
          </button>
          <button className={view === 'seasonal' ? 'nav-link active' : 'nav-link'} onClick={() => setView('seasonal')}>
            Seasonal
          </button>
          <button type="button" className="theme-toggle" onClick={toggleTheme}
            aria-label="Toggle light/dark theme" title="Toggle light/dark theme">
            {theme === 'dark' ? '☀' : '☾'}
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
      {view === 'war' && <GeoAnalyzer onAnalyseTicker={analyseTicker} />}
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

// The models predict RELATIVE performance vs the Nifty (OUTPERFORM /
// UNDERPERFORM). Map those onto the existing green/red signal styles.
const sigClass = (s) => {
  const v = String(s || '').toUpperCase();
  if (v === 'OUTPERFORM') return 'buy';
  if (v === 'UNDERPERFORM') return 'sell';
  return 'hold';          // NO_CLEAR_EDGE and anything unknown
};
const sigLabel = (s) => {
  const v = String(s || '').toUpperCase();
  if (v === 'OUTPERFORM') return 'Outperform';
  if (v === 'UNDERPERFORM') return 'Underperform';
  if (v === 'NO_CLEAR_EDGE') return 'No clear edge';
  return s;
};
const pct1 = (p) => (p == null ? '—' : `${(p * 100).toFixed(1)}%`);

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

  // Phase 3: a ticker handed off from the Geopolitical Analyzer auto-fills and auto-runs.
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

      <SectionNav />

      {/* Hero: overall score ring + how each factor contributes */}
      <div id="sec-overview" className="anchor" />
      <Hero result={result} tone={getScoreClass(result.overall_score.score)} />

      {/* Price chart */}
      <div id="sec-stats" className="anchor" />
      <StockSnapshot snap={result.snapshot} ticker={result.ticker} />

      <PriceChart history={result.price_history} ticker={result.ticker} />

      {/* Analysis grid with explainability */}
      <div id="sec-factors" className="anchor" />
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
          // Surface which scorer actually ran. A silent drop to the keyword
          // engine used to be invisible, hiding a badly degraded signal.
          badge={se.engine === 'finbert'
            ? { text: 'FinBERT', tone: 'good' }
            : se.engine
              ? { text: 'keyword fallback', tone: 'bad' }
              : null}
          explain={sentExplain} />
        <AnalysisCard title="Volumetric" score={v.score} signal={v.signal} explain={volExplain} />
        <AnalysisCard title="Sector" score={sc.score} signal={`${sc.vs_market?.outperformance ?? 0}% vs Market`}
          items={[`Sector: ${sc.sector ?? 'N/A'}`]} explain={sectorExplain} />
        <AnalysisCard title="Geopolitical" score={g.score} signal={g.signal}
          items={g.exposed
            ? [g.summary,
               `Percentile ${g.calibration?.percentile ?? '—'} of ${g.calibration?.universe ?? '—'} covered stocks`,
               `Exposure strength ${Math.round((g.strength ?? 0) * 100)}%`]
            : ['No exposure to tracked conflicts']}
          badge={g.exposed ? { text: 'calibrated', tone: 'good' } : null}
          explain={[...geoExplain, ...(g.calibration?.note ? [{ text: g.calibration.note, tone: 'neutral' }] : [])]} />
      </div>

      {/* Dual-model AI section */}
      <div id="sec-ai" className="anchor" />
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
      <div id="sec-merits" className="anchor" />
      {result.merits_analysis && <MeritsCard merits={result.merits_analysis} />}

      {/* Better-performing sector peers as alternatives */}
      {result.peer_analysis && <PeerAlternatives peers={result.peer_analysis} onAnalysePeer={onAnalysePeer} />}

      {/* Portfolio allocation (profile-driven) */}
      <div id="sec-portfolio" className="anchor" />
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


// ── Price chart (1Y, drawn from backend data) ──────────────────────────────
function PriceChart({ history, ticker }) {
  const [hover, setHover] = useState(null);
  const pts = history || [];
  const link = `https://in.tradingview.com/chart/?symbol=${encodeURIComponent(tvSymbol(ticker))}`;
  if (pts.length < 2) {
    return <div className="chart-wrap chart-wrap--empty"><a className="chart-fallback" href={link} target="_blank" rel="noopener noreferrer">Open {tvSymbol(ticker)} on TradingView ↗</a></div>;
  }
  const W = 720, H = 220, PX = 8, PY = 14;
  const cs = pts.map(p => p.c);
  const lo = Math.min(...cs), hi = Math.max(...cs);
  const x = (i) => PX + (i / (pts.length - 1)) * (W - 2 * PX);
  const y = (v) => PY + (1 - (v - lo) / ((hi - lo) || 1)) * (H - 2 * PY);
  const line = pts.map((p, i) => `${i ? 'L' : 'M'}${x(i).toFixed(1)},${y(p.c).toFixed(1)}`).join(' ');
  const area = `${line} L${x(pts.length - 1).toFixed(1)},${H - PY} L${x(0).toFixed(1)},${H - PY} Z`;
  const first = cs[0], last = cs[cs.length - 1];
  const up = last >= first;
  const onMove = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width * W - PX) / (W - 2 * PX) * (pts.length - 1));
    setHover(Math.max(0, Math.min(pts.length - 1, i)));
  };
  const cur = hover != null ? pts[hover] : pts[pts.length - 1];
  return (
    <div className="chart-wrap">
      <div className="chart-head">
        <div>
          <span className="chart-price">₹{cur.c.toLocaleString('en-IN')}</span>
          <span className="chart-date">{cur.d}</span>
        </div>
      </div>
      <svg className={`chart-svg ${up ? 'up' : 'down'}`} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none"
        onMouseMove={onMove} onMouseLeave={() => setHover(null)} role="img" aria-label={`${ticker} one-year price chart`}>
        <path className="chart-area" d={area} />
        <path className="chart-line" d={line} />
        {hover != null && (
          <g>
            <line className="chart-cross" x1={x(hover)} x2={x(hover)} y1={PY} y2={H - PY} />
            <circle className="chart-dot" cx={x(hover)} cy={y(pts[hover].c)} r="4" />
          </g>
        )}
      </svg>
      <div className="chart-foot">
        <span>52w low ₹{lo.toLocaleString('en-IN')}</span>
        <a href={link} target="_blank" rel="noopener noreferrer">Full chart on TradingView ↗</a>
        <span>52w high ₹{hi.toLocaleString('en-IN')}</span>
      </div>
    </div>
  );
}


// ── Key stats snapshot (basic info at a glance) ─────────────────────────────
const fmtCr = (cr) => {
  if (cr == null) return '—';
  if (cr >= 100000) return `₹${(cr / 100000).toFixed(2)}L Cr`;
  return `₹${cr.toLocaleString('en-IN')} Cr`;
};
const fmtNum = (v, d = 1, suf = '') => (v == null ? '—' : `${Number(v).toFixed(d)}${suf}`);
const fmtVol = (v) => (v == null ? '—' : v >= 1e7 ? `${(v / 1e7).toFixed(2)} Cr` : v >= 1e5 ? `${(v / 1e5).toFixed(1)} L` : Math.round(v).toLocaleString('en-IN'));

function StockSnapshot({ snap, ticker }) {
  if (!snap) return null;
  const up = (snap.day_change_pct ?? 0) >= 0;
  const stats = [
    ['Market cap', fmtCr(snap.market_cap_cr)],
    ['P/E (trailing)', fmtNum(snap.pe)],
    ['P/E (forward)', fmtNum(snap.forward_pe)],
    ['Price / book', fmtNum(snap.pb)],
    ['EPS', snap.eps == null ? '—' : `₹${snap.eps.toFixed(1)}`],
    ['Return on equity', fmtNum(snap.roe_pct, 1, '%')],
    ['Profit margin', fmtNum(snap.profit_margin_pct, 1, '%')],
    ['Revenue growth', fmtNum(snap.revenue_growth_pct, 1, '%')],
    ['Debt / equity', fmtNum(snap.debt_to_equity, 0, '%')],
    ['Beta', fmtNum(snap.beta, 2)],
    ['Avg volume', fmtVol(snap.avg_volume)],
  ];
  const rets = [['1M', snap.return_1m_pct], ['6M', snap.return_6m_pct]];
  return (
    <div className="snapshot">
      <div className="snap-head">
        <div>
          <div className="snap-name">{snap.name || ticker}</div>
          <div className="snap-tags">
            {snap.sector && <span className="tag">{snap.sector}</span>}
            {snap.industry && <span className="tag">{snap.industry}</span>}
            <span className="tag tag--mono">{ticker}</span>
          </div>
        </div>
        <div className="snap-price">
          <span className="snap-price-val">{snap.price != null ? `₹${snap.price.toLocaleString('en-IN')}` : '—'}</span>
          {snap.day_change_pct != null && <span className={`snap-chg ${up ? 'up' : 'down'}`}>{up ? '▲' : '▼'} {Math.abs(snap.day_change_pct).toFixed(2)}% today</span>}
        </div>
      </div>

      {snap.week52_low != null && snap.week52_high != null && (
        <div className="range52">
          <span>₹{snap.week52_low.toLocaleString('en-IN')}</span>
          <div className="range52-track">
            <div className="range52-dot" style={{ left: `${Math.max(0, Math.min(100, snap.range_position_pct ?? 50))}%` }} />
          </div>
          <span>₹{snap.week52_high.toLocaleString('en-IN')}</span>
          <em>52-week range</em>
        </div>
      )}

      <div className="stat-grid">
        {stats.map(([k, v]) => (
          <div key={k} className="stat-cell"><span>{k}</span><strong>{v}</strong></div>
        ))}
      </div>

      <div className="ret-row">
        {rets.map(([k, v]) => (
          <div key={k} className={`ret-chip ${v == null ? '' : v >= 0 ? 'up' : 'down'}`}>
            <span>{k} return</span><strong>{v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(1)}%`}</strong>
          </div>
        ))}
      </div>
    </div>
  );
}


const SECTIONS = [
  ['sec-overview', 'Overview'], ['sec-stats', 'Key stats'], ['sec-factors', 'Factors'],
  ['sec-ai', 'AI models'], ['sec-merits', 'Why buy · why not'], ['sec-portfolio', 'Portfolio'],
];
function SectionNav() {
  const go = (id) => document.getElementById(id)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
  return (
    <nav className="section-nav" aria-label="Jump to section">
      {SECTIONS.map(([id, label]) => <button type="button" key={id} onClick={() => go(id)}>{label}</button>)}
    </nav>
  );
}


// ── Hero: score ring + per-factor contribution ────────────────────────────
function ScoreRing({ score }) {
  const r = 54, c = 2 * Math.PI * r;
  const pct = Math.max(0, Math.min(100, Number(score) || 0));
  return (
    <svg className="ring" viewBox="0 0 132 132" role="img" aria-label={`Overall score ${pct} out of 100`}>
      <circle className="ring-track" cx="66" cy="66" r={r} />
      <circle className="ring-fill" cx="66" cy="66" r={r}
        strokeDasharray={c} strokeDashoffset={c * (1 - pct / 100)} transform="rotate(-90 66 66)" />
      <text x="66" y="68" className="ring-num" textAnchor="middle">{Math.round(pct)}</text>
      <text x="66" y="88" className="ring-max" textAnchor="middle">/ 100</text>
    </svg>
  );
}

function Hero({ result, tone }) {
  const bd = result.overall_score?.breakdown || {};
  const w = result.overall_score?.weights_used || {};
  const rows = [
    ['technical', 'Technical'], ['fundamental', 'Fundamental'], ['sentiment', 'Sentiment'],
    ['volumetric', 'Volume'], ['sector', 'Sector'], ['geopolitical', 'Geopolitical'],
  ].filter(([k]) => (w[k] ?? 0) > 0);
  const label = (sc) => (sc >= 60 ? 'good' : sc < 45 ? 'bad' : 'neutral');
  return (
    <div className={`hero ${tone}`}>
      <div className="hero-ring">
        <ScoreRing score={result.overall_score.score} />
        <div className="hero-ring-label">Overall score</div>
        <div className="risk-level">{result.risk_level}</div>
      </div>
      <div className="hero-factors">
        <div className="hero-factors-head">
          <span>Factor scores</span><span>weight in overall</span>
        </div>
        {rows.map(([k, name]) => (
          <div key={k} className="hf-row">
            <span className="hf-name">{name}</span>
            <div className="hf-track"><div className={`hf-fill hf-fill--${label(bd[k])}`} style={{ width: `${bd[k] ?? 0}%` }} /></div>
            <span className="hf-score">{Math.round(bd[k] ?? 0)}</span>
            <span className="hf-weight">{Math.round((w[k] ?? 0) * 100)}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}


// ── Analysis card with expandable explainability ───────────────────────────
function AnalysisCard({ title, score, signal, items = [], explain = [], badge = null }) {
  const [open, setOpen] = useState(false);
  const tone = toneOfScore(score);
  return (
    <div className={`analysis-card ac--${tone}`}>
      <div className="ac-head">
        <h4>{title}{badge && <span className={`ac-badge ac-badge--${badge.tone}`}>{badge.text}</span>}</h4>
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
    </div>
  );
}


// ── 3-Model AI Intelligence ────────────────────────────────────────────────
function ModelReliability({ cal }) {
  const [open, setOpen] = useState(false);
  if (!cal) return null;
  const tier = cal.skill_tier || 'none';
  const ci = cal.oos_auc_ci95;
  const headline = !cal.applied
    ? 'Probabilities are uncalibrated'
    : tier === 'none' ? 'No demonstrated predictive edge'
    : tier === 'weak' ? 'Weak predictive edge' : 'Moderate predictive edge';
  return (
    <div className={`reliability reliability--${cal.applied ? tier : 'none'}`}>
      <div className="reliability-head">
        <span className="reliability-dot" />
        <div>
          <div className="reliability-title">Model reliability · {headline}</div>
          <p className="reliability-sub">
            {cal.applied ? (
              <>Tested on {cal.n_oos?.toLocaleString()} unseen predictions ({cal.oos_period?.[0]?.slice(0, 4)}–{cal.oos_period?.[1]?.slice(0, 4)}),
              the ensemble scores an out-of-sample AUC of <strong>{cal.oos_auc?.toFixed(3)}</strong>
              {ci && <> (95% CI {ci[0].toFixed(3)}–{ci[1].toFixed(3)})</>} — 0.500 is a coin flip.
              Probabilities below are calibrated to those results, so they stay close to 50% by design.</>
            ) : 'Run models/calibrate.py to calibrate these probabilities against out-of-sample results.'}
          </p>
        </div>
        {cal.applied && (
          <button type="button" className="reliability-more" onClick={() => setOpen(o => !o)} aria-expanded={open}>
            {open ? 'Hide' : 'Details'}
          </button>
        )}
      </div>
      {open && cal.applied && (
        <div className="reliability-grid">
          <div><span>Raw model range (5–95%)</span><strong>{cal.raw_range ? `${pct1(cal.raw_range[0])} – ${pct1(cal.raw_range[1])}` : '—'}</strong></div>
          <div><span>Calibrated range (5–95%)</span><strong>{cal.calibrated_range ? `${pct1(cal.calibrated_range[0])} – ${pct1(cal.calibrated_range[1])}` : '—'}</strong></div>
          <div><span>Base rate (stocks beating the Nifty)</span><strong>{pct1(cal.base_rate)}</strong></div>
          <div><span>Brier score vs always-50%</span><strong>{cal.brier_calibrated ?? '—'} vs {cal.brier_constant ?? '—'}</strong></div>
        </div>
      )}
    </div>
  );
}


function AISection({ result }) {
  const consensus = result.consensus;
  const models = (consensus?.models || []).filter(m => m.signal);

  const cardMod = {
    'Calibrated RF': 'ai-signal-card--rf',
    'XGBoost': 'ai-signal-card--xgb',
    'Hybrid (rule-aware)': 'ai-signal-card--hgb',
  };

  const drivers = [
    { name: 'Calibrated RF',       data: result.feature_importance,     fill: 'drivers-bar-fill--rf',  dot: 'drivers-dot--rf',  head: 'drivers-card-header--rf' },
    { name: 'XGBoost',             data: result.xgb_feature_importance, fill: 'drivers-bar-fill--xgb', dot: 'drivers-dot--xgb', head: 'drivers-card-header--xgb' },
    { name: 'Hybrid (rule-aware)', data: result.hgb_feature_importance, fill: 'drivers-bar-fill--hgb', dot: 'drivers-dot--hgb', head: 'drivers-card-header--hgb' },
  ].filter(d => d.data?.length > 0);

  return (
    <div className="ai-section">
      <div className="ai-section-title">3-Model AI Intelligence</div>
      <p className="ai-section-sub">
        Each model estimates whether this stock will <strong>beat or lag the Nifty 50</strong> over
        the next ~1 month — a relative call, not a forecast that the price rises or falls.
      </p>

      <ModelReliability cal={result.model_calibration} />

      {consensus && <ConsensusPanel consensus={consensus} ensemble={result.ensemble}
        outperformProbability={result.outperform_probability ?? result.ensemble?.outperform_probability} />}

      <div className="xgb-section-heading">Individual models · votes &amp; calibrated odds</div>
      <div className="ai-signals-row">
        {models.map((m, i) => (
          <div key={i} className={`ai-signal-card ${cardMod[m.name] || 'ai-signal-card--rf'}`}>
            <div className="ai-signal-model-label">{m.name}</div>
            <div className={`ml-signal ${sigClass(m.signal)}`}>{sigLabel(m.signal)}</div>
            <div className="ai-signal-conf">P(beat Nifty) <strong>{pct1(m.outperform_probability)}</strong></div>
            {m.raw_confidence != null && (
              <div className="ai-signal-raw">raw model confidence {(m.raw_confidence * 100).toFixed(0)}% → calibrated {(m.confidence * 100).toFixed(1)}%</div>
            )}
          </div>
        ))}
      </div>

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


function ConsensusPanel({ consensus, ensemble, outperformProbability }) {
  const models = (consensus.models || []).filter(m => m.signal);
  if (models.length === 0) return null;
  const agree = consensus.agree;
  const rule = consensus.rule_signal;
  const pOut = outperformProbability;
  const verdict = ensemble?.prediction || consensus.ensemble_prediction;
  // Zoomed 40–60% scale: calibrated odds live in a narrow band, so a 0–100 bar
  // would pin the marker at the middle and hide the (small) differences.
  const LO = 0.40, HI = 0.60;
  const pos = pOut == null ? 50 : Math.max(0, Math.min(100, ((pOut - LO) / (HI - LO)) * 100));
  const tone = verdict === 'OUTPERFORM' ? 'up' : verdict === 'UNDERPERFORM' ? 'down' : 'flat';
  const band = ((ensemble?.neutral_band ?? 0.02) * 100).toFixed(0);
  return (
    <>
      <div className="xgb-section-heading">Ensemble verdict · vs Nifty 50</div>
      <div className={`verdict verdict--${tone}`}>
        <div className="verdict-main">
          <div className="verdict-label">{ensemble?.edge_label || sigLabel(verdict)}</div>
          <div className="verdict-sub">Calibrated probability of beating the Nifty over ~1 month</div>
        </div>
        <div className="verdict-value">{pct1(pOut)}</div>
      </div>

      {pOut != null && (
        <div className="odds-block">
          <div className="odds-scale">
            <div className="odds-zone odds-zone--down" />
            <div className="odds-zone odds-zone--flat" />
            <div className="odds-zone odds-zone--up" />
            <div className="odds-marker" style={{ left: `${pos}%` }} title={pct1(pOut)} />
          </div>
          <div className="odds-ticks"><span>40%</span><span>45%</span><span>50% · coin flip</span><span>55%</span><span>60%</span></div>
          <p className="odds-note">
            The grey middle band is “no clear edge” (±{band} points around the {pct1(ensemble?.centre)} base rate). The raw ensemble output was{' '}
            {pct1(ensemble?.raw_outperform_probability)} before calibration.
          </p>
        </div>
      )}

      <div className={`disagree-panel ${agree ? 'ok' : 'warn'}`}>
        <div className="disagree-header">
          <span className="disagree-icon">{agree ? '●' : '△'}</span>
          <h4>{agree
            ? 'All three models vote the same way'
            : `Split vote — ${consensus.agreement} agree · majority ${sigLabel(consensus.majority)}`}</h4>
        </div>
        <div className="disagree-signals">
          {models.map((m, i) => (
            <React.Fragment key={i}>
              {i > 0 && <div className="disagree-vs">vs</div>}
              <div className="disagree-model">
                <span className="disagree-model-name">{m.name}</span>
                <span className={`disagree-signal-chip ${sigClass(m.signal)}`}>{sigLabel(m.signal)}</span>
              </div>
            </React.Fragment>
          ))}
        </div>
        {rule && (
          <p className="consensus-rule">
            Rule-engine cross-check: <strong className={sigClass(rule)}>{sigLabel(rule)}</strong>
          </p>
        )}
        <p className={`disagree-msg ${agree ? 'ok' : 'warn'}`}>
          {agree
            ? 'Models that share the same inputs agreeing is not independent confirmation — read it alongside the reliability note above.'
            : 'The models diverge — treat any lean as tentative.'}
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


// ── Merits / Demerits — why buy vs why not ─────────────────────────────────
function MeritsCard({ merits }) {
  const pros = merits?.merits || [];
  const cons = merits?.demerits || [];
  if (pros.length === 0 && cons.length === 0) return null;
  return (
    <div className="merits-card">
      <div className="merits-head">
        <div className="section-label" style={{ margin: 0 }}>Why Buy · Why Not</div>
        <span className={`stance stance--${merits.stance || 'hold'}`}>
          {merits.stance === 'buy' ? 'Buy-leaning' : merits.stance === 'avoid' ? 'Caution-leaning' : 'Balanced'}
          {' · '}{pros.length} for · {cons.length} against
        </span>
      </div>
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
