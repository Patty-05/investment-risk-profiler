import React, { useState, useEffect } from 'react';
import { getConflicts, analyzeWar, getIndiaTrade, useApi } from './services';

const titleize = (s) => String(s || '').replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase());
const bn = (v) => (v == null ? '—' : `$${v >= 100 ? v.toFixed(0) : v.toFixed(1)}bn`);
const signed = (v) => (v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(0)}%`);

export default function GeoAnalyzer({ onAnalyseTicker }) {
  const [tab, setTab] = useState('trade'); // 'trade' | 'scenario'
  return (
    <div className="geo">
      <div className="page-head">
        <div>
          <h2>Geopolitical Analyzer</h2>
          <p className="page-sub">
            How world events reach Indian markets — built on India’s own customs filings (UN Comtrade),
            World Bank macro data and India’s bilateral ties.
          </p>
        </div>
        <div className="segmented segmented--tabs">
          <button type="button" className={tab === 'trade' ? 'seg active' : 'seg'} onClick={() => setTab('trade')}>India trade map</button>
          <button type="button" className={tab === 'scenario' ? 'seg active' : 'seg'} onClick={() => setTab('scenario')}>Scenario analyzer</button>
        </div>
      </div>
      {tab === 'trade' ? <TradeMap /> : <ScenarioAnalyzer onAnalyseTicker={onAnalyseTicker} />}
    </div>
  );
}


// ── India trade map (official customs data) ────────────────────────────────
function TradeMap() {
  const api = useApi(getIndiaTrade);
  const [flow, setFlow] = useState('import');
  useEffect(() => { api.execute(); }, []); // eslint-disable-line

  const d = api.data;
  if (api.loading || (!d && !api.error)) return <div className="skeleton-block"><div className="sk-line sk-line--lg" /><div className="sk-grid">{[0, 1, 2, 3].map(i => <div key={i} className="sk-card" />)}</div></div>;
  if (api.error || !d?.success) return <div className="error">{api.error || d?.error || 'Could not load trade data.'}</div>;

  const sectors = d.sectors.filter(s => s.flow === flow).sort((a, b) => b.value_usd_bn - a.value_usd_bn);
  const maxVal = Math.max(...sectors.map(s => s.value_usd_bn), 1);
  const macro = Object.values(d.macro || {});

  return (
    <>
      {macro.length > 0 && (
        <div className="stat-tiles">
          {macro.map(m => (
            <div key={m.label} className="stat-tile">
              <span className="stat-label">{m.label}</span>
              <strong className="stat-value">{m.value.toLocaleString('en-IN')}</strong>
              <span className="stat-sub">{m.year}</span>
            </div>
          ))}
        </div>
      )}

      <div className="section-row">
        <div className="section-label" style={{ margin: 0 }}>Sector trade · what India buys &amp; sells</div>
        <div className="segmented segmented--sm">
          <button type="button" className={flow === 'import' ? 'seg active' : 'seg'} onClick={() => setFlow('import')}>Imports</button>
          <button type="button" className={flow === 'export' ? 'seg active' : 'seg'} onClick={() => setFlow('export')}>Exports</button>
        </div>
      </div>

      <div className="trade-list">
        {sectors.map(s => (
          <div key={s.key} className="trade-row">
            <div className="trade-main">
              <div className="trade-name">{s.label}<span className="trade-hs">HS {s.hs} · {s.year}</span></div>
              <div className="trade-bar"><div className="trade-bar-fill" style={{ width: `${(s.value_usd_bn / maxVal) * 100}%` }} /></div>
            </div>
            <div className="trade-val">
              <strong>{bn(s.value_usd_bn)}</strong>
              {s.change_pct != null && <span className={s.change_pct >= 0 ? 'up' : 'down'}>{signed(s.change_pct)} vs {s.change_base_year}</span>}
            </div>
            <div className="trade-partners">
              {s.top_partners.slice(0, 4).map(p => (
                <span key={p.name} className={`partner-chip ${p.in_conflict ? 'partner-chip--conflict' : ''}`}
                  title={p.in_conflict ? 'Party to a live tracked conflict' : undefined}>
                  {p.name} <b>{p.share_pct.toFixed(0)}%</b>
                </span>
              ))}
            </div>
            <div className="trade-risk">
              {s.conflict_party_share_pct >= 1
                ? <span className={`risk-badge ${s.conflict_party_share_pct >= 20 ? 'risk-badge--high' : 'risk-badge--mid'}`}
                    title={`From: ${s.conflict_parties.join(', ')}`}>{s.conflict_party_share_pct.toFixed(0)}% from conflict parties</span>
                : <span className="risk-badge risk-badge--low">No live-conflict exposure</span>}
              {s.concentration_hhi != null && s.concentration_hhi >= 2500 &&
                <span className="risk-badge risk-badge--mid" title={`HHI ${s.concentration_hhi}`}>concentrated supplier base</span>}
            </div>
          </div>
        ))}
      </div>

      <div className="section-label">Trading partners · across the tracked sectors</div>
      <div className="table-container">
        <table>
          <thead><tr><th>Nation</th><th>Imports</th><th>Exports</th><th>Key sectors</th><th>Live conflicts</th></tr></thead>
          <tbody>
            {d.partners.map(p => (
              <tr key={p.name}>
                <td><strong>{p.name}</strong></td>
                <td>{p.import_usd_bn ? bn(p.import_usd_bn) : '—'}</td>
                <td>{p.export_usd_bn ? bn(p.export_usd_bn) : '—'}</td>
                <td className="wrap">{p.top_sectors.join(' · ')}</td>
                <td className="wrap">{p.conflicts.length ? p.conflicts.join(', ') : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="source-foot">
        Sources: {d.sources.join(' · ')}. {d.notes.join(' ')}
      </p>
    </>
  );
}


// ── Scenario analyzer ──────────────────────────────────────────────────────
function ScenarioAnalyzer({ onAnalyseTicker }) {
  const conflictsApi = useApi(getConflicts);
  const analysisApi = useApi(analyzeWar);
  const [scenario, setScenario] = useState('');
  const [useAi, setUseAi] = useState(false);
  const [activeId, setActiveId] = useState(null);

  useEffect(() => { conflictsApi.execute(); }, []); // eslint-disable-line

  const conflicts = conflictsApi.data?.conflicts || [];
  const res = analysisApi.data;

  const runConflict = (id) => { setActiveId(id); setScenario(''); analysisApi.execute({ conflictId: id, useAi }); };
  const runScenario = (e) => {
    e.preventDefault();
    if (!scenario.trim()) return;
    setActiveId(null);
    analysisApi.execute({ scenarioText: scenario.trim(), useAi });
  };

  return (
    <div>
      <div className="section-label" style={{ marginTop: 6 }}>Describe a scenario</div>
      <form className="war-scenario" onSubmit={runScenario}>
        <input type="text" value={scenario} onChange={(e) => setScenario(e.target.value)}
          placeholder="e.g. What if the Strait of Hormuz closes? · Taiwan blockade cuts chip supply" />
        <button type="submit" className="btn btn--primary" disabled={analysisApi.loading || !scenario.trim()}>
          {analysisApi.loading ? 'Analysing…' : 'Analyse'}
        </button>
      </form>
      <label className="check-row">
        <input type="checkbox" checked={useAi} onChange={(e) => setUseAi(e.target.checked)} />
        <span>Add an AI-written narrative <em>(optional · slower · needs Gemini)</em></span>
      </label>

      <div className="section-label">Or pick a tracked conflict</div>
      {conflictsApi.loading && <p>Loading watchlist…</p>}
      {conflictsApi.error && <div className="error">{conflictsApi.error}</div>}
      <div className="war-grid">
        {conflicts.map(c => (
          <button key={c.id} type="button" className={`war-card ${activeId === c.id ? 'active' : ''}`}
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

      {analysisApi.error && <div className="error">{analysisApi.error}</div>}
      {analysisApi.loading && <div className="result-container"><p>Reading India’s customs data and mapping sector links…</p></div>}
      {!analysisApi.loading && res && !res.success && <div className="error">{res.error || 'Analysis failed.'}</div>}
      {!analysisApi.loading && res?.success && <Report res={res} onAnalyseTicker={onAnalyseTicker} />}
    </div>
  );
}


function Report({ res, onAnalyseTicker }) {
  const report = res.report;
  const biasClass = { 'risk-on': 'good', 'risk-off': 'bad', mixed: 'neutral' }[report.net_bias] || 'neutral';
  const magDots = (m) => '●'.repeat(Math.max(0, Math.min(5, m || 0))) + '○'.repeat(5 - Math.max(0, Math.min(5, m || 0)));
  const official = Object.entries(res.official_data || {});
  const bilateral = Object.entries(res.bilateral || {});

  return (
    <div className="result-container">
      <div className="result-head">
        <div>
          <h2>{res.conflict?.name}</h2>
          <p className="result-sub">
            {(res.conflict?.parties || []).join(res.conflict?.id === 'user_scenario' ? ' · ' : '  vs  ')}
            {res.conflict?.matched_conflict && <> · closest tracked conflict: {res.conflict.matched_conflict}</>}
          </p>
        </div>
        <span className={`pill pill--${biasClass}`}>{report.net_bias} · {report.confidence} confidence</span>
      </div>

      <p className="war-summary">{report.india_summary}</p>
      {res.ai_narrative && <div className="callout callout--ai"><h4>AI narrative</h4><p>{res.ai_narrative}</p></div>}
      {res.ai_error && <p className="source-foot">{res.ai_error}</p>}

      {official.length > 0 && (
        <>
          <div className="section-label">India’s exposure · official customs data</div>
          <div className="exposure-grid">
            {official.map(([key, f]) => (
              <div key={key} className="exposure-card">
                <div className="exposure-top">
                  <span className="exposure-name">{titleize(key)}</span>
                  <span className="exposure-val">{bn(f.india_trade_value_usd_bn)}</span>
                </div>
                <div className="exposure-sub">{f.flow} · {f.reporting_year}</div>
                <div className="exposure-bar" title={`${f.share_from_conflict_parties_pct}% from conflict parties`}>
                  <div className="exposure-bar-fill" style={{ width: `${Math.min(100, f.share_from_conflict_parties_pct || 0)}%` }} />
                </div>
                <div className="exposure-note">
                  <b>{(f.share_from_conflict_parties_pct || 0).toFixed(0)}%</b> from the parties here ·
                  top: {Object.entries(f.top_named_partners_pct || {}).slice(0, 3).map(([k, v]) => `${k} ${v.toFixed(0)}%`).join(', ')}
                </div>
              </div>
            ))}
          </div>
        </>
      )}

      {bilateral.length > 0 && (
        <>
          <div className="section-label">India’s ties with the parties</div>
          <div className="exposure-grid">
            {bilateral.map(([name, r]) => (
              <div key={name} className="exposure-card">
                <div className="exposure-top"><span className="exposure-name">{name}</span></div>
                <p className="ties-text">{r.relationship}</p>
                {r.stance && <p className="ties-text ties-text--muted">{r.stance}</p>}
                {r.india_imports_from?.length > 0 && <div className="exposure-note"><b>Imports:</b> {r.india_imports_from.map(titleize).join(', ')}</div>}
                {r.india_exports_to?.length > 0 && <div className="exposure-note"><b>Exports:</b> {r.india_exports_to.map(titleize).join(', ')}</div>}
              </div>
            ))}
          </div>
        </>
      )}

      <div className="war-section-head">
        <div className="section-label" style={{ margin: 0 }}>Transmission channels</div>
        <span className="war-tip">Tap a ticker to run its full analysis →</span>
      </div>
      {report.transmission_channels?.map((ch, i) => (
        <div key={i} className="war-channel">
          <div className="war-channel-head">
            <span className="war-driver">{ch.driver}</span>
            <span className={`war-dir war-dir--${ch.direction}`}>{ch.direction === 'up' ? '↑' : '↓'} {ch.direction}</span>
          </div>
          {ch.evidence && (
            <p className="channel-evidence">
              Customs data: India {ch.evidence.flow} {bn(ch.evidence.value_usd_bn)} ({ch.evidence.year}),
              {' '}{ch.evidence.from_conflict_parties_pct.toFixed(0)}% from the conflict parties.
            </p>
          )}
          <div className="war-affected">
            {ch.affected?.map((a, j) => (
              <div key={j} className={`war-impact war-impact--${a.impact}`}>
                <div className="war-impact-top">
                  <span className="war-sector">{a.sector}</span>
                  <span className="war-mag" title={`Severity ${a.magnitude}/5`}>{magDots(a.magnitude)}</span>
                </div>
                <p className="war-rationale">{a.rationale}</p>
                {a.tickers?.length > 0 && (
                  <div className="war-tickers">
                    {a.tickers.map((t, k) => (
                      <button key={k} type="button" className="war-ticker war-ticker--btn"
                        onClick={() => onAnalyseTicker && onAnalyseTicker(t)} title={`Analyse ${t}`}>
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
        <div className="callout"><h4>Watch items — what would confirm or escalate this</h4>
          <ul className="callout-list">{report.watch_items.map((w, i) => <li key={i}>{w}</li>)}</ul></div>
      )}
      {report.caveats?.length > 0 && (
        <div className="callout callout--ai"><h4>Caveats</h4>
          <ul className="callout-list">{report.caveats.map((w, i) => <li key={i}>{w}</li>)}</ul></div>
      )}
    </div>
  );
}
