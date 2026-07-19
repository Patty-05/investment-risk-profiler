/**
 * SeasonalAnalysis.jsx
 * Indian Equity Seasonal Analysis Tab
 * Place in: frontend/src/SeasonalAnalysis.jsx
 *
 * HOW TO ADD TO App.jsx:
 *   1. import SeasonalAnalysis from './SeasonalAnalysis';
 *   2. Add tab: { key: 'seasonal', label: '🌿 Seasonal' }
 *   3. Add: {activeTab === 'seasonal' && <SeasonalAnalysis />}
 *
 * HOW TO ADD service:
 *   In frontend/src/services/index.js add:
 *   export { getSeasonalAnalysis } from './seasonalService';
 */

import React, { useState, useEffect, useRef } from 'react';
import { getSeasonalAnalysis } from './services/seasonalService';

const MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];
const QUARTERS = ["Q1","Q2","Q3","Q4"];
const QUARTER_MONTHS = {
  "Q1": ["Jan","Feb","Mar"],
  "Q2": ["Apr","May","Jun"],
  "Q3": ["Jul","Aug","Sep"],
  "Q4": ["Oct","Nov","Dec"]
};

// ── Colour helpers ─────────────────────────────────────────────────────────

function returnColor(val, alpha = 1) {
  if (val === null || val === undefined) return `rgba(30,35,48,${alpha})`;
  if (val >= 3)    return `rgba(16,185,129,${alpha})`;
  if (val >= 1.5)  return `rgba(52,211,153,${alpha})`;
  if (val >= 0.5)  return `rgba(110,231,183,${alpha})`;
  if (val >= 0)    return `rgba(167,243,208,${alpha * 0.6})`;
  if (val >= -0.5) return `rgba(252,165,165,${alpha * 0.6})`;
  if (val >= -1.5) return `rgba(248,113,113,${alpha})`;
  return `rgba(239,68,68,${alpha})`;
}

function returnTextColor(val) {
  if (val === null || val === undefined) return 'var(--text-muted)';
  if (Math.abs(val) >= 1) return '#fff';
  return val >= 0 ? '#6ee7b7' : '#fca5a5';
}

function confColor(conf) {
  if (conf === 'High')   return { bg: 'rgba(16,185,129,0.15)', color: '#10b981', border: 'rgba(16,185,129,0.3)' };
  if (conf === 'Medium') return { bg: 'rgba(245,158,11,0.15)', color: '#f59e0b', border: 'rgba(245,158,11,0.3)' };
  return                        { bg: 'rgba(107,114,128,0.15)', color: '#6b7280', border: 'rgba(107,114,128,0.3)' };
}

// ── Sub-components ─────────────────────────────────────────────────────────

function LoadingState() {
  return (
    <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
      <div style={{
        width: 40, height: 40, border: '2px solid var(--border)',
        borderTopColor: 'var(--accent)', borderRadius: '50%',
        animation: 'spin 0.8s linear infinite', margin: '0 auto 16px'
      }} />
      <p style={{ fontFamily: 'var(--font-mono)', fontSize: '0.85rem' }}>
        Fetching 10 years of Indian sector data…
      </p>
      <p style={{ fontSize: '0.75rem', marginTop: 8 }}>
        This may take 20–40s on first load (cached for 6h after)
      </p>
    </div>
  );
}

function SectionHeading({ children }) {
  return (
    <div style={{
      fontFamily: 'var(--font-display)', fontSize: '0.72rem', fontWeight: 700,
      textTransform: 'uppercase', letterSpacing: '0.1em', color: 'var(--text-muted)',
      margin: '28px 0 14px', display: 'flex', alignItems: 'center', gap: 10
    }}>
      {children}
      <div style={{ flex: 1, height: 1, background: 'var(--border)' }} />
    </div>
  );
}

// Sector Heatmap — sector × month grid
function SectorHeatmap({ heatmap }) {
  if (!heatmap) return null;
  const { sectors, sector_names, sector_icons, months, matrix } = heatmap;

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{
        width: '100%', borderCollapse: 'separate', borderSpacing: 2,
        fontFamily: 'var(--font-mono)', fontSize: '0.72rem', minWidth: 700
      }}>
        <thead>
          <tr>
            <th style={{ textAlign: 'left', padding: '6px 10px', color: 'var(--text-muted)',
              fontFamily: 'var(--font-display)', letterSpacing: '0.06em',
              fontSize: '0.68rem', whiteSpace: 'nowrap' }}>Sector</th>
            {months.map(m => (
              <th key={m} style={{
                padding: '6px 4px', color: 'var(--text-muted)',
                fontFamily: 'var(--font-display)', letterSpacing: '0.06em',
                fontSize: '0.68rem', textAlign: 'center',
                background: m === MONTHS[new Date().getMonth()]
                  ? 'rgba(0,229,160,0.1)' : 'transparent',
                borderRadius: 4
              }}>{m}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sectors.map((sector, si) => (
            <tr key={sector}>
              <td style={{ padding: '4px 10px', whiteSpace: 'nowrap',
                color: 'var(--text-secondary)', fontSize: '0.75rem' }}>
                <span style={{ marginRight: 6 }}>{sector_icons[si]}</span>
                {sector}
              </td>
              {matrix[si].map((val, mi) => (
                <td key={mi} style={{
                  padding: '5px 3px', textAlign: 'center', borderRadius: 4,
                  background: returnColor(val, 0.85),
                  color: returnTextColor(val),
                  fontWeight: Math.abs(val) >= 2 ? 600 : 400,
                  minWidth: 42,
                  outline: months[mi] === MONTHS[new Date().getMonth()]
                    ? '1px solid rgba(0,229,160,0.4)' : 'none'
                }}>
                  {val > 0 ? `+${val}` : val}%
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: 10,
        fontFamily: 'var(--font-mono)' }}>
        Average monthly returns (%) across 10 years · Highlighted column = current month
      </p>
    </div>
  );
}

// Top 3 picks cards
function TopPicks({ picks, currentMonth }) {
  if (!picks || picks.length === 0) return null;

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(260px, 1fr))', gap: 14 }}>
      {picks.map((pick, i) => {
        const cf = confColor(pick.confidence);
        return (
          <div key={pick.sector} style={{
            background: 'var(--surface-2)',
            border: `1px solid ${i === 0 ? 'rgba(0,229,160,0.35)' : 'var(--border-2)'}`,
            borderRadius: 'var(--radius-lg)',
            padding: '20px 18px',
            position: 'relative', overflow: 'hidden',
            transition: 'transform 0.2s',
          }}>
            {/* Rank badge */}
            <div style={{
              position: 'absolute', top: 12, right: 14,
              fontFamily: 'var(--font-mono)', fontSize: '1.4rem', fontWeight: 700,
              color: i === 0 ? 'var(--accent)' : 'var(--text-muted)', opacity: 0.5
            }}>#{i + 1}</div>

            {/* Top accent line for #1 */}
            {i === 0 && (
              <div style={{
                position: 'absolute', top: 0, left: 0, right: 0, height: 2,
                background: 'linear-gradient(90deg, #00ffcc, #1e90ff, transparent)'
              }} />
            )}

            <div style={{ fontSize: '1.6rem', marginBottom: 8 }}>{pick.icon}</div>
            <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700,
              fontSize: '0.95rem', color: 'var(--text-primary)', marginBottom: 4 }}>
              {pick.name}
            </div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)',
              fontFamily: 'var(--font-mono)', marginBottom: 14 }}>
              {pick.sector}
            </div>

            {/* Stats row */}
            <div style={{ display: 'flex', gap: 12, marginBottom: 14, flexWrap: 'wrap' }}>
              <div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: '0.06em' }}>Avg Return</div>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem',
                  color: pick.avg_return >= 0 ? 'var(--green)' : 'var(--red)'
                }}>
                  {pick.avg_return >= 0 ? '+' : ''}{pick.avg_return}%
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: '0.06em' }}>Win Rate</div>
                <div style={{ fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem',
                  color: 'var(--text-primary)' }}>
                  {pick.positive_rate}%
                </div>
              </div>
              <div>
                <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                  textTransform: 'uppercase', letterSpacing: '0.06em' }}>Momentum</div>
                <div style={{
                  fontFamily: 'var(--font-mono)', fontWeight: 700, fontSize: '1.1rem',
                  color: pick.momentum >= 0 ? 'var(--green)' : 'var(--red)'
                }}>
                  {pick.momentum >= 0 ? '↑' : '↓'} {Math.abs(pick.momentum)}
                </div>
              </div>
            </div>

            {/* Confidence badge */}
            <div style={{
              display: 'inline-block', padding: '3px 10px', borderRadius: 20,
              background: cf.bg, color: cf.color,
              border: `1px solid ${cf.border}`,
              fontSize: '0.7rem', fontWeight: 600, fontFamily: 'var(--font-mono)',
              letterSpacing: '0.05em', marginBottom: 12
            }}>
              {pick.confidence} Confidence
            </div>

            {/* Best months pills */}
            <div style={{ marginBottom: 12 }}>
              <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                textTransform: 'uppercase', letterSpacing: '0.06em', marginBottom: 6 }}>
                Best Months
              </div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                {pick.best_months.map(m => (
                  <span key={m} style={{
                    padding: '2px 8px', borderRadius: 4,
                    background: m === currentMonth
                      ? 'rgba(0,229,160,0.2)' : 'var(--surface-3)',
                    color: m === currentMonth ? 'var(--accent)' : 'var(--text-secondary)',
                    border: m === currentMonth ? '1px solid rgba(0,229,160,0.3)' : '1px solid var(--border)',
                    fontSize: '0.72rem', fontFamily: 'var(--font-mono)', fontWeight: 500
                  }}>{m}</span>
                ))}
              </div>
            </div>

            {/* Reason */}
            <div style={{
              fontSize: '0.75rem', color: 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)', lineHeight: 1.6,
              borderTop: '1px solid var(--border)', paddingTop: 10
            }}>
              {pick.reason}
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Monthly bar chart for a single sector
function MonthlyBarChart({ monthly, color, currentMonth }) {
  if (!monthly) return null;
  const avgs = MONTHS.map((_, i) => monthly.avg_by_month[i + 1] || 0);
  const maxAbs = Math.max(...avgs.map(Math.abs), 0.1);

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-end', gap: 4, height: 80, marginBottom: 4 }}>
        {MONTHS.map((m, i) => {
          const val = avgs[i];
          const barH = Math.max(2, (Math.abs(val) / maxAbs) * 70);
          const isCurrent = m === currentMonth;
          return (
            <div key={m} style={{ flex: 1, display: 'flex', flexDirection: 'column',
              alignItems: 'center', position: 'relative' }}
              title={`${m}: ${val > 0 ? '+' : ''}${val}%`}>
              <div style={{
                width: '100%',
                height: barH,
                background: isCurrent
                  ? 'rgba(0,229,160,0.9)'
                  : val >= 0 ? 'rgba(46,213,115,0.7)' : 'rgba(255,71,87,0.7)',
                borderRadius: '2px 2px 0 0',
                border: isCurrent ? '1px solid #00e5a0' : 'none',
                transition: 'height 0.6s ease',
                alignSelf: 'flex-end'
              }} />
            </div>
          );
        })}
      </div>
      <div style={{ display: 'flex', gap: 4 }}>
        {MONTHS.map((m, i) => (
          <div key={m} style={{
            flex: 1, textAlign: 'center', fontSize: '0.6rem',
            color: m === currentMonth ? 'var(--accent)' : 'var(--text-muted)',
            fontFamily: 'var(--font-mono)', fontWeight: m === currentMonth ? 700 : 400
          }}>{m.slice(0, 1)}</div>
        ))}
      </div>
    </div>
  );
}

// Quarterly radar-style horizontal bars
function QuarterlyBars({ quarterly }) {
  if (!quarterly) return null;
  const quarterKeys = [1, 2, 3, 4];
  const labels = ['Q1 Jan-Mar', 'Q2 Apr-Jun', 'Q3 Jul-Sep', 'Q4 Oct-Dec'];
  const maxAbs = Math.max(...quarterKeys.map(q => Math.abs(quarterly[q]?.avg || 0)), 0.1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {quarterKeys.map((q, i) => {
        const val = quarterly[q]?.avg || 0;
        const pos = quarterly[q]?.positive_rate || 50;
        const pct = Math.min(100, (Math.abs(val) / maxAbs) * 100);
        return (
          <div key={q}>
            <div style={{ display: 'flex', justifyContent: 'space-between',
              fontSize: '0.72rem', color: 'var(--text-secondary)',
              fontFamily: 'var(--font-mono)', marginBottom: 4 }}>
              <span>{labels[i]}</span>
              <span style={{ color: val >= 0 ? 'var(--green)' : 'var(--red)', fontWeight: 600 }}>
                {val >= 0 ? '+' : ''}{val}% · {pos}% wins
              </span>
            </div>
            <div style={{ height: 7, background: 'var(--surface-3)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                width: `${pct}%`, height: '100%', borderRadius: 4,
                background: val >= 0
                  ? 'linear-gradient(90deg, #2ed573, #00ffcc)'
                  : 'linear-gradient(90deg, #ff4757, #ff6b81)',
                transition: 'width 0.8s ease'
              }} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

// Sector detail card
function SectorDetailCard({ sectorKey, stats, currentMonth, expanded, onToggle }) {
  if (!stats || stats.error) return null;
  const bestMonth = stats.best_periods?.best_months?.[0];
  const overallColor = stats.overall_avg >= 0 ? 'var(--green)' : 'var(--red)';

  return (
    <div style={{
      background: 'var(--surface-2)', border: '1px solid var(--border-2)',
      borderRadius: 'var(--radius-lg)', overflow: 'hidden',
      transition: 'border-color 0.2s'
    }}>
      {/* Header — always visible */}
      <div
        onClick={onToggle}
        style={{
          padding: '16px 18px', cursor: 'pointer', display: 'flex',
          alignItems: 'center', gap: 12, userSelect: 'none'
        }}
      >
        <span style={{ fontSize: '1.3rem' }}>{stats.icon}</span>
        <div style={{ flex: 1 }}>
          <div style={{ fontFamily: 'var(--font-display)', fontWeight: 700,
            fontSize: '0.88rem', color: 'var(--text-primary)' }}>
            {stats.name}
          </div>
          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)' }}>
            Best month: {bestMonth} · Overall avg: {stats.overall_avg >= 0 ? '+' : ''}{stats.overall_avg}%/mo
          </div>
        </div>

        {/* Mini monthly bars */}
        <div style={{ width: 100, flexShrink: 0 }}>
          <MonthlyBarChart monthly={stats.monthly} color={stats.color} currentMonth={currentMonth} />
        </div>

        {/* Avg return badge */}
        <div style={{
          fontFamily: 'var(--font-mono)', fontSize: '1rem', fontWeight: 700,
          color: overallColor, minWidth: 60, textAlign: 'right'
        }}>
          {stats.overall_avg >= 0 ? '+' : ''}{stats.overall_avg}%
        </div>

        <div style={{ color: 'var(--text-muted)', fontSize: '0.8rem' }}>
          {expanded ? '▲' : '▼'}
        </div>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div style={{ padding: '0 18px 20px', borderTop: '1px solid var(--border)' }}>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20,
            marginTop: 16 }}>

            {/* Monthly avg table */}
            <div>
              <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-display)',
                textTransform: 'uppercase', letterSpacing: '0.08em',
                color: 'var(--text-muted)', marginBottom: 10 }}>
                Monthly Average Returns
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4,1fr)', gap: 4 }}>
                {MONTHS.map((m, i) => {
                  const val = stats.monthly.avg_by_month[i + 1] || 0;
                  const pos = stats.monthly.positive_rate[i + 1] || 50;
                  const isCurrent = m === currentMonth;
                  return (
                    <div key={m} style={{
                      padding: '6px 4px', borderRadius: 6, textAlign: 'center',
                      background: returnColor(val, 0.75),
                      border: isCurrent ? '1px solid rgba(0,229,160,0.5)' : '1px solid transparent',
                    }}>
                      <div style={{ fontSize: '0.62rem', color: 'rgba(255,255,255,0.6)',
                        fontFamily: 'var(--font-display)', fontWeight: 600 }}>{m}</div>
                      <div style={{ fontSize: '0.75rem', fontFamily: 'var(--font-mono)',
                        fontWeight: 700, color: returnTextColor(val) }}>
                        {val >= 0 ? '+' : ''}{val}%
                      </div>
                      <div style={{ fontSize: '0.58rem', color: 'rgba(255,255,255,0.5)' }}>
                        {pos}%↑
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Quarterly + Best window */}
            <div>
              <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-display)',
                textTransform: 'uppercase', letterSpacing: '0.08em',
                color: 'var(--text-muted)', marginBottom: 10 }}>
                Quarterly Breakdown
              </div>
              <QuarterlyBars quarterly={stats.quarterly} />

              <div style={{ marginTop: 18 }}>
                <div style={{ fontSize: '0.68rem', fontFamily: 'var(--font-display)',
                  textTransform: 'uppercase', letterSpacing: '0.08em',
                  color: 'var(--text-muted)', marginBottom: 8 }}>
                  Best 3-Month Window
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
                  {stats.best_periods.best_window.map((m, i) => (
                    <React.Fragment key={m}>
                      <span style={{
                        padding: '4px 10px', borderRadius: 6,
                        background: 'rgba(0,229,160,0.1)',
                        border: '1px solid rgba(0,229,160,0.25)',
                        color: 'var(--accent)', fontFamily: 'var(--font-mono)',
                        fontSize: '0.8rem', fontWeight: 600
                      }}>{m}</span>
                      {i < 2 && <span style={{ color: 'var(--text-muted)', fontSize: '0.7rem' }}>→</span>}
                    </React.Fragment>
                  ))}
                </div>
              </div>

              <div style={{ marginTop: 16, display: 'flex', gap: 16 }}>
                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                    textTransform: 'uppercase', letterSpacing: '0.06em' }}>Best Months</div>
                  <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                    {stats.best_periods.best_months.map(m => (
                      <span key={m} style={{
                        padding: '2px 7px', borderRadius: 4,
                        background: 'rgba(46,213,115,0.15)',
                        color: 'var(--green)', fontSize: '0.72rem',
                        border: '1px solid rgba(46,213,115,0.25)',
                        fontFamily: 'var(--font-mono)'
                      }}>{m}</span>
                    ))}
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                    textTransform: 'uppercase', letterSpacing: '0.06em' }}>Avoid Months</div>
                  <div style={{ display: 'flex', gap: 4, marginTop: 4 }}>
                    {stats.best_periods.worst_months.map(m => (
                      <span key={m} style={{
                        padding: '2px 7px', borderRadius: 4,
                        background: 'rgba(255,71,87,0.12)',
                        color: 'var(--red)', fontSize: '0.72rem',
                        border: '1px solid rgba(255,71,87,0.2)',
                        fontFamily: 'var(--font-mono)'
                      }}>{m}</span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Current month ranking bar chart
function CurrentMonthRanking({ rankings, currentMonth }) {
  if (!rankings || rankings.length === 0) return null;
  const maxAbs = Math.max(...rankings.map(r => Math.abs(r.avg_return)), 0.1);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
      {rankings.map((r, i) => {
        const pct = Math.min(100, (Math.abs(r.avg_return) / maxAbs) * 100);
        const isPositive = r.avg_return >= 0;
        return (
          <div key={r.sector}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 4 }}>
              <span style={{ fontSize: '1rem', width: 22 }}>{r.icon}</span>
              <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)',
                fontFamily: 'var(--font-mono)', minWidth: 90 }}>{r.sector}</span>
              <div style={{ flex: 1, height: 10, background: 'var(--surface-3)',
                borderRadius: 5, overflow: 'hidden' }}>
                <div style={{
                  width: `${pct}%`, height: '100%', borderRadius: 5,
                  background: isPositive
                    ? 'linear-gradient(90deg, #2ed573, #00ffcc)'
                    : 'linear-gradient(90deg, #ff4757, #ff6b81)',
                  transition: 'width 0.8s ease'
                }} />
              </div>
              <span style={{
                fontFamily: 'var(--font-mono)', fontSize: '0.8rem', fontWeight: 700,
                color: isPositive ? 'var(--green)' : 'var(--red)',
                minWidth: 52, textAlign: 'right'
              }}>
                {isPositive ? '+' : ''}{r.avg_return}%
              </span>
              <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)',
                fontFamily: 'var(--font-mono)', minWidth: 40 }}>
                {r.positive_rate}%↑
              </span>
            </div>
          </div>
        );
      })}
      <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)',
        fontFamily: 'var(--font-mono)', marginTop: 4 }}>
        Historical avg return in {currentMonth} · Win rate = % of years with positive return
      </p>
    </div>
  );
}

// ── Main Component ─────────────────────────────────────────────────────────

export default function SeasonalAnalysis() {
  const [data, setData]           = useState(null);
  const [loading, setLoading]     = useState(false);
  const [error, setError]         = useState(null);
  const [expandedSector, setExpandedSector] = useState(null);
  const [activeView, setActiveView] = useState('overview'); // overview | heatmap | sectors | compare
  const hasFetched = useRef(false);

  const fetchData = async (refresh = false) => {
    setLoading(true);
    setError(null);
    try {
      const res = await getSeasonalAnalysis(refresh);
      setData(res.data);
    } catch (err) {
      setError(err.message || 'Failed to load seasonal data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!hasFetched.current) {
      hasFetched.current = true;
      fetchData();
    }
  }, []);

  const d            = data;
  const currentMonth = d?.current_month;
  const sectorStats  = d?.sector_stats || {};

  const views = [
    { key: 'overview',  label: '📊 Overview'  },
    { key: 'heatmap',   label: '🟩 Heatmap'   },
    { key: 'sectors',   label: '📂 All Sectors' },
    { key: 'compare',   label: '⚖️ Compare'   },
  ];

  return (
    <div>
      <p>
        Historical seasonal patterns of Indian equity sectors — 10 years of data (2014–2024).
        Identifies the best months and quarters to consider each sector.
      </p>

      {/* Controls row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 12,
        marginTop: 16, marginBottom: 4, flexWrap: 'wrap' }}>
        <button className="btn" onClick={() => fetchData()} disabled={loading}>
          {loading ? 'Loading…' : '↻ Load Data'}
        </button>
        {d && (
          <button className="btn" onClick={() => fetchData(true)} disabled={loading}
            style={{ background: 'transparent', border: '1px solid var(--border)',
              color: 'var(--text-secondary)' }}>
            ⟳ Refresh
          </button>
        )}
        {d && (
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)',
            fontFamily: 'var(--font-mono)' }}>
            {d.years_analyzed}y data · Current: {d.current_month} {d.current_quarter}
          </span>
        )}
      </div>

      {error && (
        <div className="error" style={{ marginTop: 16 }}>{error}</div>
      )}

      {loading && <LoadingState />}

      {!loading && d && (
        <>
          {/* Sub-nav tabs */}
          <div style={{
            display: 'flex', gap: 2, margin: '20px 0 0',
            borderBottom: '1px solid var(--border)'
          }}>
            {views.map(v => (
              <button key={v.key}
                onClick={() => setActiveView(v.key)}
                style={{
                  background: 'transparent', border: 'none',
                  color: activeView === v.key ? '#00ffcc' : '#aaa',
                  padding: '8px 16px', cursor: 'pointer',
                  borderBottom: activeView === v.key ? '2px solid #00ffcc' : '2px solid transparent',
                  fontSize: '0.82rem', fontFamily: 'var(--font-mono)',
                  transition: 'color 0.2s'
                }}>
                {v.label}
              </button>
            ))}
          </div>

          {/* ── OVERVIEW ── */}
          {activeView === 'overview' && (
            <div style={{ marginTop: 20 }}>

              {/* Top 3 picks */}
              <SectionHeading>🏆 Top 3 Sector Picks for {currentMonth}</SectionHeading>
              <TopPicks picks={d.top_picks} currentMonth={currentMonth} />

              {/* Current month ranking */}
              <SectionHeading>📈 All Sectors · {currentMonth} Historical Performance</SectionHeading>
              <div style={{
                background: 'var(--surface-2)', border: '1px solid var(--border-2)',
                borderRadius: 'var(--radius-lg)', padding: '20px 22px'
              }}>
                <CurrentMonthRanking rankings={d.current_rankings} currentMonth={currentMonth} />
              </div>

              {/* Quick summary cards */}
              <SectionHeading>📋 Sector Summary</SectionHeading>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px,1fr))',
                gap: 10 }}>
                {d.comparison.map(c => (
                  <div key={c.sector} style={{
                    background: 'rgba(255,255,255,0.04)',
                    border: '1px solid rgba(255,255,255,0.08)',
                    borderRadius: 12, padding: 14,
                    transition: 'transform 0.2s, box-shadow 0.2s',
                    cursor: 'default'
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.transform = 'translateY(-3px)';
                    e.currentTarget.style.boxShadow = `0 8px 24px ${c.color}22`;
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.transform = '';
                    e.currentTarget.style.boxShadow = '';
                  }}>
                    <div style={{ fontSize: '1.2rem', marginBottom: 6 }}>{c.icon}</div>
                    <div style={{ fontSize: '0.72rem', fontWeight: 700,
                      color: 'var(--text-secondary)', fontFamily: 'var(--font-display)',
                      marginBottom: 8 }}>{c.sector}</div>

                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: '1.1rem',
                      fontWeight: 700,
                      color: c.overall_avg >= 0 ? 'var(--green)' : 'var(--red)',
                      marginBottom: 4 }}>
                      {c.overall_avg >= 0 ? '+' : ''}{c.overall_avg}%
                    </div>
                    <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)',
                      fontFamily: 'var(--font-mono)' }}>avg/month</div>

                    <div style={{ marginTop: 10, display: 'flex', justifyContent: 'space-between',
                      fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                      <span>🟢 {c.best_month}</span>
                      <span>🔴 {c.worst_month}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── HEATMAP ── */}
          {activeView === 'heatmap' && (
            <div style={{ marginTop: 20 }}>
              <SectionHeading>🟩 Sector × Month Return Heatmap (10-Year Avg)</SectionHeading>
              <div style={{
                background: 'var(--surface-2)', border: '1px solid var(--border-2)',
                borderRadius: 'var(--radius-lg)', padding: '20px 18px'
              }}>
                <SectorHeatmap heatmap={d.heatmap} />
              </div>

              {/* Legend */}
              <div style={{ display: 'flex', gap: 8, marginTop: 14,
                flexWrap: 'wrap', alignItems: 'center' }}>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)',
                  fontFamily: 'var(--font-mono)' }}>Legend:</span>
                {[
                  { label: '≥ +3%', bg: 'rgba(16,185,129,0.85)' },
                  { label: '+1.5%', bg: 'rgba(52,211,153,0.85)' },
                  { label: '+0.5%', bg: 'rgba(110,231,183,0.6)' },
                  { label: '~0%',   bg: 'rgba(167,243,208,0.4)' },
                  { label: '-0.5%', bg: 'rgba(252,165,165,0.6)' },
                  { label: '-1.5%', bg: 'rgba(248,113,113,0.85)' },
                  { label: '≤ -3%', bg: 'rgba(239,68,68,0.85)' },
                ].map(l => (
                  <div key={l.label} style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                    <div style={{ width: 14, height: 14, borderRadius: 3,
                      background: l.bg }} />
                    <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)',
                      fontFamily: 'var(--font-mono)' }}>{l.label}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* ── ALL SECTORS ── */}
          {activeView === 'sectors' && (
            <div style={{ marginTop: 20, display: 'flex', flexDirection: 'column', gap: 10 }}>
              <SectionHeading>📂 All Sectors · Detailed Breakdown</SectionHeading>
              {Object.entries(sectorStats).map(([key, stats]) => (
                <SectorDetailCard
                  key={key}
                  sectorKey={key}
                  stats={stats}
                  currentMonth={currentMonth}
                  expanded={expandedSector === key}
                  onToggle={() => setExpandedSector(expandedSector === key ? null : key)}
                />
              ))}
            </div>
          )}

          {/* ── COMPARE ── */}
          {activeView === 'compare' && (
            <div style={{ marginTop: 20 }}>
              <SectionHeading>⚖️ Cross-Sector Comparison</SectionHeading>

              {/* Overall avg return ranking */}
              <div style={{
                background: 'var(--surface-2)', border: '1px solid var(--border-2)',
                borderRadius: 'var(--radius-lg)', padding: '20px 22px', marginBottom: 16
              }}>
                <h4>Overall Monthly Average Return (10-Year)</h4>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {d.comparison.map((c, i) => {
                    const maxAvg = Math.max(...d.comparison.map(x => Math.abs(x.overall_avg)), 0.1);
                    const pct = Math.min(100, (Math.abs(c.overall_avg) / maxAvg) * 100);
                    return (
                      <div key={c.sector} style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                        <span style={{ width: 20, textAlign: 'center', fontSize: '0.7rem',
                          color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                          {i + 1}
                        </span>
                        <span style={{ fontSize: '1rem', width: 22 }}>{c.icon}</span>
                        <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)',
                          fontFamily: 'var(--font-mono)', minWidth: 80 }}>{c.sector}</span>
                        <div style={{ flex: 1, height: 10, background: 'var(--surface-3)',
                          borderRadius: 5, overflow: 'hidden' }}>
                          <div style={{
                            width: `${pct}%`, height: '100%', borderRadius: 5,
                            background: c.overall_avg >= 0
                              ? 'linear-gradient(90deg, #2ed573, #00ffcc)'
                              : 'linear-gradient(90deg, #ff4757, #ff6b81)',
                            transition: 'width 0.8s ease'
                          }} />
                        </div>
                        <span style={{
                          fontFamily: 'var(--font-mono)', fontSize: '0.82rem', fontWeight: 700,
                          color: c.overall_avg >= 0 ? 'var(--green)' : 'var(--red)',
                          minWidth: 52, textAlign: 'right'
                        }}>
                          {c.overall_avg >= 0 ? '+' : ''}{c.overall_avg}%
                        </span>
                        <span style={{ fontSize: '0.68rem', color: 'var(--text-muted)',
                          fontFamily: 'var(--font-mono)', minWidth: 44 }}>
                          ±{c.overall_std}%
                        </span>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Best window table */}
              <div style={{
                background: 'var(--surface-2)', border: '1px solid var(--border-2)',
                borderRadius: 'var(--radius-lg)', padding: '20px 22px'
              }}>
                <h4>Best Entry Windows Per Sector</h4>
                <div style={{ overflowX: 'auto' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse',
                    fontFamily: 'var(--font-mono)', fontSize: '0.78rem' }}>
                    <thead>
                      <tr style={{ borderBottom: '1px solid var(--border)' }}>
                        {['Sector', 'Best Month', 'Worst Month', 'Best 3-Month Window', 'Avg/Month', 'Volatility'].map(h => (
                          <th key={h} style={{ padding: '8px 12px', textAlign: 'left',
                            color: 'var(--text-muted)', fontSize: '0.68rem',
                            fontFamily: 'var(--font-display)', fontWeight: 700,
                            textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {d.comparison.map(c => (
                        <tr key={c.sector}
                          style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                          <td style={{ padding: '10px 12px', color: 'var(--text-secondary)' }}>
                            <span style={{ marginRight: 6 }}>{c.icon}</span>{c.sector}
                          </td>
                          <td style={{ padding: '10px 12px' }}>
                            <span style={{
                              padding: '2px 8px', borderRadius: 4,
                              background: 'rgba(46,213,115,0.15)',
                              color: 'var(--green)',
                              border: '1px solid rgba(46,213,115,0.25)'
                            }}>{c.best_month}</span>
                          </td>
                          <td style={{ padding: '10px 12px' }}>
                            <span style={{
                              padding: '2px 8px', borderRadius: 4,
                              background: 'rgba(255,71,87,0.12)',
                              color: 'var(--red)',
                              border: '1px solid rgba(255,71,87,0.2)'
                            }}>{c.worst_month}</span>
                          </td>
                          <td style={{ padding: '10px 12px', color: 'var(--accent)',
                            fontWeight: 600 }}>{c.best_window}</td>
                          <td style={{ padding: '10px 12px', fontWeight: 700,
                            color: c.overall_avg >= 0 ? 'var(--green)' : 'var(--red)' }}>
                            {c.overall_avg >= 0 ? '+' : ''}{c.overall_avg}%
                          </td>
                          <td style={{ padding: '10px 12px', color: 'var(--text-muted)' }}>
                            ±{c.overall_std}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {/* Disclaimer */}
          <div style={{
            marginTop: 28, padding: '12px 16px',
            background: 'rgba(255,165,2,0.06)',
            border: '1px solid rgba(255,165,2,0.2)',
            borderRadius: 'var(--radius)', fontSize: '0.75rem',
            color: 'var(--text-muted)', fontFamily: 'var(--font-mono)',
            lineHeight: 1.6
          }}>
            ⚠️ Seasonal patterns are based on historical averages and do not guarantee future returns.
            Past performance is not indicative of future results. This is for informational purposes only —
            not financial advice. Always conduct due diligence before investing.
          </div>
        </>
      )}
    </div>
  );
}
