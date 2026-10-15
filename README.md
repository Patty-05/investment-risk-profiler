# Investment Risk Profiler

A full-stack analysis tool for **Indian (NSE/BSE) equities**. Enter a ticker and get a six-factor score,
a calibrated machine-learning read against the Nifty 50, a "why buy · why not" breakdown, sector peers,
a profile-shaped portfolio allocation, and a **Geopolitical Analyzer** grounded in India's own customs data.

> Educational project. Nothing here is investment advice.

---

## Features

### Stock Analysis
- **Six-factor score** — Technical, Fundamental, Sentiment (FinBERT on Indian news), Volume, Sector, Geopolitical —
  with a "Why this score?" explanation per factor.
- **Key stats snapshot** — price, market cap, P/E, P/B, EPS, ROE, margins, debt/equity, beta, 52-week range.
- **3-model ML ensemble** (Random Forest, XGBoost, rule-aware HistGradientBoosting) predicting whether a stock
  will **beat or lag the Nifty 50 over ~20 trading days**.
- **Calibrated confidence** — probabilities are Platt-calibrated on purged walk-forward, out-of-sample results
  (`backend/models/calibrate.py`). The measured skill is shown in a reliability banner.
- **Why buy · why not** — deterministic pros/cons, kept consistent with the overall call.
- **Sector peers**, **profile-driven portfolio allocation** (risk appetite + horizon), AI insight (Gemini, optional).

### Geopolitical Analyzer
- **India trade map** — 26 sectors and the main partner nations, from UN Comtrade customs filings
  (India as reporter) and World Bank indicators: value, 5-year change, supplier concentration, and the share
  sourced from countries in live conflicts.
- **Scenario analyzer** — type a scenario ("What if the Strait of Hormuz closes?") or pick a tracked conflict.
  Built locally and instantly from customs data + India's bilateral ties + curated sector/ticker maps.
  An optional AI-written narrative (Gemini) can be added.

### Seasonal Analysis
Market-relative (excess vs Nifty) seasonality with significance testing and Indian calendar drivers.

---

## An honest note on the ML

Walk-forward testing on ~168k unseen predictions (2022–2026, 150 NSE stocks) gave the ensemble an
out-of-sample **AUC of 0.505 (95% CI 0.495–0.516)** — statistically a coin flip. Technical features and the
geopolitical signal both showed no incremental predictive power (see `docs/ALPHA_RESEARCH_FINDINGS.md`).

So the app **does not inflate confidence**: probabilities are calibrated to what was actually observed,
and the usual verdict is "No clear edge vs the Nifty". Treat the ML read as a weak tie-breaker, not a forecast.
The geopolitical score is an *exposure lens* (percentile among covered stocks), not a return prediction.

---

## Project structure

```
investment-risk-profiler/
├── backend/
│   ├── app.py                      # Flask API
│   ├── modules/                    # Analyzers: technical, fundamental, sentiment, volume, sector,
│   │                               #   geopolitical, merits, peers, seasonal, stock_analyzer (orchestrator)
│   ├── models/
│   │   ├── stock_model.py          # 3-model ensemble + calibration at inference
│   │   ├── calibrate.py            # walk-forward calibration  -> calibration.json
│   │   ├── walkforward.py          # purged time-ordered validation
│   │   ├── gov_data.py             # UN Comtrade + World Bank ingestion (cached)
│   │   ├── india_trade.py          # trade-map aggregation
│   │   ├── geo_features.py         # time-varying geo signal (research)
│   │   └── portfolio_optimizer.py
│   ├── services/                   # war_service, geo_engine (scenario engine), gemini_service
│   └── data/
│       ├── geopolitics/            # curated conflicts, relations, sector maps
│       └── gov/                    # cached official customs / macro data
├── frontend/                       # React app (App.jsx, GeoAnalyzer.jsx, SeasonalAnalysis.jsx)
├── docs/                           # SRS, research findings, reports
└── start.bat                       # launches backend + frontend (Windows)
```

---

## Setup

**Requirements:** Python 3.11, Node.js 18+.

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
```
Create `backend/.env` (optional — only needed for the AI narrative/insight):
```
GEMINI_API_KEY=your_key_here
```
Run:
```bash
python app.py        # http://localhost:5000
```
> On Windows set `PYTHONUTF8=1` if you see encoding errors in the console.

**First run:** the trained model file (`models/stock_model.pkl`, ~200 MB) is not stored in git. If it is missing,
the backend trains it on first use (about 5–8 minutes, needs internet for NSE price data).
Calibration (`models/calibration.json`) is committed; regenerate it after any retrain with
`python -m models.calibrate` (builds on the research panel cache; `python -m models.panel` rebuilds that).

### 2. Frontend
```bash
cd frontend
npm install
npm start            # http://localhost:3000
```

Or on Windows, run `start.bat` from the project root.

---

## API (selected)

| Method | Endpoint | Purpose |
|---|---|---|
| POST | `/api/analyze-stock` | Full analysis `{ticker, risk_appetite, horizon}` |
| GET | `/api/india-trade` | India sector/partner trade overview (official data) |
| GET | `/api/conflicts` | Tracked conflict watchlist |
| POST | `/api/war-analysis` | Scenario / conflict impact `{conflict_id | scenario_text, use_ai}` |
| GET | `/api/health` | Health check |

---

## Data sources
- **yfinance** — prices and company fundamentals
- **UN Comtrade** — India's customs trade statistics (keyless public preview, cached in `backend/data/gov/`)
- **World Bank Indicators API** — India macro series
- **FinBERT + RSS/news** — sentiment
- **Gemini** — optional narrative only

## Known limits
- Customs data is annual and lags ~12 months.
- Scenario matching is keyword-based — name a country and a commodity for best results.
- Fundamentals from yfinance can be incomplete for some tickers.
- Research scripts (`tuning*.py`, `ic_report.py`, `geo_alpha_test.py`) are kept for reproducibility, not used at serve time.

## Docs
- `docs/ALPHA_RESEARCH_FINDINGS.md` — walk-forward validation and null-result research
- `docs/SRS.md` — requirements specification
- `docs/Investment_Risk_Profiler_Report.pdf` — project report
