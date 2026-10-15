# Alpha research findings — geo factor and technical model

Date: 2026-08-18
Branch: `feature/india-model-and-war-room`
Target under test: **will this NSE stock outperform the Nifty 50 over the next 20 trading days?**
Protocol: purged, time-ordered walk-forward with a 20-session embargo (`models/walkforward.py`),
150 NSE tickers, 348,408 rows, 2016-11 → 2026-07.

---

## Bottom line

| Question | Answer |
|---|---|
| Can technicals be tuned above 0.52 OOS AUC? | **No.** 20+ configurations all land 0.496–0.508. |
| Does the geo factor add incremental alpha? | **No.** ΔAUC +0.0023 / −0.0018 — fold noise, and the two models disagree on sign. |
| Is the pipeline broken, or is the signal absent? | **Signal absent.** A positive control recovers a planted signal perfectly (AUC 1.0000). |

The single most useful number in this document: a **pure-noise feature scores AUC 0.5065**
on this harness. The best of every real configuration tried was **0.5054**. The models are
not underperforming their potential — there is close to nothing there to extract.

---

## 1. The harness is sound (positive control)

Before believing any null result, a feature that is a deliberately corrupted copy of the
label was planted and the noise dialled up:

| planted feature | test AUC |
|---|---|
| 0% noise (the label) | 1.0000 |
| 50% noise | 0.8731 |
| 80% noise | 0.6812 |
| 90% noise | 0.5990 |
| 95% noise | 0.5508 |
| **100% noise (control)** | **0.5065** |

Detection degrades smoothly and the control sits at ~0.50. **If the technicals carried even
5% of the label's information, the harness would have reported ~0.55.** It reports 0.507.

Supporting evidence — the models can fit, they just cannot generalise:

| model | train AUC | test AUC | gap |
|---|---|---|---|
| default XGB | 0.6849 | 0.5072 | 0.1777 |
| regularised XGB | 0.5759 | 0.5081 | 0.0679 |

That gap is memorised noise, not lost signal.

---

## 2. Phase B — technical tuning (20+ configurations, all flat)

Round 1 (model/preprocessing sweep), best-to-worst, mean OOS AUC:

| configuration | AUC | edge |
|---|---|---|
| raw feats + regularised XGB | 0.5081 | −0.0251 |
| regularised LGBM + rank | 0.5077 | −0.0255 |
| baseline: raw feats, default XGB | 0.5072 | −0.0239 |
| + cross-sectional rank | 0.5068 | −0.0241 |
| rank + reg XGB + 2% deadband | 0.5059 | −0.0245 |
| rank + logistic | 0.4967 | −0.0255 |

Round 3 (feature-set surgery):

| feature set | n | AUC |
|---|---|---|
| atr_pct alone | 1 | 0.5054 |
| all 26 features | 26 | 0.5053 |
| top-8 by IC | 8 | 0.5010 |
| 17 IC-significant only | 17 | 0.5008 |
| drop vix_z + market_trend | 24 | 0.4998 |

**Every accuracy edge is negative** — the models are marginally worse than always predicting
the majority class.

Things tried that did not help: cross-sectional percentile ranking (the highest-conviction
fix, since the label is relative but the features are absolute levels), training-only
deadbands at 2% and 5%, recency weighting, heavy regularisation, depth-2 stumps, logistic
regression, LightGBM, IC-based feature selection, and decimating to non-overlapping labels
(which made it *worse*: 0.4997).

### One real defect found

`vix_z` has **exactly one unique value per date**. It is the India VIX z-score — a property
of the market, not of the stock — so every name shares it on any given day. Against a
*cross-sectional* target it cannot express a preference between two stocks; it can only
encode regime, which is exactly what fits in-sample and dies out-of-sample. `market_trend`
is nearly as bad at 2 distinct values per day.

This is the same failure mode the project already hit and fixed once by deleting the absolute
market series. `vix_z` survived that purge because it looked like a regime feature rather
than a level. For a relative target the distinction does not matter.

Removing it did not improve AUC (nothing does), but it should go on principle.

---

## 3. Phase A — the geopolitical factor

### What was built

The existing geo score was a static per-ticker lookup from the knowledge base: ONGC scored 90
today, last year, and next year until someone edits the JSON. **A per-ticker constant cannot
produce alpha — it is a ticker fixed effect**, and feeding it to a panel model just teaches it
which 40 names happened to do well.

It was rebuilt as a genuinely time-varying, point-in-time signal (`models/geo_features.py`):

```
geo_signal(ticker, t) = Σ_themes  direction(ticker, theme)      # from the KB
                                × weight(theme, t)              # from UN Comtrade
                                × pressure(theme, t)            # from traded prices
                      / Σ_themes |weight|
```

- **Pressure** — causal, trailing-window z-score of the driver the conflict disrupts (Brent
  for crude channels, NG for gas/fertiliser, gold for safe-haven, USDINR for risk-off, SOXX
  for semis, REMX for rare earths, ITA for defence…). A conflict nobody is pricing scores ~0.
- **Weight** — the commodity's share of India's merchandise imports × the share sourced from
  countries party to a live conflict, from **UN Comtrade customs data** (`models/gov_data.py`).
- **Direction** — benefits/hurts, from the KB theme → sector → ticker maps.

### Result: no incremental alpha

| model | technicals | technicals+geo | Δ | geo only |
|---|---|---|---|---|
| XGBoost | 0.5072 | 0.5095 | **+0.0023** | 0.5023 |
| LightGBM | 0.5080 | 0.5062 | **−0.0018** | 0.5000 |

The deltas are inside fold noise and the two models disagree on sign. Geo-only sits at chance.

Model-free IC confirms it: `geo_signal` has **IC 0.0002, Newey-West t = 0.02**. That is as
close to exactly zero as this kind of measurement gets.

### The one part that is not dead

`geo_crude_beta` — the crude channel isolated, on the 31% of rows the KB actually covers —
scores **IC 0.0144, naive t 3.50, Newey-West t 1.32**, with a 53.4% hit rate. It does not
clear significance, but it is the same order of magnitude as the best technical factor in the
book, and it is economically coherent (crude is India's dominant exposure). `geo_gross` runs
*negative* on covered names (IC −0.0145) — names facing more geopolitical turbulence
underperform, which is a sensible risk relationship even if it is not a tradeable one.

If any geo work continues, it should be **the crude channel specifically**, not the aggregate.

### Honest caveat, stated up front

The knowledge base was authored in 2026 with full hindsight — which conflicts to track, which
tickers to attach to which theme. That cannot be removed without re-authoring it from period
sources, so **this test was biased in favour of the geo factor**. That asymmetry is what makes
the null result strong: it failed even with hindsight helping it.

Coverage is also a real limit: only **63 of 150 tickers (42%)** have any KB exposure at all.

---

## 4. The final IC table (Newey-West corrected)

The naive t-stat treats 2,387 daily ICs as 2,387 independent observations. They are not —
each label is a 20-day forward return, so consecutive days share 19/20 of their outcome
window and the effective sample is closer to 120. Correcting this is the difference between
"17 significant features" and the truth:

**20 of 30 signals clear |t| > 2 naively. Three survive Newey-West.**

| signal | IC | t-naive | t-NW | hit |
|---|---|---|---|---|
| atr_pct | 0.0297 | 8.69 | **2.52** | 0.566 |
| momentum_5d | −0.0154 | −5.63 | **−2.58** | 0.469 |
| returns | −0.0112 | −4.37 | **−3.86** | 0.472 |
| dist_52w_low | 0.0219 | 6.69 | 1.86 | 0.552 |
| bollinger_position | −0.0154 | −5.62 | −1.94 | 0.473 |
| GEO geo_crude_beta (covered) | 0.0144 | 3.50 | 1.32 | 0.534 |
| GEO geo_signal | 0.0002 | 0.07 | 0.02 | 0.489 |

All three survivors are the same effect family: **short-term reversal** (`returns`,
`momentum_5d` both negative — recent winners give it back) **plus a volatility premium**
(`atr_pct` positive — higher-ATR names beat the index in this sample). Even the best has
IC 0.030 and t-NW 2.52; a deployable single factor usually wants IC 0.03+ with t-NW above 3,
sustained across regimes.

---

## 5. What this means for the product

The UI presents an **"outperform probability"** from the tri-model ensemble as a headline
number. On this evidence that number has **no demonstrated out-of-sample validity** — it is
not distinguishable from a coin flip on held-out years. It should not be shown to a user as a
prediction without a prominent statement to that effect.

Recommended, in priority order:

1. **Stop reporting random-split CV as model quality.** The 0.811 ROC-AUC in the project
   history is a training-set number; the honest figure is ~0.507.
2. **Reframe the ML panel** as descriptive/explanatory rather than predictive, or gate it
   behind a validated-signal check.
3. **Drop `vix_z`** (and reconsider `market_trend`) — zero cross-sectional variance against a
   cross-sectional target.
4. **If pursuing alpha further**, the only threads with any life are short-term reversal,
   the volatility premium, and the crude-exposure channel. All are marginal.
5. The War Room's value is **explanatory, not predictive** — and that value went up
   substantially with real customs data behind it (below). That is where it should be pitched.

---

## 6. What the government-data work delivered regardless

Independent of the alpha question, the War Room is now grounded in real customs data instead
of hand-typed prose. `data/geopolitics/trade_exposure.json` said things like *"imports ~85–88%
of crude needs"* — plausible, unsourced, undated, unusable quantitatively.

It now carries India's actual reported figures, and some are genuinely non-obvious:

| commodity | India imports (2024) | from Russia+Ukraine |
|---|---|---|
| sunflower oil | $3.64bn | **85.4%** |
| crude oil | $141.53bn | 37.3% |
| fertilizers | $7.71bn | 20.9% |
| thermal coal | $31.53bn | 12.6% |
| natural gas | $29.77bn | **0.2%** |

The gas row is the point: a naive read of "Russia-Ukraine disrupts energy" would flag India's
LNG exposure, and the customs data says that channel is essentially nil while the *edible oil*
channel is the acute one. These figures now feed the Gemini prompt as authoritative grounding.

### Data sources

- **UN Comtrade** (`comtradeapi.un.org`) — official customs statistics India reports to the UN.
  140 commodity-years cached, 14 HS codes × 2015–2024. Free, keyless, rate-limited to ~1 call
  per 7s. Filter `partner2Code == 0` or totals triple-count.
- **World Bank Indicators API** — India macro context. Free, keyless.
- **Deliberately not used:** PPAC (Indian crude basket is PDF-only behind a download shim, and
  tracks Brent at r>0.99); FRED (unreachable from this network); data.gov.in (needs a key).

**Point-in-time discipline:** annual customs data for year Y is not published until well into
Y+1, so `gov_data.as_of()` applies a 12-month reporting lag. Standing on 2024-03-01, the
newest usable vintage is 2022. Conflicts also contribute nothing before their `since` year.

### Known limitations

- Partner shares only name countries in the M49 map, so coverage varies: crude 96.8%,
  gas 94.1%, semis 96.7% — but wheat 6.6%, potash 37.9%. The coverage % is published
  alongside every figure so an unlisted remainder is never read as zero.
- **HS 93 ("arms and ammunition") badly understates defence.** Military aircraft are HS 88 and
  warships HS 89, so the $0.19bn figure is roughly an order of magnitude low and says nothing
  about the Russian platform dependence that actually matters. Carried for partner *mix*, not
  level.

---

## Files

| file | what it is |
|---|---|
| `backend/models/gov_data.py` | UN Comtrade + World Bank ingestion, cached, point-in-time |
| `backend/models/geo_features.py` | time-varying geo signal (pressure × gov weight × direction) |
| `backend/models/panel.py` | cached research panel + cross-sectional rank transform |
| `backend/models/geo_alpha_test.py` | IC + incremental-AUC test for the geo block |
| `backend/models/tuning.py` | Phase B round 1 — model/preprocessing sweep |
| `backend/models/tuning_diagnose.py` | Phase B round 2 — positive control, train/test, IC screen |
| `backend/models/tuning_features.py` | Phase B round 3 — feature-set surgery |
| `backend/models/ic_report.py` | final Newey-West IC table |
| `backend/data/gov/` | 140 cached Comtrade commodity-years + 5 World Bank series |

Reproduce: `cd backend && python -m models.ic_report` (seconds, from cache).
Rebuild the panel: `python -m models.panel` (~4 min, needs network).
