"""
Geopolitical Exposure Analysis (War Room phase 2) - CALIBRATED.

Turns the War Room knowledge base into a per-stock factor: for a given ticker,
find every way it is exposed to a tracked conflict (benefit or hurt) and produce
a 0-100 score consistent with the other factors (higher = more favourably
positioned / lower geopolitical risk).

WHAT "CALIBRATED" MEANS HERE
----------------------------
The previous score was `50 + 40 * direction_bias`, which had three problems:

  1. It ignored *how much* exposure there is. A stock with one latent, low-relevance
     link scored the same +/-40 as one wired into three active conflicts.
  2. It was a pure function of the ticker - it never reflected whether the
     conflict channel is actually being priced right now.
  3. The scale meant nothing: 90 was not "90th percentile of anything".

The score is now built in three steps:

  raw   = bias * strength           bias in [-1,1] (net direction), strength =
                                    total/(total+K) in [0,1) - evidence-weighted,
                                    so thin exposure is shrunk toward neutral.
        [blended 70/30 with the live pressure signal]
                                    the current time-varying signal from
                                    models/geo_features.py (traded driver prices x
                                    UN Comtrade weights), once it has loaded.
  score = percentile of `raw` within every stock the knowledge base covers,
          mapped to [10, 90]. 50 = the median covered stock. Unexposed stocks
          sit at exactly 50 and are excluded from the overall score.

This makes the number interpretable ("better positioned than 80% of covered
names") and comparable across stocks. It does NOT make it predictive: the
walk-forward research (docs/ALPHA_RESEARCH_FINDINGS.md) found no out-of-sample
predictive power in the geo signal (IC ~0). The factor is an exposure/risk lens,
and the output says so.

Deterministic - no LLM call. Live pressure is loaded in a background thread
(`warm()`); until it is ready the score uses the static exposure only, so a
request is never blocked on a market-data download.
"""

from __future__ import annotations

import threading

import numpy as np

from services import war_service

# Evidence shrinkage constant: total exposure weight at which strength = 0.5.
_STRENGTH_K = 0.8
# Share of the raw score taken from live market pressure when available.
_LIVE_SHARE = 0.30

_lock = threading.Lock()
_state = {"universe": None, "live": None, "warming": False}


# -- Static exposure ---------------------------------------------------------
def _weight(e, status_w, rel_w):
    return status_w.get(e["status"], 0.5) * rel_w.get(e["india_relevance"], 0.5)


def _static_components(exposures):
    """(bias, strength, total) for one ticker's exposure list."""
    status_w, rel_w = war_service.exposure_weights()
    net = total = 0.0
    for e in exposures:
        w = _weight(e, status_w, rel_w)
        net += (1 if e["direction"] == "benefit" else -1) * w
        total += w
    bias = net / total if total else 0.0
    strength = total / (total + _STRENGTH_K) if total else 0.0
    return bias, strength, total


# -- Live pressure (background) ---------------------------------------------
def _compute_live(tickers):
    """{ticker: normalised live signal in [-1,1]} from the time-varying feature."""
    import pandas as pd
    from models import geo_features as gf

    idx = pd.DatetimeIndex([pd.Timestamp.today().normalize()])
    raw = {}
    for tk in tickers:
        try:
            raw[tk] = float(gf.geo_series(tk, idx)["geo_signal"].iloc[0])
        except Exception:
            continue
    if not raw:
        return {}
    sd = float(np.std(list(raw.values()))) or 1.0
    return {tk: float(np.tanh(v / sd)) for tk, v in raw.items()}


def warm():
    """Load live pressure without blocking the caller. Safe to call repeatedly."""
    with _lock:
        if _state["live"] is not None or _state["warming"]:
            return
        _state["warming"] = True

    def _run():
        try:
            live = _compute_live(list(war_service._theme_ticker_index()))
            with _lock:
                _state["live"] = live
                _state["universe"] = None  # rebuild the percentile table with live included
            print(f"[geo] live pressure loaded for {len(live)} tickers")
        except Exception as e:
            print(f"[geo] live pressure unavailable: {e}")
            with _lock:
                _state["live"] = {}
        finally:
            with _lock:
                _state["warming"] = False

    threading.Thread(target=_run, daemon=True, name="geo-warm").start()


def _raw_score(ticker, exposures):
    bias, strength, total = _static_components(exposures)
    raw = bias * strength
    live = (_state["live"] or {}).get(ticker)
    used_live = live is not None
    if used_live:
        raw = (1 - _LIVE_SHARE) * raw + _LIVE_SHARE * live
    return raw, bias, strength, total, used_live


def _universe():
    """Sorted raw scores for every covered ticker (the calibration distribution)."""
    with _lock:
        if _state["universe"] is not None:
            return _state["universe"]
    scores = []
    for tk in war_service._theme_ticker_index():
        ex = war_service.ticker_exposures(tk)
        if ex:
            scores.append(_raw_score(tk, ex)[0])
    arr = np.sort(np.array(scores, dtype=float))
    with _lock:
        _state["universe"] = arr
    return arr


def _percentile(raw, universe):
    """Mid-rank percentile in [0,1]; ties (many stocks share a profile) split evenly."""
    if len(universe) == 0:
        return 0.5
    below = np.searchsorted(universe, raw - 1e-9, side="left")
    upto = np.searchsorted(universe, raw + 1e-9, side="right")
    return float((below + 0.5 * (upto - below)) / len(universe))


class GeopoliticalAnalyzer:
    def __init__(self, ticker):
        self.ticker = (ticker or "").upper().strip()

    def get_geopolitical_score(self):
        exposures = war_service.ticker_exposures(self.ticker)

        if not exposures:
            return {
                "score": 50,
                "signal": "No Active Exposure",
                "exposed": False,
                "conflict_count": 0,
                "drivers": [],
                "strength": 0.0,
                "summary": "No direct exposure to tracked geopolitical conflicts.",
                "calibration": {
                    "method": "unexposed",
                    "note": "Neutral by construction; excluded from the overall score.",
                },
            }

        warm()  # no-op once loaded; never blocks
        raw, bias, strength, total, used_live = _raw_score(self.ticker, exposures)
        uni = _universe()
        pct = _percentile(raw, uni)
        score = round(10 + 80 * pct, 1)

        if score >= 62:
            signal = "Geopolitically Favoured"
        elif score <= 38:
            signal = "Geopolitically Exposed"
        else:
            signal = "Mixed Exposure"

        # De-duplicate drivers for display (one row per conflict+direction+sector)
        status_w, rel_w = war_service.exposure_weights()
        seen, drivers = set(), []
        for e in sorted(exposures, key=lambda x: _weight(x, status_w, rel_w), reverse=True):
            key = (e["conflict_id"], e["direction"], e["sector"])
            if key in seen:
                continue
            seen.add(key)
            drivers.append({
                "conflict": e["conflict"],
                "status": e["status"],
                "direction": e["direction"],
                "sector": e["sector"],
                "rationale": e["rationale"],
            })

        conflict_count = len({e["conflict_id"] for e in exposures})
        n_benefit = sum(1 for d in drivers if d["direction"] == "benefit")
        n_hurt = sum(1 for d in drivers if d["direction"] == "hurt")

        return {
            "score": score,
            "signal": signal,
            "exposed": True,
            "bias": round(bias, 3),
            "strength": round(strength, 3),
            "conflict_count": conflict_count,
            "drivers": drivers[:8],
            "summary": (
                f"Exposed to {conflict_count} tracked conflict(s) - "
                f"{n_benefit} beneficial, {n_hurt} adverse channel(s)."
            ),
            "calibration": {
                "method": "percentile",
                "percentile": round(pct * 100, 1),
                "universe": int(len(uni)),
                "raw": round(raw, 4),
                "live_pressure": used_live,
                "note": (
                    f"Percentile {pct * 100:.0f} of {len(uni)} covered NSE stocks "
                    "(50 = median; higher = better positioned). An exposure lens, not a return forecast - back-tests found no "
                    "out-of-sample predictive power."
                ),
            },
        }
