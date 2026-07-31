"""
Geopolitical Exposure Analysis (War Room phase 2).

Turns the War Room knowledge base into a per-stock factor: for a given ticker,
find every way it is exposed to a tracked conflict (benefit or hurt), weight each
exposure by the conflict's status (active > frozen > latent) and India-relevance,
and produce a 0-100 score consistent with the other analysis factors
(higher = more favourably positioned / lower risk).

This is deterministic (no LLM call) so it adds negligible latency to the stock
analysis pipeline. The War Room page keeps the richer LLM-generated reports.
"""

from __future__ import annotations

from services import war_service


class GeopoliticalAnalyzer:
    def __init__(self, ticker):
        self.ticker = (ticker or "").upper().strip()

    def get_geopolitical_score(self):
        exposures = war_service.ticker_exposures(self.ticker)
        status_w, rel_w = war_service.exposure_weights()

        if not exposures:
            return {
                "score": 50,
                "signal": "No Active Exposure",
                "exposed": False,
                "conflict_count": 0,
                "drivers": [],
                "summary": "No direct exposure to tracked geopolitical conflicts.",
            }

        def weight(e):
            return status_w.get(e["status"], 0.5) * rel_w.get(e["india_relevance"], 0.5)

        net = total = 0.0
        for e in exposures:
            w = weight(e)
            net += (1 if e["direction"] == "benefit" else -1) * w
            total += w

        bias = net / total if total else 0.0            # [-1, 1]
        score = round(max(0.0, min(100.0, 50 + bias * 40)), 2)  # ~[10, 90]

        if score >= 60:
            signal = "Geopolitically Favoured"
        elif score <= 40:
            signal = "Geopolitically Exposed"
        else:
            signal = "Mixed Exposure"

        # De-duplicate drivers for display (one row per conflict+direction+sector)
        seen = set()
        drivers = []
        for e in sorted(exposures, key=weight, reverse=True):
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
            "conflict_count": conflict_count,
            "drivers": drivers[:8],
            "summary": (
                f"Exposed to {conflict_count} tracked conflict(s) — "
                f"{n_benefit} beneficial, {n_hurt} adverse channel(s)."
            ),
        }
