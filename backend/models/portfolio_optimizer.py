import numpy as np


class PortfolioOptimizer:
    """
    ML-driven Portfolio Allocation Engine.

    Combines the model signal + confidence + overall risk score to size the
    equity position, then shapes the defensive split (bonds / gold / cash) using
    the investor's own risk profile (risk appetite + horizon). This makes the
    allocation personal: the same stock produces a heavier equity tilt for an
    aggressive, long-horizon investor and a more defensive mix for a
    conservative one.
    """

    # Max fraction of the portfolio that may sit in the analysed equity,
    # by stated risk appetite.
    _APPETITE_EQUITY_CAP = {"low": 0.35, "medium": 0.55, "high": 0.80}

    # How the *defensive* remainder is divided, by risk appetite.
    _APPETITE_DEFENSIVE_SPLIT = {
        "low":    {"bonds": 0.65, "gold": 0.10, "cash": 0.25},
        "medium": {"bonds": 0.60, "gold": 0.20, "cash": 0.20},
        "high":   {"bonds": 0.50, "gold": 0.30, "cash": 0.20},
    }

    def __init__(self):
        pass

    def optimize(self, stock_result, risk_appetite=None, horizon=None):
        try:
            ml_signal = stock_result.get("ml_prediction", "UNDERPERFORM")
            risk_score = stock_result.get("overall_score", {}).get("score", 50)
            sector = stock_result.get("sector_comparison", {}).get("sector", "Unknown")
            ticker = stock_result.get("ticker")

            # P(beats the Nifty over the next ~month), from the soft-vote
            # ensemble. This is the natural sizing input for a relative-
            # performance model: 0.5 = coin flip, higher = stronger edge.
            ensemble = stock_result.get("ensemble") or {}
            p_out = ensemble.get("outperform_probability")
            if p_out is None:
                confidence = stock_result.get("ml_confidence", 0.5)
                p_out = confidence if ml_signal == "OUTPERFORM" else 1.0 - confidence
            p_out = max(0.0, min(1.0, float(p_out)))

            appetite = (risk_appetite or "medium").strip().lower()
            if appetite not in self._APPETITE_EQUITY_CAP:
                appetite = "medium"

            # ── Base equity weight from the edge over the index ───────────────
            # Scale the edge (p_out - 0.5) rather than the raw probability, so a
            # coin-flip read lands on a modest base position and only a genuine
            # edge earns a heavier one.
            edge = (p_out - 0.5) * 2.0            # -1 .. +1
            weight = 0.30 + 0.35 * edge           # ~0.30 at 50/50, up to ~0.65

            # Scale by the stock's own QUALITY. `overall_score` is a health score
            # (higher = better / lower-risk), so a stronger stock earns MORE
            # equity. (An earlier `1 - score/100` inverted this and starved good
            # names of weight.)
            quality_factor = 0.4 + 0.6 * (risk_score / 100)   # score 40→0.64, 70→0.82
            weight *= quality_factor

            # ── Personalise: apply the investor's risk-appetite cap ───────────
            equity_cap = self._APPETITE_EQUITY_CAP[appetite]

            # Horizon nudges the cap: longer horizons tolerate more equity
            reasons = []
            horizon = self._safe_int(horizon)
            if horizon is not None:
                if horizon >= 15:
                    equity_cap = min(0.90, equity_cap + 0.10)
                    reasons.append(f"long {horizon}y horizon raises the equity ceiling")
                elif horizon >= 7:
                    equity_cap = min(0.90, equity_cap + 0.05)
                    reasons.append(f"{horizon}y horizon allows a modestly higher equity ceiling")
                elif horizon <= 2:
                    equity_cap = max(0.10, equity_cap - 0.10)
                    reasons.append(f"short {horizon}y horizon lowers the equity ceiling")

            capped = weight > equity_cap
            weight = min(weight, equity_cap)
            weight = min(max(weight, 0.05), 0.90)

            # ── Defensive split shaped by appetite ────────────────────────────
            split = self._APPETITE_DEFENSIVE_SPLIT[appetite]
            remaining = 1 - weight

            portfolio = [
                {"asset": ticker, "weight": round(weight * 100, 2),
                 "type": "Equity", "sector": sector},
                {"asset": "Debt / G-Secs (bonds, debt funds)", "weight": round(remaining * split["bonds"] * 100, 2),
                 "type": "Debt"},
                {"asset": "Gold (SGB / gold ETF)", "weight": round(remaining * split["gold"] * 100, 2),
                 "type": "Commodity"},
                {"asset": "Cash / Liquid fund", "weight": round(remaining * split["cash"] * 100, 2),
                 "type": "Cash"},
            ]

            # ── Diversification score — honest, Herfindahl-based ──────────────
            # Effective number of asset-class positions = 1 / Σ wᵢ². Normalised
            # so an equal 4-way split ≈ 100 and an all-in-one-bucket mix ≈ 0.
            # A heavy single-stock tilt therefore scores LOW (as it should),
            # instead of the old formula that peaked at a 50% equity weight.
            fractions = [max(0.0, item["weight"] / 100.0) for item in portfolio]
            hhi = sum(f * f for f in fractions) or 1.0
            eff_n = 1.0 / hhi
            n_buckets = len(portfolio)
            diversification_score = round(
                max(0.0, min(100.0, (eff_n - 1.0) / (n_buckets - 1.0) * 100.0)), 1
            )
            if diversification_score >= 70:
                div_label = "Well spread across asset classes"
            elif diversification_score >= 45:
                div_label = "Moderately concentrated"
            else:
                div_label = "Concentrated — high single-name risk"

            # ── Suggestions ───────────────────────────────────────────────────
            suggestions = []
            if p_out >= 0.55:
                suggestions.append(
                    f"Models put ~{round(p_out * 100)}% odds on beating the Nifty — "
                    "a staggered (SIP-style) entry beats a lump sum")
            elif p_out <= 0.45:
                suggestions.append(
                    f"Models put only ~{round(p_out * 100)}% odds on beating the Nifty — "
                    "a plain index fund may serve this slot better")
            else:
                suggestions.append(
                    "Models see roughly coin-flip odds against the index — "
                    "no clear edge either way in the next month")
            if risk_score > 70:
                suggestions.append("Elevated stock risk — lean on the debt/gold sleeve for ballast")
            if weight >= 0.5:
                suggestions.append("This is a heavy single-name bet — hold it inside a diversified core (index funds/ETFs)")

            # ── How your profile SHOULD be shaped (prescriptive, educational) ──
            guidance = self._profile_guidance(
                appetite, horizon, risk_score, ml_signal, weight
            )

            return {
                "allocation": portfolio,
                "diversification_score": diversification_score,
                "diversification_label": div_label,
                "suggestions": suggestions,
                # Prescriptive, educational guidance on how the investor's own
                # profile should be shaped given THIS stock's risk — not
                # personalised financial advice.
                "profile_basis": {
                    "risk_appetite": appetite,
                    "horizon": horizon,
                    "equity_cap_pct": round(equity_cap * 100, 1),
                    "capped": capped,
                    "guidance": guidance,
                },
            }

        except Exception as e:
            return {"allocation": [], "error": str(e)}

    # ── Prescriptive profile guidance ──────────────────────────────────────
    @staticmethod
    def _profile_guidance(appetite, horizon, risk_score, ml_signal, weight):
        """
        General, educational pointers on how an investor's profile *should* be
        shaped to hold a stock with this risk character. Non-directive: no
        buy/sell/allocate instruction, framed as what tends to suit whom.
        """
        g = []

        # Risk character of the stock → temperament / horizon it suits
        if risk_score < 45:
            g.append("This stock carries above-average risk — it tends to suit investors with a "
                     "longer horizon and the temperament to sit through sharp drawdowns.")
        elif risk_score >= 65:
            g.append("This is a relatively steady name — it fits even conservative, capital-preservation "
                     "oriented investors as a core holding.")
        else:
            g.append("This stock sits in the middle of the risk spectrum — comfortable for a balanced "
                     "investor who accepts moderate swings for moderate growth.")

        # Horizon shaping
        if horizon is not None:
            if horizon <= 3:
                g.append(f"A {horizon}-year horizon is short for single-stock equity — a longer runway "
                         "(7y+) or a staggered SIP entry would reduce timing risk.")
            elif horizon >= 10:
                g.append(f"Your {horizon}-year horizon is a genuine edge here — it lets you ride out "
                         "volatility that would hurt a short-term holder.")
        else:
            g.append("Set a clear horizon before committing — single-stock equity rewards patience "
                     "measured in years, not months.")

        # Appetite shaping
        if appetite == "low":
            g.append("A low risk appetite argues for anchoring most of the portfolio in debt/gold and "
                     "keeping this equity a small satellite position.")
        elif appetite == "high":
            g.append("An aggressive appetite can carry a heavier equity tilt — but discipline (stop "
                     "levels, position limits) matters more, not less, as size grows.")
        else:
            g.append("A medium appetite is well matched — keep this within a diversified core rather "
                     "than letting one conviction name dominate.")

        # Concentration reminder
        if weight >= 0.5:
            g.append("At this suggested weight the portfolio leans heavily on one stock — broadening "
                     "into index funds/ETFs would make your profile more resilient.")

        return g

    @staticmethod
    def _safe_int(value):
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None
