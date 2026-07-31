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
            ml_signal = stock_result.get("ml_prediction", "HOLD")
            confidence = stock_result.get("ml_confidence", 0.5)
            risk_score = stock_result.get("overall_score", {}).get("score", 50)
            sector = stock_result.get("sector_comparison", {}).get("sector", "Unknown")
            ticker = stock_result.get("ticker")

            appetite = (risk_appetite or "medium").strip().lower()
            if appetite not in self._APPETITE_EQUITY_CAP:
                appetite = "medium"

            # ── Base equity weight from the model signal ──────────────────────
            base_weight = {"BUY": 0.6, "HOLD": 0.3}.get(ml_signal, 0.1)

            # Scale by model confidence, then by the stock's own risk
            weight = base_weight * (0.5 + confidence)
            risk_factor = max(0.2, 1 - (risk_score / 100))
            weight *= risk_factor

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

            if capped:
                reasons.append(f"{appetite} risk appetite caps equity near {round(equity_cap*100)}%")
            else:
                reasons.append(f"{appetite} risk appetite; equity below the {round(equity_cap*100)}% cap")

            # ── Defensive split shaped by appetite ────────────────────────────
            split = self._APPETITE_DEFENSIVE_SPLIT[appetite]
            remaining = 1 - weight

            portfolio = [
                {"asset": ticker, "weight": round(weight * 100, 2),
                 "type": "Equity", "sector": sector},
                {"asset": "Safe Bonds / Debt", "weight": round(remaining * split["bonds"] * 100, 2),
                 "type": "Debt"},
                {"asset": "Gold / Commodities", "weight": round(remaining * split["gold"] * 100, 2),
                 "type": "Commodity"},
                {"asset": "Cash Reserve", "weight": round(remaining * split["cash"] * 100, 2),
                 "type": "Cash"},
            ]

            diversification_score = round(100 - abs(weight - 0.5) * 100, 2)

            # ── Suggestions ───────────────────────────────────────────────────
            suggestions = []
            if ml_signal == "BUY":
                suggestions.append("Model favours the stock — increase exposure gradually")
            elif ml_signal == "SELL":
                suggestions.append("Model is bearish — reduce exposure or avoid entry")
            if risk_score > 70:
                suggestions.append("Elevated stock risk — lean on the defensive allocation")
            if confidence < 0.5:
                suggestions.append("Low model confidence — avoid aggressive positions")
            if appetite == "low":
                suggestions.append("Conservative profile — capital preservation prioritised")
            elif appetite == "high":
                suggestions.append("Aggressive profile — higher equity tilt accepted for growth")

            return {
                "allocation": portfolio,
                "diversification_score": diversification_score,
                "suggestions": suggestions,
                # Explainability: how the profile shaped this allocation
                "profile_basis": {
                    "risk_appetite": appetite,
                    "horizon": horizon,
                    "equity_cap_pct": round(equity_cap * 100, 1),
                    "reasons": reasons,
                },
            }

        except Exception as e:
            return {"allocation": [], "error": str(e)}

    @staticmethod
    def _safe_int(value):
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except (TypeError, ValueError):
            return None
