"""
Merits / Demerits — the "why to buy vs why not" read.

Turns the completed multi-factor analysis into two plain-language lists framed
around what an Indian retail investor actually cares about: growth momentum,
financial strength, stability of the ride, valuation/entry, the crowd's mood,
and macro/geopolitical exposure. Purely deterministic (derived from the factor
scores + signals) — no LLM, no buy/sell directive. Educational only.
"""


def _tone_from_score(score, good=60, bad=45):
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "neutral"
    if s >= good:
        return "merit"
    if s <= bad:
        return "demerit"
    return "neutral"


def build_merits(result: dict) -> dict:
    """Return {'merits': [...], 'demerits': [...]} of {title, detail} items."""
    merits, demerits = [], []

    ov   = result.get("overall_score", {}) or {}
    bd   = ov.get("breakdown", {}) or {}
    tech = result.get("technical_analysis", {}) or {}
    fund = result.get("fundamental_analysis", {}) or {}
    sent = result.get("sentiment_analysis", {}) or {}
    vol  = result.get("volumetric_analysis", {}) or {}
    sec  = result.get("sector_comparison", {}) or {}
    geo  = result.get("geopolitical_analysis", {}) or {}
    feats = result.get("features_used", {}) or {}
    consensus = result.get("consensus") or {}

    def add(bucket, title, detail, w=1.0):
        bucket.append({"title": title, "detail": detail, "_w": w})

    # ── Momentum / technical (what momentum-chasers want) ──────────────────
    t_tone = _tone_from_score(bd.get("technical"))
    if t_tone == "merit":
        add(merits, "Momentum is on its side",
            f"Technicals score {round(float(bd.get('technical', 50)))}/100 "
            f"({tech.get('signal', 'positive')}) — the trend and RSI back an entry rather than fight it.")
    elif t_tone == "demerit":
        add(demerits, "Chart is working against you",
            f"Technicals score {round(float(bd.get('technical', 50)))}/100 "
            f"({tech.get('signal', 'weak')}) — momentum buyers are likely to be caught in a downtrend.")

    # ── Fundamentals (what long-term / value investors want) ───────────────
    f_tone = _tone_from_score(bd.get("fundamental"))
    if f_tone == "merit":
        add(merits, "Financially sound business",
            f"Fundamentals score {round(float(bd.get('fundamental', 50)))}/100 — the balance-sheet and "
            "profitability signals suit an investor who wants to hold, not just trade.")
    elif f_tone == "demerit":
        add(demerits, "Shaky fundamentals",
            f"Fundamentals score {round(float(bd.get('fundamental', 50)))}/100 — weak financial-health "
            "metrics make this hard to justify for a buy-and-hold SIP investor.")

    # ── Volatility / stability (what conservative investors want) ──────────
    volatility = feats.get("volatility")
    if volatility is not None:
        if volatility <= 0.28:
            add(merits, "A steadier ride",
                "Historical volatility is on the lower side — smoother drawdowns suit investors "
                "who lose sleep over sharp swings.")
        elif volatility >= 0.45:
            add(demerits, "Expect a bumpy ride",
                "Elevated volatility means larger drawdowns — uncomfortable for capital-preservation "
                "or short-horizon money.")

    # ── Sector vs market (relative strength) ───────────────────────────────
    vs_mkt = (sec.get("vs_market", {}) or {}).get("outperformance")
    if vs_mkt is not None:
        try:
            vs_mkt = float(vs_mkt)
            if vs_mkt >= 3:
                add(merits, "Leading its sector / the index",
                    f"Outperforming the market by {vs_mkt:.1f}% — relative strength that momentum and "
                    "growth investors pay up for.")
            elif vs_mkt <= -3:
                add(demerits, "Lagging the broader market",
                    f"Trailing the market by {abs(vs_mkt):.1f}% — the money is flowing elsewhere for now.")
        except (TypeError, ValueError):
            pass

    # ── Sentiment + divergence (the crowd's mood) ──────────────────────────
    s_tone = _tone_from_score(bd.get("sentiment"))
    div = (sent.get("divergence", {}) or {})
    if s_tone == "merit":
        add(merits, "News flow is supportive",
            "FinBERT reads recent coverage as net-positive — a tailwind for sentiment-driven re-rating.")
    elif s_tone == "demerit":
        add(demerits, "Negative news overhang",
            "Recent coverage skews negative — a headwind until the narrative turns.")
    if div.get("label") and str(div.get("label")).lower() not in ("aligned", "neutral", "n/a"):
        add(demerits if "weaker" in str(div.get("label")).lower() else merits,
            "Stock-vs-sector divergence",
            f"The stock's mood is {div.get('label')} its sector — {div.get('note', 'worth watching as sector moves lead the stock')}.")

    # ── Geopolitical (India-specific macro) ────────────────────────────────
    if geo.get("exposed"):
        drivers = geo.get("drivers", []) or []
        benefit = [d for d in drivers if d.get("direction") == "benefit"]
        hurt    = [d for d in drivers if d.get("direction") != "benefit"]
        if benefit:
            d = benefit[0]
            add(merits, "Geopolitical tailwind for India",
                f"{d.get('conflict')} works in this stock's favour via {d.get('sector')} — an edge tied to "
                "India's positioning.")
        if hurt:
            d = hurt[0]
            add(demerits, "Geopolitical risk exposure",
                f"{d.get('conflict')} pressures this stock via {d.get('sector')} — a macro risk outside "
                "the company's control.")

    # ── Valuation / income / balance sheet (from the key-stats snapshot) ────
    snap = result.get("snapshot", {}) or {}
    pe, pb = snap.get("pe"), snap.get("pb")
    if pe is not None and pe > 0:
        if pe <= 18:
            add(merits, "Reasonably priced earnings",
                f"Trailing P/E of {pe:.1f} is modest next to India's large-cap average — less valuation risk if growth slips.", 1.1)
        elif pe >= 45:
            add(demerits, "Rich valuation",
                f"Trailing P/E of {pe:.1f} prices in a lot of growth — any earnings miss can hurt the share price disproportionately.", 1.1)
    if pb is not None and pb > 0:
        if pb <= 2:
            add(merits, "Trades close to book value", f"Price/book of {pb:.1f} leaves some asset cushion.", 0.7)
        elif pb >= 8:
            add(demerits, "Premium to book value", f"Price/book of {pb:.1f} means little asset backing at this price.", 0.7)
    roe = snap.get("roe_pct")
    if roe is not None:
        if roe >= 15:
            add(merits, "High return on equity", f"ROE of {roe:.0f}% — management turns shareholder capital into profit efficiently.", 1.0)
        elif roe < 8:
            add(demerits, "Low return on equity", f"ROE of {roe:.0f}% is thin — capital is not being put to work very productively.", 1.0)
    de = snap.get("debt_to_equity")
    if de is not None:
        if de <= 50:
            add(merits, "Conservative balance sheet", f"Debt/equity of {de:.0f}% leaves room to ride out a downturn.", 0.9)
        elif de >= 150:
            add(demerits, "Heavily leveraged", f"Debt/equity of {de:.0f}% — interest costs and refinancing risk rise if conditions tighten.", 0.9)
    rg = snap.get("revenue_growth_pct")
    if rg is not None:
        if rg >= 10:
            add(merits, "Growing top line", f"Revenue is growing about {rg:.0f}% year on year.", 0.9)
        elif rg < 0:
            add(demerits, "Shrinking revenue", f"Revenue is down about {abs(rg):.0f}% year on year.", 0.9)
    pm = snap.get("profit_margin_pct")
    if pm is not None:
        if pm >= 15:
            add(merits, "Healthy profit margins", f"Net margin of {pm:.0f}% shows pricing power or cost discipline.", 0.8)
        elif pm < 4:
            add(demerits, "Thin profit margins", f"Net margin of {pm:.0f}% leaves little buffer against cost inflation.", 0.8)
    beta = snap.get("beta")
    if beta is not None:
        if beta <= 0.8:
            add(merits, "Defensive beta", f"Beta of {beta:.2f} — it has tended to move less than the market.", 0.7)
        elif beta >= 1.4:
            add(demerits, "High market sensitivity", f"Beta of {beta:.2f} — it has tended to amplify market swings, both ways.", 0.7)
    pos = snap.get("range_position_pct")
    if pos is not None:
        if pos <= 25:
            add(merits, "Near its 52-week low", f"Trading in the bottom {pos:.0f}% of its 52-week range — a better entry than chasing highs, if the thesis holds.", 0.6)
        elif pos >= 92:
            add(demerits, "Stretched near its 52-week high", f"Trading at {pos:.0f}% of its 52-week range — little margin for error on entry.", 0.6)

    # ── Model consensus (the quant read, relative to the Nifty) ────────────
    # Uses the CALIBRATED ensemble. Out of sample the models show little skill,
    # so a near-50% probability is reported as "no clear edge" rather than
    # dressed up as a merit or a demerit.
    if consensus:
        ens = result.get("ensemble") or {}
        verdict = ens.get("prediction")
        p_out = ens.get("outperform_probability")
        odds = f" (calibrated ~{round(p_out * 100, 1)}%)" if p_out is not None else ""
        if verdict == "OUTPERFORM":
            add(merits, "Models lean towards beating the index",
                f"The calibrated ensemble gives a slight lean to outperforming the Nifty{odds}. "
                "Back-tests show this lean is weak — treat it as a tie-breaker, not a thesis.")
        elif verdict == "UNDERPERFORM":
            add(demerits, "Models lean towards lagging the index",
                f"The calibrated ensemble gives a slight lean to trailing the Nifty{odds}. "
                "Back-tests show this lean is weak — treat it as a caution, not a verdict.")
        else:
            add(demerits, "No reliable ML edge",
                f"Calibrated odds of beating the Nifty are about even{odds}; out-of-sample testing "
                "found the models cannot reliably separate winners from losers at this horizon.")
        if not consensus.get("agree"):
            add(demerits, "Models don't fully agree",
                "The three models are split — another reason not to lean on the ML read.")

    # ── Fallbacks so the panel is never empty ──────────────────────────────
    if not merits:
        add(merits, "Balanced risk profile",
            "No single factor stands out as a strong positive — a neutral, watch-and-wait candidate.")
    if not demerits:
        add(demerits, "No glaring red flags",
            "No single factor stands out as a strong negative — but no analysis removes market risk.")

    # ── Keep the lists consistent with the overall call ────────────────────
    # A stock that scores as a buy should show more reasons to buy than not to,
    # and vice versa. Items are ranked by strength and the *weaker* side is
    # trimmed; nothing is invented to pad the stronger side.
    score = float(ov.get("score", 50) or 50)
    stance = "buy" if score >= 60 else "avoid" if score <= 45 else "hold"

    merits.sort(key=lambda m: -m["_w"])
    demerits.sort(key=lambda m: -m["_w"])
    merits, demerits = merits[:6], demerits[:6]
    if stance == "buy":
        if len(merits) <= len(demerits):
            add(merits, "Composite score supports it",
                f"The six-factor score is {score:.0f}/100 — strong enough to sit in the buy band.", 0.5)
        cap = max(1, len(merits) - 1)
        demerits = demerits[:cap]
    elif stance == "avoid":
        if len(demerits) <= len(merits):
            add(demerits, "Composite score argues for caution",
                f"The six-factor score is {score:.0f}/100 — weak enough to sit in the caution/avoid band.", 0.5)
        cap = max(1, len(demerits) - 1)
        merits = merits[:cap]
    else:
        merits, demerits = merits[:max(1, min(len(merits), len(demerits) + 1))], demerits[:max(1, min(len(demerits), len(merits) + 1))]

    strip = lambda items: [{"title": i["title"], "detail": i["detail"]} for i in items]
    return {"merits": strip(merits), "demerits": strip(demerits), "stance": stance, "score": round(score, 1)}
