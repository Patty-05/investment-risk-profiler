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

    def add(bucket, title, detail):
        bucket.append({"title": title, "detail": detail})

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

    # ── Model consensus (the quant read) ───────────────────────────────────
    if consensus:
        maj = consensus.get("majority")
        agreement = consensus.get("agreement", "")
        if maj == "BUY":
            add(merits, "Models lean constructive",
                f"{agreement} of the models point to BUY — the quant signals agree with a constructive stance.")
        elif maj == "SELL":
            add(demerits, "Models lean cautious",
                f"{agreement} of the models point to SELL — the quant signals flag downside risk.")
        if not consensus.get("agree"):
            add(demerits, "Models don't fully agree",
                "The three models are split — a reason to wait for cleaner confirmation before sizing up.")

    # ── Fallbacks so the panel is never empty ──────────────────────────────
    if not merits:
        add(merits, "Balanced risk profile",
            "No single factor stands out as a strong positive — a neutral, watch-and-wait candidate.")
    if not demerits:
        add(demerits, "No glaring red flags",
            "No single factor stands out as a strong negative — but no analysis removes market risk.")

    return {"merits": merits, "demerits": demerits}
