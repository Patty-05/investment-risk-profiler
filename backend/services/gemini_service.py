"""
Gemini reasoning layer — India-aware AI investment insight.

Takes the full multi-factor analysis result and produces a concise, grounded,
Indian-market-framed narrative. This is prompt + context engineering (no model
fine-tuning): "India-awareness" comes from the role/instructions and the
structured context we feed it, and the model is told to reason ONLY from that
context. Educational analysis, not personalised financial advice.

Tunable by hand:
  • GEMINI_MODEL env var (default gemini-flash-latest)
  • _GENERATION_CONFIG (temperature / length)
  • the prompt template in _build_prompt()
"""

import os
import google.generativeai as genai

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# 'gemini-1.5-flash' is retired; 'gemini-flash-latest' tracks the current stable
# flash model and works on current API keys. Override via GEMINI_MODEL.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-flash-latest")

_GENERATION_CONFIG = {
    "temperature": 0.4,        # low → consistent, grounded; raise for more flair
    # gemini-flash-latest is a "thinking" model — internal reasoning tokens draw
    # from this budget too, so keep it generous or the visible answer truncates.
    "max_output_tokens": 2048,
}

model = genai.GenerativeModel(GEMINI_MODEL)


def _fmt(v, nd=1):
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return "N/A"


def _build_context(d):
    """Compact, structured, India-specific context for the model to reason over."""
    ov   = d.get("overall_score", {}) or {}
    bd   = ov.get("breakdown", {}) or {}
    tech = d.get("technical_analysis", {}) or {}
    fund = d.get("fundamental_analysis", {}) or {}
    sent = d.get("sentiment_analysis", {}) or {}
    vol  = d.get("volumetric_analysis", {}) or {}
    sec  = d.get("sector_comparison", {}) or {}
    geo  = d.get("geopolitical_analysis", {}) or {}
    port = d.get("portfolio", {}) or {}
    basis = port.get("profile_basis", {}) or {}

    lines = []
    lines.append(f"Ticker (NSE): {d.get('ticker')}")
    lines.append(f"Overall risk score: {_fmt(ov.get('score'))}/100  ->  {d.get('risk_level')}")
    lines.append(
        "Factor breakdown (0-100): "
        f"technical {_fmt(bd.get('technical'))}, fundamental {_fmt(bd.get('fundamental'))}, "
        f"sentiment {_fmt(bd.get('sentiment'))}, volumetric {_fmt(bd.get('volumetric'))}, "
        f"sector {_fmt(bd.get('sector'))}, geopolitical {_fmt(bd.get('geopolitical'))}"
    )

    # ML tri-model consensus (all trained on Indian NSE data).
    # TARGET: relative performance — will this stock beat the Nifty 50 over the
    # next ~20 trading days? NOT an absolute up/down or buy/sell call.
    cons = d.get("consensus") or {}
    ens = d.get("ensemble") or {}
    if cons.get("models"):
        parts = []
        for m in cons["models"]:
            if m.get("signal") is not None:
                parts.append(f"{m.get('name')} = {m.get('signal')} (conf {_fmt(m.get('confidence'), 2)})")
        agree_txt = ("all three agree" if cons.get("agree")
                     else f"{cons.get('agreement')} agree; majority {cons.get('majority')}")
        p_out = ens.get("outperform_probability")
        p_txt = (f" Ensemble probability of BEATING the Nifty over the next ~1 month: "
                 f"{_fmt((p_out or 0) * 100)}%." if p_out is not None else "")
        lines.append(
            "Three ML models (trained on Indian NSE data) predict RELATIVE performance — "
            "whether this stock OUTPERFORMS or UNDERPERFORMS the Nifty 50 over the next "
            "~20 trading days (not whether it rises in absolute terms): "
            + "; ".join(parts)
            + f". Consensus: {agree_txt}.{p_txt} The third model is rule-aware (blends the "
              "technical rule engine into its features)."
        )
    else:
        lines.append(
            f"ML model (Indian NSE data, relative to Nifty): {d.get('ml_prediction')} "
            f"(conf {_fmt(d.get('ml_confidence'), 2)})."
        )

    # Technical / fundamental
    lines.append(f"Technical signal: {tech.get('signal')} (RSI status {tech.get('details', {}).get('rsi_status', 'N/A')}, "
                 f"trend {tech.get('details', {}).get('trend', 'N/A')}).")
    lines.append(f"Fundamental score: {_fmt(fund.get('score'))} ({fund.get('signal')}).")

    # Sentiment + divergence (stock vs sector)
    div = sent.get("divergence", {}) or {}
    ssent = sent.get("sector_sentiment", {}) or {}
    lines.append(
        f"News sentiment (FinBERT): stock {_fmt(sent.get('score'))} ({sent.get('sentiment')}), "
        f"sector {_fmt(ssent.get('score'))} ({ssent.get('label')}), "
        f"divergence: {div.get('label', 'N/A')} ({div.get('note', '')})."
    )

    # Sector
    lines.append(f"Sector: {sec.get('sector', 'N/A')}, {_fmt(sec.get('vs_market', {}).get('outperformance'))}% vs market.")

    # Geopolitical (India-specific)
    if geo.get("exposed"):
        drivers = geo.get("drivers", []) or []
        dtxt = "; ".join(
            f"{x.get('conflict')} ({'benefit' if x.get('direction') == 'benefit' else 'hurt'} via {x.get('sector')})"
            for x in drivers[:4]
        )
        lines.append(
            f"Geopolitical exposure: {_fmt(geo.get('score'))} ({geo.get('signal')}) — {geo.get('summary')} "
            f"Key channels: {dtxt}."
        )
    else:
        lines.append("Geopolitical exposure: no direct exposure to tracked conflicts.")

    # Portfolio + investor profile
    alloc = port.get("allocation", []) or []
    if alloc:
        alloc_txt = ", ".join(f"{a.get('asset')} {a.get('weight')}%" for a in alloc)
        lines.append(
            f"Personalised allocation ({basis.get('risk_appetite', 'medium')} risk appetite, "
            f"{basis.get('horizon', 'N/A')}y horizon): {alloc_txt}."
        )

    # Rule-based recommendation
    recs = d.get("recommendations", []) or []
    if recs:
        lines.append(f"Rule-based rating: {recs[0].get('message', '')}")

    return "\n".join(lines)


def _build_prompt(d):
    context = _build_context(d)
    return f"""You are a sell-side equity research analyst in Mumbai who covers the INDIAN market (NSE/BSE) for domestic retail investors. You think in Indian Rupees (INR) and in the Indian macro frame: the RBI rate/repo cycle and liquidity, FII vs DII flows, the rupee (USD/INR), the monsoon and rural demand, the Union Budget and GST, festive/wedding-season demand, and where the Nifty 50 / Sensex and the relevant sector index (Nifty Bank, Nifty IT, Nifty Auto, Nifty Pharma, Nifty FMCG, Nifty Metal, etc.) sit. You know Indian retail investors mostly hold via SIPs and think in multi-year horizons.

Below is a completed multi-factor analysis for one Indian stock. Reason ONLY from this data — do not invent prices, news, or numbers that are not given. Where you add Indian macro colour, keep it to well-known standing context (e.g. "IT earnings track the rupee and US demand", "OMCs are hurt by a crude spike"), never fabricated events.

ANALYSIS DATA
{context}

Write a tight 5–6 sentence insight in clear, confident plain English with an Indian-market voice (INR, NSE framing). Structure it as:
1. BOTTOM LINE (first sentence): the overall stance in a few words, the single biggest reason, and what the THREE ML models say together about whether it BEATS OR LAGS THE NIFTY over the next month (agree / split, the majority, and the probability). Be precise that this is a call relative to the index, not a prediction that the stock rises or falls.
2. The key supporting and opposing factors — name the 2–3 that matter most (e.g. sector-relative strength vs weak fundamentals, or a stock-vs-sector sentiment divergence). Be specific.
3. One line of INDIAN macro context that actually bears on this stock (rate cycle, rupee, crude, monsoon, flows, or Budget/GST) — only if genuinely relevant to its sector.
4. The GEOPOLITICAL read, India-specific: name the actual conflict channels and whether they help or hurt this stock (only when there is exposure).
5. Who this suits and how their profile should be shaped to hold it (risk appetite, horizon, SIP vs lump sum) — reflect the suggested equity weight, framed as general guidance.

Stay balanced (one clear positive, one clear risk). Do NOT tell the reader to buy or sell or give a personalised allocation instruction. Avoid hedging filler ("it is important to note", "investors should carefully consider"). Close with one short sentence: educational analysis, not financial advice. Plain prose only — no markdown, asterisks, bullets, or bold."""


def generate_ai_insight(data):
    try:
        prompt = _build_prompt(data)
        response = model.generate_content(prompt, generation_config=_GENERATION_CONFIG)
        return response.text
    except Exception as e:
        return f"AI insight unavailable: {str(e)}"
