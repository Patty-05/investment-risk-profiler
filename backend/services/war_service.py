"""
War Room — India-specific geopolitical conflict analysis (RAG).

Pipeline:
    retrieve grounded facts from the knowledge base
        (conflict entry + both parties' India relations + trade exposure of the
         relevant commodities + linked sector/ticker maps)
    -> assemble a grounded prompt
    -> Gemini reasons over ONLY those facts (+ its own knowledge)
    -> structured India-impact report (transmission channels -> sectors -> tickers)

The knowledge base is plain editable JSON under data/geopolitics/ so it can grow
over time. Scenarios come from the curated watchlist OR a free-text user scenario
(which is matched against known countries/commodities for grounding).

Educational geopolitical-risk analysis only — not financial advice.
"""

from __future__ import annotations

import os
import re
import json
import traceback
import datetime as _dt

_KB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "geopolitics")

# Which sector-map themes each commodity activates
_COMMODITY_THEMES = {
    "crude_oil":           ["crude_oil_up", "energy_importer_relief"],
    "natural_gas":         ["natural_gas_up", "fertilizer_input_up"],
    "lng":                 ["natural_gas_up"],
    "wheat":               ["wheat_export_window"],
    "sunflower_oil":       ["edible_oil_up"],
    "palm_oil":            ["edible_oil_up"],
    "fertilizers":         ["fertilizer_input_up"],
    "potash":              ["fertilizer_input_up"],
    "defense":             ["defense_indigenization", "defense_exports"],
    "semiconductors":      ["semiconductor_disruption", "china_plus_one"],
    "electronics":         ["china_plus_one", "semiconductor_disruption"],
    "rare_earths":         ["rare_earths_squeeze"],
    "pharmaceuticals_api": ["china_plus_one", "pharma_generic_demand"],
    "machinery":           ["china_plus_one"],
    "thermal_coal":        ["coal_up"],
    "gold":                ["gold_safe_haven"],
    "shipping":            ["shipping_freight_up"],
    "textiles":            ["textile_export_gain"],
}

# Conflicts where India is itself a belligerent add a broad risk-off channel
_INDIA_DIRECT = {"india_pakistan", "india_china"}

# Weights used by the per-stock geopolitical exposure score (phase 2)
_STATUS_WEIGHT = {
    "active": 1.0, "recurring": 0.9, "frozen": 0.6, "latent": 0.4, "user_scenario": 0.7,
}
_RELEVANCE_WEIGHT = {
    "very_high": 1.0, "high": 0.85, "moderate": 0.6, "low": 0.4, "unknown": 0.5,
}

_kb_cache = {}


# ── Knowledge base loading ──────────────────────────────────────────────────
def _load(name):
    if name in _kb_cache:
        return _kb_cache[name]
    path = os.path.join(_KB_DIR, f"{name}.json")
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    _kb_cache[name] = data
    return data


def list_conflicts():
    """Watchlist for the UI — id, name, parties, status, relevance."""
    conflicts = _load("conflicts")["conflicts"]
    return [
        {k: c[k] for k in ("id", "name", "parties", "status", "since",
                           "india_relevance", "summary")}
        for c in conflicts
    ]


# ── Retrieval ───────────────────────────────────────────────────────────────
def _find_conflict(conflict_id):
    for c in _load("conflicts")["conflicts"]:
        if c["id"] == conflict_id:
            return c
    return None


def _scenario_from_text(text):
    """Free-text scenario -> conflict-shaped context (countries, commodities, nearest tracked conflict)."""
    from services import geo_engine
    return geo_engine.parse_scenario(
        text, _load("india_relations")["relations"], _load("trade_exposure")["commodities"],
        _load("conflicts")["conflicts"])


def _conflict_theme_ids(conflict):
    """The sector-map theme ids activated by a conflict's commodities."""
    theme_ids = []
    for c in conflict.get("key_commodities", []):
        theme_ids.extend(_COMMODITY_THEMES.get(c, []))
    if conflict.get("id") in _INDIA_DIRECT or conflict.get("india_direct"):
        theme_ids.extend(["defense_indigenization", "risk_off_rupee_weak"])
    # Broad safe-haven / risk-off channels for any major *active* conflict:
    # IT exporters and gold are the classic risk-off winners.
    if conflict.get("status") == "active" and conflict.get("india_relevance") in ("high", "very_high"):
        theme_ids.extend(["risk_off_rupee_weak", "gold_safe_haven"])
    return list(dict.fromkeys(theme_ids))  # de-dupe, preserve order


def _retrieve(conflict):
    """Gather the grounded facts relevant to this conflict."""
    relations = _load("india_relations")["relations"]
    exposures = _load("trade_exposure")["commodities"]
    themes = _load("sector_map")["themes"]

    party_relations = {p: relations[p] for p in conflict.get("parties", []) if p in relations}

    commodities = conflict.get("key_commodities", [])
    commodity_exposure = {c: exposures[c] for c in commodities if c in exposures}

    theme_ids = _conflict_theme_ids(conflict)
    relevant_themes = {tid: themes[tid] for tid in theme_ids if tid in themes}

    return {
        "party_relations": party_relations,
        "commodity_exposure": commodity_exposure,
        "sector_themes": relevant_themes,
        "official_trade_data": _gov_trade_facts(commodities, conflict.get("parties", [])),
    }


def _gov_trade_facts(commodities, parties):
    """
    Hard customs numbers for each commodity, from UN Comtrade via models.gov_data.

    The KB's own trade_exposure.json describes dependence in prose ("imports
    ~85-88% of crude needs"), which is fine for a human but gives the model
    nothing to reason quantitatively with, and nothing anyone can audit. This
    adds India's actual reported import value, its top suppliers, and crucially
    the share sourced from the conflict's own parties -- which is the number that
    decides whether a conflict is a genuine supply threat to India or a headline.

    Degrades to {} if the cache has not been built; the War Room still works,
    just without the sourced figures.
    """
    if not commodities:
        return {}
    try:
        from models import gov_data
    except Exception:
        return {}

    party_set = set(parties or [])
    facts = {}
    for c in commodities:
        if c not in gov_data.COMMODITIES:
            continue
        try:
            year, shares, total = gov_data.as_of(c, _dt.date.today())
        except Exception:
            continue
        if not shares or not total:
            continue

        top = sorted(shares.items(), key=lambda kv: -kv[1])[:5]
        party_share = sum(v for k, v in shares.items() if k in party_set)

        # gov_data only names countries in its M49 map, so the partner table can
        # account for less than 100% of the flow. That is fine for crude (~97%
        # covered) and misleading for something like wheat exports (~7%), which
        # goes to many small buyers. Publishing the coverage keeps the model and
        # the reader from reading an unlisted remainder as zero.
        coverage = sum(shares.values())
        facts[c] = {
            "reporting_year": year,
            "india_trade_value_usd_bn": round(total / 1e9, 2),
            "flow": "imports" if gov_data.COMMODITIES[c]["flow"] == "M" else "exports",
            "top_named_partners_pct": {k: round(v * 100, 1) for k, v in top},
            "named_partner_coverage_pct": round(coverage * 100, 1),
            "share_from_conflict_parties_pct": round(party_share * 100, 1),
            "source": "UN Comtrade (India customs filings), HS {}".format(
                gov_data.COMMODITIES[c]["hs"]),
        }
    return facts


# ── Reverse lookup: ticker -> conflict exposures (phase 2) ──────────────────
def _theme_ticker_index():
    """Map every ticker in the sector map to its (theme, direction, sector, rationale)."""
    themes = _load("sector_map")["themes"]
    idx = {}
    for tid, theme in themes.items():
        for direction, key in (("benefit", "benefits"), ("hurt", "hurts")):
            for entry in theme.get(key, []):
                for ticker in entry.get("tickers", []):
                    idx.setdefault(ticker.upper().strip(), []).append({
                        "theme": tid,
                        "direction": direction,
                        "sector": entry.get("sector"),
                        "rationale": entry.get("rationale"),
                    })
    return idx


def ticker_exposures(ticker):
    """
    For a stock ticker, return every way it is exposed to a tracked conflict:
    a list of {conflict, conflict_id, status, india_relevance, direction
    (benefit|hurt), sector, theme, rationale}. Empty if the ticker isn't mapped.
    """
    ticker = (ticker or "").upper().strip()
    hits = _theme_ticker_index().get(ticker, [])
    if not hits:
        return []

    exposures = []
    for c in _load("conflicts")["conflicts"]:
        conflict_themes = set(_conflict_theme_ids(c))
        for hit in hits:
            if hit["theme"] in conflict_themes:
                exposures.append({
                    "conflict_id": c["id"],
                    "conflict": c["name"],
                    "status": c.get("status", "latent"),
                    "india_relevance": c.get("india_relevance", "unknown"),
                    "direction": hit["direction"],
                    "sector": hit["sector"],
                    "theme": hit["theme"],
                    "rationale": hit["rationale"],
                })
    return exposures


def exposure_weights():
    """Expose the scoring weights so the analyzer stays in sync with the KB."""
    return _STATUS_WEIGHT, _RELEVANCE_WEIGHT


# ── Prompt + LLM ────────────────────────────────────────────────────────────
def _build_prompt(conflict, facts):
    grounded = {
        "conflict": {k: conflict.get(k) for k in
                     ("name", "parties", "status", "since", "summary",
                      "key_commodities", "escalation_triggers", "user_scenario_text")
                     if conflict.get(k) is not None},
        "india_bilateral_relations": facts["party_relations"],
        "india_trade_exposure": facts["commodity_exposure"],
        "official_customs_data": facts.get("official_trade_data", {}),
        "candidate_sector_ticker_maps": facts["sector_themes"],
    }

    return f"""You are a geopolitical risk analyst for INDIAN equity investors.
Analyse how the following conflict scenario would affect Indian markets.

Use the GROUNDED FACTS below as your primary evidence. Where
`official_customs_data` is present it comes from India's own customs filings to
UN Comtrade -- treat those figures as authoritative and cite them concretely
(e.g. the share of the commodity India sources from the parties to this
conflict) rather than speaking in vague terms about dependence. Prefer the tickers listed
in the sector maps. Stay India-specific: reason through India's bilateral
relations with each party and its import/export exposure. Be concrete about the
direction of impact (which sectors benefit vs. get hurt and why).

GROUNDED FACTS (JSON):
{json.dumps(grounded, indent=2)}

Respond with ONLY valid JSON (no markdown fences) in exactly this schema:
{{
  "india_summary": "2-3 sentence net view of what this means for Indian markets",
  "net_bias": "risk-on | risk-off | mixed",
  "transmission_channels": [
    {{
      "driver": "the mechanism, e.g. 'Crude oil price spike'",
      "direction": "up | down",
      "affected": [
        {{
          "sector": "sector name",
          "impact": "positive | negative",
          "magnitude": 1,
          "rationale": "one concrete sentence",
          "tickers": ["ONGC.NS"]
        }}
      ]
    }}
  ],
  "watch_items": ["what would confirm or escalate this"],
  "confidence": "low | medium | high",
  "caveats": ["key uncertainties"]
}}
magnitude is 1 (small) to 5 (severe). Keep it grounded and India-specific."""


def _extract_json(text):
    """Pull the first JSON object out of the model response, tolerating fences."""
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the outermost {...}
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(cleaned[start:end + 1])
        raise


def _call_gemini(prompt):
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_name = os.getenv("GEMINI_MODEL", "gemini-flash-latest")
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(prompt)
    return resp.text


# ── Public entry point ──────────────────────────────────────────────────────
def _theme_commodity_map():
    from models import geo_features
    return {tid: spec[2] for tid, spec in geo_features.THEME_DRIVERS.items()}


def analyze(conflict_id=None, scenario_text=None, use_ai=False):
    """
    Analyse a conflict from the watchlist (conflict_id) or a free-text scenario.

    The report is built locally from official customs data + the curated maps
    (instant, no API key). `use_ai=True` additionally asks Gemini for a short
    narrative; if that fails the local report is still returned.
    """
    from services import geo_engine
    try:
        if conflict_id:
            conflict = _find_conflict(conflict_id)
            if not conflict:
                return {"success": False, "error": f"Unknown conflict '{conflict_id}'"}
        elif scenario_text and scenario_text.strip():
            conflict = _scenario_from_text(scenario_text)
        else:
            return {"success": False, "error": "Provide a conflict_id or scenario_text"}

        facts = _retrieve(conflict)
        report = geo_engine.build_report(conflict, facts, _theme_commodity_map())

        result = {
            "success": True,
            "engine": "local",
            "conflict": {
                "id": conflict.get("id"), "name": conflict.get("name"),
                "parties": conflict.get("parties"), "status": conflict.get("status"),
                "matched_conflict": conflict.get("matched_conflict"),
                "commodities": conflict.get("key_commodities", []),
            },
            "report": report,
            "official_data": facts.get("official_trade_data", {}),
            "bilateral": {
                p: {k: r.get(k) for k in ("relationship", "stance", "energy", "defense_dependence",
                                          "india_imports_from", "india_exports_to")}
                for p, r in facts.get("party_relations", {}).items()
            },
            "grounding": _grounding_summary(conflict, facts),
        }

        if use_ai:
            try:
                brief = json.dumps({"scenario": conflict.get("name"), "report": report,
                                    "official_data": result["official_data"]})[:6000]
                result["ai_narrative"] = _call_gemini(
                    "You are an analyst for Indian equity investors. In <=120 words, narrate what this "
                    "India-impact report means, citing the customs figures given. No financial advice. "
                    + brief)
                result["engine"] = "local+gemini"
            except Exception as e:
                result["ai_error"] = f"AI narrative unavailable ({type(e).__name__}); showing the data-driven report."
        return result

    except Exception as e:
        print(f"[war] analyze failed: {traceback.format_exc()}")
        return {"success": False, "error": str(e)}


def _grounding_summary(conflict, facts):
    """What the report was grounded on — for transparency in the UI."""
    return {
        "commodities": conflict.get("key_commodities", []),
        "parties_profiled": list(facts["party_relations"].keys()),
        "themes_considered": list(facts["sector_themes"].keys()),
        "official_data": {
            k: "{} {}: ${}bn, {}% from conflict parties".format(
                v["reporting_year"], v["flow"], v["india_trade_value_usd_bn"],
                v["share_from_conflict_parties_pct"])
            for k, v in facts.get("official_trade_data", {}).items()
        },
    }
