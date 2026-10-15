"""
Deterministic India-impact engine for the Geopolitical Analyzer.

Replaces the Gemini round-trip (60-100 s, needs a key, can fail) with a report
built directly from the data the app already holds:

  * official customs figures (UN Comtrade) - value, supplier mix, and the share of
    each commodity India sources from the parties to the scenario,
  * India's bilateral-relations notes for every party,
  * the curated theme -> sector -> ticker maps,
  * the conflict knowledge base (status, escalation triggers).

Free-text scenarios are matched by keyword to countries, commodities and live
conflicts; the report schema is the same one the UI already renders, so the
front end does not care which engine produced it. Gemini is an optional extra
(`use_ai`) that only writes a short narrative on top of this report.
"""

from __future__ import annotations

import math
import re

# keyword -> commodity keys (matches trade_exposure.json / gov_data.COMMODITIES)
_COMMODITY_KEYWORDS = {
    "crude_oil":           ["oil", "crude", "petroleum", "opec", "brent", "hormuz", "refiner"],
    "natural_gas":         ["natural gas", "lng", " gas "],
    "thermal_coal":        ["coal"],
    "wheat":               ["wheat", "grain", "food supply", "food price"],
    "sunflower_oil":       ["sunflower", "edible oil", "vegetable oil", "cooking oil"],
    "palm_oil":            ["palm oil", "palm"],
    "fertilizers":         ["fertili", "urea", "potash", "dap "],
    "semiconductors":      ["chip", "semiconductor", "tsmc", "wafer"],
    "electronics":         ["electronics", "smartphone"],
    "rare_earths":         ["rare earth", "magnet", "lithium", "critical mineral"],
    "pharmaceuticals_api": ["pharma", " api ", "drug", "medicine"],
    "defense":             ["defence", "defense", "weapon", "arms", "missile", "fighter", "s-400"],
    "gold":                ["gold", "safe haven", "safe-haven"],
    "shipping":            ["shipping", "red sea", "suez", "freight", "strait", "maritime",
                            "blockade", "hormuz", "port", "tanker"],
    "textiles":            ["textile", "garment", "apparel", "cotton"],
}

# places that imply parties not always spelled as countries
_PLACE_PARTIES = {
    "hormuz": ["Iran", "Saudi Arabia", "UAE"], "persian gulf": ["Iran", "Saudi Arabia", "UAE"],
    "gulf": ["Saudi Arabia", "UAE", "Iran"], "red sea": ["Saudi Arabia"], "houthi": ["Saudi Arabia"],
    "kashmir": ["Pakistan"], "ladakh": ["China"], "galwan": ["China"], "lac": ["China"],
    "taiwan strait": ["Taiwan", "China"], "south china sea": ["China", "Philippines"],
    "crimea": ["Russia", "Ukraine"], "black sea": ["Russia", "Ukraine"],
    "gaza": ["Israel"], "tehran": ["Iran"], "beijing": ["China"], "moscow": ["Russia"],
}
_COUNTRY_ALIASES = {"usa": "United States", "u.s.": "United States", "america": "United States",
                    "uae": "UAE", "emirates": "UAE", "saudi": "Saudi Arabia", "s. korea": "South Korea"}

_WAR_WORDS = ("war", "attack", "strike", "invasion", "clash", "blockade", "sanction", "closure",
              "closes", "close", "shutdown", "embargo", "escalat", "conflict", "tension", "ban")

# themes whose headline driver falls rather than rises
_DOWN_THEMES = {"semiconductor_disruption"}

_DRIVER_LABEL = {
    "crude_oil_up": "Crude oil price spike", "energy_importer_relief": "Discounted-crude advantage for refiners",
    "natural_gas_up": "Gas / LNG price spike", "edible_oil_up": "Edible-oil price spike",
    "fertilizer_input_up": "Fertiliser & gas input costs", "wheat_export_window": "Global grain shortage - export window",
    "defense_indigenization": "Defence self-reliance push", "defense_exports": "Indian defence exports",
    "china_plus_one": "China+1 supply-chain shift", "semiconductor_disruption": "Semiconductor supply disruption",
    "rare_earths_squeeze": "Rare-earth export squeeze", "shipping_freight_up": "Freight & insurance costs",
    "coal_up": "Coal price spike", "gold_safe_haven": "Safe-haven demand for gold",
    "risk_off_rupee_weak": "Risk-off flows & rupee weakness", "pharma_generic_demand": "Generic pharma demand",
    "textile_export_gain": "Apparel orders shifting to India",
}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def parse_scenario(text, relations, commodities, conflicts):
    """Scenario text -> a conflict-shaped context (parties, commodities, status)."""
    low = " " + re.sub(r"\s+", " ", text.lower()) + " "
    parties = []
    for country in relations:
        if country.lower() in low:
            parties.append(country)
    for alias, canon in _COUNTRY_ALIASES.items():
        if alias in low and canon not in parties and canon in relations:
            parties.append(canon)
    for place, ps in _PLACE_PARTIES.items():
        if place in low:
            parties += [p for p in ps if p not in parties]
    if re.search(r"\bindia\b", low) and not parties:
        parties.append("India")

    found = {c for c, kws in _COMMODITY_KEYWORDS.items() if any(k in low for k in kws)}
    found |= {c for c in commodities if c.replace("_", " ") in low}

    # Inherit from a tracked conflict when the scenario names >=1 of its parties
    # and >=1 of its commodities (or >=2 parties) - gives it status + triggers.
    best, best_score = None, 0
    for c in conflicts:
        pm = len(set(c.get("parties", [])) & set(parties))
        cm = len(found & set(c.get("key_commodities", [])))
        score = pm * 2 + cm
        if pm and score > best_score:
            best, best_score = c, score
    if best:
        found |= set(best.get("key_commodities", [])) if len(found) == 0 else set()

    return {
        "id": "user_scenario",
        "name": text.strip()[:120],
        "parties": [p for p in parties if p != "India"] or parties,
        "status": (best or {}).get("status", "user_scenario"),
        "since": (best or {}).get("since"),
        "india_relevance": (best or {}).get("india_relevance", "unknown"),
        "summary": "User-supplied scenario matched against the knowledge base and India's customs data.",
        "key_commodities": sorted(found),
        "escalation_triggers": (best or {}).get("escalation_triggers", []),
        "user_scenario_text": text.strip(),
        "matched_conflict": (best or {}).get("name"),
        # India as a belligerent is risk-off by construction (rupee, FPI outflows).
        "india_direct": bool(best and best.get("id") in ("india_pakistan", "india_china")) or "India" in parties,
        "has_war_words": any(w in low for w in _WAR_WORDS),
    }


def _magnitude(fact, party_share_pct):
    """1-5 severity from the commodity's weight in India's trade + supplier exposure."""
    if not fact:
        return 2
    value = fact.get("india_trade_value_usd_bn") or 0
    weight = min(1.0, math.sqrt(value / 60.0))                  # $60bn+ ~ systemic
    exposure = _clip((party_share_pct or 0) / 60.0, 0, 1)       # 60%+ from belligerents ~ max
    return int(_clip(round(1 + 4 * (0.55 * weight + 0.45 * exposure)), 1, 5))


def build_report(conflict, facts, theme_commodity):
    """Assemble the UI report dict from grounded facts. `theme_commodity` maps theme id -> commodity key."""
    official = facts.get("official_trade_data", {})
    channels, n_neg, n_pos = [], 0.0, 0.0

    for tid, theme in facts.get("sector_themes", {}).items():
        com = theme_commodity.get(tid)
        fact = official.get(com) if com else None
        party_share = (fact or {}).get("share_from_conflict_parties_pct", 0)
        mag = _magnitude(fact, party_share)

        affected = []
        for key, impact in (("benefits", "positive"), ("hurts", "negative")):
            for entry in theme.get(key, []):
                m = mag if key == "hurts" else max(1, mag - 1)
                affected.append({
                    "sector": entry.get("sector"), "impact": impact, "magnitude": m,
                    "rationale": entry.get("rationale"), "tickers": entry.get("tickers", []),
                })
                if impact == "negative":
                    n_neg += m
                else:
                    n_pos += m
        if affected:
            channels.append({
                "driver": _DRIVER_LABEL.get(tid, theme.get("description", tid)),
                "direction": "down" if tid in _DOWN_THEMES else "up",
                "theme": tid, "commodity": com,
                "evidence": ({
                    "value_usd_bn": fact["india_trade_value_usd_bn"], "year": fact["reporting_year"],
                    "flow": fact["flow"], "from_conflict_parties_pct": party_share,
                    "top_partners": fact["top_named_partners_pct"],
                } if fact else None),
                "affected": sorted(affected, key=lambda a: -a["magnitude"]),
            })
    channels.sort(key=lambda c: -max((a["magnitude"] for a in c["affected"]), default=0))

    total = n_neg + n_pos
    if conflict.get("india_direct"):
        bias = "risk-off"
    elif total == 0:
        bias = "mixed"
    elif n_neg > n_pos * 1.25:
        bias = "risk-off"
    elif n_pos > n_neg * 1.25:
        bias = "risk-on"
    else:
        bias = "mixed"

    # ── Summary: lead with the hardest number ──
    lines = []
    exposed = sorted(official.items(),
                     key=lambda kv: -((kv[1].get("share_from_conflict_parties_pct") or 0)
                                      * (kv[1].get("india_trade_value_usd_bn") or 0)))
    for com, f in exposed[:2]:
        verb = "imported" if f["flow"] == "imports" else "exported"
        sp = f.get("share_from_conflict_parties_pct", 0)
        tail = (f", {sp:.0f}% of it with {'/'.join(conflict.get('parties') or ['the parties'])}"
                if sp >= 1 else "")
        lines.append(f"India {verb} ${f['india_trade_value_usd_bn']:.1f}bn of {com.replace('_', ' ')} "
                     f"in {f['reporting_year']}{tail}.")
    n_sectors_neg = sum(1 for c in channels for a in c["affected"] if a["impact"] == "negative")
    n_sectors_pos = sum(1 for c in channels for a in c["affected"] if a["impact"] == "positive")
    if channels:
        lines.append(f"Net read: {n_sectors_neg} sector groups pressured, {n_sectors_pos} supported - "
                     f"{bias.replace('-', ' ')} for Indian equities.")
    else:
        lines.append("No mapped transmission channel - the scenario did not match a tracked commodity. "
                     "Try naming a commodity (oil, gas, wheat, chips...) or a country.")

    covered = [c for c in conflict.get("key_commodities", []) if c in official]
    max_share = max([(official[c].get("share_from_conflict_parties_pct") or 0) for c in covered] or [0])
    confidence = "high" if len(covered) >= 2 and max_share >= 10 else "medium" if covered else "low"

    watch = list(conflict.get("escalation_triggers") or [])
    for com in conflict.get("key_commodities", [])[:3]:
        watch.append(f"Price and shipment data for {com.replace('_', ' ')} - India's landed cost reacts first.")
    watch.append("RBI intervention and the rupee: a sustained slide would amplify every import channel.")

    caveats = [
        "Customs data is annual and lags ~12 months; recent supplier shifts may not show yet.",
        "Sector and ticker links come from a curated map, not a statistical model - they explain exposure, they do not forecast returns.",
    ]
    low_cov = [c for c, f in official.items() if f.get("named_partner_coverage_pct", 100) < 70]
    if low_cov:
        caveats.append("Partner tables explain <70% of the flow for: " + ", ".join(
            c.replace("_", " ") for c in low_cov) + " - an unlisted remainder is not zero.")
    if conflict.get("id") == "user_scenario":
        caveats.append("Scenario text is matched by keyword to countries and commodities, so be explicit about both.")

    return {
        "india_summary": " ".join(lines),
        "net_bias": bias,
        "confidence": confidence,
        "transmission_channels": channels,
        "watch_items": watch[:6],
        "caveats": caveats,
    }
