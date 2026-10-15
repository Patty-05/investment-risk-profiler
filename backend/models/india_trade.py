"""
India trade overview for the Geopolitical Analyzer.

Everything here is read from the cached UN Comtrade customs tables
(`data/gov/comtrade/`, India as reporter) and the World Bank indicators already
ingested by `models/gov_data.py`. Nothing is typed in by hand.

Per sector it reports India's trade value, the named partner nations with their
share, supplier concentration (Herfindahl index), the 5-year change, and how much
of the flow comes from countries that are party to a tracked conflict - the
number that decides whether a headline is a real supply risk to India.

Aggregating the same tables by country gives the "who does India trade with"
view: each partner's total value across the tracked sectors and which sectors
it supplies.

WTO's stats API needs a registered key; UN Comtrade is the same customs data
India files, and is keyless, so it is the source used.
"""

from __future__ import annotations

import datetime as _dt
import json
import os

from models import gov_data as g

_KB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "geopolitics", "conflicts.json")

# Sectors beyond gov_data.COMMODITIES, fetched by chapter-level HS code.
# key -> (hs, flow, label)
EXTRA_SECTORS = {
    "mineral_fuels":     ("27", "M", "Mineral fuels, oils (all)"),
    "electronics_elec":  ("85", "M", "Electrical machinery & electronics"),
    "machinery":         ("84", "M", "Industrial machinery"),
    "organic_chemicals": ("29", "M", "Organic chemicals"),
    "plastics":          ("39", "M", "Plastics"),
    "iron_steel":        ("72", "M", "Iron & steel"),
    "vehicles":          ("87", "M", "Vehicles & parts"),
    "pharma_exports":    ("30", "X", "Pharmaceutical products"),
    "gems_jewellery":    ("71", "X", "Gems & jewellery"),
    "cereals_rice":      ("10", "X", "Cereals (incl. rice)"),
    "knit_apparel":      ("61", "X", "Apparel, knitted"),
    "footwear":          ("64", "X", "Footwear"),
}

_NOT_SUPPLY_RISK = {"us_china_tech_war"}

# Narrower commodity tables that sit inside a chapter-level sector; counting both
# would double the partner totals.
_NESTED = {"crude_oil", "natural_gas", "thermal_coal", "potash", "semiconductors"}

_LABELS = {
    "crude_oil": "Crude oil", "natural_gas": "Natural gas / LNG", "thermal_coal": "Coal",
    "wheat": "Wheat", "palm_oil": "Palm oil", "sunflower_oil": "Sunflower oil",
    "fertilizers": "Fertilizers", "potash": "Potash", "gold": "Gold",
    "semiconductors": "Semiconductors (ICs)", "rare_earths": "Rare earths",
    "defense": "Arms & ammunition", "pharmaceuticals_api": "Pharma APIs",
    "textiles": "Apparel (woven)",
}


def _spec(key):
    """(hs, flow, label) for a sector key from either source."""
    if key in g.COMMODITIES:
        c = g.COMMODITIES[key]
        return c["hs"], c["flow"], _LABELS.get(key, c["desc"])
    hs, flow, label = EXTRA_SECTORS[key]
    return hs, flow, label


def _shares(hs, flow, year):
    rec = g.fetch_comtrade(hs, year, flow=flow, cache_only=True)
    if not rec or not rec.get("total_usd"):
        return None, None
    total = rec["total_usd"]
    named = {}
    for code, val in rec["partners_usd"].items():
        name = g.CODE_TO_COUNTRY.get(int(code))
        if name:
            named[name] = val / total
    return named, total


def _conflict_parties():
    """{country: [conflict names]} for conflicts that are live (not merely latent)."""
    try:
        with open(_KB, "r", encoding="utf-8") as fh:
            conflicts = json.load(fh)["conflicts"]
    except Exception:
        return {}
    out = {}
    for c in conflicts:
        # The US-China tech war is a policy contest, not a supply-disruption risk to
        # India's US trade; flagging the US everywhere would swamp the signal.
        if c.get("status") in ("active", "recurring", "frozen") and c.get("id") not in _NOT_SUPPLY_RISK:
            for p in c.get("parties", []):
                out.setdefault(p, []).append(c.get("name"))
    return out


def _latest(key, hs, flow):
    """Newest cached (year, shares, total) walking back from the published year."""
    top = g.latest_published_year(_dt.date.today())
    for yr in range(top, top - 6, -1):
        sh, total = _shares(hs, flow, yr)
        if sh is not None:
            return yr, sh, total
    return None, None, None


def _earlier_total(hs, flow, year):
    """Total value ~5 years before `year` (nearest cached year within +-1)."""
    for yr in (year - 5, year - 6, year - 4):
        rec = g.fetch_comtrade(hs, yr, flow=flow, cache_only=True)
        if rec and rec.get("total_usd"):
            return yr, rec["total_usd"]
    return None, None


def overview():
    parties = _conflict_parties()
    sectors, by_partner = [], {}

    keys = list(g.COMMODITIES) + list(EXTRA_SECTORS)
    for key in keys:
        hs, flow, label = _spec(key)
        year, shares, total = _latest(key, hs, flow)
        if not shares:
            continue
        ranked = sorted(shares.items(), key=lambda kv: -kv[1])
        coverage = sum(shares.values())
        # Herfindahl over named partners only, rescaled to their combined share so
        # a thin partner map does not read as "diversified".
        hhi = (sum((v / coverage) ** 2 for v in shares.values()) * 10000) if coverage else None
        flagged = [(k, v) for k, v in ranked if k in parties]
        base_year, base_total = _earlier_total(hs, flow, year)
        change = ((total / base_total - 1) * 100) if base_total else None

        sectors.append({
            "key": key, "label": label, "hs": hs, "flow": "import" if flow == "M" else "export",
            "year": year, "value_usd_bn": round(total / 1e9, 2),
            "change_pct": round(change, 1) if change is not None else None,
            "change_base_year": base_year,
            "top_partners": [{"name": k, "share_pct": round(v * 100, 1),
                              "in_conflict": k in parties} for k, v in ranked[:5]],
            "named_coverage_pct": round(coverage * 100, 1),
            "concentration_hhi": round(hhi) if hhi is not None else None,
            "conflict_party_share_pct": round(sum(v for _, v in flagged) * 100, 1),
            "conflict_parties": [k for k, _ in flagged],
            "scope_note": ("Small arms & ammunition only - military aircraft/ships sit in other chapters."
                           if key == "defense" else None),
        })

        for name, share in ({} if key in _NESTED else shares).items():
            slot = by_partner.setdefault(name, {"import_usd": 0.0, "export_usd": 0.0, "sectors": []})
            slot["import_usd" if flow == "M" else "export_usd"] += share * total
            slot["sectors"].append((label, share * total, flow))

    partners = []
    for name, slot in by_partner.items():
        top = sorted(slot["sectors"], key=lambda s: -s[1])[:3]
        partners.append({
            "name": name,
            "import_usd_bn": round(slot["import_usd"] / 1e9, 2),
            "export_usd_bn": round(slot["export_usd"] / 1e9, 2),
            "top_sectors": [f"{lbl} ({'imp' if f == 'M' else 'exp'})" for lbl, _, f in top],
            "conflicts": parties.get(name, []),
        })
    partners.sort(key=lambda p: -(p["import_usd_bn"] + p["export_usd_bn"]))

    macro = {}
    for name, label in (("merch_imports_usd", "Merchandise imports (USD bn)"),
                        ("merch_exports_usd", "Merchandise exports (USD bn)"),
                        ("fuel_import_share", "Fuel share of imports (%)"),
                        ("energy_import_dep", "Net energy imports (% of use)")):
        s = g.fetch_worldbank(name)
        if s:
            yr = max(s)
            v = s[yr] / 1e9 if "usd" in name else s[yr]
            macro[name] = {"label": label, "year": yr, "value": round(v, 1)}

    return {
        "success": True,
        "sectors": sectors,
        "partners": partners[:14],
        "macro": macro,
        "sources": [
            "UN Comtrade - official customs statistics filed by India (reporter 699)",
            "World Bank Indicators API - India national accounts / trade",
        ],
        "notes": [
            "Latest published customs year lags by ~12 months.",
            "Partner shares cover only named countries; 'coverage' shows how much of each flow they explain.",
        ],
    }
