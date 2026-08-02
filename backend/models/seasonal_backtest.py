"""
Out-of-sample validation of the significant seasonal edges.

Honest test of whether the market-relative seasonal patterns are real or just
in-sample curve-fitting:
  1. Split each sector's history into IN-SAMPLE (older years) and OUT-OF-SAMPLE
     (recent years the discovery never sees).
  2. On IN-SAMPLE only, flag the significant sector-month edges (t-test p<0.10
     on excess-vs-Nifty returns) — exactly what the live tool surfaces.
  3. On OUT-OF-SAMPLE, measure whether those flagged edges PERSISTED: did an
     "outperform" edge keep a positive mean excess, and an "underperform" edge a
     negative one? Aggregate the persistence hit-rate and the OOS excess of the
     flagged edges vs the all-cells baseline.

Caveat: seasonality has only ~10 annual observations per month, so per-edge OOS
samples are tiny (~4). The aggregate across all flagged edges is the meaningful
read; individual edges are directional, not definitive.

Run:  python -m models.seasonal_backtest
"""

from __future__ import annotations

import numpy as np

from modules.seasonal_analysis import IndianSeasonalAnalyzer, SECTOR_CONFIG, MONTHS

try:
    from scipy import stats as _stats
    _SCIPY = True
except ImportError:
    _SCIPY = False

_P = 0.10          # in-sample significance bar
_MIN_INSAMPLE = 4  # min years needed in-sample for a t-test


def run(split_frac=0.6):
    an = IndianSeasonalAnalyzer(years=11)
    bench = an._benchmark_monthly()

    flagged = []      # in-sample-significant edges, with their OOS outcome
    all_oos_means = []  # baseline: OOS mean excess of every sector-month cell

    for key in SECTOR_CONFIG:
        df = an._fetch_sector_data(key)
        if df is None:
            continue
        sm = an._monthly_returns(df)
        merged = sm.merge(bench, on=["year", "month"], how="inner")
        merged["excess"] = merged["ret"] - merged["bench"]

        years = sorted(merged["year"].unique())
        if len(years) < 6:
            continue
        cut = years[int(len(years) * split_frac) - 1]
        ins = merged[merged["year"] <= cut]
        oos = merged[merged["year"] > cut]

        for m in range(1, 13):
            ins_ex = ins.loc[ins["month"] == m, "excess"]
            oos_ex = oos.loc[oos["month"] == m, "excess"]
            if len(oos_ex) > 0:
                all_oos_means.append(float(oos_ex.mean()))
            if len(ins_ex) < _MIN_INSAMPLE or ins_ex.std() < 1e-9 or not _SCIPY:
                continue
            t, p = _stats.ttest_1samp(ins_ex, 0.0)
            if p < _P and len(oos_ex) >= 2:
                ins_mean = float(ins_ex.mean())
                oos_mean = float(oos_ex.mean())
                direction = "outperform" if ins_mean > 0 else "underperform"
                persisted = (oos_mean > 0) if ins_mean > 0 else (oos_mean < 0)
                flagged.append({
                    "sector": key, "month": MONTHS[m - 1], "direction": direction,
                    "ins_mean": ins_mean, "ins_p": float(p),
                    "oos_mean": oos_mean, "oos_n": int(len(oos_ex)),
                    "persisted": bool(persisted),
                })

    # ── Report ──
    print("\n" + "=" * 74)
    print(f"SEASONAL EDGE OUT-OF-SAMPLE VALIDATION (split {int(split_frac*100)}% in / rest out)")
    print("=" * 74)
    if not flagged:
        print("No in-sample-significant edges found.")
        return flagged

    baseline = float(np.mean(all_oos_means)) if all_oos_means else 0.0
    persist_rate = np.mean([e["persisted"] for e in flagged]) * 100
    outp = [e for e in flagged if e["direction"] == "outperform"]
    undp = [e for e in flagged if e["direction"] == "underperform"]

    print(f"In-sample-significant edges found : {len(flagged)}  "
          f"({len(outp)} outperform, {len(undp)} underperform)")
    print(f"Baseline OOS mean excess (all cells): {baseline:+.3f}%")
    print(f"\nPERSISTENCE (kept the correct sign out-of-sample): "
          f"{persist_rate:.0f}%  ({sum(e['persisted'] for e in flagged)}/{len(flagged)})")
    if outp:
        oos_mean_out = np.mean([e["oos_mean"] for e in outp])
        print(f"  Outperform edges — OOS mean excess {oos_mean_out:+.3f}%  "
              f"(vs baseline {baseline:+.3f}%);  {sum(e['oos_mean']>0 for e in outp)}/{len(outp)} stayed positive")
    if undp:
        oos_mean_und = np.mean([e["oos_mean"] for e in undp])
        print(f"  Underperform edges — OOS mean excess {oos_mean_und:+.3f}%;  "
              f"{sum(e['oos_mean']<0 for e in undp)}/{len(undp)} stayed negative")

    print("\nEdge-by-edge (in-sample -> out-of-sample):")
    print(f"  {'Sector':13} {'Mo':4} {'dir':13} {'IS mean':>9} {'IS p':>7} {'OOS mean':>9} {'OOSn':>5} {'held?':>6}")
    for e in sorted(flagged, key=lambda x: x["ins_p"]):
        print(f"  {e['sector']:13} {e['month']:4} {e['direction']:13} "
              f"{e['ins_mean']:+8.2f}% {e['ins_p']:>7.3f} {e['oos_mean']:+8.2f}% "
              f"{e['oos_n']:>5} {'YES' if e['persisted'] else 'no':>6}")
    return flagged


if __name__ == "__main__":
    run()
