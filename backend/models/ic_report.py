"""
Final IC table -- every technical feature plus the geo block, honestly scored.

This is the summary artefact for the research round. It exists separately from
the diagnostic scripts because it answers the one question that survives all the
model-fiddling: does ANY of these signals actually rank Indian stocks by their
next 20 days of performance relative to the Nifty?

Two things make this table different from the earlier screen:

  * VECTORISED IC. Ranking within each date with groupby and computing the
    correlation from grouped moments replaces ~62,000 scipy.spearmanr calls, so
    the whole 29-signal table builds in seconds rather than ten minutes. The
    numbers are identical -- Spearman IS Pearson on ranks.
  * NEWEY-WEST t-STATS. The naive t-stat treats 2,387 daily ICs as 2,387
    independent observations. They are not: each label is a 20-day forward
    return, so consecutive days share 19/20 of their outcome window and the
    effective sample is closer to 120. Correcting this is the difference between
    "17 significant features" and the truth.

Run:
    cd backend && python -m models.ic_report
"""

from __future__ import annotations

import sys
import numpy as np
import pandas as pd

from models.panel import load_panel, attach_geo
from models.feature_engineer import FEATURE_NAMES
from models.geo_alpha_test import newey_west_t

HORIZON = 20


def daily_ic_series(sig, dates, fwd, min_names=15):
    """
    Cross-sectional Spearman IC per date, vectorised.

    Spearman is Pearson on ranks, so ranking within date and then computing the
    correlation from per-date sums gives exactly the same answer as calling
    spearmanr on every group, at a fraction of the cost.
    """
    df = pd.DataFrame({"d": dates, "s": np.asarray(sig, float), "f": np.asarray(fwd, float)})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    if df.empty:
        return np.array([])

    g = df.groupby("d")
    df["rs"] = g["s"].rank(pct=True)
    df["rf"] = g["f"].rank(pct=True)
    df["nuniq"] = g["s"].transform("nunique")
    df["n"] = g["s"].transform("size")
    df = df[(df["n"] >= min_names) & (df["nuniq"] >= 3)].copy()
    if df.empty:
        return np.array([])

    # Precompute the cross-products as columns so every aggregation below is a
    # vectorised groupby-sum. Using .apply here would put a Python call back in
    # the inner loop, which is the exact cost this function exists to avoid.
    df["rs2"] = df["rs"] * df["rs"]
    df["rf2"] = df["rf"] * df["rf"]
    df["rsf"] = df["rs"] * df["rf"]

    agg = df.groupby("d")[["rs", "rf", "rs2", "rf2", "rsf"]].agg(["sum", "size"])
    n = agg[("rs", "size")]
    sx, sy = agg[("rs", "sum")], agg[("rf", "sum")]
    sxx, syy, sxy = agg[("rs2", "sum")], agg[("rf2", "sum")], agg[("rsf", "sum")]

    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    denom = np.sqrt(vx * vy)
    ic = (cov / denom.replace(0, np.nan)).dropna()
    return ic.values


def score(sig, dates, fwd, label):
    ics = daily_ic_series(sig, dates, fwd)
    if len(ics) < 30:
        return {"label": label, "days": len(ics), "ic": np.nan,
                "t_naive": np.nan, "t_nw": np.nan, "hit": np.nan}
    ic = float(ics.mean())
    t_naive = float(ic / (ics.std(ddof=1) / np.sqrt(len(ics))))
    return {
        "label": label, "days": int(len(ics)), "ic": ic,
        "t_naive": t_naive, "t_nw": newey_west_t(ics, HORIZON),
        "hit": float((ics > 0).mean()),
    }


def main():
    print("=" * 82)
    print("FINAL IC TABLE -- technicals + geo, Newey-West corrected")
    print("=" * 82)

    X, y, meta = load_panel()
    G, geo_names = attach_geo(meta)

    dates = pd.to_datetime(meta["date"]).values
    fwd = meta["excess_return"].values

    rows = [score(X[:, i], dates, fwd, nm) for i, nm in enumerate(FEATURE_NAMES)]
    rows += [score(G[:, j], dates, fwd, "GEO " + nm) for j, nm in enumerate(geo_names)]

    exposed = np.abs(G).sum(axis=1) > 0
    rows += [score(G[exposed, j], dates[exposed], fwd[exposed], "GEO " + nm + " (covered)")
             for j, nm in enumerate(geo_names)]

    rows = [r for r in rows if np.isfinite(r["ic"])]
    rows.sort(key=lambda r: -abs(r["ic"]))

    print("\n{:<34}{:>7}{:>10}{:>10}{:>9}{:>8}".format(
        "signal", "days", "IC", "t-naive", "t-NW", "hit"))
    print("-" * 82)
    for r in rows:
        mark = "  <-- survives NW" if abs(r["t_nw"]) > 2 else ""
        print("{:<34}{:>7}{:>10.4f}{:>10.2f}{:>9.2f}{:>8.3f}{}".format(
            r["label"], r["days"], r["ic"], r["t_naive"], r["t_nw"], r["hit"], mark))
    print("-" * 82)

    n_naive = sum(1 for r in rows if abs(r["t_naive"]) > 2)
    n_nw = sum(1 for r in rows if abs(r["t_nw"]) > 2)
    print("\n{}/{} signals clear |t| > 2 naively; {}/{} survive Newey-West.".format(
        n_naive, len(rows), n_nw, len(rows)))
    print("\nFor scale: a deployable single equity factor is usually IC ~0.03+")
    print("with a Newey-West t-stat above 3, sustained across regimes.")


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
