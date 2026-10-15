import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance
import os
import pickle
from models.data_builder import DataBuilder
from models.feature_engineer import FEATURE_NAMES
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter

# The hybrid model is trained on the standard feature set PLUS one extra column:
# the rule engine's net technical score. So its vote reflects how the learned
# patterns line up with the rule system — "a mix of model + rule".
HYBRID_FEATURE_NAMES = list(FEATURE_NAMES) + ["rule_signal"]

# Binary target: does the stock beat the Nifty 50 over the next 20 trading days?
# Index 0 = UNDERPERFORM, 1 = OUTPERFORM (matches DataBuilder's label encoding).
LABELS = ["UNDERPERFORM", "OUTPERFORM"]

# Out-of-sample probability calibration, produced by `python -m models.calibrate`.
# Raw model probabilities are over-confident out of sample (train AUC ~0.68 vs
# walk-forward test AUC ~0.51), so predict() maps them through a Platt curve fit
# on purged walk-forward predictions. If the file is absent predict() falls back
# to the raw numbers and flags `calibration.applied = False`.
CALIBRATION_PATH = os.path.join(os.path.dirname(__file__), "calibration.json")
# Calibrated P(outperform) within +/- this of 0.5 is reported as "no clear edge".
NEUTRAL_BAND = 0.02
_calibration_cache = {"mtime": None, "data": None}


def load_calibration():
    """Cached read of calibration.json (reloaded if the file changes)."""
    try:
        mtime = os.path.getmtime(CALIBRATION_PATH)
    except OSError:
        return None
    if _calibration_cache["mtime"] != mtime:
        try:
            import json
            with open(CALIBRATION_PATH, "r", encoding="utf-8") as fh:
                _calibration_cache["data"] = json.load(fh)
            _calibration_cache["mtime"] = mtime
        except Exception as e:
            print(f"[warn] could not read calibration.json: {e}")
            return None
    return _calibration_cache["data"]


def apply_calibration(p_out, key, cal):
    """Platt-map a raw P(outperform) for model `key` (rf/xgb/hgb/ens)."""
    if not cal or key not in cal.get("models", {}):
        return float(p_out)
    m = cal["models"][key]
    p = float(np.clip(p_out, 1e-4, 1 - 1e-4))
    z = m["a"] * np.log(p / (1 - p)) + m["b"]
    return float(1.0 / (1.0 + np.exp(-z)))

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("[warn] xgboost not installed — XGB features disabled. Run: pip install xgboost")


# ── Training universe: Indian equities only (NSE, .NS) ──────────────────────
# Broadened to a Nifty-200-style set spanning large + mid caps across sectors.
# No US tickers — the model is trained purely on Indian market data.
TRAINING_TICKERS = [
    # ── IT ───────────────────────────────────────────────────────────────────
    "TCS.NS", "INFY.NS", "HCLTECH.NS", "WIPRO.NS", "TECHM.NS", "LTIM.NS",
    "PERSISTENT.NS", "COFORGE.NS", "MPHASIS.NS", "LTTS.NS",

    # ── Banks ────────────────────────────────────────────────────────────────
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "INDUSINDBK.NS", "BANKBARODA.NS", "PNB.NS", "IDFCFIRSTB.NS", "FEDERALBNK.NS",
    "AUBANK.NS",

    # ── NBFC / Insurance / Financials ────────────────────────────────────────
    "BAJFINANCE.NS", "BAJAJFINSV.NS", "SBICARD.NS", "CHOLAFIN.NS", "SHRIRAMFIN.NS",
    "MUTHOOTFIN.NS", "LICHSGFIN.NS", "PFC.NS", "RECLTD.NS", "HDFCLIFE.NS",
    "SBILIFE.NS", "ICICIPRULI.NS", "ICICIGI.NS", "HDFCAMC.NS",

    # ── FMCG ─────────────────────────────────────────────────────────────────
    "HINDUNILVR.NS", "NESTLEIND.NS", "BRITANNIA.NS", "DABUR.NS", "MARICO.NS",
    "GODREJCP.NS", "ITC.NS", "TATACONSUM.NS", "COLPAL.NS", "VBL.NS", "UBL.NS",
    "PGHH.NS",

    # ── Auto & Ancillaries ───────────────────────────────────────────────────
    "MARUTI.NS", "TATAMOTORS.NS", "BAJAJ-AUTO.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
    "M&M.NS", "TVSMOTOR.NS", "ASHOKLEY.NS", "BHARATFORG.NS", "MOTHERSON.NS",
    "BOSCHLTD.NS", "MRF.NS", "APOLLOTYRE.NS", "BALKRISIND.NS",

    # ── Pharma & Healthcare ──────────────────────────────────────────────────
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "AUROPHARMA.NS",
    "TORNTPHARM.NS", "LUPIN.NS", "ZYDUSLIFE.NS", "ALKEM.NS", "BIOCON.NS",
    "APOLLOHOSP.NS", "MAXHEALTH.NS", "LAURUSLABS.NS",

    # ── Realty ───────────────────────────────────────────────────────────────
    "DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PRESTIGE.NS", "PHOENIXLTD.NS",
    "BRIGADE.NS",

    # ── Energy / Power / Oil & Gas ───────────────────────────────────────────
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    "BPCL.NS", "IOC.NS", "GAIL.NS", "OIL.NS", "PETRONET.NS", "ADANIGREEN.NS",
    "ADANIENSOL.NS", "TATAPOWER.NS", "JSWENERGY.NS", "NHPC.NS", "IGL.NS",

    # ── Metals & Mining ──────────────────────────────────────────────────────
    "TATASTEEL.NS", "HINDALCO.NS", "JSWSTEEL.NS", "VEDL.NS", "NMDC.NS",
    "JINDALSTEL.NS", "SAIL.NS", "NATIONALUM.NS", "HINDZINC.NS", "APLAPOLLO.NS",

    # ── Cement ───────────────────────────────────────────────────────────────
    "ULTRACEMCO.NS", "SHREECEM.NS", "AMBUJACEM.NS", "ACC.NS", "DALBHARAT.NS",

    # ── Infra / Capital Goods / Defense ──────────────────────────────────────
    "LT.NS", "ADANIPORTS.NS", "ADANIENT.NS", "SIEMENS.NS", "ABB.NS",
    "CUMMINSIND.NS", "HAVELLS.NS", "POLYCAB.NS", "DIXON.NS", "BEL.NS",
    "HAL.NS", "BDL.NS", "MAZDOCK.NS", "GRASIM.NS",

    # ── Chemicals / Agri-inputs ──────────────────────────────────────────────
    "SRF.NS", "DEEPAKNTR.NS", "PIIND.NS", "AARTIIND.NS", "NAVINFLUOR.NS",
    "TATACHEM.NS", "UPL.NS", "COROMANDEL.NS",

    # ── Consumer / Retail / Paints ───────────────────────────────────────────
    "ASIANPAINT.NS", "BERGEPAINT.NS", "TITAN.NS", "PIDILITIND.NS", "TRENT.NS",
    "DMART.NS", "PAGEIND.NS", "JUBLFOOD.NS", "NYKAA.NS",

    # ── Telecom / Media / New-age / Travel ───────────────────────────────────
    "BHARTIARTL.NS", "IDEA.NS", "INDUSTOWER.NS", "ZOMATO.NS", "NAUKRI.NS",
    "IRCTC.NS", "INDIGO.NS", "INDHOTEL.NS",

    # ── Textiles ─────────────────────────────────────────────────────────────
    "KPRMILL.NS", "TRIDENT.NS", "WELSPUNLIV.NS",
]
# ── Signal threshold constants ─────────────────────────────────────────────
_STRONG_SELL_RSI    = 72
_STRONG_BUY_RSI     = 32
_STRONG_SELL_SHARPE = -0.3
_STRONG_BUY_SHARPE  = 0.6
_BEARISH_TREND      = -1
_BULLISH_TREND      = 1
_NEGATIVE_RETURNS   = -0.005
_POSITIVE_RETURNS   = 0.003
# bollinger_position is now a z-score centred at 0 (range ~[-1.5, 1.5]):
# strongly above the mean band ≈ overbought, strongly below ≈ oversold.
_LOW_BOLLINGER      = -0.75
_HIGH_BOLLINGER     = 0.75


class StockRiskModel:
    def __init__(self):
        # Single source of truth for feature order (matches training & serving)
        self.feature_names = list(FEATURE_NAMES)
        self.scaler      = RobustScaler()
        self.trained     = False
        self.xgb_model   = None
        self.xgb_trained = False
        # Third model: rule-aware HistGradientBoosting hybrid
        self.hgb_model      = None
        self.hgb_trained    = False
        self.hgb_importances = []   # permutation importance, computed at train
        self.metrics     = {}

    # ── Rule-based cross-check ─────────────────────────────────────────────
    def _rule_based_signal(self, feature_dict: dict):
        """
        Scores multiple technical indicators. Returns 'UNDERPERFORM'/'OUTPERFORM'
        when >= 2 indicators align, otherwise None (let ML decide). The signal is
        relative — strong technicals suggest the stock beats the index, not that
        it rises in absolute terms.
        """
        score = 0
        rsi          = feature_dict.get("rsi", 50)
        returns      = feature_dict.get("returns", 0)
        sharpe       = feature_dict.get("sharpe_ratio", 0)
        market_trend = feature_dict.get("market_trend", 0)
        bollinger    = feature_dict.get("bollinger_position", 0.0)
        sentiment    = feature_dict.get("sentiment_score", 50)
        volatility   = feature_dict.get("volatility", 0.3)

        if rsi > _STRONG_SELL_RSI:               score -= 1
        if returns < _NEGATIVE_RETURNS:          score -= 1
        if sharpe < _STRONG_SELL_SHARPE:         score -= 1
        if market_trend == _BEARISH_TREND:       score -= 1
        if bollinger > _HIGH_BOLLINGER:          score -= 1
        if sentiment < 35:                       score -= 1
        if volatility > 0.6 and returns < 0:    score -= 1

        if rsi < _STRONG_BUY_RSI:               score += 1
        if returns > _POSITIVE_RETURNS:         score += 1
        if sharpe > _STRONG_BUY_SHARPE:         score += 1
        if market_trend == _BULLISH_TREND:      score += 1
        if bollinger < _LOW_BOLLINGER:          score += 1
        if sentiment > 65:                      score += 1

        if score <= -2: return "UNDERPERFORM"
        if score >= 2:  return "OUTPERFORM"
        return None

    # ── Numeric rule score (price-derived only) for the hybrid model ───────
    # Same technical logic as _rule_based_signal but WITHOUT sentiment (which
    # can't be reconstructed historically), returned as a signed net score so
    # it can be fed to the hybrid model as an extra feature. Computed from real
    # (unscaled) feature values at both train and inference → consistent.
    @staticmethod
    def _rule_score_column(X: np.ndarray) -> np.ndarray:
        """Vectorised net rule score over a raw feature matrix (N × len(FEATURE_NAMES))."""
        idx = {n: i for i, n in enumerate(FEATURE_NAMES)}
        rsi    = X[:, idx["rsi"]]
        returns = X[:, idx["returns"]]
        vol    = X[:, idx["volatility"]]
        trend  = X[:, idx["market_trend"]]
        boll   = X[:, idx["bollinger_position"]]
        sharpe = X[:, idx["sharpe_ratio"]]

        score = np.zeros(X.shape[0], dtype=np.float32)
        score -= (rsi > _STRONG_SELL_RSI)
        score -= (returns < _NEGATIVE_RETURNS)
        score -= (sharpe < _STRONG_SELL_SHARPE)
        score -= (trend == _BEARISH_TREND)
        score -= (boll > _HIGH_BOLLINGER)
        score -= ((vol > 0.6) & (returns < 0))
        score += (rsi < _STRONG_BUY_RSI)
        score += (returns > _POSITIVE_RETURNS)
        score += (sharpe > _STRONG_BUY_SHARPE)
        score += (trend == _BULLISH_TREND)
        score += (boll < _LOW_BOLLINGER)
        return score

    @staticmethod
    def _rule_score_row(feature_dict: dict) -> float:
        """Scalar net rule score for one inference bar (price-derived only)."""
        rsi    = feature_dict.get("rsi", 50)
        returns = feature_dict.get("returns", 0)
        vol    = feature_dict.get("volatility", 0.3)
        trend  = feature_dict.get("market_trend", 0)
        boll   = feature_dict.get("bollinger_position", 0.0)
        sharpe = feature_dict.get("sharpe_ratio", 0)
        s = 0.0
        if rsi > _STRONG_SELL_RSI:        s -= 1
        if returns < _NEGATIVE_RETURNS:   s -= 1
        if sharpe < _STRONG_SELL_SHARPE:  s -= 1
        if trend == _BEARISH_TREND:       s -= 1
        if boll > _HIGH_BOLLINGER:        s -= 1
        if vol > 0.6 and returns < 0:     s -= 1
        if rsi < _STRONG_BUY_RSI:         s += 1
        if returns > _POSITIVE_RETURNS:   s += 1
        if sharpe > _STRONG_BUY_SHARPE:   s += 1
        if trend == _BULLISH_TREND:       s += 1
        if boll < _LOW_BOLLINGER:         s += 1
        return float(s)

    # ── Feature importance from CalibratedClassifierCV ─────────────────────
    def _extract_rf_importances(self):
        m = self.model
        if hasattr(m, 'feature_importances_'):
            return m.feature_importances_
        if hasattr(m, 'calibrated_classifiers_'):
            importances = []
            for cc in m.calibrated_classifiers_:
                base = getattr(cc, 'estimator', None) or getattr(cc, 'base_estimator', None)
                if base is not None and hasattr(base, 'feature_importances_'):
                    importances.append(base.feature_importances_)
            if importances:
                return np.mean(importances, axis=0)
        return None

    # ── Train ──────────────────────────────────────────────────────────────
    def train(self):
        print("[model] Training on Indian NSE market data (real data only)...")

        X_all, y_all = [], []
        ok, skipped = 0, 0
        for ticker in TRAINING_TICKERS:
            try:
                builder = DataBuilder(ticker)
                X, y = builder.build()
                if X is not None and len(X) > 0:
                    X_all.append(X)
                    y_all.append(y)
                    ok += 1
                    print(f"  [ok]   {ticker}: {len(X)} samples")
                else:
                    skipped += 1
                    print(f"  [skip] {ticker}: no data")
            except Exception as e:
                skipped += 1
                print(f"  [err]  {ticker}: {e}")

        # Real Indian data only — no synthetic fallback. Fail loudly if the
        # fetch produced nothing rather than fabricating training rows.
        if not X_all:
            raise RuntimeError(
                "No real NSE training data could be fetched — aborting "
                "(synthetic data has been removed by design)."
            )

        X = np.vstack(X_all)
        y = np.concatenate(y_all)
        dist = Counter(int(v) for v in y)
        print(f"[model] Real NSE data: {len(X)} samples from {ok} tickers "
              f"({skipped} skipped)")
        n_under, n_out = dist.get(0, 0), dist.get(1, 0)
        total = max(1, n_under + n_out)
        print(f"[model] Class distribution (vs Nifty, {DataBuilder.HORIZON}d) — "
              f"UNDERPERFORM:{n_under} OUTPERFORM:{n_out} "
              f"(baseline {max(n_under, n_out) / total:.3f})")

        X_scaled = self.scaler.fit_transform(X)

        # ── Calibrated RF ─────────────────────────────────────────────────
        # The relative-performance target is near 50/50 by construction, but
        # class_weight='balanced' is kept so any residual skew is handled
        # without fabricating rows.
        base_rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features='sqrt',
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        self.model = CalibratedClassifierCV(base_rf, method='isotonic', cv=5)
        self.model.fit(X_scaled, y)
        print("[model] Calibrated RF trained")

        # CV
        try:
            cv_rf = RandomForestClassifier(
                n_estimators=300, max_depth=12,
                min_samples_split=6, min_samples_leaf=3,
                class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(cv_rf, X_scaled, y, cv=skf, scoring='accuracy')
            print(f"  RF CV accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        except Exception as e:
            print(f"  [warn] RF CV skipped: {e}")
            skf = None

        # ── XGBoost ───────────────────────────────────────────────────────
        # XGBoost has no class_weight; use balanced per-sample weights instead.
        xgb_sample_weight = compute_sample_weight('balanced', y)
        if XGB_AVAILABLE:
            try:
                self.xgb_model = XGBClassifier(
                    n_estimators=400, max_depth=5, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.1, reg_lambda=1.5,
                    min_child_weight=3, gamma=0.1,
                    eval_metric='logloss',
                    use_label_encoder=False,
                    random_state=42, verbosity=0, n_jobs=-1
                )
                self.xgb_model.fit(X_scaled, y, sample_weight=xgb_sample_weight)
                self.xgb_trained = True
                print("[model] XGBoost trained")

                if skf is not None:
                    try:
                        xgb_cv = XGBClassifier(
                            n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.7,
                            eval_metric='logloss', random_state=42, verbosity=0
                        )
                        xgb_scores = cross_val_score(xgb_cv, X_scaled, y, cv=skf, scoring='accuracy')
                        print(f"  XGB CV accuracy: {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")
                    except Exception as e:
                        print(f"  [warn] XGB CV skipped: {e}")
            except Exception as e:
                print(f"[warn] XGBoost training failed: {e}")
                self.xgb_trained = False
        else:
            print("[warn] XGBoost not available — skipping")

        # ── Hybrid rule-aware HistGradientBoosting ─────────────────────────
        # Trained on the scaled feature set PLUS the rule engine's net score,
        # so it blends learned patterns with the rule system. HGB natively
        # handles class weighting via sample_weight and is fast on Windows.
        try:
            rule_col = self._rule_score_column(X).reshape(-1, 1)
            X_hybrid = np.hstack([X_scaled, rule_col])
            hgb_sample_weight = compute_sample_weight('balanced', y)
            self.hgb_model = HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.05, max_depth=6,
                l2_regularization=1.0, max_leaf_nodes=31,
                early_stopping=True, validation_fraction=0.1,
                random_state=42,
            )
            self.hgb_model.fit(X_hybrid, y, sample_weight=hgb_sample_weight)
            self.hgb_trained = True
            print("[model] Hybrid HistGradientBoosting trained")

            if skf is not None:
                try:
                    hgb_cv = HistGradientBoostingClassifier(
                        max_iter=400, learning_rate=0.05, max_depth=6,
                        l2_regularization=1.0, random_state=42,
                    )
                    hgb_scores = cross_val_score(hgb_cv, X_hybrid, y, cv=skf, scoring='accuracy')
                    print(f"  Hybrid CV accuracy: {hgb_scores.mean():.3f} +/- {hgb_scores.std():.3f}")
                except Exception as e:
                    print(f"  [warn] Hybrid CV skipped: {e}")

            # HGB has no native feature_importances_ — compute permutation
            # importance once on a subsample and persist it (global, like the
            # other models' importances shown in the UI).
            try:
                n = min(4000, X_hybrid.shape[0])
                sel = np.random.RandomState(42).choice(X_hybrid.shape[0], n, replace=False)
                perm = permutation_importance(
                    self.hgb_model, X_hybrid[sel], y[sel],
                    n_repeats=5, random_state=42, scoring='accuracy', n_jobs=-1
                )
                imp = np.clip(perm.importances_mean, 0, None)
                total = imp.sum()
                imp = (imp / total) if total > 0 else imp
                self.hgb_importances = [
                    {"feature": n_, "importance": float(s)}
                    for n_, s in zip(HYBRID_FEATURE_NAMES, imp)
                ]
                print("[model] Hybrid permutation importance computed")
            except Exception as e:
                print(f"  [warn] Hybrid importance skipped: {e}")
                self.hgb_importances = []
        except Exception as e:
            print(f"[warn] Hybrid HGB training failed: {e}")
            self.hgb_trained = False

        y_pred = self.model.predict(X_scaled)
        self.metrics = {
            "accuracy":         round(accuracy_score(y, y_pred), 3),
            "precision":        round(precision_score(y, y_pred, average='weighted', zero_division=0), 3),
            "recall":           round(recall_score(y, y_pred, average='weighted', zero_division=0), 3),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }
        # ROC-AUC is the honest headline for a binary target: it measures how
        # well the model RANKS outperformers, independent of the 0.5 cutoff.
        try:
            self.metrics["roc_auc"] = round(
                roc_auc_score(y, self.model.predict_proba(X_scaled)[:, 1]), 3
            )
        except Exception as e:
            print(f"  [warn] ROC-AUC skipped: {e}")

        self.trained = True
        print(f"[ok] Model trained — train accuracy: {self.metrics['accuracy']} "
              f"| ROC-AUC: {self.metrics.get('roc_auc', 'n/a')}")

    # ── Predict ────────────────────────────────────────────────────────────
    def predict(self, features, feature_dict: dict = None):
        if not self.trained:
            loaded = self.load()
            if not loaded:
                self.train()
                self.save()

        features_scaled = self.scaler.transform(features)
        labels = LABELS

        # Map any model's proba (in its own class order) to an
        # [UNDERPERFORM, OUTPERFORM] vector.
        def _to_vec(proba, classes):
            v = np.zeros(2, dtype=float)
            for i, cls in enumerate(classes):
                v[int(cls)] = float(proba[i])
            return v

        cal = load_calibration()

        def _member(name, key, vec):
            """Raw vote + calibrated confidence for one model's [P(under), P(out)] vector."""
            raw_p = float(vec[1])
            cal_p = apply_calibration(raw_p, key, cal)
            # Vote relative to the out-of-sample base rate: Platt shifts the
            # midpoint to it, so 0.5 would call everything an outperformer.
            centre = float((cal or {}).get('base_rate') or 0.5)
            vote = labels[int(cal_p >= centre)]
            return {
                "name": name, "key": key, "signal": vote,
                # Calibrated confidence in the model's own direction. Out of
                # sample this hugs 50% - that is the honest number.
                "confidence": float(max(cal_p, 1 - cal_p)),
                "outperform_probability": cal_p,
                "raw_confidence": float(max(raw_p, 1 - raw_p)),
            }

        rf_vec = _to_vec(self.model.predict_proba(features_scaled)[0], self.model.classes_)
        members = [_member("Calibrated RF", "rf", rf_vec)]
        vecs = [rf_vec]

        # -- XGBoost
        xgb_m = None
        if self.xgb_trained and self.xgb_model is not None:
            try:
                xgb_vec = _to_vec(self.xgb_model.predict_proba(features_scaled)[0],
                                  self.xgb_model.classes_)
                xgb_m = _member("XGBoost", "xgb", xgb_vec)
                vecs.append(xgb_vec)
            except Exception as e:
                print(f"[warn] XGBoost inference failed: {e}")
        members.append(xgb_m or {"name": "XGBoost", "key": "xgb", "signal": None, "confidence": None})

        # -- Hybrid rule-aware HGB
        hgb_m = None
        if self.hgb_trained and self.hgb_model is not None:
            try:
                rule_val = self._rule_score_row(feature_dict) if feature_dict else 0.0
                hybrid_input = np.hstack([features_scaled, [[rule_val]]])
                hgb_vec = _to_vec(self.hgb_model.predict_proba(hybrid_input)[0],
                                  self.hgb_model.classes_)
                hgb_m = _member("Hybrid (rule-aware)", "hgb", hgb_vec)
                vecs.append(hgb_vec)
            except Exception as e:
                print(f"[warn] Hybrid HGB inference failed: {e}")
        members.append(hgb_m or {"name": "Hybrid (rule-aware)", "key": "hgb", "signal": None, "confidence": None})

        # Rule-based cross-check: shown alongside, never mixed into the
        # probability (the old +0.25 nudge manufactured confidence).
        rule_signal = self._rule_based_signal(feature_dict) if feature_dict else None

        # -- Soft-voting ensemble, then calibrated
        ens_raw = float(np.mean([v[1] for v in vecs]))
        ens_p = apply_calibration(ens_raw, "ens", cal)
        # Centre the neutral band on the out-of-sample base rate (the share of
        # stocks that beat the Nifty), not on 0.5: a stock at the base rate has no edge.
        centre = float((cal or {}).get('base_rate') or 0.5)
        edge = ens_p - centre
        if edge >= NEUTRAL_BAND:
            ens_pred, edge_label = "OUTPERFORM", "Slight lean to outperform"
        elif edge <= -NEUTRAL_BAND:
            ens_pred, edge_label = "UNDERPERFORM", "Slight lean to underperform"
        else:
            ens_pred, edge_label = "NO_CLEAR_EDGE", "No clear edge vs the Nifty"

        skill = (cal or {}).get("skill")
        ens_cal = (cal or {}).get("models", {}).get("ens", {})
        calibration = {
            "applied": bool(cal),
            "skill_tier": skill.get("tier") if skill else None,
            "skill_label": skill.get("label") if skill else "Uncalibrated",
            "oos_auc": skill.get("auc") if skill else None,
            "oos_auc_ci95": skill.get("auc_ci95") if skill else None,
            "n_oos": (cal or {}).get("n_oos"),
            "oos_period": (cal or {}).get("oos_period"),
            "base_rate": (cal or {}).get("base_rate"),
            "calibrated_range": ens_cal.get("cal_p_range"),
            "raw_range": ens_cal.get("raw_p_range"),
            "brier_calibrated": ens_cal.get("brier_calibrated"),
            "brier_constant": ens_cal.get("brier_constant"),
            "ece_raw": ens_cal.get("ece_raw"),
            "ece_calibrated": ens_cal.get("ece_calibrated"),
            "generated": (cal or {}).get("generated"),
        }

        ensemble = {
            "prediction": ens_pred,
            "edge_label": edge_label,
            "confidence": float(max(ens_p, 1 - ens_p)),
            # P(beats the Nifty), CALIBRATED against walk-forward out-of-sample outcomes.
            "outperform_probability": ens_p,
            "raw_outperform_probability": ens_raw,
            "neutral_band": NEUTRAL_BAND,
            "centre": centre,
            "probabilities": {labels[0]: 1 - ens_p, labels[1]: ens_p},
        }

        result = {
            # Headline = calibrated ensemble (kept under the old keys for consumers).
            "prediction":    ens_pred,
            "confidence":    ensemble["confidence"],
            "probabilities": ensemble["probabilities"],
            "rule_override": False,
            "rule_signal":   rule_signal,
            "xgb_prediction":  xgb_m["signal"] if xgb_m else None,
            "xgb_confidence":  xgb_m["confidence"] if xgb_m else None,
            "hgb_prediction":  hgb_m["signal"] if hgb_m else None,
            "hgb_confidence":  hgb_m["confidence"] if hgb_m else None,
            "ensemble": ensemble,
            "calibration": calibration,
        }

        # -- 3-model consensus (raw votes, calibrated confidences)
        votes = [m["signal"] for m in members if m["signal"] is not None]
        if votes:
            tally = Counter(votes)
            majority, top_n = tally.most_common(1)[0]
            all_agree = len(tally) == 1
            result["consensus"] = {
                "models":      members,
                "agree":       all_agree,
                "agreement":   f"{top_n}/{len(votes)}",
                "majority":    majority,
                "n_models":    len(votes),
                "rule_signal": rule_signal,
                "ensemble_prediction": ens_pred,
                "ensemble_confidence": ensemble["confidence"],
            }
        else:
            result["consensus"] = None

        # Kept for any older consumers: True if the models are not unanimous.
        result["model_disagreement"] = bool(votes) and len(set(votes)) > 1

        return result

    # ── Feature importance ────────────────────────────────────────────────
    def get_feature_importance(self, feature_names):
        if not self.trained:
            return []
        importances = self._extract_rf_importances()
        if importances is None:
            return []
        return [{"feature": n, "importance": float(s)}
                for n, s in zip(feature_names, importances)]

    def get_xgb_feature_importance(self, feature_names):
        if not self.xgb_trained or self.xgb_model is None:
            return []
        try:
            return [{"feature": n, "importance": float(s)}
                    for n, s in zip(feature_names, self.xgb_model.feature_importances_)]
        except Exception as e:
            print(f"[warn] XGBoost feature importance failed: {e}")
            return []

    def get_hgb_feature_importance(self):
        """Permutation importance computed at train time (includes 'rule_signal')."""
        return list(self.hgb_importances or [])

    # ── Persistence ───────────────────────────────────────────────────────
    # Dict format (v3 — binary relative-performance target) so the bundle can grow without positional
    # tuple breakage. Also carries the feature list so a stale pkl trained on a
    # different feature set is detected and retrained rather than served wrong.
    def save(self, path="models/stock_model.pkl"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            bundle = {
                "version":         3,
                "feature_names":   list(FEATURE_NAMES),
                "model":           self.model,
                "scaler":          self.scaler,
                "xgb_model":       self.xgb_model,
                "xgb_trained":     self.xgb_trained,
                "hgb_model":       self.hgb_model,
                "hgb_trained":     self.hgb_trained,
                "hgb_importances": self.hgb_importances,
            }
            with open(path, "wb") as f:
                pickle.dump(bundle, f)
            print(f"[ok] Model saved to {path}")
        except Exception as e:
            print(f"[warn] Could not save model: {e}")

    def load(self, path="models/stock_model.pkl"):
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                loaded = pickle.load(f)

            if not isinstance(loaded, dict) or loaded.get("version") != 3:
                print("[warn] Old model format — retraining with Indian data...")
                return False

            if loaded.get("feature_names") != list(FEATURE_NAMES):
                print("[warn] Feature set changed since this pkl was trained — retraining...")
                return False

            self.model           = loaded["model"]
            self.scaler          = loaded["scaler"]
            self.xgb_model       = loaded.get("xgb_model")
            self.xgb_trained     = loaded.get("xgb_trained", False)
            self.hgb_model       = loaded.get("hgb_model")
            self.hgb_trained     = loaded.get("hgb_trained", False)
            self.hgb_importances = loaded.get("hgb_importances", [])
            self.trained = True
            print("[ok] Loaded trained model from disk (RF + XGBoost + Hybrid HGB)")
            return True
        except Exception as e:
            print(f"[warn] Could not load model: {e}")
        return False