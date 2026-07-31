import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
import os
import pickle
from models.data_builder import DataBuilder
from models.feature_engineer import FEATURE_NAMES
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
from collections import Counter

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ xgboost not installed — XGB features disabled. Run: pip install xgboost")


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
        self.metrics     = {}

    # ── Rule-based override — fixes HOLD bias ──────────────────────────────
    def _rule_based_signal(self, feature_dict: dict):
        """
        Scores multiple technical indicators. Returns 'SELL'/'BUY' when
        >= 2 indicators align, otherwise None (let ML decide).
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

        if score <= -2: return "SELL"
        if score >= 2:  return "BUY"
        return None

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
        print(f"[model] Class distribution — SELL:{dist.get(0,0)} "
              f"HOLD:{dist.get(1,0)} BUY:{dist.get(2,0)}")

        X_scaled = self.scaler.fit_transform(X)

        # ── Calibrated RF ─────────────────────────────────────────────────
        # class_weight='balanced' handles the natural HOLD-heavy skew of real
        # 5-day-forward-return labels without any synthetic rebalancing.
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
                    eval_metric='mlogloss',
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
                            eval_metric='mlogloss', random_state=42, verbosity=0
                        )
                        xgb_scores = cross_val_score(xgb_cv, X_scaled, y, cv=skf, scoring='accuracy')
                        print(f"  XGB CV accuracy: {xgb_scores.mean():.3f} ± {xgb_scores.std():.3f}")
                    except Exception as e:
                        print(f"  ⚠️ XGB CV skipped: {e}")
            except Exception as e:
                print(f"⚠️ XGBoost training failed: {e}")
                self.xgb_trained = False
        else:
            print("⚠️ XGBoost not available — skipping")

        y_pred = self.model.predict(X_scaled)
        self.metrics = {
            "accuracy":         round(accuracy_score(y, y_pred), 3),
            "precision":        round(precision_score(y, y_pred, average='weighted', zero_division=0), 3),
            "recall":           round(recall_score(y, y_pred, average='weighted', zero_division=0), 3),
            "confusion_matrix": confusion_matrix(y, y_pred).tolist()
        }
        self.trained = True
        print(f"✅ Model trained — train accuracy: {self.metrics['accuracy']}")

    # ── Predict ────────────────────────────────────────────────────────────
    def predict(self, features, feature_dict: dict = None):
        if not self.trained:
            loaded = self.load()
            if not loaded:
                self.train()
                self.save()

        features_scaled = self.scaler.transform(features)
        pred  = self.model.predict(features_scaled)[0]
        probs = self.model.predict_proba(features_scaled)[0]
        labels = ["SELL", "HOLD", "BUY"]

        prob_dict = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
        for i, cls in enumerate(self.model.classes_):
            prob_dict[labels[cls]] = float(probs[i])

        # Rule-based override
        rule_signal = self._rule_based_signal(feature_dict) if feature_dict else None
        final_prediction = rule_signal if rule_signal is not None else labels[pred]

        if rule_signal is not None and rule_signal != labels[pred]:
            override_idx = labels.index(rule_signal)
            boosted      = min(1.0, probs[override_idx] + 0.35)
            remaining    = 1.0 - boosted
            other_sum    = sum(probs[i] for i in range(len(probs)) if i != override_idx)
            new_probs    = []
            for i in range(len(probs)):
                if i == override_idx:
                    new_probs.append(boosted)
                else:
                    new_probs.append(remaining * (probs[i] / other_sum) if other_sum > 0 else remaining / 2)
            for i, cls in enumerate(self.model.classes_):
                prob_dict[labels[cls]] = float(new_probs[i])

        result = {
            "prediction":    final_prediction,
            "confidence":    float(np.max(list(prob_dict.values()))) * 0.8 + 0.2,
            "probabilities": prob_dict,
            "rule_override": rule_signal is not None,
        }

        # XGBoost
        if self.xgb_trained and self.xgb_model is not None:
            try:
                xgb_pred  = self.xgb_model.predict(features_scaled)[0]
                xgb_proba = self.xgb_model.predict_proba(features_scaled)[0]

                xgb_prob_dict = {"SELL": 0.0, "HOLD": 0.0, "BUY": 0.0}
                for i, cls in enumerate(self.xgb_model.classes_):
                    xgb_prob_dict[labels[int(cls)]] = float(xgb_proba[i])

                xgb_label = rule_signal if rule_signal is not None else labels[int(xgb_pred)]

                result["xgb_prediction"]     = xgb_label
                result["xgb_confidence"]     = float(np.max(xgb_proba))
                result["xgb_probabilities"]  = xgb_prob_dict
                result["model_disagreement"] = (final_prediction != xgb_label)
            except Exception as e:
                print(f"⚠️ XGBoost inference failed: {e}")
                result["xgb_prediction"]     = None
                result["xgb_confidence"]     = None
                result["xgb_probabilities"]  = None
                result["model_disagreement"] = False
        else:
            result["xgb_prediction"]     = None
            result["xgb_confidence"]     = None
            result["xgb_probabilities"]  = None
            result["model_disagreement"] = False

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
            print(f"⚠️ XGBoost feature importance failed: {e}")
            return []

    # ── Persistence ───────────────────────────────────────────────────────
    def save(self, path="models/stock_model.pkl"):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump((self.model, self.scaler, self.xgb_model, self.xgb_trained), f)
            print(f"✅ Model saved to {path}")
        except Exception as e:
            print(f"⚠️ Could not save model: {e}")

    def load(self, path="models/stock_model.pkl"):
        if os.path.exists(path):
            try:
                with open(path, "rb") as f:
                    loaded = pickle.load(f)
                if isinstance(loaded, tuple) and len(loaded) == 4:
                    self.model, self.scaler, self.xgb_model, self.xgb_trained = loaded
                    print("✅ Loaded trained model from disk (with XGBoost)")
                    self.trained = True
                    return True
                else:
                    print("⚠️ Old model format — retraining with Indian data...")
                    return False
            except Exception as e:
                print(f"⚠️ Could not load model: {e}")
        return False