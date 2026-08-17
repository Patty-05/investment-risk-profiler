"""
Import this FIRST — before numpy / scikit-learn / xgboost.

WHY
---
Two native libraries here ship their own OpenMP runtime, and on Windows both
break if Intel MKL's OpenMP (pulled in by numpy/scikit-learn/xgboost) is loaded
first. Neither failure is graceful:

  * torch     -> OSError: [WinError 1114] ... torch\\lib\\c10.dll
                 Consequence is SILENT: finbert_scorer catches it and the whole
                 sentiment pipeline quietly degrades to the shallow keyword
                 scorer, so the app looks healthy on a much weaker signal.
  * lightgbm  -> OSError: exception: access violation reading 0x0 (a hard
                 segfault inside LGBM_DatasetSetField, at .fit() time)

Setting KMP_DUPLICATE_LIB_OK does NOT fix either (verified). Import order is the
only reliable remedy, so this module centralises it: every entry point that may
touch sentiment or LightGBM does `import torch_bootstrap` on its first line.

Exposes:
    TORCH_OK / TORCH_ERROR   — whether torch imported cleanly
    LGBM_OK  / LGBM_ERROR    — whether lightgbm imported cleanly
"""

TORCH_OK = False
TORCH_ERROR = None
LGBM_OK = False
LGBM_ERROR = None

# torch first — it is the most fragile of the two.
try:
    import torch  # noqa: F401  (imported for side-effect: DLL load order)
    TORCH_OK = True
except Exception as e:                      # pragma: no cover - platform specific
    TORCH_ERROR = str(e)
    print(f"[bootstrap] torch unavailable — FinBERT sentiment will fall back "
          f"to the keyword scorer. Reason: {e}")

# lightgbm next, still ahead of xgboost/sklearn.
try:
    import lightgbm  # noqa: F401  (imported for side-effect: DLL load order)
    LGBM_OK = True
except Exception as e:                      # pragma: no cover - platform specific
    LGBM_ERROR = str(e)
    print(f"[bootstrap] lightgbm unavailable: {e}")


def status() -> dict:
    """Small dict so callers/health checks can report the real engine state."""
    return {
        "torch_ok": TORCH_OK, "torch_error": TORCH_ERROR,
        "lightgbm_ok": LGBM_OK, "lightgbm_error": LGBM_ERROR,
    }
