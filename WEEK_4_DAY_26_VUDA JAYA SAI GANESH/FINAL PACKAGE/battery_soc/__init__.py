"""battery_soc - frozen State-of-Charge (SOC) and charging-time estimation package.

Author: VUDA JAYA SAI GANESH (Week 4, Day 26 - final software package)
"""
from .config import RAW_FEATURES, CHARGE_FEATURES, SOC_CLIP

__version__ = "1.0.0"
__all__ = ["RAW_FEATURES", "CHARGE_FEATURES", "SOC_CLIP", "load_models", "predict_soc", "predict_charge_time"]


def __getattr__(name):          # lazy import keeps `python -m battery_soc.predict` warning-free
    if name in ("load_models", "predict_soc", "predict_charge_time"):
        from . import predict
        return getattr(predict, name)
    raise AttributeError(name)
