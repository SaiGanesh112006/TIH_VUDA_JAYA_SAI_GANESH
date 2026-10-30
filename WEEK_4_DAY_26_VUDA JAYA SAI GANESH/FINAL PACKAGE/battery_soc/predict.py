"""Inference entry point.  python -m battery_soc.predict --input in.csv --output out.csv"""
import argparse, os
import joblib, numpy as np, pandas as pd
from .config import RAW_FEATURES, CHARGE_FEATURES, SOC_CLIP, SOC_MODEL_FILE, CHARGE_MODEL_FILE
from .data import check_columns


def load_models(models_dir="models"):
    soc = joblib.load(os.path.join(models_dir, SOC_MODEL_FILE))
    cpath = os.path.join(models_dir, CHARGE_MODEL_FILE)
    charge = joblib.load(cpath) if os.path.exists(cpath) else None
    return soc, charge


def predict_soc(pipe, df):
    """SOC in % (0-100) from raw sensor columns."""
    check_columns(df, RAW_FEATURES)
    return np.clip(pipe.predict(df[RAW_FEATURES]), *SOC_CLIP)


def predict_charge_time(pipe, df):
    """Remaining charging time in seconds (never negative). Lower confidence than SOC."""
    check_columns(df, CHARGE_FEATURES)
    return np.maximum(pipe.predict(df[CHARGE_FEATURES]), 0.0)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Predict SOC (and charging time) from a CSV of sensor readings.")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default="predictions.csv")
    ap.add_argument("--models", default="models")
    a = ap.parse_args(argv)
    df = pd.read_csv(a.input)
    soc, charge = load_models(a.models)
    out = df.copy()
    out["predicted_soc_percent"] = predict_soc(soc, df).round(2)
    if charge is not None and all(c in df.columns for c in CHARGE_FEATURES):
        out["predicted_remaining_charge_time_s"] = predict_charge_time(charge, df).round(0)
    out.to_csv(a.output, index=False)
    print(f"Wrote {len(out)} predictions -> {a.output}")


if __name__ == "__main__":
    main()
