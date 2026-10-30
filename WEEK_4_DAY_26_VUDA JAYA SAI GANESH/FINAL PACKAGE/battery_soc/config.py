"""Single source of truth for every frozen setting (Day 12 split, Day 22 freeze)."""

TARGET = "soc_percent"
RAW_FEATURES = ["voltage_v", "current_a", "cell_temp_c", "power_w", "ambient_temp_c"]

CHARGE_TARGET = "remaining_charge_time_s"
CHARGE_FEATURES = ["voltage_v", "current_a", "cell_temp_c", "ambient_temp_c"]

SEED = 42
SOC_CLIP = (0.0, 100.0)
CV_SPLITS = 5
CV_TOLERANCE = 0.10          # keep models within 10 % of the best CV MAE
ROBUSTNESS_FACTOR = 2.0      # worst-fold MAE must be <= 2 x mean CV MAE

# Columns that would leak the answer (cumulative capacity / energy / counters) - never used as inputs.
LEAKAGE_COLUMNS = ["capacity_ah", "wh_accu", "cnt", "step", "cycle", "cycle_level",
                   "procedure", "status", "prog_time", "step_time"]

# SOC bands used to show where the model is weak (the low-SOC "knee" is the known problem area).
SOC_BINS = [0, 15, 30, 50, 70, 85, 100.0001]
SOC_BIN_LABELS = ["0-15", "15-30", "30-50", "50-70", "70-85", "85-100"]

# Day 19/20 grouped-validation hyper-parameters (never re-tuned on the test set).
RF_PARAMS = dict(n_estimators=300, max_depth=8, min_samples_leaf=4)
HGB_PARAMS = dict(max_iter=300, max_depth=6, learning_rate=0.05)
CHARGE_RF_PARAMS = dict(n_estimators=100, max_depth=10)

SOC_MODEL_FILE = "soc_final_pipeline.joblib"
CHARGE_MODEL_FILE = "charging_time_final_pipeline.joblib"
CONFIG_FILE = "frozen_config.json"
