"""Grouped-CV model selection + freeze.   python -m battery_soc.train --train ... --out models/"""
import argparse, datetime, json, os, platform, time
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold

from . import config as C
from .data import md5, load_split, build_charge_frame
from .models import make_pipe, soc_candidates, charge_estimator


def cross_validate(df_train, n_splits=C.CV_SPLITS):
    X, y, groups = df_train[C.RAW_FEATURES], df_train[C.TARGET], df_train["file_id"]
    gkf = GroupKFold(n_splits=min(n_splits, groups.nunique()))
    rows = []
    for name, (factory, rank) in soc_candidates().items():
        fm, fr, f2, ft = [], [], [], []
        for tr, va in gkf.split(X, y, groups):
            t0 = time.perf_counter()
            p = make_pipe(factory(), C.RAW_FEATURES).fit(X.iloc[tr], y.iloc[tr])
            ft.append(time.perf_counter() - t0)
            pred = np.clip(p.predict(X.iloc[va]), *C.SOC_CLIP)
            fm.append(mean_absolute_error(y.iloc[va], pred))
            fr.append(np.sqrt(mean_squared_error(y.iloc[va], pred)))
            f2.append(r2_score(y.iloc[va], pred))
        rows.append({"model": name, "complexity_rank": rank, "cv_MAE": np.mean(fm), "cv_MAE_std": np.std(fm),
                     "cv_RMSE": np.mean(fr), "cv_R2": np.mean(f2), "worst_fold_MAE": np.max(fm), "fit_s": np.mean(ft)})
    return pd.DataFrame(rows).set_index("model"), gkf.n_splits


def select_model(cv):
    """Rule fixed before the test set: within 10 % of best CV MAE -> simplest model."""
    cand = cv.drop("Mean baseline")
    eligible = cand[cand["cv_MAE"] <= (1 + C.CV_TOLERANCE) * cand["cv_MAE"].min()]
    chosen = eligible.sort_values(["complexity_rank", "cv_RMSE", "worst_fold_MAE"]).index[0]
    row = cv.loc[chosen]
    robust_ok = bool(row["worst_fold_MAE"] <= C.ROBUSTNESS_FACTOR * row["cv_MAE"])
    return chosen, list(eligible.index), robust_ok


def train(train_path, out_dir, full_path=None, test_path=None, n_splits=C.CV_SPLITS):
    os.makedirs(out_dir, exist_ok=True)
    df_train, df_test = load_split(train_path, test_path)
    train_ids = sorted(df_train["file_id"].unique().tolist())

    cv, k = cross_validate(df_train, n_splits)
    chosen, eligible, robust_ok = select_model(cv)
    cv.to_csv(os.path.join(out_dir, "cv_results_soc.csv"))

    final_pipe = make_pipe(soc_candidates()[chosen][0](), C.RAW_FEATURES).fit(df_train[C.RAW_FEATURES], df_train[C.TARGET])
    joblib.dump(final_pipe, os.path.join(out_dir, C.SOC_MODEL_FILE))

    charge_frozen = False
    if full_path and os.path.exists(full_path):
        c_tr = build_charge_frame(pd.read_csv(full_path), train_ids)
        if c_tr["file_id"].nunique() >= 3:
            cp = make_pipe(charge_estimator(), C.CHARGE_FEATURES).fit(c_tr[C.CHARGE_FEATURES], c_tr[C.CHARGE_TARGET])
            joblib.dump(cp, os.path.join(out_dir, C.CHARGE_MODEL_FILE))
            charge_frozen = True

    config = {
        "frozen_on": str(datetime.datetime.now()),
        "dataset": {"train_md5": md5(train_path), "train_shape": list(df_train.shape), "train_file_ids": train_ids,
                    "test_md5": md5(test_path) if test_path else None,
                    "split": "GroupShuffleSplit by file_id, test_size=0.2, random_state=42 (Day 12)"},
        "target": C.TARGET, "features": C.RAW_FEATURES, "excluded_as_leakage": C.LEAKAGE_COLUMNS,
        "preprocessing": "column selection by name only; no scaling/imputation; predictions clipped to [0,100]",
        "selection": {"cv": f"GroupKFold({k}) on train files", "eligible": eligible, "chosen": chosen,
                      "estimator": repr(final_pipe.named_steps["model"]), "robustness_pass": robust_ok},
        "charging_time_model": {"frozen": charge_frozen, "confidence": "low - sparse CC-CV data"},
        "seed": C.SEED,
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "numpy": np.__version__, "pandas": pd.__version__},
    }
    with open(os.path.join(out_dir, C.CONFIG_FILE), "w") as f:
        json.dump(config, f, indent=2)
    return cv, chosen, robust_ok, charge_frozen


def main(argv=None):
    ap = argparse.ArgumentParser(description="Select and freeze the SOC model (train files only).")
    ap.add_argument("--train", required=True)
    ap.add_argument("--full", default=None, help="full CSV (needed for the charging-time model)")
    ap.add_argument("--test", default=None, help="only used to record its md5 hash - never for fitting")
    ap.add_argument("--out", default="models")
    ap.add_argument("--splits", type=int, default=C.CV_SPLITS)
    a = ap.parse_args(argv)
    cv, chosen, robust_ok, ch = train(a.train, a.out, a.full, a.test, a.splits)
    print(cv.round(4).to_string())
    print(f"\nFINAL SOC MODEL: {chosen} | robustness: {'PASS' if robust_ok else 'FLAG'} | charge model frozen: {ch}")


if __name__ == "__main__":
    main()
