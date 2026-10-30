"""One-shot test evaluation of the FROZEN models + figures.  python -m battery_soc.evaluate --test ..."""
import argparse, json, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from . import config as C
from .data import build_charge_frame
from .predict import load_models, predict_soc, predict_charge_time

NAVY, TEAL, ORANGE, GREY = "#1F3A5F", "#1B998B", "#F46036", "#8A8F98"


def metrics(y, p):
    return {"MAE": float(mean_absolute_error(y, p)),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": float(r2_score(y, p))}


def evaluate(test_path, models_dir, out_dir, train_path=None, full_path=None, cv_path=None):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    soc_pipe, charge_pipe = load_models(models_dir)
    te = pd.read_csv(test_path)
    y, p = te[C.TARGET].to_numpy(), predict_soc(soc_pipe, te)
    te = te.assign(pred=p, err=p - y, abs_err=np.abs(p - y))

    summary = {"n_test_rows": int(len(te)), "n_test_files": int(te["file_id"].nunique()),
               "soc_test": metrics(y, p)}
    if train_path:   # honest reference: always predicting the training mean
        base = np.full(len(te), pd.read_csv(train_path, usecols=[C.TARGET])[C.TARGET].mean())
        summary["mean_baseline_test"] = metrics(y, base)

    te["soc_band"] = pd.cut(te[C.TARGET], C.SOC_BINS, labels=C.SOC_BIN_LABELS, include_lowest=True)
    band = te.groupby("soc_band", observed=True)["abs_err"].agg(["mean", "count"]).rename(columns={"mean": "MAE"})
    per_file = te.groupby("file_id")["abs_err"].mean().rename("MAE").to_frame()
    band.to_csv(os.path.join(out_dir, "test_error_by_soc_band.csv"))
    per_file.to_csv(os.path.join(out_dir, "test_error_by_file.csv"))
    summary["worst_soc_band"] = str(band["MAE"].idxmax())
    summary["mae_by_soc_band"] = {str(k): float(v) for k, v in band["MAE"].items()}
    summary["mae_by_file"] = {str(k): float(v) for k, v in per_file["MAE"].items()}

    # ---- figures -------------------------------------------------------------------------------
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    ax.scatter(y, p, s=4, alpha=0.25, color=TEAL)
    ax.plot([0, 100], [0, 100], color=ORANGE, lw=2, label="perfect prediction")
    ax.set(xlabel="True SOC (%)", ylabel="Predicted SOC (%)", title="Predicted vs true SOC (test files)")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(f"{fig_dir}/fig1_pred_vs_true.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.hist(te["err"], bins=60, color=NAVY, alpha=0.9)
    ax.axvline(0, color=ORANGE, lw=2)
    ax.set(xlabel="Prediction error (pred - true, SOC %)", ylabel="Rows", title="Where the errors fall")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig2_error_hist.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    cols = [ORANGE if b == band["MAE"].idxmax() else TEAL for b in band.index]
    ax.bar(band.index.astype(str), band["MAE"], color=cols)
    ax.set(xlabel="True SOC band (%)", ylabel="MAE (SOC %)", title="Error by SOC band (worst band in orange)")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig3_error_by_band.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.bar(per_file.index.astype(str), per_file["MAE"], color=NAVY)
    ax.set(xlabel="Test file", ylabel="MAE (SOC %)", title="Error by held-out battery file")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout(); fig.savefig(f"{fig_dir}/fig4_error_by_file.png", dpi=160); plt.close(fig)

    if cv_path and os.path.exists(cv_path):
        cv = pd.read_csv(cv_path, index_col=0)
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.barh(cv.index, cv["cv_MAE"], xerr=cv["cv_MAE_std"], color=[GREY if "Mean" in n else TEAL for n in cv.index])
        ax.set(xlabel="Grouped-CV MAE (SOC %), train files only", title="Model selection")
        ax.invert_yaxis(); fig.tight_layout(); fig.savefig(f"{fig_dir}/fig5_cv_models.png", dpi=160); plt.close(fig)

    # ---- charging-time model (kept separate, lower confidence) ----------------------------------
    if charge_pipe is not None and full_path and os.path.exists(full_path):
        c_te = build_charge_frame(pd.read_csv(full_path), te["file_id"].unique())
        if len(c_te):
            cp = predict_charge_time(charge_pipe, c_te)
            m = metrics(c_te[C.CHARGE_TARGET], cp)
            summary["charge_time_test"] = {"rows": int(len(c_te)), "MAE_min": m["MAE"] / 60,
                                           "RMSE_min": m["RMSE"] / 60, "R2": m["R2"]}

    with open(os.path.join(out_dir, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description="Evaluate frozen models on the held-out test files.")
    ap.add_argument("--test", required=True)
    ap.add_argument("--train", default=None)
    ap.add_argument("--full", default=None)
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args(argv)
    s = evaluate(a.test, a.models, a.out, a.train, a.full, os.path.join(a.models, "cv_results_soc.csv"))
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
