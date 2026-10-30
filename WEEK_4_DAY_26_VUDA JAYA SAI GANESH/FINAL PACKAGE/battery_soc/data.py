"""Loading, integrity checks and the charging-time target."""
import hashlib
import pandas as pd
from .config import CHARGE_TARGET, CHARGE_FEATURES


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_split(train_path, test_path=None):
    """Load train (and optionally test) and make sure no battery file appears in both."""
    train = pd.read_csv(train_path)
    if test_path is None:
        return train, None
    test = pd.read_csv(test_path)
    shared = set(train["file_id"]).intersection(test["file_id"])
    if shared:
        raise AssertionError(f"Train/test share file_ids {sorted(shared)[:5]} - split is broken!")
    return train, test


def check_columns(df, columns, name="input"):
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def build_charge_frame(full_df, file_ids):
    """Remaining-charge-time target = (end of the CHA segment - now), in seconds.

    Only constant-current/constant-voltage charging rows ('CHA') are used, one segment at a time.
    """
    d = full_df.copy()
    d["prog_time_td"] = pd.to_timedelta(d["prog_time"], errors="coerce")
    d = d.sort_values(["file_id", "prog_time_td"]).reset_index(drop=True)
    d["status_change"] = (d["status"] != d["status"].shift(1)) | (d["file_id"] != d["file_id"].shift(1))
    d["segment_id"] = d["status_change"].cumsum()
    dc = d[d["status"] == "CHA"].copy()
    dc[CHARGE_TARGET] = (dc.groupby("segment_id")["prog_time_td"].transform("max")
                         - dc["prog_time_td"]).dt.total_seconds()
    dc = dc.dropna(subset=[CHARGE_TARGET] + CHARGE_FEATURES)
    return dc[dc["file_id"].isin(set(file_ids))]
