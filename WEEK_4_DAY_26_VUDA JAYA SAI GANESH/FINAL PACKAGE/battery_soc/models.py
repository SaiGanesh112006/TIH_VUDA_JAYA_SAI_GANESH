"""Pipeline factory. Preprocessing is frozen as 'select these columns by name' (no scaling needed)."""
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from .config import SEED, RF_PARAMS, HGB_PARAMS, CHARGE_RF_PARAMS


def make_pipe(estimator, features):
    sel = ColumnTransformer([("keep", "passthrough", list(features))], remainder="drop")
    return Pipeline([("select", sel), ("model", estimator)])


def soc_candidates():
    """name -> (estimator factory, complexity rank; lower = simpler)."""
    return {
        "Mean baseline":        (lambda: DummyRegressor(strategy="mean"), 0),
        "Linear (raw)":         (lambda: LinearRegression(), 1),
        "RandomForest (tuned)": (lambda: RandomForestRegressor(random_state=SEED, n_jobs=-1, **RF_PARAMS), 2),
        "HistGradBoost (raw)":  (lambda: HistGradientBoostingRegressor(random_state=SEED, **HGB_PARAMS), 3),
    }


def charge_estimator():
    return RandomForestRegressor(random_state=SEED, n_jobs=-1, **CHARGE_RF_PARAMS)
