"""Smoke tests: python -m pytest -q   (needs models/ to exist - run battery_soc.train first)."""
import numpy as np, pandas as pd, pytest
from battery_soc import RAW_FEATURES, SOC_CLIP, load_models, predict_soc


def _sample(n=50):
    rng = np.random.default_rng(0)
    return pd.DataFrame({"voltage_v": rng.uniform(3.0, 4.2, n), "current_a": rng.uniform(-3, 3, n),
                         "cell_temp_c": rng.uniform(20, 35, n), "power_w": rng.uniform(-10, 10, n),
                         "ambient_temp_c": rng.uniform(24, 26, n)})


def test_output_is_clipped_and_sized():
    soc, _ = load_models("models")
    out = predict_soc(soc, _sample())
    assert out.shape == (50,) and out.min() >= SOC_CLIP[0] and out.max() <= SOC_CLIP[1]


def test_missing_column_raises():
    soc, _ = load_models("models")
    with pytest.raises(ValueError):
        predict_soc(soc, _sample().drop(columns=["voltage_v"]))


def test_deterministic():
    soc, _ = load_models("models")
    s = _sample()
    assert np.allclose(predict_soc(soc, s), predict_soc(soc, s))
