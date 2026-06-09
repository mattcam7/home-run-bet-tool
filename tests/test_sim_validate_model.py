"""Tests for sim_validate_model helpers."""
import numpy as np
import pandas as pd
import pytest


def _make_training_cache(n_train: int = 500, n_test: int = 200) -> pd.DataFrame:
    """Synthetic training cache with v1 + v2 features + hit_hr + season."""
    rng = np.random.default_rng(99)
    n = n_train + n_test
    seasons = [2022] * (n_train // 2) + [2023] * (n_train // 2) + [2025] * n_test
    brl = rng.uniform(4, 18, n)
    ev = rng.uniform(85, 95, n)
    hard = rng.uniform(30, 55, n)
    iso = rng.uniform(0.10, 0.35, n)
    bat_speed = rng.uniform(64, 75, n)
    park = rng.uniform(0.85, 1.20, n)
    same_hand = rng.integers(0, 2, n).astype(float)
    hr9 = rng.uniform(0.8, 2.0, n)
    fb_pct = rng.uniform(0.25, 0.50, n)
    hr_fb = rng.uniform(0.05, 0.30, n)
    logit = -3.5 + 0.06 * brl + 8.0 * iso + 2.0 * hr_fb
    prob = 1.0 / (1.0 + np.exp(-logit))
    hit_hr = rng.binomial(1, prob).astype(float)
    return pd.DataFrame({
        # v1 features
        "brl_percent": brl, "avg_hit_speed": ev, "ev95percent": hard,
        "iso": iso, "bat_speed": bat_speed, "park_factor": park,
        "same_hand": same_hand, "pitcher_hr9": hr9,
        "fb_pct": fb_pct, "hr_fb": hr_fb,
        # v2 features
        "brl_pct_vs_hand": brl * rng.uniform(0.8, 1.2, n),
        "iso_vs_hand": iso * rng.uniform(0.8, 1.2, n),
        "fb_pct_vs_hand": fb_pct * rng.uniform(0.8, 1.2, n),
        "hr_fb_vs_hand": hr_fb * rng.uniform(0.8, 1.2, n),
        "rolling_brl_pct": brl * rng.uniform(0.9, 1.1, n),
        "rolling_avg_ev": ev * rng.uniform(0.98, 1.02, n),
        "rolling_pitcher_hr9": hr9 * rng.uniform(0.9, 1.1, n),
        "pitcher_gb_pct": rng.uniform(0.35, 0.55, n),
        "lineup_slot": rng.uniform(1, 9, n),
        "hit_hr": hit_hr,
        "season": seasons,
    })


def test_train_eval_returns_metric_keys():
    from agents.sim_validate_model import _train_eval, FEATURES_V1
    df = _make_training_cache()
    train = df[df["season"] < 2025]
    test = df[df["season"] == 2025]
    result = _train_eval(train, test, FEATURES_V1)
    assert "auc" in result
    assert "brier" in result
    assert "logloss" in result
    assert "n_train" in result
    assert result["n_train"] > 0
    assert 0.0 < result["auc"] < 1.0


def test_train_eval_empty_test_returns_nan():
    from agents.sim_validate_model import _train_eval, FEATURES_V1
    df = _make_training_cache()
    train = df[df["season"] < 2025]
    test = df[df["season"] == 2099]  # no rows
    result = _train_eval(train, test, FEATURES_V1)
    assert result["n_test"] == 0
    assert result["auc"] != result["auc"]  # NaN check


def test_validate_model_main_exits_cleanly(tmp_path):
    from agents.sim_validate_model import main
    import agents.sim_validate_model as vm
    cache = _make_training_cache(400, 150)
    cache_path = tmp_path / "cache.parquet"
    cache.to_parquet(cache_path, index=False)
    import unittest.mock as mock
    with mock.patch.object(vm, "TRAINING_CACHE_PATH", cache_path):
        # Should not raise; may exit 0 or 1 depending on synthetic data
        try:
            main()
        except SystemExit as e:
            assert e.code in (0, 1)
