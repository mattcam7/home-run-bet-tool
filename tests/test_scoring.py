import math
import pandas as pd
import pytest
from agents.scoring import compute_bet_score, _SIGMOID_K, _ANCHOR_WEIGHTS, _sim_multiplier


def _slate(rows):
    """Build a minimal DataFrame for scoring tests."""
    base = {
        "ev_pct": 0.05,
        "pinnacle_prob": 0.18,
        "anchor_quality": "pinnacle",
        "over_only": False,
        "kelly_units": 0.5,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_compute_bet_score_adds_columns():
    df = _slate([{}, {}])
    result = compute_bet_score(df)
    assert "bet_score" in result.columns
    assert "bet_grade" in result.columns


def test_over_only_play_always_scores_zero():
    df = _slate([{"anchor_quality": "pinnacle_over_only", "over_only": True, "ev_pct": 0.30}])
    result = compute_bet_score(df)
    assert result.iloc[0]["bet_score"] == 0


def test_highest_ev_play_scores_highest_on_slate():
    df = _slate([
        {"ev_pct": 0.20, "anchor_quality": "pinnacle"},
        {"ev_pct": 0.02, "anchor_quality": "pinnacle"},
        {"ev_pct": -0.01, "anchor_quality": "betonlineag"},
    ])
    result = compute_bet_score(df)
    assert result.iloc[0]["bet_score"] > result.iloc[1]["bet_score"]
    assert result.iloc[1]["bet_score"] > result.iloc[2]["bet_score"]


def test_pinnacle_anchor_scores_higher_than_unknown_at_same_ev():
    df = _slate([
        {"ev_pct": 0.10, "anchor_quality": "pinnacle"},
        {"ev_pct": 0.10, "anchor_quality": "unknown"},
    ])
    result = compute_bet_score(df)
    assert result.iloc[0]["bet_score"] > result.iloc[1]["bet_score"]


def test_strong_grade_at_80_plus():
    # Force a high z-score play to be 80+
    df = _slate([
        {"ev_pct": 0.40, "anchor_quality": "pinnacle", "pinnacle_prob": 0.25},
        {"ev_pct": -0.10, "anchor_quality": "unknown", "pinnacle_prob": 0.10},
    ])
    result = compute_bet_score(df)
    assert result.iloc[0]["bet_grade"] == "Strong"


def test_grade_thresholds():
    from agents.scoring import _grade
    assert _grade(80) == "Strong"
    assert _grade(79) == "Solid"
    assert _grade(60) == "Solid"
    assert _grade(59) == "Marginal"
    assert _grade(40) == "Marginal"
    assert _grade(39) == "Skip"
    assert _grade(0) == "Skip"


def test_uniform_slate_scores_50():
    # When all plays have identical composite, all z-scores are 0 → score=50
    df = _slate([{"ev_pct": 0.05} for _ in range(5)])
    result = compute_bet_score(df)
    assert all(result["bet_score"] == 50)


def test_single_row_slate_scores_50():
    df = _slate([{}])
    result = compute_bet_score(df)
    assert result.iloc[0]["bet_score"] == 50


# ---------------------------------------------------------------------------
# Sim dampener unit tests
# ---------------------------------------------------------------------------

def test_sim_multiplier_no_penalty_above_threshold():
    assert _sim_multiplier(0.0) == 1.0
    assert _sim_multiplier(-0.04) == 1.0
    assert _sim_multiplier(None) == 1.0


def test_sim_multiplier_partial_penalty_at_minus_10pp():
    # sim_edge = -0.10: halfway into penalty range → penalty = 0.20
    m = _sim_multiplier(-0.10)
    assert abs(m - 0.80) < 1e-9


def test_sim_multiplier_max_penalty_at_minus_15pp_plus():
    assert abs(_sim_multiplier(-0.15) - 0.60) < 1e-9
    assert abs(_sim_multiplier(-0.30) - 0.60) < 1e-9  # capped at 40% penalty


def test_negative_sim_edge_reduces_score():
    """A player with large negative sim_edge should score lower than one without."""
    base = {
        "ev_pct": 0.15, "pinnacle_prob": 0.22, "anchor_quality": "pinnacle",
        "over_only": False, "kelly_units": 1.0,
    }
    df = pd.DataFrame([
        {**base, "sim_prob": 0.08},   # sim_edge = -0.14: heavy penalty
        {**base, "sim_prob": 0.22},   # sim_edge = 0.00: no penalty
    ])
    result = compute_bet_score(df)
    assert result.iloc[1]["bet_score"] > result.iloc[0]["bet_score"]


def test_positive_sim_edge_does_not_boost_score():
    """Sim being more bullish than Pinnacle should not inflate the score."""
    base = {
        "ev_pct": 0.10, "pinnacle_prob": 0.18, "anchor_quality": "pinnacle",
        "over_only": False, "kelly_units": 0.5,
    }
    df = pd.DataFrame([
        {**base, "sim_prob": 0.30},   # sim_edge = +0.12: bullish
        {**base, "sim_prob": 0.18},   # sim_edge = 0.00: neutral
    ])
    result = compute_bet_score(df)
    # Bullish sim must not push score above neutral — multiplier is 1.0 for both
    assert result.iloc[0]["bet_score"] == result.iloc[1]["bet_score"]


def test_missing_sim_prob_is_graceful():
    """When sim_prob column is absent, score is computed without penalty."""
    df = _slate([{"ev_pct": 0.20}, {"ev_pct": 0.05}])
    result = compute_bet_score(df)
    assert result.iloc[0]["bet_score"] > result.iloc[1]["bet_score"]


def test_nan_sim_prob_treated_as_no_penalty():
    """NaN sim_prob should behave identically to no sim column."""
    base = {
        "ev_pct": 0.10, "pinnacle_prob": 0.18, "anchor_quality": "pinnacle",
        "over_only": False, "kelly_units": 0.5,
    }
    df_with_nan = pd.DataFrame([{**base, "sim_prob": float("nan")}])
    df_without = pd.DataFrame([base])
    r_nan = compute_bet_score(df_with_nan).iloc[0]["bet_score"]
    r_none = compute_bet_score(df_without).iloc[0]["bet_score"]
    assert r_nan == r_none
