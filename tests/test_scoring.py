import math
import pandas as pd
import pytest
from agents.scoring import compute_bet_score, _SIGMOID_K, _ANCHOR_WEIGHTS


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
