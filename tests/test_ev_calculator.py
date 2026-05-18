import pytest
import pandas as pd
from datetime import datetime, timezone
from agents.ev_calculator import calculate_ev

COMMENCE = datetime(2026, 5, 15, 23, 5, tzinfo=timezone.utc)
GAME = "New York Yankees @ Boston Red Sox"

RETAIL_DF = pd.DataFrame([
    {"player_name": "Aaron Judge",   "game": GAME, "commence_time": COMMENCE, "bookmaker": "DraftKings", "american_odds": 450, "implied_prob": 1/5.5},
    {"player_name": "Aaron Judge",   "game": GAME, "commence_time": COMMENCE, "bookmaker": "FanDuel",    "american_odds": 420, "implied_prob": 1/5.2},
    {"player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE, "bookmaker": "DraftKings", "american_odds": 600, "implied_prob": 1/7.0},
    {"player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE, "bookmaker": "FanDuel",    "american_odds": 580, "implied_prob": 1/6.8},
])

PINNACLE_DF = pd.DataFrame([
    {"player_name": "Aaron Judge",   "game": GAME, "commence_time": COMMENCE, "pinnacle_odds": 380, "pinnacle_prob": 1/4.8},
    {"player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE, "pinnacle_odds": 520, "pinnacle_prob": 1/6.2},
])

def test_empty_pinnacle_raises_clear_error():
    with pytest.raises(ValueError, match="Pinnacle"):
        calculate_ev(RETAIL_DF, pd.DataFrame())

def test_empty_retail_raises_clear_error():
    with pytest.raises(ValueError, match="retail"):
        calculate_ev(pd.DataFrame(), PINNACLE_DF)

def test_one_row_per_player():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    assert len(df) == 2

def test_excludes_players_not_at_pinnacle():
    extra = pd.concat([RETAIL_DF, pd.DataFrame([{
        "player_name": "Ghost Player", "game": GAME, "commence_time": COMMENCE,
        "bookmaker": "DraftKings", "american_odds": 800, "implied_prob": 0.11,
    }])])
    df = calculate_ev(extra, PINNACLE_DF)
    assert "Ghost Player" not in df["player_name"].values

def test_ev_formula():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    expected = (1/4.8 * 5.5) - 1
    assert abs(judge["ev_pct"] - expected) < 0.001

def test_best_retail_selects_highest_decimal():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    assert judge["best_retail_odds"] == 450  # DK +450 beats FD +420

def test_best_retail_book_is_tracked():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    assert judge["best_retail_book"] == "DraftKings"

def test_composite_score():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    assert abs(judge["composite_score"] - (judge["ev_pct"] * judge["pinnacle_prob"])) < 0.0001

def test_composite_z_mean_is_zero():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    assert abs(df["composite_z"].mean()) < 0.0001

def test_sorted_by_composite_z_descending():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    assert df["composite_z"].iloc[0] >= df["composite_z"].iloc[-1]

def test_quarter_kelly_units_and_stake():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    # best retail +450 -> dec 5.5 -> b=4.5 ; p=1/4.8 (Pinnacle no-vig)
    b = 5.5 - 1
    p = 1 / 4.8
    f_star = (b * p - (1 - p)) / b
    expected_units = round(max(f_star / 4, 0) * 100 * 2) / 2
    assert abs(judge["kelly_units"] - expected_units) < 1e-9
    assert abs(judge["stake_usd"] - expected_units * 25) < 1e-9

def test_kelly_units_capped_at_3_and_floored_at_0():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    assert (df["kelly_units"] >= 0).all()
    assert (df["kelly_units"] <= 3.0).all()
