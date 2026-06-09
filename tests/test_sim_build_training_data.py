"""Tests for sim_build_training_data helpers."""
import numpy as np
import pandas as pd
import pytest


def _make_statcast_df(n: int = 200) -> pd.DataFrame:
    """Minimal synthetic Statcast pitch-level DataFrame."""
    rng = np.random.default_rng(42)
    batter_ids = rng.choice([592450, 660271, 605141], n)
    pitcher_ids = rng.choice([543037, 477132], n)
    game_pks = rng.choice([1001, 1002, 1003, 1004], n)
    game_dates = rng.choice(
        pd.date_range("2024-04-01", periods=20).astype(str).tolist(), n
    )
    p_throws = rng.choice(["L", "R"], n)
    stand = rng.choice(["L", "R"], n)
    bb_types = rng.choice(["fly_ball", "ground_ball", "line_drive", None], n, p=[0.2, 0.3, 0.2, 0.3])
    events = rng.choice(
        ["home_run", "field_out", "strikeout", "single", "double", None],
        n, p=[0.03, 0.20, 0.20, 0.10, 0.05, 0.42],
    )
    barrels = np.where(
        (bb_types == "fly_ball") & (rng.random(n) < 0.15), 1, 0
    ).astype(float)
    launch_speeds = np.where(
        [b is not None for b in bb_types],
        rng.uniform(70, 110, n),
        np.nan,
    )
    bat_orders = rng.choice(range(1, 10), n).astype(float)
    return pd.DataFrame({
        "batter": batter_ids,
        "pitcher": pitcher_ids,
        "game_pk": game_pks,
        "game_date": game_dates,
        "home_team": "NYY",
        "p_throws": p_throws,
        "stand": stand,
        "events": events,
        "bb_type": bb_types,
        "bat_order": bat_orders,
        "launch_speed": launch_speeds,
        "barrel": barrels,
    })


def test_fetch_batter_splits_returns_expected_columns():
    from agents.sim_build_training_data import _fetch_batter_splits
    sc = _make_statcast_df(400)
    result = _fetch_batter_splits(sc, season=2024)
    assert not result.empty
    required = {"player_id", "season", "vs_hand", "brl_pct", "iso", "fb_pct", "hr_fb"}
    assert required.issubset(result.columns), f"Missing: {required - set(result.columns)}"


def test_fetch_batter_splits_vs_hand_values():
    from agents.sim_build_training_data import _fetch_batter_splits
    sc = _make_statcast_df(400)
    result = _fetch_batter_splits(sc, season=2024)
    assert result["vs_hand"].isin(["L", "R"]).all()


def test_fetch_batter_splits_fb_pct_in_range():
    from agents.sim_build_training_data import _fetch_batter_splits
    sc = _make_statcast_df(400)
    result = _fetch_batter_splits(sc, season=2024)
    valid = result["fb_pct"].dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_aggregate_lineup_slot_captured():
    from agents.sim_build_training_data import _aggregate_to_player_game
    sc = _make_statcast_df(200)
    result = _aggregate_to_player_game(sc)
    assert "lineup_slot" in result.columns
    valid = result["lineup_slot"].dropna()
    assert (valid >= 1).all() and (valid <= 9).all()


def test_compute_rolling_batter_features_no_data_leakage():
    """Rolling stats for game N must not include game N itself (shift(1))."""
    from agents.sim_build_training_data import _compute_rolling_batter_features
    # One batter, 15 sequential games with known per-game barrel counts
    rng = np.random.default_rng(7)
    n_games = 15
    contact = pd.DataFrame({
        "player_id": [592450] * n_games,
        "game_pk": list(range(1001, 1001 + n_games)),
        "game_date": pd.date_range("2024-04-01", periods=n_games).astype(str).tolist(),
        "game_brl_pct": rng.uniform(5, 15, n_games),
        "game_avg_ev": rng.uniform(86, 94, n_games),
    })
    result = _compute_rolling_batter_features(contact)
    # First game should have NaN rolling stats (no prior games)
    first_row = result[result["game_pk"] == 1001]
    assert first_row["rolling_brl_pct"].isna().all()
    # Game 11+ should have rolling values (min 5 games in window)
    game_11 = result[result["game_pk"] == 1011]
    assert game_11["rolling_brl_pct"].notna().all()


def test_compute_rolling_pitcher_features_gb_pct_in_range():
    from agents.sim_build_training_data import _compute_rolling_pitcher_features
    sc = _make_statcast_df(600)
    result = _compute_rolling_pitcher_features(sc)
    valid_gb = result["rolling_pitcher_gb_pct"].dropna()
    assert (valid_gb >= 0.0).all() and (valid_gb <= 1.0).all()


def test_compute_rolling_pitcher_features_returns_expected_columns():
    from agents.sim_build_training_data import _compute_rolling_pitcher_features
    sc = _make_statcast_df(600)
    result = _compute_rolling_pitcher_features(sc)
    assert "pitcher" in result.columns
    assert "game_pk" in result.columns
    assert "rolling_pitcher_hr9" in result.columns
    assert "rolling_pitcher_gb_pct" in result.columns
