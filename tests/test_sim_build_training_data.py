"""Tests for agents.sim_build_training_data — mocks all pybaseball calls."""
import pytest
import pandas as pd
import numpy as np


class TestAggregateToPlayerGame:
    """_aggregate_to_player_game converts pitch rows to player-game binary outcomes."""

    def _make_pitch_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            # Game 1, batter 123 hits a HR (two PAs)
            {"batter": 123, "pitcher": 500, "game_pk": 1001, "game_date": "2022-04-01",
             "home_team": "NYY", "p_throws": "R", "stand": "L", "events": "home_run", "bat_speed": 74.0},
            {"batter": 123, "pitcher": 500, "game_pk": 1001, "game_date": "2022-04-01",
             "home_team": "NYY", "p_throws": "R", "stand": "L", "events": "single", "bat_speed": 70.0},
            # Game 1, batter 456 does not hit a HR
            {"batter": 456, "pitcher": 500, "game_pk": 1001, "game_date": "2022-04-01",
             "home_team": "NYY", "p_throws": "R", "stand": "R", "events": "strikeout", "bat_speed": 66.0},
            # Game 2, batter 123 does not hit a HR
            {"batter": 123, "pitcher": 501, "game_pk": 1002, "game_date": "2022-04-02",
             "home_team": "BOS", "p_throws": "L", "stand": "L", "events": "groundout", "bat_speed": 71.0},
        ])

    def test_hit_hr_is_1_when_batter_hits_home_run(self):
        from agents.sim_build_training_data import _aggregate_to_player_game
        df = self._make_pitch_df()
        result = _aggregate_to_player_game(df)
        judge_game1 = result[(result["player_id"] == 123) & (result["game_pk"] == 1001)]
        assert len(judge_game1) == 1
        assert judge_game1.iloc[0]["hit_hr"] == 1

    def test_hit_hr_is_0_when_no_home_run(self):
        from agents.sim_build_training_data import _aggregate_to_player_game
        df = self._make_pitch_df()
        result = _aggregate_to_player_game(df)
        judge_game2 = result[(result["player_id"] == 123) & (result["game_pk"] == 1002)]
        assert len(judge_game2) == 1
        assert judge_game2.iloc[0]["hit_hr"] == 0

    def test_same_hand_is_1_when_stand_equals_p_throws(self):
        from agents.sim_build_training_data import _aggregate_to_player_game
        df = self._make_pitch_df()
        result = _aggregate_to_player_game(df)
        # batter 456 stands R vs pitcher R — same_hand = 1
        row = result[(result["player_id"] == 456) & (result["game_pk"] == 1001)].iloc[0]
        assert row["same_hand"] == 1

    def test_same_hand_is_0_when_stand_differs_from_p_throws(self):
        from agents.sim_build_training_data import _aggregate_to_player_game
        df = self._make_pitch_df()
        result = _aggregate_to_player_game(df)
        # batter 123 stands L vs pitcher R — same_hand = 0
        row = result[(result["player_id"] == 123) & (result["game_pk"] == 1001)].iloc[0]
        assert row["same_hand"] == 0

    def test_bat_speed_mean_is_averaged_across_plate_appearances(self):
        from agents.sim_build_training_data import _aggregate_to_player_game
        df = self._make_pitch_df()
        result = _aggregate_to_player_game(df)
        row = result[(result["player_id"] == 123) & (result["game_pk"] == 1001)].iloc[0]
        assert abs(row["bat_speed_mean"] - 72.0) < 0.01  # mean of 74.0 and 70.0

    def test_produces_one_row_per_player_game(self):
        from agents.sim_build_training_data import _aggregate_to_player_game
        df = self._make_pitch_df()
        result = _aggregate_to_player_game(df)
        assert len(result) == 3  # batter 123 game1, batter 456 game1, batter 123 game2

    def test_missing_bat_speed_column_produces_nan(self):
        from agents.sim_build_training_data import _aggregate_to_player_game
        df = self._make_pitch_df().drop(columns=["bat_speed"])
        result = _aggregate_to_player_game(df)
        assert result["bat_speed_mean"].isna().all()


class TestBuildSeasonUsesCheckpoint:
    def test_loads_checkpoint_without_network_calls(self, tmp_path, monkeypatch):
        import agents.sim_build_training_data as builder
        monkeypatch.setattr(builder, "CHECKPOINT_DIR", tmp_path)

        # Write fake checkpoints — _build_season returns early when both exist,
        # so no pybaseball calls are made regardless of network availability.
        pg_df = pd.DataFrame([{
            "player_id": 123, "game_pk": 1001, "game_date": "2022-04-01",
            "hit_hr": 1, "brl_percent": 10.0, "avg_hit_speed": 90.0,
            "ev95percent": 44.0, "iso": 0.22, "bat_speed": 69.5,
            "park_factor": 1.0, "same_hand": 0, "pitcher_hr9": 1.30, "season": 2022,
        }])
        bs_df = pd.DataFrame([{"player_id": 123, "season": 2022, "Name": "Aaron Judge", "avg_bat_speed": 74.2}])
        pg_df.to_parquet(tmp_path / "training_2022.parquet", index=False)
        bs_df.to_parquet(tmp_path / "bat_speed_2022.parquet", index=False)

        park_factors = {"NYY": 1.05}
        result_pg, result_bs = builder._build_season(2022, park_factors)
        assert len(result_pg) == 1
        assert result_pg.iloc[0]["hit_hr"] == 1
        assert result_bs.iloc[0]["Name"] == "Aaron Judge"
