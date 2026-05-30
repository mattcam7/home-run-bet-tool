# tests/test_simulation.py
"""Unit tests for agents.simulation — name normalization and matching."""
import pytest
from agents.simulation import _normalize_name, _match_player


class TestNormalizeName:
    def test_lowercases_then_titlecases(self):
        assert _normalize_name("AARON JUDGE") == "Aaron Judge"

    def test_strips_jr_suffix(self):
        assert _normalize_name("Vladimir Guerrero Jr.") == "Vladimir Guerrero"

    def test_strips_iii_suffix(self):
        assert _normalize_name("Cal Ripken III") == "Cal Ripken"

    def test_strips_accents(self):
        assert _normalize_name("Yandy Díaz") == "Yandy Diaz"

    def test_strips_leading_trailing_whitespace(self):
        assert _normalize_name("  Aaron Judge  ") == "Aaron Judge"

    def test_already_normal(self):
        assert _normalize_name("Aaron Judge") == "Aaron Judge"


class TestMatchPlayer:
    CANDIDATES = ["Aaron Judge", "Rafael Devers", "Shohei Ohtani"]

    def test_exact_match(self):
        assert _match_player("Aaron Judge", self.CANDIDATES) == "Aaron Judge"

    def test_fuzzy_match_typo(self):
        # "Aron Judge" is close enough
        result = _match_player("Aron Judge", self.CANDIDATES)
        assert result == "Aaron Judge"

    def test_no_match_returns_none(self):
        assert _match_player("Totally Unknown Player", self.CANDIDATES) is None

    def test_case_insensitive_exact(self):
        assert _match_player("aaron judge", self.CANDIDATES) == "Aaron Judge"


import pandas as pd
from unittest.mock import patch, MagicMock
from agents.simulation import (
    _get_weighted_batter_stats,
    _get_pitcher_stats_by_name,
    BATTER_FEATURES,
)


class TestGetWeightedBatterStats:
    """Tests weighted batter stat lookup (mocks pybaseball calls)."""

    def _make_batter_df(self, name: str, vals: dict) -> pd.DataFrame:
        row = {"Name": name, "HR": 20, "G": 140, "PA": 500}
        row.update({"Barrel%": 8.0, "ISO": 0.200, "FB%": 38.0, "Hard%": 42.0, "EV": 89.5})
        row.update(vals)
        return pd.DataFrame([row])

    def test_returns_none_for_unknown_player(self):
        batter_dfs = {
            2024: self._make_batter_df("Aaron Judge", {}),
            2025: self._make_batter_df("Aaron Judge", {}),
            2026: pd.DataFrame(columns=["Name"] + BATTER_FEATURES + ["HR", "G", "PA"]),
        }
        result = _get_weighted_batter_stats("Totally Unknown Player", batter_dfs)
        assert result is None

    def test_single_season_returns_that_seasons_stats(self):
        batter_dfs = {
            2024: pd.DataFrame(columns=["Name"] + BATTER_FEATURES + ["HR", "G", "PA"]),
            2025: pd.DataFrame(columns=["Name"] + BATTER_FEATURES + ["HR", "G", "PA"]),
            2026: self._make_batter_df("Aaron Judge", {"Barrel%": 10.0}),
        }
        result = _get_weighted_batter_stats("Aaron Judge", batter_dfs)
        assert result is not None
        assert abs(result["Barrel%"] - 10.0) < 0.01

    def test_weighted_average_across_seasons(self):
        """With 2024=0.10, 2025=0.30, 2026=0.60 weights, confirm weighted avg."""
        # Barrel%: 2024=4.0, 2025=8.0, 2026=12.0 -> weighted avg = 0.1*4+0.3*8+0.6*12 = 10.0
        df_2024 = pd.DataFrame([{"Name": "Aaron Judge", "Barrel%": 4.0, "ISO": 0.2, "FB%": 38.0, "Hard%": 42.0, "EV": 88.0, "HR": 15, "G": 140, "PA": 490}])
        df_2025 = pd.DataFrame([{"Name": "Aaron Judge", "Barrel%": 8.0, "ISO": 0.2, "FB%": 38.0, "Hard%": 42.0, "EV": 88.0, "HR": 20, "G": 140, "PA": 490}])
        df_2026 = pd.DataFrame([{"Name": "Aaron Judge", "Barrel%": 12.0, "ISO": 0.2, "FB%": 38.0, "Hard%": 42.0, "EV": 88.0, "HR": 12, "G": 70, "PA": 245}])
        batter_dfs = {2024: df_2024, 2025: df_2025, 2026: df_2026}
        result = _get_weighted_batter_stats("Aaron Judge", batter_dfs)
        assert result is not None
        assert abs(result["Barrel%"] - 10.0) < 0.01


class TestGetPitcherStatsByName:
    def _make_pitcher_df(self, name: str, hr9: float = 1.2, ip: float = 120.0) -> pd.DataFrame:
        return pd.DataFrame([{"Name": name, "HR/9": hr9, "HR/FB": 0.12, "xFIP": 3.80, "IP": ip}])

    def test_returns_none_for_unknown_pitcher(self):
        pitcher_dfs = {2025: self._make_pitcher_df("Gerrit Cole")}
        assert _get_pitcher_stats_by_name("Unknown Pitcher", pitcher_dfs) is None

    def test_finds_pitcher_by_name(self):
        pitcher_dfs = {2025: self._make_pitcher_df("Gerrit Cole", hr9=1.5)}
        result = _get_pitcher_stats_by_name("Gerrit Cole", pitcher_dfs)
        assert result is not None
        assert abs(result["HR/9"] - 1.5) < 0.01

    def test_fuzzy_match_finds_close_name(self):
        pitcher_dfs = {2025: self._make_pitcher_df("Gerrit Cole", hr9=1.5)}
        result = _get_pitcher_stats_by_name("Gerrit A. Cole", pitcher_dfs)
        assert result is not None
