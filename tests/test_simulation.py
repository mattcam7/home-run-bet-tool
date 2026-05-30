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


import numpy as np
from agents.simulation import HRRateModel, BATTER_FEATURES


def _make_training_df(n: int = 50) -> pd.DataFrame:
    """
    Synthetic training data with plausible feature ranges and realistic signal.
    hr_per_game is derived from features (like real data) so that Ridge can
    learn positive coefficients for power metrics (Barrel%, ISO, etc.).
    """
    rng = np.random.default_rng(42)
    barrel = rng.uniform(4, 18, n)
    iso = rng.uniform(0.10, 0.35, n)
    fb = rng.uniform(25, 55, n)
    hard = rng.uniform(30, 55, n)
    ev = rng.uniform(85, 95, n)
    # hr_per_game has a realistic positive relationship with power metrics
    hr_per_game = (
        0.004 * barrel
        + 0.15 * iso
        + 0.001 * fb
        + 0.001 * hard
        + 0.001 * ev
        + rng.normal(0, 0.005, n)  # small noise
    ).clip(0)
    data = {
        "Barrel%": barrel,
        "ISO": iso,
        "FB%": fb,
        "Hard%": hard,
        "EV": ev,
        "HR": (hr_per_game * 140).astype(int),
        "G": np.full(n, 140),
        "hr_per_game": hr_per_game,
    }
    return pd.DataFrame(data)


class TestHRRateModel:
    def test_fit_and_predict_returns_nonneg_float(self):
        model = HRRateModel()
        train_df = _make_training_df(50)
        model.fit(train_df)
        features = {"Barrel%": 10.0, "ISO": 0.22, "FB%": 38.0, "Hard%": 44.0, "EV": 90.0}
        result = model.predict(features)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_higher_barrel_pct_predicts_higher_hr_rate(self):
        model = HRRateModel()
        train_df = _make_training_df(200)
        model.fit(train_df)
        base = {"ISO": 0.22, "FB%": 38.0, "Hard%": 44.0, "EV": 90.0}
        low = model.predict({**base, "Barrel%": 4.0})
        high = model.predict({**base, "Barrel%": 18.0})
        assert high > low

    def test_save_and_load_round_trip(self, tmp_path):
        model = HRRateModel()
        train_df = _make_training_df(50)
        model.fit(train_df)
        path = str(tmp_path / "model.pkl")
        model.save(path)
        model2 = HRRateModel()
        model2.load(path)
        features = {"Barrel%": 10.0, "ISO": 0.22, "FB%": 38.0, "Hard%": 44.0, "EV": 90.0}
        assert abs(model.predict(features) - model2.predict(features)) < 1e-9

    def test_predict_before_fit_raises(self):
        model = HRRateModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict({"Barrel%": 10.0, "ISO": 0.2, "FB%": 38.0, "Hard%": 42.0, "EV": 90.0})


from agents.simulation import (
    _get_park_factor,
    _get_platoon_factor,
    add_simulation,
)


class TestGetParkFactor:
    def test_known_home_team_returns_factor(self):
        park_factors = {"BOS": 1.065, "SFG": 0.828}
        result = _get_park_factor("Texas Rangers @ Boston Red Sox", park_factors)
        assert abs(result - 1.065) < 0.001

    def test_unknown_home_team_returns_neutral(self):
        park_factors = {"BOS": 1.065}
        result = _get_park_factor("Texas Rangers @ Unknown Team", park_factors)
        assert result == 1.0

    def test_malformed_game_string_returns_neutral(self):
        result = _get_park_factor("", {})
        assert result == 1.0

    def test_oracle_park_suppresses_hr(self):
        park_factors = {"SFG": 0.828}
        result = _get_park_factor("Los Angeles Dodgers @ San Francisco Giants", park_factors)
        assert result < 1.0

    def test_coors_field_boosts_hr(self):
        park_factors = {"COL": 1.198}
        result = _get_park_factor("Los Angeles Dodgers @ Colorado Rockies", park_factors)
        assert result > 1.0


class TestGetPlatoonFactor:
    def test_opposite_hand_favorable(self):
        assert _get_platoon_factor("L", "R") == 1.05
        assert _get_platoon_factor("R", "L") == 1.05

    def test_same_hand_unfavorable(self):
        assert _get_platoon_factor("L", "L") == 0.95
        assert _get_platoon_factor("R", "R") == 0.95

    def test_switch_hitter_neutral(self):
        assert _get_platoon_factor("S", "R") == 1.0
        assert _get_platoon_factor("S", "L") == 1.0

    def test_unknown_hand_neutral(self):
        assert _get_platoon_factor("", "R") == 1.0
        assert _get_platoon_factor("R", "") == 1.0
        assert _get_platoon_factor("", "") == 1.0


class TestAddSimulation:
    def _make_final_df(self) -> pd.DataFrame:
        from datetime import datetime, timezone
        return pd.DataFrame([
            {
                "player_name": "Aaron Judge",
                "team": "NYY",
                "game": "Texas Rangers @ New York Yankees",
                "commence_time": datetime(2026, 5, 30, 20, 0, tzinfo=timezone.utc),
                "pinnacle_prob": 0.18,
                "pinnacle_odds": 400,
                "sharp_anchor": "pinnacle",
                "best_retail_odds": 450,
                "best_retail_decimal": 5.5,
                "ev_pct": 0.05,
                "composite_score": 0.009,
                "composite_z": 1.2,
                "kelly_units": 0.5,
                "stake_usd": 12.5,
            }
        ])

    def test_returns_df_unchanged_on_import_error(self, monkeypatch):
        """If pybaseball is unavailable, add_simulation returns df unchanged."""
        import sys
        monkeypatch.setitem(sys.modules, "pybaseball", None)
        df = self._make_final_df()
        result = add_simulation(df)
        assert list(result.columns) == list(df.columns)

    def test_adds_sim_columns_when_data_available(self, tmp_path, monkeypatch):
        """With mocked pybaseball and network, sim columns are added with valid sim_prob."""
        import pybaseball as _pybaseball
        import agents.simulation as sim_mod

        monkeypatch.setattr(sim_mod, "CACHE_DIR", tmp_path / "sim_cache")
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "sim_model.pkl")
        monkeypatch.setattr(sim_mod, "UNMATCHED_LOG", tmp_path / "sim_unmatched.log")

        # Build a small but non-degenerate batter DataFrame (need >=2 rows for Ridge)
        import numpy as np
        rng = np.random.default_rng(0)
        n = 10
        mock_batter_df = pd.DataFrame({
            "Name": ["Aaron Judge"] + [f"Player{i}" for i in range(n - 1)],
            "Barrel%": [18.0] + list(rng.uniform(4, 18, n - 1)),
            "ISO": [0.340] + list(rng.uniform(0.10, 0.35, n - 1)),
            "FB%": [42.0] + list(rng.uniform(25, 55, n - 1)),
            "Hard%": [58.0] + list(rng.uniform(30, 55, n - 1)),
            "EV": [95.0] + list(rng.uniform(85, 95, n - 1)),
            "HR": [25] + list(rng.integers(5, 40, n - 1)),
            "G": [80] + list(rng.integers(80, 162, n - 1)),
            "PA": [300] + list(rng.integers(200, 600, n - 1)),
        })

        cole_row = {
            "Name": "Gerrit Cole", "HR/9": 1.1, "HR/FB": 0.10, "xFIP": 3.20, "IP": 80.0
        }
        mock_pitcher_df = pd.DataFrame([cole_row])

        monkeypatch.setattr(_pybaseball, "batting_stats", lambda s, qual=50: mock_batter_df)
        monkeypatch.setattr(_pybaseball, "pitching_stats", lambda s, qual=1: mock_pitcher_df)

        # Mock network-dependent functions to avoid live API calls
        monkeypatch.setattr(sim_mod, "_fetch_probable_starters", lambda today: {})
        monkeypatch.setattr(sim_mod, "_fetch_batter_hands", lambda: {})

        df = self._make_final_df()
        result = add_simulation(df)
        assert "sim_prob" in result.columns
        assert "sim_edge" in result.columns
        assert "convergence" in result.columns
        # Aaron Judge should be matched with valid sim_prob in [0.01, 0.60]
        matched = result["sim_prob"].dropna()
        assert len(matched) > 0, "No players were matched to FanGraphs data"
        assert matched.between(0.01, 0.60).all()
        # With elite stats (Barrel 18, ISO .340), sim_prob should be non-trivial
        judge_prob = result.loc[result.index[0], "sim_prob"]
        assert 0.05 < judge_prob < 0.55, f"sim_prob {judge_prob:.3f} is implausibly outside 0.05-0.55"


def test_add_simulation_importable_from_run():
    """Confirm add_simulation is imported in run.py so the pipeline can call it."""
    import importlib, ast, pathlib
    source = pathlib.Path("run.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "agents.simulation"
    ]
    assert imports, "run.py does not import from agents.simulation"
    names = [alias.name for imp in imports for alias in imp.names]
    assert "add_simulation" in names, f"add_simulation not imported; found: {names}"
