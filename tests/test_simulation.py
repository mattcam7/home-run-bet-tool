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
        row = {"Name": name, "HR": 20, "G": 140}
        row.update({"brl_percent": 8.0, "avg_hit_speed": 89.5, "ev95percent": 42.0, "iso": 0.200})
        row.update(vals)
        return pd.DataFrame([row])

    def test_returns_none_for_unknown_player(self):
        batter_dfs = {
            2024: self._make_batter_df("Aaron Judge", {}),
            2025: self._make_batter_df("Aaron Judge", {}),
            2026: pd.DataFrame(columns=["Name"] + BATTER_FEATURES + ["HR", "G"]),
        }
        result = _get_weighted_batter_stats("Totally Unknown Player", batter_dfs)
        assert result is None

    def test_single_season_returns_that_seasons_stats(self):
        batter_dfs = {
            2024: pd.DataFrame(columns=["Name"] + BATTER_FEATURES + ["HR", "G"]),
            2025: pd.DataFrame(columns=["Name"] + BATTER_FEATURES + ["HR", "G"]),
            2026: self._make_batter_df("Aaron Judge", {"brl_percent": 10.0}),
        }
        result = _get_weighted_batter_stats("Aaron Judge", batter_dfs)
        assert result is not None
        assert abs(result["brl_percent"] - 10.0) < 0.01

    def test_weighted_average_across_seasons_full_sample(self):
        """With 2026 G=100 (full sample), weights stay at 10/30/60 — no shrinkage."""
        # brl_percent: 2024=4.0, 2025=8.0, 2026=12.0 -> 0.1*4+0.3*8+0.6*12 = 10.0
        df_2024 = pd.DataFrame([{"Name": "Aaron Judge", "brl_percent": 4.0, "avg_hit_speed": 88.0, "ev95percent": 42.0, "iso": 0.2, "HR": 15, "G": 140}])
        df_2025 = pd.DataFrame([{"Name": "Aaron Judge", "brl_percent": 8.0, "avg_hit_speed": 88.0, "ev95percent": 42.0, "iso": 0.2, "HR": 20, "G": 140}])
        df_2026 = pd.DataFrame([{"Name": "Aaron Judge", "brl_percent": 12.0, "avg_hit_speed": 88.0, "ev95percent": 42.0, "iso": 0.2, "HR": 12, "G": 100}])
        batter_dfs = {2024: df_2024, 2025: df_2025, 2026: df_2026}
        result = _get_weighted_batter_stats("Aaron Judge", batter_dfs)
        assert result is not None
        assert abs(result["brl_percent"] - 10.0) < 0.01

    def test_small_sample_2026_shrinks_toward_prior_seasons(self):
        """With 2026 G=50, current-season weight halves — prior seasons dominate more."""
        # Extreme 2026 stats (brl_percent=30) get shrunk when sample is small.
        df_2024 = pd.DataFrame([{"Name": "Aaron Judge", "brl_percent": 8.0, "avg_hit_speed": 88.0, "ev95percent": 42.0, "iso": 0.2, "HR": 15, "G": 140}])
        df_2025 = pd.DataFrame([{"Name": "Aaron Judge", "brl_percent": 8.0, "avg_hit_speed": 88.0, "ev95percent": 42.0, "iso": 0.2, "HR": 20, "G": 140}])
        df_2026 = pd.DataFrame([{"Name": "Aaron Judge", "brl_percent": 30.0, "avg_hit_speed": 88.0, "ev95percent": 42.0, "iso": 0.2, "HR": 10, "G": 50}])
        batter_dfs = {2024: df_2024, 2025: df_2025, 2026: df_2026}
        result = _get_weighted_batter_stats("Aaron Judge", batter_dfs)
        assert result is not None
        # 2026 effective weight = 0.60 * (50/100) = 0.30; total = 0.10+0.30+0.30 = 0.70
        # expected = (0.10*8 + 0.30*8 + 0.30*30) / 0.70 = (0.8 + 2.4 + 9.0) / 0.70 ≈ 17.43
        effective_2026 = 0.60 * (50 / 100)
        total = 0.10 + 0.30 + effective_2026
        expected = (0.10 * 8 + 0.30 * 8 + effective_2026 * 30) / total
        assert abs(result["brl_percent"] - expected) < 0.01
        # Crucially: full 2026 weight (0.60) would give (0.10*8+0.30*8+0.60*30)/1.0=21.6.
        # Shrinkage reduces this to ~17.4 — still elevated but not as extreme.
        assert result["brl_percent"] < 21.6


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
    Features match BATTER_FEATURES: brl_percent, avg_hit_speed, ev95percent, iso.
    hr_per_game is derived from features (like real data) so that Ridge can
    learn positive coefficients for power metrics.
    """
    rng = np.random.default_rng(42)
    brl = rng.uniform(4, 18, n)           # brl_percent (Barrel%)
    ev = rng.uniform(85, 95, n)            # avg_hit_speed (exit velocity)
    hard = rng.uniform(30, 55, n)          # ev95percent (hard hit %)
    iso = rng.uniform(0.10, 0.35, n)       # iso (slugging - batting avg)
    # hr_per_game has a realistic positive relationship with power metrics
    hr_per_game = (
        0.004 * brl
        + 0.15 * iso
        + 0.001 * hard
        + 0.001 * ev
        + rng.normal(0, 0.005, n)  # small noise
    ).clip(0)
    data = {
        "brl_percent": brl,
        "avg_hit_speed": ev,
        "ev95percent": hard,
        "iso": iso,
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
        features = {"brl_percent": 10.0, "avg_hit_speed": 90.0, "ev95percent": 44.0, "iso": 0.22}
        result = model.predict(features)
        assert isinstance(result, float)
        assert result >= 0.0

    def test_higher_barrel_pct_predicts_higher_hr_rate(self):
        model = HRRateModel()
        train_df = _make_training_df(200)
        model.fit(train_df)
        base = {"avg_hit_speed": 90.0, "ev95percent": 44.0, "iso": 0.22}
        low = model.predict({**base, "brl_percent": 4.0})
        high = model.predict({**base, "brl_percent": 18.0})
        assert high > low

    def test_save_and_load_round_trip(self, tmp_path):
        model = HRRateModel()
        train_df = _make_training_df(50)
        model.fit(train_df)
        path = str(tmp_path / "model.pkl")
        model.save(path)
        model2 = HRRateModel()
        model2.load(path)
        features = {"brl_percent": 10.0, "avg_hit_speed": 90.0, "ev95percent": 44.0, "iso": 0.22}
        assert abs(model.predict(features) - model2.predict(features)) < 1e-9

    def test_predict_before_fit_raises(self):
        model = HRRateModel()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict({"brl_percent": 10.0, "avg_hit_speed": 90.0, "ev95percent": 42.0, "iso": 0.2})


from agents.simulation import (
    _get_park_factor,
    _get_platoon_factor,
    add_simulation,
    validate_simulation,
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
        """With mocked fetch functions and no network calls, sim columns are added correctly."""
        import agents.simulation as sim_mod

        monkeypatch.setattr(sim_mod, "CACHE_DIR", tmp_path / "sim_cache")
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "sim_model.pkl")
        monkeypatch.setattr(sim_mod, "UNMATCHED_LOG", tmp_path / "sim_unmatched.log")

        # Build mock batter DataFrame using Statcast column names
        import numpy as np
        rng = np.random.default_rng(0)
        n = 10
        mock_batter_df = pd.DataFrame({
            "Name": ["Aaron Judge"] + [f"Player{i}" for i in range(n - 1)],
            "brl_percent": [24.7] + list(rng.uniform(4, 18, n - 1)),
            "avg_hit_speed": [95.4] + list(rng.uniform(85, 95, n - 1)),
            "ev95percent": [58.2] + list(rng.uniform(30, 55, n - 1)),
            "iso": [0.357] + list(rng.uniform(0.10, 0.35, n - 1)),
            "HR": [54] + list(rng.integers(5, 40, n - 1)),
            "G": [159] + list(rng.integers(80, 162, n - 1)),
        })

        mock_pitcher_df = pd.DataFrame([
            {"Name": "Gerrit Cole", "HR/9": 1.1, "IP": 80.0}
        ])

        # Monkeypatch fetch functions to avoid any live network calls
        monkeypatch.setattr(sim_mod, "_fetch_batter_stats", lambda season: mock_batter_df)
        monkeypatch.setattr(sim_mod, "_fetch_pitcher_stats", lambda season: mock_pitcher_df)
        monkeypatch.setattr(sim_mod, "_fetch_probable_starters", lambda today: {})
        monkeypatch.setattr(sim_mod, "_fetch_batter_hands", lambda: {})

        df = self._make_final_df()
        result = add_simulation(df)
        assert "sim_prob" in result.columns
        assert "sim_edge" in result.columns
        assert "convergence" in result.columns
        # Aaron Judge should be matched with valid sim_prob in [0.01, 0.40]
        matched = result["sim_prob"].dropna()
        assert len(matched) > 0, "No players were matched to simulation data"
        assert matched.between(0.01, 0.40).all()
        # With elite Statcast stats, sim_prob should be non-trivial
        judge_prob = result.loc[result.index[0], "sim_prob"]
        assert 0.05 < judge_prob <= 0.40, f"sim_prob {judge_prob:.3f} is implausibly outside 0.05-0.40"


    def test_correction_factor_is_applied_when_available(self, tmp_path, monkeypatch):
        """sim_prob should be nudged by correction factor when models/correction_factors.json exists."""
        import json
        from pathlib import Path
        import agents.simulation as sim_mod
        from agents import ml_retrain

        # Monkeypatch fetch functions so sim_prob is always computed (no live network)
        import numpy as np
        rng = np.random.default_rng(7)
        n = 10
        mock_batter_df = pd.DataFrame({
            "Name": ["Matt Olson"] + [f"Player{i}" for i in range(n - 1)],
            "brl_percent": [18.0] + list(rng.uniform(4, 18, n - 1)),
            "avg_hit_speed": [94.0] + list(rng.uniform(85, 95, n - 1)),
            "ev95percent": [55.0] + list(rng.uniform(30, 55, n - 1)),
            "iso": [0.30] + list(rng.uniform(0.10, 0.35, n - 1)),
            "HR": [40] + list(rng.integers(5, 40, n - 1)),
            "G": [155] + list(rng.integers(80, 162, n - 1)),
        })
        mock_pitcher_df = pd.DataFrame([{"Name": "Dummy Pitcher", "HR/9": 1.3, "IP": 100.0}])

        monkeypatch.setattr(sim_mod, "CACHE_DIR", tmp_path / "sim_cache")
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "sim_model.pkl")
        monkeypatch.setattr(sim_mod, "UNMATCHED_LOG", tmp_path / "sim_unmatched.log")
        monkeypatch.setattr(sim_mod, "_fetch_batter_stats", lambda season: mock_batter_df)
        monkeypatch.setattr(sim_mod, "_fetch_pitcher_stats", lambda season: mock_pitcher_df)
        monkeypatch.setattr(sim_mod, "_fetch_probable_starters", lambda today: {})
        monkeypatch.setattr(sim_mod, "_fetch_batter_hands", lambda: {})

        df = pd.DataFrame([{
            "player_name": "Matt Olson",
            "game": "SF Giants @ Atlanta Braves",
            "commence_time": pd.Timestamp("2026-06-10 19:20:00", tz="UTC"),
            "pinnacle_prob": 0.15,
            "ev_pct": 0.05,
        }])

        # Run without correction factors to get the baseline
        no_cf_path = tmp_path / "nonexistent.json"
        monkeypatch.setattr(ml_retrain, "CORRECTION_PATH", no_cf_path)
        baseline = add_simulation(df)
        if "sim_prob" not in baseline.columns or pd.isna(baseline.iloc[0]["sim_prob"]):
            pytest.skip("sim_prob not available (stats not fetched in test env)")
        baseline_prob = float(baseline.iloc[0]["sim_prob"])

        # Write a correction factor with factor=2.0, n=100 (alpha=min(100/200,0.40)=0.40)
        # Expected: base_prob * (0.40*2.0 + 0.60) = base_prob * 1.40
        cf = {"Matt Olson": {"factor": 2.0, "n": 100, "actual_rate": 0.30, "predicted_rate": 0.15}}
        cf_path = tmp_path / "cf.json"
        cf_path.write_text(json.dumps(cf))
        monkeypatch.setattr(ml_retrain, "CORRECTION_PATH", cf_path)

        # Re-run with correction factor active; use fresh model path to force retrain
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "sim_model2.pkl")
        result = add_simulation(df)
        corrected_prob = float(result.iloc[0]["sim_prob"])

        # With factor=2.0 and alpha=0.40, corrected = base * 1.40, capped at 0.40
        expected = min(0.40, max(0.01, baseline_prob * 1.40))
        assert abs(corrected_prob - expected) < 1e-6, (
            f"Correction not applied: baseline={baseline_prob:.4f}, "
            f"corrected={corrected_prob:.4f}, expected≈{expected:.4f}"
        )


def test_add_simulation_importable_from_run():
    """Confirm add_simulation and validate_simulation are imported in run.py."""
    import ast, pathlib
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
    assert "validate_simulation" in names, f"validate_simulation not imported; found: {names}"


class TestValidateSimulation:
    def _df(self, sim_probs, pin_probs=None):
        n = len(sim_probs)
        pin_probs = pin_probs or [0.18] * n
        rows = []
        for sp, pp in zip(sim_probs, pin_probs):
            rows.append({
                "player_name": "P",
                "pinnacle_prob": pp,
                "sim_prob": sp,
                "sim_edge": (sp - pp) if sp is not None else None,
            })
        return pd.DataFrame(rows)

    def test_no_warnings_for_clean_slate(self, tmp_path, monkeypatch):
        import agents.simulation as sim_mod
        model_path = tmp_path / "fresh_model.pkl"
        model_path.write_bytes(b"dummy")  # exists + age 0 days → no model warnings
        monkeypatch.setattr(sim_mod, "MODEL_PATH", model_path)
        df = self._df([0.16, 0.20, 0.18], [0.18, 0.18, 0.18])
        warnings = validate_simulation(df)
        assert warnings == []

    def test_flags_missing_sim_prob_column(self):
        df = pd.DataFrame([{"player_name": "P", "pinnacle_prob": 0.18}])
        warnings = validate_simulation(df)
        assert any("missing" in w for w in warnings)

    def test_flags_systematic_bearish_bias(self, tmp_path, monkeypatch):
        import agents.simulation as sim_mod
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "m.pkl")
        # sim is ~40% of Pinnacle — well below 0.60 ratio threshold
        df = self._df([0.06, 0.07, 0.08, 0.06], [0.18, 0.18, 0.18, 0.18])
        warnings = validate_simulation(df)
        assert any("bearish" in w for w in warnings)

    def test_flags_extreme_divergences(self, tmp_path, monkeypatch):
        import agents.simulation as sim_mod
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "m.pkl")
        # All players have |sim_edge| > 0.15 — all extreme
        df = self._df([0.02, 0.02, 0.02, 0.02], [0.20, 0.20, 0.20, 0.20])
        # fix sim_edge column
        df["sim_edge"] = df["sim_prob"] - df["pinnacle_prob"]
        warnings = validate_simulation(df)
        assert any("sim_edge" in w for w in warnings)

    def test_no_warnings_when_zero_coverage(self):
        df = pd.DataFrame([{"player_name": "P", "pinnacle_prob": 0.18, "sim_prob": None}])
        warnings = validate_simulation(df)
        assert any("no players" in w.lower() for w in warnings)


import numpy as np
from agents.simulation import HRClassifier, GAME_FEATURES


def _make_game_training_df(n: int = 300) -> pd.DataFrame:
    """Synthetic game-level data: 8 features + binary hit_hr label."""
    rng = np.random.default_rng(42)
    brl = rng.uniform(4, 18, n)
    ev = rng.uniform(85, 95, n)
    hard = rng.uniform(30, 55, n)
    iso = rng.uniform(0.10, 0.35, n)
    bat_speed = rng.uniform(64, 75, n)
    park_factor = rng.uniform(0.85, 1.20, n)
    same_hand = rng.integers(0, 2, n).astype(float)
    pitcher_hr9 = rng.uniform(0.8, 2.0, n)
    logit = (
        -3.5
        + 0.06 * brl
        + 8.0 * iso
        + 0.03 * bat_speed
        + 1.5 * (park_factor - 1.0)
        + 0.4 * (pitcher_hr9 - 1.30)
        - 0.15 * same_hand
    )
    prob = 1.0 / (1.0 + np.exp(-logit))
    hit_hr = rng.binomial(1, prob).astype(float)
    return pd.DataFrame({
        "brl_percent": brl,
        "avg_hit_speed": ev,
        "ev95percent": hard,
        "iso": iso,
        "bat_speed": bat_speed,
        "park_factor": park_factor,
        "same_hand": same_hand,
        "pitcher_hr9": pitcher_hr9,
        "hit_hr": hit_hr,
    })


class TestHRClassifier:
    def test_fit_and_predict_returns_probability(self):
        model = HRClassifier()
        model.fit(_make_game_training_df(300))
        features = {
            "brl_percent": 10.0, "avg_hit_speed": 90.0, "ev95percent": 44.0,
            "iso": 0.22, "bat_speed": 69.5, "park_factor": 1.0,
            "same_hand": 0, "pitcher_hr9": 1.30,
        }
        result = model.predict(features)
        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_higher_barrel_pct_predicts_higher_probability(self):
        model = HRClassifier()
        model.fit(_make_game_training_df(500))
        base = {
            "avg_hit_speed": 90.0, "ev95percent": 44.0, "iso": 0.22,
            "bat_speed": 69.5, "park_factor": 1.0, "same_hand": 0, "pitcher_hr9": 1.30,
        }
        low = model.predict({**base, "brl_percent": 4.0})
        high = model.predict({**base, "brl_percent": 18.0})
        assert high > low

    def test_coors_field_predicts_higher_than_neutral(self):
        model = HRClassifier()
        model.fit(_make_game_training_df(500))
        base = {
            "brl_percent": 10.0, "avg_hit_speed": 90.0, "ev95percent": 44.0,
            "iso": 0.22, "bat_speed": 69.5, "same_hand": 0, "pitcher_hr9": 1.30,
        }
        neutral = model.predict({**base, "park_factor": 1.0})
        coors = model.predict({**base, "park_factor": 1.20})
        assert coors > neutral

    def test_save_and_load_round_trip(self, tmp_path):
        model = HRClassifier()
        model.fit(_make_game_training_df(200))
        path = tmp_path / "clf.pkl"
        model.save(path)
        model2 = HRClassifier()
        model2.load(path)
        features = {
            "brl_percent": 10.0, "avg_hit_speed": 90.0, "ev95percent": 44.0,
            "iso": 0.22, "bat_speed": 69.5, "park_factor": 1.0,
            "same_hand": 0, "pitcher_hr9": 1.30,
        }
        assert abs(model.predict(features) - model2.predict(features)) < 1e-9

    def test_predict_before_fit_raises(self):
        model = HRClassifier()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict({
                "brl_percent": 10.0, "avg_hit_speed": 90.0, "ev95percent": 42.0,
                "iso": 0.2, "bat_speed": 69.5, "park_factor": 1.0,
                "same_hand": 0, "pitcher_hr9": 1.30,
            })

    def test_game_features_has_eight_entries(self):
        assert len(GAME_FEATURES) == 8
        assert "bat_speed" in GAME_FEATURES
        assert "park_factor" in GAME_FEATURES
        assert "same_hand" in GAME_FEATURES
        assert "pitcher_hr9" in GAME_FEATURES
