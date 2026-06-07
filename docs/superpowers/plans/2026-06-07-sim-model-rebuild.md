# Simulation Model Rebuild — Game-Level Logistic Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the ad-hoc multiplier-based HR simulation model with a logistic regression trained on 2022–2025 game-level Statcast data where all coefficients are learned from actual binary HR outcomes.

**Architecture:** `HRClassifier` (LogisticRegression) replaces `HRRateModel` (Ridge). Training data comes from a one-time build script that pulls Statcast pitch-level data, aggregates to player-game rows, and joins all 8 features by MLBAM numeric ID. Daily inference assembles the same 8-feature vector from daily-cached stats and calls `model.predict_proba()`. No post-hoc multipliers. No `apply_correction()`.

**Tech Stack:** pybaseball, scikit-learn (LogisticRegression, StandardScaler, Pipeline), pandas, pickle, pytest

**Spec:** `docs/superpowers/specs/2026-06-07-sim-model-rebuild-design.md`

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `agents/simulation.py` | Modify | HRClassifier class, updated constants, updated _get_or_train_model(), updated add_simulation() |
| `agents/sim_build_training_data.py` | Create | One-time Statcast pull and aggregation script |
| `tests/test_simulation.py` | Modify | HRClassifier tests, updated add_simulation test, remove old multiplier tests |
| `tests/test_sim_build_training_data.py` | Create | Unit tests for training data builder helpers |

**Public interface unchanged:** `add_simulation(df)` and `validate_simulation(df)` signatures stay identical. `run.py` needs no changes.

---

## Task 1: Add `HRClassifier` Class + `GAME_FEATURES` Constant

**Files:**
- Modify: `agents/simulation.py` — add after line 405 (end of HRRateModel class)
- Modify: `tests/test_simulation.py` — add new test class

---

- [ ] **Step 1.1: Write the failing tests for HRClassifier**

Add this class at the bottom of `tests/test_simulation.py`, after the existing `TestHRRateModel` class:

```python
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
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```
pytest tests/test_simulation.py::TestHRClassifier -v
```

Expected: `ImportError` — `HRClassifier` and `GAME_FEATURES` not yet defined.

- [ ] **Step 1.3: Add `GAME_FEATURES` constant and `HRClassifier` class to `simulation.py`**

In `agents/simulation.py`, make these changes:

**1a. Change the Ridge import to LogisticRegression** (line 23):
```python
# Replace:
from sklearn.linear_model import Ridge
# With:
from sklearn.linear_model import LogisticRegression
```

**1b. Add two new constants** after `BATTER_FEATURES = [...]` (after line 35):
```python
BATTER_FEATURES = ["brl_percent", "avg_hit_speed", "ev95percent", "iso"]
GAME_FEATURES = [
    "brl_percent", "avg_hit_speed", "ev95percent", "iso",
    "bat_speed", "park_factor", "same_hand", "pitcher_hr9",
]
LEAGUE_MEAN_BAT_SPEED = 68.9   # mph — 2024+ Statcast average on all swings
BAT_SPEED_PATH = Path("data/batter_bat_speed.parquet")
TRAINING_CACHE_PATH = Path("data/sim_training_cache.parquet")
MODEL_MAX_AGE_DAYS = 30
```

**1c. Add `HRClassifier` class** immediately after the closing of `HRRateModel` (after line 405):
```python
class HRClassifier:
    """
    Logistic regression classifier predicting P(hit_hr=1) from 8 game-level features.
    Trained on binary player-game Statcast outcomes (2022-2025).

    Features (GAME_FEATURES):
        brl_percent, avg_hit_speed, ev95percent, iso  — batter season contact quality
        bat_speed    — batter average bat speed (league mean when pre-2024)
        park_factor  — home stadium HR factor (1.0 = neutral)
        same_hand    — 1 if batter/pitcher same handedness, 0 if opposite
        pitcher_hr9  — opposing starter's season HR/9
    """

    def __init__(self) -> None:
        self._pipe: Pipeline | None = None

    def fit(self, df: pd.DataFrame) -> None:
        train = df.dropna(subset=GAME_FEATURES + ["hit_hr"])
        X = train[GAME_FEATURES]
        y = train["hit_hr"].astype(int)
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
        ])
        self._pipe.fit(X, y)

    def predict(self, features: dict) -> float:
        """Return P(hit_hr=1) as a float in (0, 1)."""
        if self._pipe is None:
            raise RuntimeError("HRClassifier is not fitted. Call fit() or load() first.")
        X = pd.DataFrame([features])[GAME_FEATURES].fillna(0.0)
        return float(self._pipe.predict_proba(X)[0, 1])

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipe, f)

    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            self._pipe = pickle.load(f)
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```
pytest tests/test_simulation.py::TestHRClassifier -v
```

Expected: 6 tests PASS. All existing tests should still pass too:

```
pytest tests/test_simulation.py -v
```

Expected: All tests pass (HRRateModel tests still run — it hasn't been removed yet).

- [ ] **Step 1.5: Commit**

```
git add agents/simulation.py tests/test_simulation.py
git commit -m "feat(sim): add HRClassifier and GAME_FEATURES — game-level logistic regression"
```

---

## Task 2: Create `agents/sim_build_training_data.py`

**Files:**
- Create: `agents/sim_build_training_data.py`
- Create: `tests/test_sim_build_training_data.py`

This is a standalone one-time script. It does NOT run during the daily dashboard. Run it once manually after implementation is complete (`python -m agents.sim_build_training_data`). It takes ~2 hours.

---

- [ ] **Step 2.1: Write the failing tests**

Create `tests/test_sim_build_training_data.py`:

```python
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
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```
pytest tests/test_sim_build_training_data.py -v
```

Expected: `ModuleNotFoundError: No module named 'agents.sim_build_training_data'`

- [ ] **Step 2.3: Create `agents/sim_build_training_data.py`**

```python
"""
One-time training data builder for the game-level HR logistic regression model.

Usage:
    python -m agents.sim_build_training_data

Takes ~2 hours to pull Statcast for 2022-2025.
Checkpoints by season — safe to re-run after interruption.

Outputs:
    data/sim_training_cache.parquet   — player-game rows with all 8 GAME_FEATURES + hit_hr
    data/batter_bat_speed.parquet     — (player_id, season, Name, avg_bat_speed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from agents.simulation import (
    PITCHER_LEAGUE_HR9,
    TEAM_NAME_TO_ABBREV,
    LEAGUE_MEAN_BAT_SPEED,
    GAME_FEATURES,
    _normalize_team_abbrev,
    _reverse_statcast_name,
)

TRAINING_SEASONS = [2022, 2023, 2024, 2025]
CACHE_PATH = Path("data/sim_training_cache.parquet")
BAT_SPEED_PATH = Path("data/batter_bat_speed.parquet")
PARK_FACTORS_PATH = Path("data/park_factors.json")
CHECKPOINT_DIR = Path("data/sim_cache")

_STATCAST_BASE_COLS = [
    "batter", "pitcher", "game_pk", "game_date",
    "home_team", "p_throws", "stand", "events",
]


def _aggregate_to_player_game(sc: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Statcast pitch rows to one row per (batter, game_pk).

    Returns DataFrame with columns:
        player_id, game_pk, game_date, hit_hr, home_team,
        stand, p_throws, opp_pitcher_id, same_hand, bat_speed_mean
    """
    has_bat_speed = "bat_speed" in sc.columns

    def _agg(g: pd.DataFrame) -> pd.Series:
        opp_pitcher = g["pitcher"].mode()
        return pd.Series({
            "hit_hr": int((g["events"] == "home_run").any()),
            "home_team": g["home_team"].iloc[0],
            "stand": g["stand"].mode()[0] if not g["stand"].isna().all() else "",
            "p_throws": g["p_throws"].mode()[0] if not g["p_throws"].isna().all() else "",
            "opp_pitcher_id": int(opp_pitcher[0]) if len(opp_pitcher) > 0 else -1,
            "bat_speed_mean": float(g["bat_speed"].mean())
            if has_bat_speed and g["bat_speed"].notna().any()
            else float("nan"),
        })

    pg = sc.groupby(["batter", "game_pk", "game_date"], group_keys=False).apply(_agg).reset_index()
    pg = pg.rename(columns={"batter": "player_id"})
    pg["same_hand"] = (pg["stand"] == pg["p_throws"]).astype(int)
    return pg


def _fetch_batter_season_stats(season: int) -> pd.DataFrame:
    """Fetch (player_id, Name, brl_percent, avg_hit_speed, ev95percent, iso) for a season."""
    import pybaseball

    ev_df = pybaseball.statcast_batter_exitvelo_barrels(season, minBBE=1)
    xs_df = pybaseball.statcast_batter_expected_stats(season, minPA=1)

    ev_cols = ["player_id", "last_name, first_name", "brl_percent", "avg_hit_speed", "ev95percent"]
    xs_cols = ["player_id", "ba", "slg"]

    ev_df = ev_df[[c for c in ev_cols if c in ev_df.columns]]
    xs_df = xs_df[[c for c in xs_cols if c in xs_df.columns]]

    merged = ev_df.merge(xs_df, on="player_id", how="inner")
    merged["iso"] = merged["slg"] - merged["ba"]
    merged["Name"] = merged["last_name, first_name"].apply(_reverse_statcast_name)
    return merged[["player_id", "Name", "brl_percent", "avg_hit_speed", "ev95percent", "iso"]].copy()


def _fetch_pitcher_season_stats(season: int) -> pd.DataFrame:
    """Fetch (player_id, pitcher_hr9) for a season via baseball-reference."""
    import pybaseball

    df = pybaseball.pitching_stats_bref(season)
    if df.empty or "HR" not in df.columns or "IP" not in df.columns:
        return pd.DataFrame(columns=["player_id", "pitcher_hr9"])

    df = df.copy()
    df["pitcher_hr9"] = df.apply(
        lambda r: float(r["HR"]) / (float(r["IP"]) / 9.0)
        if pd.notna(r["IP"]) and float(r["IP"]) > 0 and pd.notna(r["HR"])
        else PITCHER_LEAGUE_HR9,
        axis=1,
    )
    if "mlbID" not in df.columns:
        return pd.DataFrame(columns=["player_id", "pitcher_hr9"])

    df = df.rename(columns={"mlbID": "player_id"})
    return df[["player_id", "pitcher_hr9"]].dropna(subset=["player_id"]).copy()


def _build_season(season: int, park_factors: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pull Statcast for `season`, aggregate to player-game level, join all features.

    Checkpoints to CHECKPOINT_DIR/training_{season}.parquet so re-runs are fast.

    Returns:
        (player_game_df, bat_speed_sidecar_df)
        player_game_df columns: GAME_FEATURES + [hit_hr, season, player_id, game_pk, game_date]
        bat_speed_sidecar_df columns: [player_id, season, Name, avg_bat_speed]
    """
    checkpoint = CHECKPOINT_DIR / f"training_{season}.parquet"
    bs_checkpoint = CHECKPOINT_DIR / f"bat_speed_{season}.parquet"

    if checkpoint.exists() and bs_checkpoint.exists():
        print(f"  [season {season}] Loading checkpoint...")
        return pd.read_parquet(checkpoint), pd.read_parquet(bs_checkpoint)

    import pybaseball  # only imported when a live pull is needed

    print(f"  [season {season}] Pulling Statcast for {season} (~20-30 min)...", flush=True)
    sc = pybaseball.statcast(f"{season}-03-01", f"{season}-11-30", verbose=False)

    keep_cols = _STATCAST_BASE_COLS[:]
    if "bat_speed" in sc.columns:
        keep_cols.append("bat_speed")
    sc = sc[[c for c in keep_cols if c in sc.columns]].copy()
    if "bat_speed" not in sc.columns:
        sc["bat_speed"] = float("nan")

    # Bat speed sidecar: per-batter season mean
    bs_raw = (
        sc.groupby("batter")["bat_speed"]
        .mean()
        .reset_index()
        .rename(columns={"batter": "player_id", "bat_speed": "avg_bat_speed"})
    )
    bs_raw["season"] = season

    # Aggregate to player-game
    print(f"  [season {season}] Aggregating to player-game level...")
    pg = _aggregate_to_player_game(sc)

    # Join batter season stats (by MLBAM numeric ID — no name matching)
    print(f"  [season {season}] Joining batter season stats...")
    batter_stats = _fetch_batter_season_stats(season)
    pg = pg.merge(
        batter_stats[["player_id", "Name", "brl_percent", "avg_hit_speed", "ev95percent", "iso"]],
        on="player_id",
        how="inner",
    )

    # Bat speed sidecar with Names for inference-time lookup
    bs_df = bs_raw.merge(batter_stats[["player_id", "Name"]], on="player_id", how="left")

    # Bat speed on player-game rows: use per-batter season mean (consistent with inference)
    bs_lookup = dict(zip(bs_raw["player_id"], bs_raw["avg_bat_speed"]))
    pg["bat_speed"] = pg["player_id"].map(bs_lookup).fillna(LEAGUE_MEAN_BAT_SPEED)

    # Join pitcher season stats (by MLBAM numeric ID)
    print(f"  [season {season}] Joining pitcher season stats...")
    pitcher_stats = _fetch_pitcher_season_stats(season)
    if not pitcher_stats.empty:
        pitcher_stats = pitcher_stats.rename(columns={"player_id": "opp_pitcher_id"})
        pitcher_stats["opp_pitcher_id"] = pitcher_stats["opp_pitcher_id"].astype("Int64")
        pg["opp_pitcher_id"] = pg["opp_pitcher_id"].astype("Int64")
        pg = pg.merge(pitcher_stats, on="opp_pitcher_id", how="left")
    if "pitcher_hr9" not in pg.columns:
        pg["pitcher_hr9"] = PITCHER_LEAGUE_HR9
    pg["pitcher_hr9"] = pg["pitcher_hr9"].fillna(PITCHER_LEAGUE_HR9)

    # Park factor
    def _park_factor(team_abbrev: str) -> float:
        normalized = _normalize_team_abbrev(str(team_abbrev))
        return park_factors.get(normalized, park_factors.get(str(team_abbrev), 1.0))

    pg["park_factor"] = pg["home_team"].apply(_park_factor)
    pg["season"] = season

    # Drop rows with any missing model feature
    pg = pg.dropna(subset=GAME_FEATURES + ["hit_hr"])

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    pg.to_parquet(checkpoint, index=False)
    bs_df.to_parquet(bs_checkpoint, index=False)
    print(f"  [season {season}] Done — {len(pg):,} player-game rows. Checkpoint saved.")
    return pg, bs_df


def main() -> None:
    print("Building game-level HR simulation training data (2022-2025)...")
    print("Expected runtime: ~2 hours. Safe to re-run — checkpoints by season.\n")

    if not PARK_FACTORS_PATH.exists():
        print(f"ERROR: {PARK_FACTORS_PATH} not found.")
        sys.exit(1)

    with PARK_FACTORS_PATH.open(encoding="utf-8") as f:
        park_factors = json.load(f)

    all_pg: list[pd.DataFrame] = []
    all_bs: list[pd.DataFrame] = []

    for season in TRAINING_SEASONS:
        print(f"\n=== Season {season} ===")
        pg_df, bs_df = _build_season(season, park_factors)
        if not pg_df.empty:
            all_pg.append(pg_df)
        if not bs_df.empty:
            all_bs.append(bs_df)

    if not all_pg:
        print("ERROR: No training data built. Check pybaseball connectivity.")
        sys.exit(1)

    training_df = pd.concat(all_pg, ignore_index=True)
    bat_speed_df = pd.concat(all_bs, ignore_index=True)

    training_df.to_parquet(CACHE_PATH, index=False)
    bat_speed_df.to_parquet(BAT_SPEED_PATH, index=False)

    n = len(training_df)
    hr_rate = float(training_df["hit_hr"].mean())
    print(f"\n=== Done ===")
    print(f"{n:,} player-game rows across {len(TRAINING_SEASONS)} seasons")
    print(f"Base HR rate: {hr_rate:.3f} ({hr_rate*100:.1f}% of games)")
    print(f"Training cache: {CACHE_PATH}")
    print(f"Bat speed sidecar: {BAT_SPEED_PATH}")
    print(f"\nNext step: delete data/sim_model.pkl if it exists, then run the dashboard.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```
pytest tests/test_sim_build_training_data.py -v
```

Expected: All tests PASS. The `TestBuildSeasonUsesCheckpoint` test confirms checkpoint loading works without hitting the network.

- [ ] **Step 2.5: Commit**

```
git add agents/sim_build_training_data.py tests/test_sim_build_training_data.py
git commit -m "feat(sim): add one-time training data builder (Statcast game-level aggregation)"
```

---

## Task 3: Update `_get_or_train_model()` to Use Parquet Cache

**Files:**
- Modify: `agents/simulation.py` — replace `_get_or_train_model(batter_dfs)` with `_get_or_train_model()`
- Modify: `tests/test_simulation.py` — add test for new function signature

---

- [ ] **Step 3.1: Write the failing test**

Add this test to `tests/test_simulation.py` after the existing `TestHRClassifier` class:

```python
class TestGetOrTrainModel:
    def test_loads_from_pkl_when_fresh(self, tmp_path, monkeypatch):
        import agents.simulation as sim_mod
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "model.pkl")

        # Save a trained classifier to the pkl path
        clf = HRClassifier()
        clf.fit(_make_game_training_df(100))
        clf.save(tmp_path / "model.pkl")

        result = sim_mod._get_or_train_model()
        assert isinstance(result, HRClassifier)

    def test_trains_from_parquet_when_pkl_missing(self, tmp_path, monkeypatch):
        import agents.simulation as sim_mod
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "model.pkl")
        monkeypatch.setattr(sim_mod, "TRAINING_CACHE_PATH", tmp_path / "cache.parquet")

        # Write a small training cache
        cache = _make_game_training_df(200)
        cache.to_parquet(tmp_path / "cache.parquet", index=False)

        result = sim_mod._get_or_train_model()
        assert isinstance(result, HRClassifier)
        assert (tmp_path / "model.pkl").exists()

    def test_raises_when_both_pkl_and_cache_missing(self, tmp_path, monkeypatch):
        import agents.simulation as sim_mod
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "model.pkl")
        monkeypatch.setattr(sim_mod, "TRAINING_CACHE_PATH", tmp_path / "cache.parquet")

        with pytest.raises(RuntimeError, match="sim_build_training_data"):
            sim_mod._get_or_train_model()
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```
pytest tests/test_simulation.py::TestGetOrTrainModel -v
```

Expected: `TypeError` — `_get_or_train_model()` currently requires `batter_dfs` argument.

- [ ] **Step 3.3: Replace `_get_or_train_model()` in `simulation.py`**

Find the existing `_get_or_train_model(batter_dfs: dict[int, pd.DataFrame]) -> HRRateModel:` function (line ~408) and replace the entire function body with:

```python
def _get_or_train_model() -> HRClassifier:
    """
    Load HRClassifier from data/sim_model.pkl if < MODEL_MAX_AGE_DAYS old.
    Otherwise load data/sim_training_cache.parquet and train a new classifier.
    If neither pkl nor cache exists, raises RuntimeError (caught by add_simulation).
    """
    model = HRClassifier()

    if MODEL_PATH.exists():
        age_days = (
            datetime.now() - datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
        ).days
        if age_days < MODEL_MAX_AGE_DAYS:
            model.load(MODEL_PATH)
            return model

    if not TRAINING_CACHE_PATH.exists():
        raise RuntimeError(
            f"[simulation] Training cache not found at {TRAINING_CACHE_PATH}. "
            "Run: python -m agents.sim_build_training_data"
        )

    print(f"[simulation] Training HRClassifier on game-level Statcast data...")
    train_df = pd.read_parquet(TRAINING_CACHE_PATH)
    model.fit(train_df)
    n = len(train_df.dropna(subset=GAME_FEATURES + ["hit_hr"]))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[simulation] Model trained on {n:,} player-game rows, saved to {MODEL_PATH}.")
    return model
```

Also update `validate_simulation()` — change the model age check (currently checks `age_days >= 7`) to:

```python
    if MODEL_PATH.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)).days
        if age_days >= MODEL_MAX_AGE_DAYS:
            warnings.append(
                f"[sim-validate] Model is {age_days} days old (refresh threshold: {MODEL_MAX_AGE_DAYS}d) — "
                "run 'python -m agents.sim_build_training_data' then delete data/sim_model.pkl to retrain"
            )
    else:
        warnings.append(
            "[sim-validate] No trained model found — "
            "run 'python -m agents.sim_build_training_data' then run the dashboard"
        )
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```
pytest tests/test_simulation.py::TestGetOrTrainModel -v
```

Expected: All 3 tests PASS.

```
pytest tests/test_simulation.py -v
```

Expected: All existing tests still pass. (The `test_adds_sim_columns_when_data_available` test may fail if it still calls `_get_or_train_model(batter_dfs)` — that will be fixed in Task 4.)

- [ ] **Step 3.5: Commit**

```
git add agents/simulation.py tests/test_simulation.py
git commit -m "feat(sim): update _get_or_train_model to load from parquet cache and use HRClassifier"
```

---

## Task 4: Update `add_simulation()` — 8-Feature Vector Assembly, Remove Multipliers

**Files:**
- Modify: `agents/simulation.py` — update `add_simulation()`, add `_load_bat_speed_lookup()`, rename `_get_pitcher_factor()` to `_get_pitcher_info()`
- Modify: `tests/test_simulation.py` — update `TestAddSimulation`

---

- [ ] **Step 4.1: Write the failing test for the new add_simulation interface**

In `tests/test_simulation.py`, find the existing `TestAddSimulation` class and **replace** its `test_adds_sim_columns_when_data_available` method with:

```python
    def test_adds_sim_columns_when_data_available(self, tmp_path, monkeypatch):
        """With mocked fetch functions and a pre-trained model, sim columns are added."""
        import agents.simulation as sim_mod

        monkeypatch.setattr(sim_mod, "CACHE_DIR", tmp_path / "sim_cache")
        monkeypatch.setattr(sim_mod, "MODEL_PATH", tmp_path / "sim_model.pkl")
        monkeypatch.setattr(sim_mod, "UNMATCHED_LOG", tmp_path / "sim_unmatched.log")

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
        mock_pitcher_df = pd.DataFrame([{"Name": "Gerrit Cole", "HR/9": 1.1, "IP": 80.0}])

        monkeypatch.setattr(sim_mod, "_fetch_batter_stats", lambda season: mock_batter_df)
        monkeypatch.setattr(sim_mod, "_fetch_pitcher_stats", lambda season: mock_pitcher_df)
        monkeypatch.setattr(sim_mod, "_fetch_probable_starters", lambda today: {})
        monkeypatch.setattr(sim_mod, "_fetch_batter_hands", lambda: {})
        monkeypatch.setattr(sim_mod, "_load_bat_speed_lookup", lambda: {"Aaron Judge": 74.2})

        # Pre-train a classifier so add_simulation doesn't need the parquet cache
        clf = HRClassifier()
        clf.fit(_make_game_training_df(200))
        monkeypatch.setattr(sim_mod, "_get_or_train_model", lambda: clf)

        df = self._make_final_df()
        result = add_simulation(df)
        assert "sim_prob" in result.columns
        assert "sim_edge" in result.columns
        assert "convergence" in result.columns
        matched = result["sim_prob"].dropna()
        assert len(matched) > 0
        assert matched.between(0.01, 0.35).all()
        judge_prob = result.loc[result.index[0], "sim_prob"]
        assert 0.05 < judge_prob <= 0.35, f"sim_prob {judge_prob:.3f} is outside 0.05-0.35"
```

Also **delete** the entire `test_correction_factor_is_applied_when_available` test method — `apply_correction()` is being removed.

- [ ] **Step 4.2: Run test to confirm it fails**

```
pytest tests/test_simulation.py::TestAddSimulation::test_adds_sim_columns_when_data_available -v
```

Expected: `AttributeError` — `_load_bat_speed_lookup` not yet defined on `sim_mod`.

- [ ] **Step 4.3: Add `_load_bat_speed_lookup()` helper to `simulation.py`**

Add this function after `_fetch_batter_hands()` (after line ~552):

```python
def _load_bat_speed_lookup() -> dict[str, float]:
    """
    Load {normalized_player_name: avg_bat_speed} from data/batter_bat_speed.parquet.
    Uses the most recent season's value per player. Returns empty dict if file missing.
    """
    if not BAT_SPEED_PATH.exists():
        return {}
    try:
        bs = pd.read_parquet(BAT_SPEED_PATH)
        if bs.empty or "Name" not in bs.columns or "avg_bat_speed" not in bs.columns:
            return {}
        latest = bs.sort_values("season").groupby("player_id").last().reset_index()
        return {
            _normalize_name(str(n)): float(s)
            for n, s in zip(latest["Name"], latest["avg_bat_speed"])
            if pd.notna(s) and pd.notna(n)
        }
    except Exception as exc:
        logger.warning("[simulation] Could not load bat speed sidecar: %s", exc)
        return {}
```

- [ ] **Step 4.4: Add `_get_pitcher_info()` helper to `simulation.py`**

Add this function immediately after `_get_pitcher_factor()`. The existing `_get_pitcher_factor()` function stays for now (removed in Task 5); we add the new cleaner version alongside it:

```python
def _get_pitcher_info(
    row: pd.Series,
    starters: dict,
    pitcher_dfs: dict[int, pd.DataFrame],
) -> tuple[str, str, float]:
    """
    Returns (pitcher_name, pitcher_hand, pitcher_hr9) for the opposing starter.
    Defaults to ("", "", PITCHER_LEAGUE_HR9) when data is unavailable.
    """
    game = str(row.get("game", ""))
    batter_team = str(row.get("team", ""))

    if " @ " not in game or not batter_team:
        return "", "", PITCHER_LEAGUE_HR9

    away_name, home_name = game.split(" @ ", 1)
    away_abbrev = TEAM_NAME_TO_ABBREV.get(away_name.strip(), "")
    home_abbrev = TEAM_NAME_TO_ABBREV.get(home_name.strip(), "")
    batter_team_norm = _normalize_team_abbrev(batter_team)

    if batter_team_norm == home_abbrev:
        opposing_abbrev = away_abbrev
    elif batter_team_norm == away_abbrev:
        opposing_abbrev = home_abbrev
    else:
        return "", "", PITCHER_LEAGUE_HR9

    starter_info = starters.get(opposing_abbrev, {})
    pitcher_name = starter_info.get("name", "")
    pitcher_hand = starter_info.get("hand", "")

    if not pitcher_name:
        return pitcher_name, pitcher_hand, PITCHER_LEAGUE_HR9

    pitcher_stats = _get_pitcher_stats_by_name(pitcher_name, pitcher_dfs)
    if pitcher_stats is None or pitcher_stats.get("IP", 0) < 5:
        return pitcher_name, pitcher_hand, PITCHER_LEAGUE_HR9

    return pitcher_name, pitcher_hand, pitcher_stats.get("HR/9", PITCHER_LEAGUE_HR9)
```

- [ ] **Step 4.5: Replace `add_simulation()` body in `simulation.py`**

Find the existing `add_simulation()` function (line ~619) and replace its entire body with:

```python
def add_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append simulation columns to final_df and return it.

    Added columns:
        sim_prob    — HRClassifier P(HR today), clipped to [0.01, 0.35]
        sim_edge    — sim_prob - pinnacle_prob (positive = sim more bullish)
        convergence — "AGREE" if |sim_edge| < 0.03, else "DIVERGE"

    Returns df unchanged if training cache is missing or any error occurs.
    """
    try:
        import pybaseball  # noqa: F401 — verify available
    except (ImportError, TypeError):
        logger.warning("[simulation] pybaseball not available — skipping simulation.")
        return df

    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        all_seasons = [2024, 2025, 2026]
        batter_dfs = {s: _fetch_batter_stats(s) for s in all_seasons}
        pitcher_dfs = {s: _fetch_pitcher_stats(s) for s in all_seasons}

        starters = _fetch_probable_starters(today)
        batter_hands = _fetch_batter_hands()
        bat_speed_lookup = _load_bat_speed_lookup()

        with PARK_FACTORS_PATH.open(encoding="utf-8") as f:
            park_factors = json.load(f)

        model = _get_or_train_model()

        sim_probs: list[float | None] = []
        for _, row in df.iterrows():
            stats = _get_weighted_batter_stats(row["player_name"], batter_dfs)
            if stats is None:
                sim_probs.append(None)
                continue

            norm_name = _normalize_name(row["player_name"])

            park_factor = _get_park_factor(str(row.get("game", "")), park_factors)
            _pitcher_name, pitcher_hand, pitcher_hr9 = _get_pitcher_info(
                row, starters, pitcher_dfs
            )
            batter_hand = batter_hands.get(norm_name, "")
            same_hand = (
                1
                if batter_hand and pitcher_hand and batter_hand not in ("", "S")
                and batter_hand == pitcher_hand
                else 0
            )
            bat_speed = bat_speed_lookup.get(norm_name, LEAGUE_MEAN_BAT_SPEED)

            features = {
                **stats,  # brl_percent, avg_hit_speed, ev95percent, iso
                "bat_speed": bat_speed,
                "park_factor": park_factor,
                "same_hand": same_hand,
                "pitcher_hr9": pitcher_hr9,
            }

            sim_prob = model.predict(features)
            sim_prob = max(0.01, min(0.35, sim_prob))
            sim_probs.append(sim_prob)

        df = df.copy()
        df["sim_prob"] = sim_probs
        df["sim_edge"] = df.apply(
            lambda r: (r["sim_prob"] - r["pinnacle_prob"]) if pd.notna(r["sim_prob"]) else None,
            axis=1,
        )
        df["convergence"] = df["sim_edge"].apply(
            lambda e: "AGREE" if pd.notna(e) and abs(e) < 0.03 else "DIVERGE"
        )

        matched = df["sim_prob"].notna().sum()
        total = len(df)
        if total > 0:
            print(
                f"[simulation] Matched {matched}/{total} players "
                f"({matched/total*100:.0f}% match rate). "
                f"See {UNMATCHED_LOG} for misses."
            )
        return df

    except Exception as exc:
        logger.exception("[simulation] Unexpected error — returning df unchanged.")
        print(f"[simulation] WARNING: {exc}. Dashboard will show simulation as unavailable.")
        return df
```

Also remove the `from agents.ml_retrain import apply_correction` import (line 27) since `apply_correction()` is no longer called.

- [ ] **Step 4.6: Run all tests**

```
pytest tests/test_simulation.py -v
```

Expected: All tests pass. The updated `test_adds_sim_columns_when_data_available` should PASS. The `test_returns_df_unchanged_on_import_error` and `test_add_simulation_importable_from_run` should still pass.

- [ ] **Step 4.7: Commit**

```
git add agents/simulation.py tests/test_simulation.py
git commit -m "feat(sim): update add_simulation to use 8-feature vector — remove arbitrary multipliers"
```

---

## Task 5: Remove Old Code and Tests

**Files:**
- Modify: `agents/simulation.py` — remove HRRateModel, _get_pitcher_factor, _get_platoon_factor, TRAIN_SEASONS
- Modify: `tests/test_simulation.py` — remove TestHRRateModel, TestGetPlatoonFactor

---

- [ ] **Step 5.1: Delete `HRRateModel` class from `simulation.py`**

Remove the entire `HRRateModel` class (lines ~361–405). It spans from `class HRRateModel:` through the closing of the `load()` method.

- [ ] **Step 5.2: Remove `_get_pitcher_factor()` and `_get_platoon_factor()` from `simulation.py`**

Remove the entire `_get_pitcher_factor()` function (was used to return a ratio multiplier — replaced by `_get_pitcher_info()`).

Remove the entire `_get_platoon_factor()` function (was returning ±5% multiplier — replaced by inline `same_hand` integer in `add_simulation()`).

- [ ] **Step 5.3: Remove `TRAIN_SEASONS` constant from `simulation.py`**

Delete the line:
```python
TRAIN_SEASONS = [2024, 2025]
```

- [ ] **Step 5.4: Remove old tests for deleted code**

In `tests/test_simulation.py`, delete these entire test classes/functions:
- `class TestHRRateModel` — all methods
- `class TestGetPlatoonFactor` — all methods
- The `_make_training_df()` helper function (was used only by TestHRRateModel)

Also delete the now-stale import at the top of the test file:
```python
from agents.simulation import HRRateModel, BATTER_FEATURES
```

Replace it with:
```python
from agents.simulation import HRClassifier, GAME_FEATURES, BATTER_FEATURES
```

(Keep `BATTER_FEATURES` imported since `TestGetWeightedBatterStats` still uses it.)

Also remove `_get_park_factor` and `_get_platoon_factor` from the import block near the bottom of the test file:
```python
# Old:
from agents.simulation import (
    _get_park_factor,
    _get_platoon_factor,
    add_simulation,
    validate_simulation,
)

# New:
from agents.simulation import (
    _get_park_factor,
    add_simulation,
    validate_simulation,
)
```

- [ ] **Step 5.5: Run the full test suite**

```
pytest tests/test_simulation.py tests/test_sim_build_training_data.py -v
```

Expected: All tests PASS. Also run the full suite to check nothing else broke:

```
pytest -v
```

Expected: All tests pass (same count as before minus the deleted test classes).

- [ ] **Step 5.6: Commit**

```
git add agents/simulation.py tests/test_simulation.py
git commit -m "refactor(sim): remove HRRateModel, multiplier functions, and apply_correction — cleanup after rebuild"
```

---

## Post-Implementation: Build Training Data

After all 5 tasks are complete and tests pass, the model needs its training data before it can produce predictions.

- [ ] **Step 6.1: Run the training data builder**

```
python -m agents.sim_build_training_data
```

**Expected output:**
```
Building game-level HR simulation training data (2022-2025)...
Expected runtime: ~2 hours. Safe to re-run — checkpoints by season.

=== Season 2022 ===
  [season 2022] Pulling Statcast for 2022 (~20-30 min)...
  [season 2022] Aggregating to player-game level...
  [season 2022] Joining batter season stats...
  [season 2022] Joining pitcher season stats...
  [season 2022] Done — 94,xxx player-game rows. Checkpoint saved.

[... seasons 2023-2025 ...]

=== Done ===
~400,000 player-game rows across 4 seasons
Base HR rate: 0.111 (11.1% of games)
Training cache: data/sim_training_cache.parquet
Bat speed sidecar: data/batter_bat_speed.parquet
```

- [ ] **Step 6.2: Delete any stale model pkl and verify auto-train**

```
del data\sim_model.pkl
python run.py
```

On first run after cache is built, you'll see:
```
[simulation] Training HRClassifier on game-level Statcast data...
[simulation] Model trained on ~400,000 player-game rows, saved to data/sim_model.pkl.
[simulation] Matched XX/YY players (ZZ% match rate).
```

Subsequent runs load from `data/sim_model.pkl` in < 1 second.

- [ ] **Step 6.3: Sanity check the output**

Run the dashboard and verify:
- sim_prob values are in [0.01, 0.35] range
- Coors Field games show slightly higher sim_prob than neutral parks
- Players with high barrel rates (Judge, Alonso) have higher sim_prob than contact hitters
- `[sim-validate]` warnings are empty or show only expected messages
- Coverage (matched players) is ≥ 75% of the slate
