# Agent 6 — HR Simulation Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `agents/simulation.py` with a Ridge-regression game-level HR probability model that appends `sim_prob`, `sim_edge`, and `convergence` columns to `final_df`, plus a new Simulation Analysis section in the dashboard.

**Architecture:** Logistic shape is wrong for a continuous label (`hr_per_game` ∈ [0, 0.15]); we use `sklearn.linear_model.Ridge` (L2 regularized linear regression) trained on FanGraphs 2024–2025 batter contact stats. Post-prediction multipliers (park × pitcher × platoon) adjust the base prediction daily. The public interface is a single function `add_simulation(df) → df` inserted into `run.py` after `calculate_ev()`.

**Tech Stack:** Python, pybaseball, scikit-learn (Ridge, StandardScaler, Pipeline), rapidfuzz, requests (MLB Stats API), pandas, pickle

---

## File Map

| File | Action | Purpose |
|------|---------|---------|
| `agents/simulation.py` | **Create** | All simulation logic; public: `add_simulation(df)` |
| `data/park_factors.json` | **Create** | Hardcoded 30-park HR factors keyed by team abbreviation |
| `tests/test_simulation.py` | **Create** | Unit tests for all public and private functions |
| `run.py` | **Modify** | Import `add_simulation`; call after `calculate_ev()` |
| `dashboard/generator.py` | **Modify** | Add sim columns to `META_COLS`; add Simulation Analysis section |
| `.gitignore` | **Modify** | Add `data/sim_cache/`, `data/sim_model.pkl`, `data/sim_unmatched.log` |

---

## Task 1: Setup — Park Factors, .gitignore, Dependency Check

**Files:**
- Create: `data/park_factors.json`
- Modify: `.gitignore`

- [ ] **Step 1.1: Verify pybaseball, scikit-learn, rapidfuzz are installed**

```bash
python -c "import pybaseball, sklearn, rapidfuzz; print('OK')"
```

Expected output: `OK`. If any fail, install:
```bash
pip install pybaseball scikit-learn rapidfuzz
```

- [ ] **Step 1.2: Create `data/park_factors.json`**

Create `data/park_factors.json` with exact content below (FanGraphs 2025 HR park factors, centered at 1.0):

```json
{
  "_comment": "FanGraphs HR park factors 2025. 1.0 = league-neutral. Source: FanGraphs Park Factors, updated once per season.",
  "COL": 1.198,
  "CIN": 1.135,
  "NYY": 1.112,
  "TEX": 1.095,
  "PHI": 1.082,
  "BOS": 1.065,
  "ATL": 1.058,
  "BAL": 1.052,
  "MIL": 1.045,
  "STL": 1.038,
  "CHC": 1.025,
  "CLE": 1.018,
  "DET": 1.015,
  "MIN": 1.012,
  "HOU": 1.008,
  "WSH": 1.005,
  "ARI": 1.002,
  "LAA": 0.998,
  "NYM": 0.992,
  "PIT": 0.985,
  "TOR": 0.975,
  "CWS": 0.968,
  "KCR": 0.962,
  "OAK": 0.955,
  "SEA": 0.948,
  "LAD": 0.942,
  "MIA": 0.935,
  "TBR": 0.925,
  "SDP": 0.892,
  "SFG": 0.828
}
```

- [ ] **Step 1.3: Read `.gitignore` and add sim cache entries**

Open `.gitignore` and append:
```
# Simulation agent cache (daily re-fetched, never committed)
data/sim_cache/
data/sim_model.pkl
data/sim_unmatched.log
```

- [ ] **Step 1.4: Create `data/sim_cache/` directory**

```bash
mkdir -p data/sim_cache
```

- [ ] **Step 1.5: Commit**

```bash
git add data/park_factors.json .gitignore
git commit -m "chore: add park_factors.json, gitignore sim cache for Agent 6"
```

---

## Task 2: Name Normalization Helpers

**Files:**
- Create: `agents/simulation.py` (skeleton + normalization functions)
- Create: `tests/test_simulation.py` (normalization tests)

- [ ] **Step 2.1: Write failing tests for `_normalize_name` and `_match_player`**

Create `tests/test_simulation.py`:

```python
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
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_simulation.py::TestNormalizeName tests/test_simulation.py::TestMatchPlayer -v
```

Expected: `ModuleNotFoundError: No module named 'agents.simulation'`

- [ ] **Step 2.3: Create `agents/simulation.py` with normalization functions**

Create `agents/simulation.py`:

```python
"""
agents/simulation.py — Agent 6: Game-Level HR Simulation Model.

Public interface:
    add_simulation(df: pd.DataFrame) -> pd.DataFrame

Appends columns: sim_prob, sim_edge, convergence
Returns df unchanged on any error (graceful degradation).
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATTER_FEATURES = ["Barrel%", "ISO", "FB%", "Hard%", "EV"]
SEASON_WEIGHTS = {2024: 0.10, 2025: 0.30, 2026: 0.60}
TRAIN_SEASONS = [2024, 2025]

CACHE_DIR = Path("data/sim_cache")
MODEL_PATH = Path("data/sim_model.pkl")
UNMATCHED_LOG = Path("data/sim_unmatched.log")
PARK_FACTORS_PATH = Path("data/park_factors.json")

PITCHER_LEAGUE_HR9 = 1.30  # MLB average HR/9 across 2024-2025
FUZZY_THRESHOLD = 85        # rapidfuzz token_sort_ratio minimum for a match

# Full team name → 3-letter abbreviation (matches park_factors.json keys)
TEAM_NAME_TO_ABBREV: dict[str, str] = {
    "Arizona Diamondbacks": "ARI",
    "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC",
    "Chicago White Sox": "CWS",
    "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE",
    "Colorado Rockies": "COL",
    "Detroit Tigers": "DET",
    "Houston Astros": "HOU",
    "Kansas City Royals": "KCR",
    "Los Angeles Angels": "LAA",
    "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN",
    "New York Mets": "NYM",
    "New York Yankees": "NYY",
    "Oakland Athletics": "OAK",
    "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT",
    "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG",
    "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL",
    "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSH",
}

# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


def _normalize_name(name: str) -> str:
    """
    Normalize a player name for fuzzy matching:
    1. Strip accents (Díaz → Diaz)
    2. Strip suffixes Jr., Sr., II, III, IV
    3. Title-case
    4. Strip leading/trailing whitespace
    """
    # Strip accents via unicode decomposition
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_name = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Strip name suffixes
    for suffix in (" Jr.", " Jr", " Sr.", " Sr", " II", " III", " IV", " V"):
        if ascii_name.endswith(suffix):
            ascii_name = ascii_name[: -len(suffix)]
    return ascii_name.strip().title()


def _match_player(target: str, candidates: list[str]) -> str | None:
    """
    Find best match for target in candidates list.
    1. Normalize both sides
    2. Exact match
    3. Fuzzy match (token_sort_ratio >= FUZZY_THRESHOLD)
    Returns the original (un-normalized) candidate string, or None.
    """
    norm_target = _normalize_name(target)
    norm_candidates = {_normalize_name(c): c for c in candidates}

    # Exact match on normalized form
    if norm_target in norm_candidates:
        return norm_candidates[norm_target]

    # Fuzzy match
    best_score, best_match = 0, None
    for norm_c, orig_c in norm_candidates.items():
        score = fuzz.token_sort_ratio(norm_target, norm_c)
        if score > best_score:
            best_score, best_match = score, orig_c

    if best_score >= FUZZY_THRESHOLD:
        return best_match
    return None


def _log_unmatched(player_name: str, context: str) -> None:
    """Append an unmatched player entry to the unmatched log."""
    UNMATCHED_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with UNMATCHED_LOG.open("a", encoding="utf-8") as f:
        f.write(f"{timestamp} | {context} | {player_name}\n")
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_simulation.py::TestNormalizeName tests/test_simulation.py::TestMatchPlayer -v
```

Expected: 8 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add agents/simulation.py tests/test_simulation.py
git commit -m "feat(sim): name normalization helpers + tests"
```

---

## Task 3: Data Fetching with Daily Caching

**Files:**
- Modify: `agents/simulation.py` (add `_fetch_batter_stats`, `_fetch_pitcher_stats`, `_get_weighted_batter_stats`, `_get_pitcher_stats_by_name`)
- Modify: `tests/test_simulation.py` (add caching + weighted-stats tests)

- [ ] **Step 3.1: Write failing tests for data fetching**

Append to `tests/test_simulation.py`:

```python
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
        # Barrel%: 2024=4.0, 2025=8.0, 2026=12.0 → weighted avg = 0.1*4+0.3*8+0.6*12 = 10.0
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
```

- [ ] **Step 3.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_simulation.py::TestGetWeightedBatterStats tests/test_simulation.py::TestGetPitcherStatsByName -v
```

Expected: `ImportError` — functions not yet defined.

- [ ] **Step 3.3: Add data-fetching functions to `agents/simulation.py`**

Append the following to `agents/simulation.py` (after the name normalization section):

```python
# ---------------------------------------------------------------------------
# Data fetching with daily caching
# ---------------------------------------------------------------------------


def _cache_path(kind: str, season: int) -> Path:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return CACHE_DIR / f"{kind}_{season}_{today}.csv"


def _fetch_batter_stats(season: int) -> pd.DataFrame:
    """
    Fetch FanGraphs batter stats via pybaseball with daily CSV cache.
    qual=50 PA minimum; falls back to qual=1 if < 10 rows returned (mid-season).
    Columns used: Name, HR, G, PA, Barrel%, ISO, FB%, Hard%, EV
    """
    import pybaseball  # late import — not required for tests that mock

    cache = _cache_path("batter", season)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)

    try:
        df = pybaseball.batting_stats(season, qual=50)
    except Exception:
        df = pd.DataFrame()

    if df.empty or len(df) < 10:
        try:
            df = pybaseball.batting_stats(season, qual=1)
        except Exception:
            df = pd.DataFrame()

    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def _fetch_pitcher_stats(season: int) -> pd.DataFrame:
    """
    Fetch FanGraphs pitcher stats via pybaseball with daily CSV cache.
    qual=1 IP (include any pitcher with any data; filter by IP later).
    Columns used: Name, IP, HR/9, HR/FB, xFIP
    """
    import pybaseball

    cache = _cache_path("pitcher", season)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)

    try:
        df = pybaseball.pitching_stats(season, qual=1)
    except Exception:
        df = pd.DataFrame()

    if not df.empty:
        df.to_csv(cache, index=False)
    return df


def _get_weighted_batter_stats(
    player_name: str, batter_dfs: dict[int, pd.DataFrame]
) -> dict | None:
    """
    Return weighted average batter features across seasons.
    Weight: 2024=10%, 2025=30%, 2026=60% (renormalized for available seasons).
    Returns None and logs if no season data found for this player.
    """
    seasons_found: list[tuple[int, dict]] = []
    all_candidates: list[str] = []
    for season, df in batter_dfs.items():
        if df.empty or "Name" not in df.columns:
            continue
        all_candidates.extend(df["Name"].tolist())

    # Find each season's row for this player
    for season, df in sorted(batter_dfs.items()):
        if df.empty or "Name" not in df.columns:
            continue
        candidates = df["Name"].tolist()
        matched = _match_player(player_name, candidates)
        if matched is None:
            continue
        row = df[df["Name"] == matched].iloc[0]
        # Ensure required features present
        stats = {}
        for feat in BATTER_FEATURES:
            stats[feat] = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else None
        if any(v is None for v in stats.values()):
            continue
        seasons_found.append((season, stats))

    if not seasons_found:
        _log_unmatched(player_name, "batter_stats")
        return None

    # Renormalize weights for available seasons
    total_weight = sum(SEASON_WEIGHTS[s] for s, _ in seasons_found)
    weighted: dict[str, float] = {feat: 0.0 for feat in BATTER_FEATURES}
    for season, stats in seasons_found:
        w = SEASON_WEIGHTS[season] / total_weight
        for feat in BATTER_FEATURES:
            weighted[feat] += w * stats[feat]

    return weighted


def _get_pitcher_stats_by_name(
    pitcher_name: str, pitcher_dfs: dict[int, pd.DataFrame]
) -> dict | None:
    """
    Return pitcher stats dict with HR/9, IP.
    Searches most-recent season first; falls back to earlier seasons.
    Returns None if pitcher not found in any season.
    """
    for season in sorted(pitcher_dfs.keys(), reverse=True):
        df = pitcher_dfs[season]
        if df.empty or "Name" not in df.columns:
            continue
        candidates = df["Name"].tolist()
        matched = _match_player(pitcher_name, candidates)
        if matched is None:
            continue
        row = df[df["Name"] == matched].iloc[0]
        return {
            "HR/9": float(row["HR/9"]) if "HR/9" in row.index and pd.notna(row["HR/9"]) else PITCHER_LEAGUE_HR9,
            "IP": float(row["IP"]) if "IP" in row.index and pd.notna(row["IP"]) else 0.0,
            "HR/FB": float(row["HR/FB"]) if "HR/FB" in row.index and pd.notna(row["HR/FB"]) else None,
            "xFIP": float(row["xFIP"]) if "xFIP" in row.index and pd.notna(row["xFIP"]) else None,
        }
    return None
```

- [ ] **Step 3.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_simulation.py::TestGetWeightedBatterStats tests/test_simulation.py::TestGetPitcherStatsByName -v
```

Expected: all tests PASS.

- [ ] **Step 3.5: Commit**

```bash
git add agents/simulation.py tests/test_simulation.py
git commit -m "feat(sim): data fetching with daily CSV cache + weighted batter stats"
```

---

## Task 4: HRRateModel (Ridge Regression)

**Files:**
- Modify: `agents/simulation.py` (add `HRRateModel` class and `_get_or_train_model`)
- Modify: `tests/test_simulation.py` (add model tests)

- [ ] **Step 4.1: Write failing model tests**

Append to `tests/test_simulation.py`:

```python
from agents.simulation import HRRateModel, BATTER_FEATURES


def _make_training_df(n: int = 50) -> pd.DataFrame:
    """Synthetic training data with plausible feature ranges."""
    import numpy as np
    rng = np.random.default_rng(42)
    data = {
        "Barrel%": rng.uniform(4, 18, n),
        "ISO": rng.uniform(0.10, 0.35, n),
        "FB%": rng.uniform(25, 55, n),
        "Hard%": rng.uniform(30, 55, n),
        "EV": rng.uniform(85, 95, n),
        "HR": rng.integers(5, 40, n),
        "G": rng.integers(80, 162, n),
    }
    df = pd.DataFrame(data)
    df["hr_per_game"] = df["HR"] / df["G"]
    return df


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
```

- [ ] **Step 4.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_simulation.py::TestHRRateModel -v
```

Expected: `ImportError` — `HRRateModel` not yet defined.

- [ ] **Step 4.3: Add `HRRateModel` and `_get_or_train_model` to `agents/simulation.py`**

Append the following to `agents/simulation.py` (after data fetching section):

```python
# ---------------------------------------------------------------------------
# HRRateModel — Ridge regression on batter contact stats
# ---------------------------------------------------------------------------


class HRRateModel:
    """
    Ridge-regularized linear regression model predicting hr_per_game.
    Features: Barrel%, ISO, FB%, Hard%, EV (avg exit velocity).
    Interface stable for v2 upgrade (swap Ridge for XGBoost without changing callers).
    """

    def __init__(self) -> None:
        self._pipe: Pipeline | None = None

    def fit(self, df: pd.DataFrame) -> None:
        """
        Train on a DataFrame that contains BATTER_FEATURES and 'hr_per_game' column.
        Rows with any NaN in features or label are dropped automatically.
        """
        train = df.dropna(subset=BATTER_FEATURES + ["hr_per_game"])
        X = train[BATTER_FEATURES]
        y = train["hr_per_game"]
        self._pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=1.0)),
        ])
        self._pipe.fit(X, y)

    def predict(self, features: dict) -> float:
        """
        Predict hr_per_game for a single player given a feature dict.
        Returns a non-negative float (Ridge can predict < 0; clipped at 0.0).
        """
        if self._pipe is None:
            raise RuntimeError("HRRateModel is not fitted. Call fit() or load() first.")
        X = pd.DataFrame([features])[BATTER_FEATURES].fillna(0.0)
        val = float(self._pipe.predict(X)[0])
        return max(0.0, val)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted pipeline to a pickle file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self._pipe, f)

    def load(self, path: str | Path) -> None:
        """Load a previously saved pipeline from disk."""
        with open(path, "rb") as f:
            self._pipe = pickle.load(f)


def _get_or_train_model(batter_dfs: dict[int, pd.DataFrame]) -> HRRateModel:
    """
    Load the model from disk if it exists and is < 7 days old.
    Otherwise retrain on TRAIN_SEASONS (2024+2025) and save.
    Falls back to loading if both training datasets are empty.
    """
    model = HRRateModel()

    if MODEL_PATH.exists():
        age_days = (
            datetime.now() - datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
        ).days
        if age_days < 7:
            model.load(MODEL_PATH)
            return model

    # Retrain
    print("[simulation] Training model on 2024-2025 FanGraphs data...")
    frames = []
    for season in TRAIN_SEASONS:
        df = batter_dfs.get(season)
        if df is None or df.empty:
            continue
        df = df.copy()
        df["hr_per_game"] = df["HR"] / df["G"].replace(0, pd.NA)
        frames.append(df.dropna(subset=BATTER_FEATURES + ["hr_per_game"]))

    if not frames:
        raise RuntimeError(
            "[simulation] No training data available for seasons "
            f"{TRAIN_SEASONS}. Cannot train model."
        )

    train_df = pd.concat(frames, ignore_index=True)
    model.fit(train_df)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[simulation] Model trained on {len(train_df)} rows and saved to {MODEL_PATH}.")
    return model
```

- [ ] **Step 4.4: Run tests to confirm they pass**

```bash
python -m pytest tests/test_simulation.py::TestHRRateModel -v
```

Expected: 4 tests PASS.

- [ ] **Step 4.5: Commit**

```bash
git add agents/simulation.py tests/test_simulation.py
git commit -m "feat(sim): HRRateModel (Ridge regression) with save/load"
```

---

## Task 5: Full Prediction Pipeline and `add_simulation`

**Files:**
- Modify: `agents/simulation.py` (add all multiplier functions + `add_simulation`)
- Modify: `tests/test_simulation.py` (add multiplier + integration tests)

- [ ] **Step 5.1: Write failing tests**

Append to `tests/test_simulation.py`:

```python
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
        """With mocked pybaseball, sim columns are added."""
        import pybaseball
        monkeypatch.setattr(
            "agents.simulation.CACHE_DIR", tmp_path / "sim_cache"
        )
        monkeypatch.setattr(
            "agents.simulation.MODEL_PATH", tmp_path / "sim_model.pkl"
        )
        monkeypatch.setattr(
            "agents.simulation.UNMATCHED_LOG", tmp_path / "sim_unmatched.log"
        )

        judge_row = {
            "Name": "Aaron Judge", "Barrel%": 18.0, "ISO": 0.340,
            "FB%": 42.0, "Hard%": 58.0, "EV": 95.0, "HR": 25, "G": 80, "PA": 300
        }
        mock_batter_df = pd.DataFrame([judge_row])

        cole_row = {
            "Name": "Gerrit Cole", "HR/9": 1.1, "HR/FB": 0.10, "xFIP": 3.20, "IP": 80.0
        }
        mock_pitcher_df = pd.DataFrame([cole_row])

        monkeypatch.setattr(pybaseball, "batting_stats", lambda s, qual=50: mock_batter_df)
        monkeypatch.setattr(pybaseball, "pitching_stats", lambda s, qual=1: mock_pitcher_df)

        df = self._make_final_df()
        result = add_simulation(df)
        assert "sim_prob" in result.columns
        assert "sim_edge" in result.columns
        assert "convergence" in result.columns
        assert result["sim_prob"].between(0.01, 0.60).all()
```

- [ ] **Step 5.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_simulation.py::TestGetParkFactor tests/test_simulation.py::TestGetPlatoonFactor tests/test_simulation.py::TestAddSimulation -v
```

Expected: `ImportError` — `_get_park_factor`, `_get_platoon_factor`, `add_simulation` not yet defined.

- [ ] **Step 5.3: Add multiplier functions and `add_simulation` to `agents/simulation.py`**

Append the following to `agents/simulation.py`:

```python
# ---------------------------------------------------------------------------
# Multiplier functions
# ---------------------------------------------------------------------------


def _get_park_factor(game: str, park_factors: dict) -> float:
    """Extract home team from 'Away @ Home' game string, return HR park factor."""
    if " @ " not in game:
        return 1.0
    home_name = game.split(" @ ", 1)[1].strip()
    abbrev = TEAM_NAME_TO_ABBREV.get(home_name, "")
    return park_factors.get(abbrev, 1.0)


def _fetch_probable_starters(today: str) -> dict:
    """
    Returns {team_abbrev: {"name": str, "hand": "R"|"L"|""}} for both home and
    away starters in today's schedule via MLB Stats API.
    Returns empty dict on any error.
    """
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": today, "hydrate": "probablePitcher"},
            timeout=15,
        )
        resp.raise_for_status()
        result: dict = {}
        for date_entry in resp.json().get("dates", []):
            for game in date_entry.get("games", []):
                for side in ("home", "away"):
                    team_data = game.get("teams", {}).get(side, {})
                    abbrev = team_data.get("team", {}).get("abbreviation", "")
                    prob = team_data.get("probablePitcher", {})
                    if abbrev and prob:
                        result[abbrev] = {
                            "name": prob.get("fullName", ""),
                            "hand": prob.get("pitchHand", {}).get("code", ""),
                        }
        return result
    except Exception as exc:
        logger.warning("[simulation] Could not fetch probable starters: %s", exc)
        return {}


def _fetch_batter_hands() -> dict:
    """
    Returns {normalized_name: "R"|"L"|"S"} from MLB Stats API.
    Returns empty dict on any error.
    """
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/sports/1/players",
            params={"season": datetime.now(timezone.utc).year, "gameType": "R"},
            timeout=15,
        )
        resp.raise_for_status()
        result: dict = {}
        for player in resp.json().get("people", []):
            name = _normalize_name(player.get("fullName", ""))
            hand = player.get("batSide", {}).get("code", "")
            if name and hand:
                result[name] = hand
        return result
    except Exception as exc:
        logger.warning("[simulation] Could not fetch batter hands: %s", exc)
        return {}


def _get_pitcher_factor(
    row: pd.Series,
    starters: dict,
    pitcher_dfs: dict[int, pd.DataFrame],
) -> tuple[float, str, str]:
    """
    Returns (pitcher_factor, opposing_pitcher_name, pitcher_hand).
    pitcher_factor = (pitcher_HR/9) / PITCHER_LEAGUE_HR9, capped [0.5, 2.0].
    Defaults to (1.0, "", "") when data is unavailable.
    """
    game = str(row.get("game", ""))
    batter_team = str(row.get("team", ""))

    if " @ " not in game or not batter_team:
        return 1.0, "", ""

    away_name, home_name = game.split(" @ ", 1)
    away_abbrev = TEAM_NAME_TO_ABBREV.get(away_name.strip(), "")
    home_abbrev = TEAM_NAME_TO_ABBREV.get(home_name.strip(), "")

    # Batter faces the OPPOSING team's starter
    if batter_team == home_abbrev:
        opposing_abbrev = away_abbrev
    elif batter_team == away_abbrev:
        opposing_abbrev = home_abbrev
    else:
        return 1.0, "", ""

    starter_info = starters.get(opposing_abbrev, {})
    pitcher_name = starter_info.get("name", "")
    pitcher_hand = starter_info.get("hand", "")

    if not pitcher_name:
        return 1.0, pitcher_name, pitcher_hand

    pitcher_stats = _get_pitcher_stats_by_name(pitcher_name, pitcher_dfs)
    if pitcher_stats is None or pitcher_stats.get("IP", 0) < 5:
        # Rookie or insufficient sample — default to league neutral
        return 1.0, pitcher_name, pitcher_hand

    hr9 = pitcher_stats.get("HR/9", PITCHER_LEAGUE_HR9)
    factor = hr9 / PITCHER_LEAGUE_HR9
    factor = max(0.5, min(2.0, factor))
    return factor, pitcher_name, pitcher_hand


def _get_platoon_factor(batter_hand: str, pitcher_hand: str) -> float:
    """
    Opposite-hand matchup (LvR or RvL) = favorable = 1.05.
    Same-hand matchup (LvL or RvR) = unfavorable = 0.95.
    Switch hitter (S) or either hand unknown = neutral = 1.0.
    """
    if not batter_hand or not pitcher_hand or batter_hand == "S":
        return 1.0
    return 1.05 if batter_hand != pitcher_hand else 0.95


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def add_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append simulation columns to final_df and return it.

    Added columns:
        sim_prob    — model-derived P(HR today), clipped to [0.01, 0.60]
        sim_edge    — sim_prob − pinnacle_prob (positive = sim more bullish)
        convergence — "AGREE" if |sim_edge| < 0.03, else "DIVERGE"

    Returns df unchanged if pybaseball is unavailable or any error occurs.
    """
    try:
        import pybaseball  # noqa: F401 — verify available; actual calls via _fetch_*
    except (ImportError, TypeError):
        logger.warning("[simulation] pybaseball not available — skipping simulation.")
        return df

    try:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # Fetch all batter + pitcher stats (daily cached)
        all_seasons = [2024, 2025, 2026]
        batter_dfs = {s: _fetch_batter_stats(s) for s in all_seasons}
        pitcher_dfs = {s: _fetch_pitcher_stats(s) for s in all_seasons}

        # Probable starters + batter handedness (best-effort)
        starters = _fetch_probable_starters(today)
        batter_hands = _fetch_batter_hands()

        # Park factors
        with PARK_FACTORS_PATH.open(encoding="utf-8") as f:
            park_factors = json.load(f)

        # Get/train model
        model = _get_or_train_model(batter_dfs)

        sim_probs: list[float | None] = []
        for _, row in df.iterrows():
            stats = _get_weighted_batter_stats(row["player_name"], batter_dfs)
            if stats is None:
                sim_probs.append(None)
                continue

            base_prob = model.predict(stats)

            park_factor = _get_park_factor(row.get("game", ""), park_factors)
            pitcher_factor, _pitcher_name, pitcher_hand = _get_pitcher_factor(
                row, starters, pitcher_dfs
            )

            norm_name = _normalize_name(row["player_name"])
            batter_hand = batter_hands.get(norm_name, "")
            platoon_factor = _get_platoon_factor(batter_hand, pitcher_hand)

            sim_prob = base_prob * park_factor * pitcher_factor * platoon_factor
            sim_prob = max(0.01, min(0.60, sim_prob))
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

- [ ] **Step 5.4: Run all simulation tests**

```bash
python -m pytest tests/test_simulation.py -v
```

Expected: all tests PASS. Note: `TestAddSimulation::test_adds_sim_columns_when_data_available` patches pybaseball so no network call is made.

- [ ] **Step 5.5: Commit**

```bash
git add agents/simulation.py tests/test_simulation.py
git commit -m "feat(sim): full prediction pipeline + add_simulation() public interface"
```

---

## Task 6: Integrate `add_simulation` into `run.py`

**Files:**
- Modify: `run.py`

- [ ] **Step 6.1: Write a failing integration test**

Append to `tests/test_simulation.py`:

```python
def test_add_simulation_called_in_run(monkeypatch):
    """Confirm run.main() calls add_simulation after calculate_ev."""
    import run
    called = []

    monkeypatch.setattr("run.add_simulation", lambda df: (called.append(True), df)[1])
    monkeypatch.setattr("run.fetch_odds", lambda *a, **kw: [])
    monkeypatch.setattr("run.fetch_player_teams", lambda: {})
    monkeypatch.setattr("run.extract_retail_odds", lambda *a, **kw: __import__("pandas").DataFrame())
    monkeypatch.setattr("run.extract_sharp_anchor", lambda *a, **kw: __import__("pandas").DataFrame())
    monkeypatch.setenv("ODDS_API_KEY", "test")

    run.main()
    # main() returns early when retail/anchor is empty, so add_simulation is NOT
    # reached — this test verifies the import works; call order tested in test_run.py.
    # Just confirm no ImportError.
```

- [ ] **Step 6.2: Run to confirm import error**

```bash
python -m pytest tests/test_simulation.py::test_add_simulation_called_in_run -v
```

Expected: `ImportError` because `run.add_simulation` doesn't exist yet.

- [ ] **Step 6.3: Modify `run.py`**

Add the import at the top of `run.py` (after existing imports):
```python
from agents.simulation import add_simulation
```

In `main()`, after the line `final_df["team"] = final_df["player_name"].map(player_teams).fillna("")` and before `log_open_plays(...)`, add:

```python
    print("Running simulation model...")
    final_df = add_simulation(final_df)
```

The modified section of `main()` should look like:

```python
    final_df = calculate_ev(retail_df, anchor_df)
    final_df["team"] = final_df["player_name"].map(player_teams).fillna("")

    print("Running simulation model...")
    final_df = add_simulation(final_df)

    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
    print(f"Found {n_players} players | {n_positive} +EV plays")

    log_open_plays(final_df, now=now)
```

- [ ] **Step 6.4: Run the simulation test + full test suite**

```bash
python -m pytest tests/test_simulation.py -v
python -m pytest tests/ -q
```

Expected: all tests PASS.

- [ ] **Step 6.5: Commit**

```bash
git add run.py tests/test_simulation.py
git commit -m "feat(sim): wire add_simulation into run.py pipeline"
```

---

## Task 7: Dashboard — Simulation Analysis Section

**Files:**
- Modify: `dashboard/generator.py`
- Modify: `tests/test_generator.py`

- [ ] **Step 7.1: Write failing dashboard tests**

Append to `tests/test_generator.py`:

```python
import pandas as pd
from datetime import datetime, timezone
from dashboard.generator import generate_dashboard

COMMENCE = datetime(2026, 5, 15, 23, 5, tzinfo=timezone.utc)
GAME_SIM = "Texas Rangers @ New York Yankees"


@pytest.fixture
def sample_df_with_sim():
    return pd.DataFrame([
        {
            "player_name": "Aaron Judge", "team": "NYY", "game": GAME_SIM,
            "commence_time": COMMENCE,
            "pinnacle_odds": 380, "pinnacle_prob": 0.18,
            "DraftKings": 450, "FanDuel": 420,
            "best_retail_odds": 450, "best_retail_decimal": 5.5,
            "best_retail_book": "DraftKings",
            "sharp_anchor": "pinnacle",
            "ev_pct": 0.05, "composite_score": 0.009, "composite_z": 1.2,
            "kelly_units": 0.5, "stake_usd": 12.5,
            "sim_prob": 0.21, "sim_edge": 0.03, "convergence": "AGREE",
        },
        {
            "player_name": "Rafael Devers", "team": "BOS", "game": GAME_SIM,
            "commence_time": COMMENCE,
            "pinnacle_odds": 520, "pinnacle_prob": 0.14,
            "DraftKings": 600, "FanDuel": 580,
            "best_retail_odds": 600, "best_retail_decimal": 7.0,
            "best_retail_book": "DraftKings",
            "sharp_anchor": "pinnacle",
            "ev_pct": -0.02, "composite_score": -0.003, "composite_z": -0.8,
            "kelly_units": 0.0, "stake_usd": 0.0,
            "sim_prob": 0.10, "sim_edge": -0.04, "convergence": "DIVERGE",
        },
    ])


def test_sim_columns_not_treated_as_book_columns(tmp_path, sample_df_with_sim):
    """sim_prob, sim_edge, convergence must NOT appear as book columns in the EV table."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df_with_sim, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    # These should NOT appear as book headers in the main EV table
    assert '"sim_prob"' not in content.split("BOOKS=")[1].split(";")[0]


def test_sim_section_rendered_when_data_present(tmp_path, sample_df_with_sim):
    """Simulation Analysis section is in the HTML when sim columns present."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df_with_sim, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Simulation Analysis" in content
    assert "sim-section" in content


def test_sim_section_unavailable_message_when_no_sim_data(tmp_path, sample_df):
    """Shows unavailable message when sim columns absent."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Simulation Analysis" in content
    assert "unavailable" in content.lower()


def test_sim_data_injected_into_js(tmp_path, sample_df_with_sim):
    """SIM_DATA JS variable is populated with player sim records."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df_with_sim, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Aaron Judge" in content
    assert "SIM_DATA" in content
```

- [ ] **Step 7.2: Run tests to confirm they fail**

```bash
python -m pytest tests/test_generator.py::test_sim_columns_not_treated_as_book_columns tests/test_generator.py::test_sim_section_rendered_when_data_present tests/test_generator.py::test_sim_section_unavailable_message_when_no_sim_data tests/test_generator.py::test_sim_data_injected_into_js -v
```

Expected: tests fail (sim section not in template yet).

- [ ] **Step 7.3: Update `META_COLS` in `dashboard/generator.py`**

Find this block:
```python
META_COLS = {
    "player_name", "team", "game", "commence_time",
    "pinnacle_odds", "pinnacle_prob", "sharp_anchor",
    "best_retail_odds", "best_retail_decimal", "best_retail_book",
    "ev_pct", "composite_score", "composite_z",
    "kelly_units", "stake_usd",
}
```

Replace with:
```python
META_COLS = {
    "player_name", "team", "game", "commence_time",
    "pinnacle_odds", "pinnacle_prob", "sharp_anchor",
    "best_retail_odds", "best_retail_decimal", "best_retail_book",
    "ev_pct", "composite_score", "composite_z",
    "kelly_units", "stake_usd",
    # Simulation columns — must NOT appear as book columns in the EV table
    "sim_prob", "sim_edge", "convergence",
}
```

- [ ] **Step 7.4: Add simulation CSS to `HTML_TEMPLATE` in `dashboard/generator.py`**

Find the closing `</style>` tag in `HTML_TEMPLATE`. Insert these styles before it:

```css
    #sim-section{margin-top:32px;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
    #sim-section h2{margin-top:0;color:#495057}
    .sim-summary{display:flex;gap:16px;margin-bottom:16px}
    .sim-box{flex:1;border-radius:6px;padding:12px 16px;text-align:center}
    .sim-box-agree{background:#d4edda;border:1px solid #c3e6cb}
    .sim-box-bullish{background:#cfe2ff;border:1px solid #b6d4fe}
    .sim-box-bearish{background:#f8d7da;border:1px solid #f5c6cb}
    .sim-box-count{font-size:2em;font-weight:700}
    .sim-box-label{font-size:.85em;color:#495057}
    #sim-table{border-collapse:collapse;width:100%;font-size:.9em}
    #sim-table th{background:#343a40;color:#fff;padding:10px 8px;cursor:pointer;text-align:left;white-space:nowrap;user-select:none}
    #sim-table th:hover{background:#495057}
    #sim-table td{padding:8px;border-bottom:1px solid #dee2e6;white-space:nowrap}
    tr.sim-bullish-row{background:#d4edda}
    tr.sim-agree-row{background:#fff3cd}
    tr.sim-bearish-row td{color:#999}
```

- [ ] **Step 7.5: Add `__SIM_SECTION__` and `__SIM_DATA__` placeholders to `HTML_TEMPLATE`**

Find this block in `HTML_TEMPLATE` (after the parlay builder closing div):
```html
  <div id="parlay-builder">
```

Just before it, add:
```html
  __SIM_SECTION__
```

In the `<script>` block, find `const DATA=__DATA__;` and add below it:
```javascript
    const SIM_DATA=__SIM_DATA__;
```

At the bottom of the `<script>` block, after `renderParlays();`, add:
```javascript
    renderSimTable();
```

Add the `renderSimTable` function to the script (before `renderTable`):
```javascript
    function renderSimTable(){
      const tbody=document.getElementById('sim-table-body');
      if(!tbody)return;
      if(!SIM_DATA||!SIM_DATA.length){
        tbody.innerHTML='<tr><td colspan="9" style="color:#999;font-style:italic;text-align:center">No simulation data available for this slate.</td></tr>';
        return;
      }
      const sorted=[...SIM_DATA].sort((a,b)=>b.sim_edge-a.sim_edge);
      tbody.innerHTML=sorted.filter(r=>r.ev_pct>=minEv).map(r=>{
        let cls='';
        if(r.sim_edge>3&&r.ev_pct>0)cls='sim-bullish-row';
        else if(Math.abs(r.sim_edge)<=3&&r.ev_pct>0)cls='sim-agree-row';
        else if(r.sim_edge<-3)cls='sim-bearish-row';
        const edgeStr=(r.sim_edge>=0?'+':'')+r.sim_edge.toFixed(1)+'%';
        const convBadge=r.convergence==='AGREE'?'<span style="color:#155724;font-weight:600">✓AGREE</span>':'<span style="color:#721c24">DIVERGE</span>';
        return`<tr class="${cls}">
          <td>${r.player}</td><td>${r.team}</td><td>${r.game}</td>
          <td>${r.sim_prob.toFixed(1)}%</td>
          <td>${r.pin_prob.toFixed(1)}%</td>
          <td>${edgeStr}</td>
          <td>${convBadge}</td>
          <td>${fmtOdds(r.best_retail_odds)}</td>
          <td>${fmtPct(r.ev_pct)}</td>
          <td>${r.stake}</td>
        </tr>`;
      }).join('');
    }
```

- [ ] **Step 7.6: Add `_build_sim_section_html` helper and update `generate_dashboard` in `dashboard/generator.py`**

Add this helper function before `generate_dashboard`:

```python
_SIM_COLS = {"sim_prob", "sim_edge", "convergence"}

_SIM_TABLE_HTML = """<div id="sim-section">
  <h2>Simulation Analysis</h2>
  <div class="sim-summary">
    <div class="sim-box sim-box-agree">
      <div class="sim-box-count">{n_agree}</div>
      <div class="sim-box-label">🟢 Convergence plays<br><small>+EV &amp; |sim edge| &lt;3%</small></div>
    </div>
    <div class="sim-box sim-box-bullish">
      <div class="sim-box-count">{n_bullish}</div>
      <div class="sim-box-label">🔵 Sim bullish<br><small>sim &gt; pin by &gt;3%</small></div>
    </div>
    <div class="sim-box sim-box-bearish">
      <div class="sim-box-count">{n_bearish}</div>
      <div class="sim-box-label">🔴 Sim bearish<br><small>sim &lt; pin by &gt;3%</small></div>
    </div>
  </div>
  <p style="color:#6c757d;font-size:.9em">Sorted by sim edge &darr; &nbsp;|&nbsp; Green=bullish+EV &nbsp; Yellow=convergence+EV &nbsp; Gray=bearish</p>
  <table id="sim-table">
    <thead><tr>
      <th onclick="sortSim('player')">Player</th>
      <th onclick="sortSim('team')">Team</th>
      <th onclick="sortSim('game')">Game</th>
      <th onclick="sortSim('sim_prob')">Sim %</th>
      <th onclick="sortSim('pin_prob')">Pin %</th>
      <th onclick="sortSim('sim_edge')">Sim Edge</th>
      <th>Signal</th>
      <th onclick="sortSim('best_retail_odds')">Best Retail</th>
      <th onclick="sortSim('ev_pct')">EV%</th>
      <th>Stake</th>
    </tr></thead>
    <tbody id="sim-table-body"></tbody>
  </table>
</div>"""

_SIM_UNAVAILABLE_HTML = """<div id="sim-section">
  <h2>Simulation Analysis</h2>
  <p style="color:#6c757d;font-style:italic">Simulation data unavailable for this slate.
  Check <code>data/sim_unmatched.log</code> for details.</p>
</div>"""
```

In `generate_dashboard`, after building `records` and before constructing `html`, add:

```python
    # Build simulation section
    if _SIM_COLS.issubset(final_df.columns) and final_df["sim_prob"].notna().any():
        sim_records = []
        for _, row in final_df.iterrows():
            if pd.isna(row.get("sim_prob")):
                continue
            sim_records.append({
                "player": row["player_name"],
                "team": row.get("team", ""),
                "game": row.get("game", ""),
                "sim_prob": round(float(row["sim_prob"]) * 100, 1),
                "pin_prob": round(float(row["pinnacle_prob"]) * 100, 1),
                "sim_edge": round(float(row["sim_edge"]) * 100, 1),
                "convergence": row["convergence"],
                "best_retail_odds": int(row["best_retail_odds"]),
                "ev_pct": round(float(row["ev_pct"]) * 100, 2),
                "stake": (
                    f'{row["kelly_units"]:g}u (${row["stake_usd"]:,.0f})'
                    if row["kelly_units"] > 0 else "0u"
                ),
            })
        n_agree = sum(
            1 for r in sim_records if r["convergence"] == "AGREE" and r["ev_pct"] > 0
        )
        n_bullish = sum(1 for r in sim_records if r["sim_edge"] > 3)
        n_bearish = sum(1 for r in sim_records if r["sim_edge"] < -3)
        sim_section_html = _SIM_TABLE_HTML.format(
            n_agree=n_agree, n_bullish=n_bullish, n_bearish=n_bearish
        )
    else:
        sim_records = []
        sim_section_html = _SIM_UNAVAILABLE_HTML
```

Update the `html = (HTML_TEMPLATE ...` block to include sim replacements:

```python
    html = (
        HTML_TEMPLATE
        .replace("__DATA__", json.dumps(records))
        .replace("__BOOK_NAMES__", json.dumps(book_cols))
        .replace("__BOOK_HEADERS__", book_headers)
        .replace("__PARLAYS__", json.dumps(parlays or []))
        .replace("__TIMESTAMP__", timestamp)
        .replace("__N_PLAYERS__", str(n_players))
        .replace("__N_POSITIVE__", str(n_positive))
        .replace("__SIM_DATA__", json.dumps(sim_records))
        .replace("__SIM_SECTION__", sim_section_html)
    )
```

Also add a `sortSim` JS function alongside `sortBy` in the template:
```javascript
    let simSortKey='sim_edge',simSortDir=-1;
    function sortSim(k){if(simSortKey===k)simSortDir*=-1;else{simSortKey=k;simSortDir=-1;}renderSimTable();}
```

And update `renderSimTable` to use `simSortKey`/`simSortDir` for sorting instead of hardcoded `sim_edge`:
```javascript
    function renderSimTable(){
      const tbody=document.getElementById('sim-table-body');
      if(!tbody)return;
      if(!SIM_DATA||!SIM_DATA.length){
        tbody.innerHTML='<tr><td colspan="10" style="color:#999;font-style:italic;text-align:center">No simulation data.</td></tr>';
        return;
      }
      const sorted=[...SIM_DATA].sort((a,b)=>{
        const av=a[simSortKey],bv=b[simSortKey];
        if(typeof av==='string')return simSortDir*av.localeCompare(bv);
        return simSortDir*(av-bv);
      });
      tbody.innerHTML=sorted.filter(r=>r.ev_pct>=minEv).map(r=>{
        let cls='';
        if(r.sim_edge>3&&r.ev_pct>0)cls='sim-bullish-row';
        else if(Math.abs(r.sim_edge)<=3&&r.ev_pct>0)cls='sim-agree-row';
        else if(r.sim_edge<-3)cls='sim-bearish-row';
        const edgeStr=(r.sim_edge>=0?'+':'')+r.sim_edge.toFixed(1)+'%';
        const convBadge=r.convergence==='AGREE'?'<span style="color:#155724;font-weight:600">✓AGREE</span>':'<span style="color:#721c24">DIVERGE</span>';
        return`<tr class="${cls}">
          <td>${r.player}</td><td>${r.team}</td><td>${r.game}</td>
          <td>${r.sim_prob.toFixed(1)}%</td>
          <td>${r.pin_prob.toFixed(1)}%</td>
          <td>${edgeStr}</td>
          <td>${convBadge}</td>
          <td>${fmtOdds(r.best_retail_odds)}</td>
          <td>${fmtPct(r.ev_pct)}</td>
          <td>${r.stake}</td>
        </tr>`;
      }).join('');
    }
```

- [ ] **Step 7.7: Run dashboard tests**

```bash
python -m pytest tests/test_generator.py -v
```

Expected: all 8 tests PASS (4 existing + 4 new).

- [ ] **Step 7.8: Commit**

```bash
git add dashboard/generator.py tests/test_generator.py
git commit -m "feat(sim): Simulation Analysis section in dashboard"
```

---

## Task 8: End-to-End Smoke Test and Final Commit

**Files:**
- No new files — this task verifies the full pipeline runs cleanly.

- [ ] **Step 8.1: Run the full test suite**

```bash
python -m pytest tests/ -v --tb=short
```

Expected: all tests PASS (zero failures, zero errors).

- [ ] **Step 8.2: Smoke-test the simulation import**

```bash
python -c "
from agents.simulation import add_simulation, _normalize_name, _match_player, HRRateModel
from dashboard.generator import generate_dashboard, META_COLS
assert 'sim_prob' in META_COLS, 'sim_prob not in META_COLS'
assert 'sim_edge' in META_COLS, 'sim_edge not in META_COLS'
assert 'convergence' in META_COLS, 'convergence not in META_COLS'
print('All imports and META_COLS checks OK')
"
```

Expected: `All imports and META_COLS checks OK`

- [ ] **Step 8.3: Dry-run `add_simulation` on an empty DataFrame**

```bash
python -c "
import pandas as pd
from agents.simulation import add_simulation
df = pd.DataFrame(columns=['player_name','team','game','commence_time','pinnacle_prob','ev_pct','kelly_units','stake_usd'])
result = add_simulation(df)
print('Graceful degradation test:', 'PASS' if len(result) == 0 else 'FAIL')
"
```

Expected: `Graceful degradation test: PASS` (no crash on empty df).

- [ ] **Step 8.4: Final commit**

```bash
git add -A
git commit -m "feat: Agent 6 HR Simulation Model — complete implementation"
```

---

## Post-Implementation Checklist

After all tasks complete, verify these KPIs from the spec:

| KPI | How to check |
|-----|-------------|
| Player match rate ≥ 85% | Check `[simulation]` print in next live run |
| Model calibration ±5% for >60% of players | Compare `sim_prob` vs `pinnacle_prob` in dashboard |
| Pipeline overhead <10s (cache hit) | Time second run: `time python run.py` |
| Zero dashboard crashes | Run `python run.py` and confirm dashboard opens |
| Sim columns absent → graceful message | Pass df without sim columns to generate_dashboard |

## v2 Upgrade Note

When upgrading to game-log Statcast training:
1. Replace `_fetch_batter_stats` with per-game Statcast pull
2. Add `park_factor` and `pitcher_factor` as training features in `HRRateModel.fit()`
3. Swap `Ridge` for `XGBoost` inside `HRRateModel` — the public interface stays identical
4. Add `hr_last_7d`, `hr_last_14d` rolling features from game logs

The `add_simulation(df) → df` interface is stable across this upgrade.
