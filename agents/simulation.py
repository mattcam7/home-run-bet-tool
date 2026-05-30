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

# Full team name -> 3-letter abbreviation (matches park_factors.json keys)
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
    1. Strip accents (Diaz -> Diaz)
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
