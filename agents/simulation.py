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
