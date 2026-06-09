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
import pickle
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from rapidfuzz import fuzz
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BATTER_FEATURES = ["brl_percent", "avg_hit_speed", "ev95percent", "iso"]
GAME_FEATURES = [
    "brl_percent", "avg_hit_speed", "ev95percent", "iso",
    "bat_speed", "park_factor", "same_hand", "pitcher_hr9",
]
LEAGUE_MEAN_BAT_SPEED = 68.9   # mph — 2024+ Statcast average on all swings
LEAGUE_MEAN_FB_PCT = 0.362     # MLB average fly ball rate 2022-2025
LEAGUE_MEAN_HR_FB = 0.138      # MLB average HR/FB rate 2022-2025
BAT_SPEED_PATH = Path("data/batter_bat_speed.parquet")
FG_STATS_PATH = Path("data/fg_batter_stats.parquet")
LEAGUE_MEAN_GB_PCT = 0.44
BATTER_SPLITS_PATH = Path("data/batter_splits.parquet")
TRAINING_CACHE_PATH = Path("data/sim_training_cache.parquet")
MODEL_MAX_AGE_DAYS = 30
SEASON_WEIGHTS = {2024: 0.10, 2025: 0.30, 2026: 0.60}

CACHE_DIR = Path("data/sim_cache")
MODEL_PATH = Path("data/sim_model.pkl")
UNMATCHED_LOG = Path("data/sim_unmatched.log")
PARK_FACTORS_PATH = Path("data/park_factors.json")

PITCHER_LEAGUE_HR9 = 1.30  # MLB average HR/9 across 2024-2025
FUZZY_THRESHOLD = 85        # rapidfuzz token_sort_ratio minimum for a match

CURRENT_SEASON = 2026
# Games needed for the current season to carry its full SEASON_WEIGHTS share.
# Below this, the weight scales linearly (Bayesian shrinkage toward prior seasons).
# 100 G ≈ a meaningful mid-season sample; at 10 G the 2026 weight is ~6% of normal.
MIN_FULL_SAMPLE_G = 100

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

# MLB Stats API uses different abbreviations from FanGraphs for 6 teams.
# This mapping converts MLB API style -> FanGraphs/park_factors.json style.
_MLB_API_TO_FG_ABBREV: dict[str, str] = {
    "KC": "KCR",   # Kansas City Royals
    "SD": "SDP",   # San Diego Padres
    "SF": "SFG",   # San Francisco Giants
    "TB": "TBR",   # Tampa Bay Rays
    "AZ": "ARI",   # Arizona Diamondbacks
    "ATH": "OAK",  # Athletics (relocated)
}


def _normalize_team_abbrev(abbrev: str) -> str:
    """Convert MLB Stats API team abbreviation to FanGraphs/park_factors.json style."""
    return _MLB_API_TO_FG_ABBREV.get(abbrev, abbrev)


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


def _reverse_statcast_name(s: str) -> str:
    """Convert 'Last, First' Statcast format to 'First Last' for matching."""
    parts = s.split(", ", 1)
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else s


def _fetch_batter_stats(season: int) -> pd.DataFrame:
    """
    Fetch batter stats from Baseball Savant (Statcast) and Baseball Reference.
    No FanGraphs dependency.

    Sources merged on player_id (MLBAM ID):
      - statcast_batter_exitvelo_barrels → brl_percent, avg_hit_speed, ev95percent
      - statcast_batter_expected_stats   → ba, slg → iso = slg - ba
      - batting_stats_bref               → HR, G for hr_per_game training label

    Returns DataFrame with: Name, brl_percent, avg_hit_speed, ev95percent, iso, HR, G
    """
    import pybaseball  # late import — not required for tests that mock

    cache = _cache_path("batter", season)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)

    # --- Statcast exit velocity / barrels (Baseball Savant) ---
    try:
        ev_df = pybaseball.statcast_batter_exitvelo_barrels(season, minBBE=1)
    except Exception:
        ev_df = pd.DataFrame()

    # --- Statcast expected stats for ISO (Baseball Savant) ---
    try:
        xs_df = pybaseball.statcast_batter_expected_stats(season, minPA=1)
    except Exception:
        xs_df = pd.DataFrame()

    if ev_df.empty or xs_df.empty:
        return pd.DataFrame()

    # Merge Statcast tables on player_id
    ev_cols = ["player_id", "last_name, first_name", "brl_percent", "avg_hit_speed", "ev95percent"]
    xs_cols = ["player_id", "ba", "slg"]
    merged = ev_df[ev_cols].merge(xs_df[xs_cols], on="player_id", how="inner")
    merged["iso"] = merged["slg"] - merged["ba"]

    # Convert "Last, First" → "First Last" for name matching against OddsAPI names
    merged["Name"] = merged["last_name, first_name"].apply(_reverse_statcast_name)

    # --- Baseball Reference for HR, G (training label) ---
    try:
        bref_df = pybaseball.batting_stats_bref(season)
    except Exception:
        bref_df = pd.DataFrame()

    if not bref_df.empty and "mlbID" in bref_df.columns:
        bref_sub = bref_df[["mlbID", "HR", "G"]].copy()
        bref_sub = bref_sub.rename(columns={"mlbID": "player_id"})
        merged = merged.merge(bref_sub, on="player_id", how="left")
    else:
        merged["HR"] = pd.NA
        merged["G"] = pd.NA

    result_cols = ["Name", "player_id", "brl_percent", "avg_hit_speed", "ev95percent", "iso", "HR", "G"]
    result = merged[result_cols].copy()
    result = result.dropna(subset=["brl_percent", "avg_hit_speed", "ev95percent", "iso"])

    if not result.empty:
        result.to_csv(cache, index=False)
    return result


def _fetch_pitcher_stats(season: int) -> pd.DataFrame:
    """
    Fetch pitcher stats from Baseball Reference (no FanGraphs dependency).
    Computes HR/9 = HR / (IP / 9) for pitcher_factor.
    Columns returned: Name, IP, HR/9, mlbID
    """
    import pybaseball

    cache = _cache_path("pitcher", season)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        return pd.read_csv(cache)

    try:
        df = pybaseball.pitching_stats_bref(season)
    except Exception:
        df = pd.DataFrame()

    if not df.empty and "HR" in df.columns and "IP" in df.columns:
        df = df.copy()
        df["HR/9"] = df.apply(
            lambda r: (float(r["HR"]) / (float(r["IP"]) / 9.0))
            if pd.notna(r["IP"]) and float(r["IP"]) > 0 and pd.notna(r["HR"])
            else PITCHER_LEAGUE_HR9,
            axis=1,
        )
        df.to_csv(cache, index=False)

    return df


def _get_weighted_batter_stats(
    player_name: str, batter_dfs: dict[int, pd.DataFrame]
) -> dict | None:
    """
    Return weighted average batter features across seasons.
    Base weights: 2024=10%, 2025=30%, 2026=60%, renormalized for available seasons.

    Current-season weight is shrunk proportionally to games played so that a player
    with only a handful of 2026 games doesn't let a hot small sample dominate.
    Effective 2026 weight = base_weight * min(G_2026 / MIN_FULL_SAMPLE_G, 1.0).
    Returns None and logs if no season data found for this player.
    """
    seasons_found: list[tuple[int, dict, int | None]] = []  # (season, features, games)

    for season, df in sorted(batter_dfs.items()):
        if df.empty or "Name" not in df.columns:
            continue
        candidates = df["Name"].tolist()
        matched = _match_player(player_name, candidates)
        if matched is None:
            continue
        row = df[df["Name"] == matched].iloc[0]
        stats = {}
        for feat in BATTER_FEATURES:
            stats[feat] = float(row[feat]) if feat in row.index and pd.notna(row[feat]) else None
        if any(v is None for v in stats.values()):
            continue
        games = int(row["G"]) if "G" in row.index and pd.notna(row["G"]) else None
        seasons_found.append((season, stats, games))

    if not seasons_found:
        _log_unmatched(player_name, "batter_stats")
        return None

    def _effective_weight(season: int, games: int | None) -> float:
        base = SEASON_WEIGHTS.get(season, 0.0)
        if season != CURRENT_SEASON or games is None:
            return base
        # Shrink toward prior seasons when current-season sample is small.
        return base * min(1.0, games / MIN_FULL_SAMPLE_G)

    total_weight = sum(_effective_weight(s, g) for s, _, g in seasons_found)
    if total_weight == 0:
        _log_unmatched(player_name, "batter_stats")
        return None

    weighted: dict[str, float] = {feat: 0.0 for feat in BATTER_FEATURES}
    for season, stats, games in seasons_found:
        w = _effective_weight(season, games) / total_weight
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


class HRClassifier:
    """
    Logistic regression classifier predicting P(hit_hr=1) from 14 game-level features.
    Trained on binary player-game Statcast outcomes (2022-2025).

    Features (GAME_FEATURES) v2:
        brl_pct_vs_hand    — barrel % vs pitcher hand (splits sidecar, falls back to season)
        iso_vs_hand        — ISO vs pitcher hand (splits sidecar, falls back to season)
        fb_pct_vs_hand     — fly ball % vs pitcher hand (splits sidecar, falls back to FG)
        hr_fb_vs_hand      — HR/FB vs pitcher hand (splits sidecar, falls back to FG)
        avg_hit_speed      — batter season avg exit velocity
        ev95percent        — batter season hard-hit rate (EV >= 95 mph)
        bat_speed          — batter average bat speed (league mean when missing)
        park_factor        — home stadium HR factor (1.0 = neutral)
        same_hand          — 1 if same handedness (platoon disadvantage), 0 if opposite
        rolling_brl_pct    — 30-day rolling barrel % (falls back to season)
        rolling_avg_ev     — 30-day rolling avg exit velocity (falls back to season)
        rolling_pitcher_hr9 — 30-day rolling pitcher HR/9 (falls back to season HR/9)
        pitcher_gb_pct     — 30-day rolling pitcher ground ball % (falls back to league mean)
        lineup_slot        — batting order slot 1-9 (defaults to 4.5 when not posted)
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


def _get_or_train_model() -> HRClassifier:
    """
    Load HRClassifier from data/sim_model.pkl if < MODEL_MAX_AGE_DAYS old.
    Otherwise load data/sim_training_cache.parquet and train a new classifier.
    Raises RuntimeError if neither pkl nor cache exists (caught by add_simulation).
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
    n = len(train_df.dropna(subset=GAME_FEATURES + ["hit_hr"]))
    model.fit(train_df)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    print(f"[simulation] Model trained on {n:,} player-game rows, saved to {MODEL_PATH}.")
    return model


# ---------------------------------------------------------------------------
# Game-context feature helpers
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

    The schedule endpoint does not include team abbreviation or pitcher hand in
    its response. We resolve team abbrev via TEAM_NAME_TO_ABBREV and pitcher
    hand via a second call to the all-players endpoint (matched by player ID).
    Returns empty dict on any error.
    """
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": today, "hydrate": "probablePitcher"},
            timeout=15,
        )
        resp.raise_for_status()

        # Collect starters: {team_abbrev: {"name": str, "pitcher_id": int}}
        result: dict = {}
        pitcher_ids: set[int] = set()
        for date_entry in resp.json().get("dates", []):
            for game in date_entry.get("games", []):
                for side in ("home", "away"):
                    team_data = game.get("teams", {}).get(side, {})
                    # Schedule endpoint returns team name, not abbreviation
                    team_name = team_data.get("team", {}).get("name", "")
                    abbrev = TEAM_NAME_TO_ABBREV.get(team_name, "")
                    prob = team_data.get("probablePitcher", {})
                    if abbrev and prob.get("fullName"):
                        abbrev = _normalize_team_abbrev(abbrev)
                        pitcher_id = prob.get("id")
                        result[abbrev] = {
                            "name": prob["fullName"],
                            "hand": "",
                            "_id": pitcher_id,
                        }
                        if pitcher_id:
                            pitcher_ids.add(pitcher_id)

        if not result:
            return result

        # Resolve pitcher handedness from the all-players endpoint
        try:
            players_resp = requests.get(
                "https://statsapi.mlb.com/api/v1/sports/1/players",
                params={"season": datetime.now(timezone.utc).year, "gameType": "R"},
                timeout=15,
            )
            players_resp.raise_for_status()
            id_to_hand: dict[int, str] = {}
            for player in players_resp.json().get("people", []):
                pid = player.get("id")
                if pid in pitcher_ids:
                    id_to_hand[pid] = player.get("pitchHand", {}).get("code", "")
            for entry in result.values():
                pid = entry.pop("_id", None) or 0
                entry["hand"] = id_to_hand.get(pid, "")
                entry["pitcher_id"] = pid
        except Exception as hand_exc:
            logger.warning("[simulation] Could not resolve pitcher hands: %s", hand_exc)
            for entry in result.values():
                entry["pitcher_id"] = entry.pop("_id", None) or 0

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


def _fetch_lineups(today: str) -> dict[str, int]:
    """
    Returns {normalized_player_name: batting_order_slot (1-9)} for today's confirmed lineups.
    Returns empty dict when lineups are not yet posted or on any error.
    """
    try:
        resp = requests.get(
            "https://statsapi.mlb.com/api/v1/schedule",
            params={"sportId": 1, "date": today, "hydrate": "lineups"},
            timeout=15,
        )
        resp.raise_for_status()

        players_resp = requests.get(
            "https://statsapi.mlb.com/api/v1/sports/1/players",
            params={"season": datetime.now(timezone.utc).year, "gameType": "R"},
            timeout=15,
        )
        players_resp.raise_for_status()
        id_to_norm_name: dict[int, str] = {
            int(p["id"]): _normalize_name(p.get("fullName", ""))
            for p in players_resp.json().get("people", [])
            if p.get("id") and p.get("fullName")
        }

        result: dict[str, int] = {}
        for date_entry in resp.json().get("dates", []):
            for game in date_entry.get("games", []):
                lineups = game.get("lineups", {})
                for side_key in ("homePlayers", "awayPlayers"):
                    for player in lineups.get(side_key, []):
                        pid = player.get("id")
                        slot = player.get("battingOrder")
                        if pid and slot is not None and pid in id_to_norm_name:
                            result[id_to_norm_name[pid]] = int(slot) // 100
        return result
    except Exception as exc:
        logger.warning("[simulation] Could not fetch lineups: %s", exc)
        return {}


def _load_bat_speed_lookup() -> dict[str, float]:
    """
    Load {normalized_player_name: avg_bat_speed} from data/batter_bat_speed.parquet.
    Uses the most recent season's value per player. Returns empty dict if file missing.
    """
    if not BAT_SPEED_PATH.exists():
        return {}
    try:
        bs = pd.read_parquet(BAT_SPEED_PATH)
        required = {"Name", "avg_bat_speed", "season", "player_id"}
        if bs.empty or not required.issubset(bs.columns):
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


def _load_fg_stats_lookup() -> dict[str, dict[str, float]]:
    """
    Load {normalized_player_name: {fb_pct, hr_fb}} from data/fg_batter_stats.parquet.
    Uses most recent season per player. Returns empty dict if file missing.
    """
    if not FG_STATS_PATH.exists():
        return {}
    try:
        fg = pd.read_parquet(FG_STATS_PATH)
        required = {"Name", "fb_pct", "hr_fb", "season", "player_id"}
        if fg.empty or not required.issubset(fg.columns):
            return {}
        latest = fg.sort_values("season").groupby("player_id").last().reset_index()
        result = {}
        for _, r in latest.iterrows():
            if pd.notna(r["Name"]) and pd.notna(r["fb_pct"]) and pd.notna(r["hr_fb"]):
                result[_normalize_name(str(r["Name"]))] = {
                    "fb_pct": float(r["fb_pct"]),
                    "hr_fb": float(r["hr_fb"]),
                }
        return result
    except Exception as exc:
        logger.warning("[simulation] Could not load FanGraphs stats sidecar: %s", exc)
        return {}


def _load_batter_splits_lookup() -> dict[tuple[str, str], dict[str, float | None]]:
    """
    Load {(normalized_name, pitcher_hand): {brl_pct, iso, fb_pct, hr_fb}}
    from data/batter_splits.parquet. Uses most recent season per player-hand combo.
    Returns empty dict if file missing or malformed.
    """
    if not BATTER_SPLITS_PATH.exists():
        return {}
    try:
        df = pd.read_parquet(BATTER_SPLITS_PATH)
        required = {"Name", "vs_hand", "brl_pct", "iso", "fb_pct", "hr_fb"}
        if df.empty or not required.issubset(df.columns):
            return {}
        latest = df.sort_values("season").groupby(["player_id", "vs_hand"]).last().reset_index()
        result: dict[tuple[str, str], dict[str, float | None]] = {}
        for _, r in latest.iterrows():
            if pd.isna(r.get("Name")):
                continue
            key = (_normalize_name(str(r["Name"])), str(r["vs_hand"]))
            result[key] = {
                "brl_pct": float(r["brl_pct"]) if pd.notna(r["brl_pct"]) else None,
                "iso": float(r["iso"]) if pd.notna(r["iso"]) else None,
                "fb_pct": float(r["fb_pct"]) if pd.notna(r["fb_pct"]) else None,
                "hr_fb": float(r["hr_fb"]) if pd.notna(r["hr_fb"]) else None,
            }
        return result
    except Exception as exc:
        logger.warning("[simulation] Could not load batter splits sidecar: %s", exc)
        return {}


def _fetch_rolling_window(days: int = 30) -> tuple[dict[str, dict], dict[int, dict]]:
    """
    Pull last `days` calendar days of Statcast (all players).
    Compute per-batter rolling stats and per-pitcher rolling stats.
    Daily-cached to data/sim_cache/rolling_{date}.parquet.

    Returns:
        batter_rolling: {normalized_name: {rolling_brl_pct, rolling_avg_ev}}
        pitcher_rolling_by_id: {pitcher_mlbam_id: {rolling_pitcher_hr9, rolling_pitcher_gb_pct}}
    Returns ({}, {}) on any error.
    """
    from datetime import timedelta

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cache_path = CACHE_DIR / f"rolling_{today_str}.parquet"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            batter_rows = cached[cached["role"] == "batter"]
            pitcher_rows = cached[cached["role"] == "pitcher"]
            batter_rolling: dict[str, dict] = {
                str(r["norm_name"]): {
                    "rolling_brl_pct": r.get("rolling_brl_pct"),
                    "rolling_avg_ev": r.get("rolling_avg_ev"),
                }
                for _, r in batter_rows.iterrows()
            }
            pitcher_rolling_by_id: dict[int, dict] = {}
            for _, r in pitcher_rows.iterrows():
                pid = r.get("_pitcher_id")
                if pd.notna(pid):
                    pitcher_rolling_by_id[int(pid)] = {
                        "rolling_pitcher_hr9": r.get("rolling_pitcher_hr9"),
                        "rolling_pitcher_gb_pct": r.get("rolling_pitcher_gb_pct"),
                    }
            return batter_rolling, pitcher_rolling_by_id
        except Exception:
            pass  # re-fetch on cache read failure

    try:
        import pybaseball
        start_date = (
            datetime.now(timezone.utc) - timedelta(days=days)
        ).strftime("%Y-%m-%d")
        sc = pybaseball.statcast(start_date, today_str, verbose=False)
    except Exception as exc:
        logger.warning("[simulation] Rolling window fetch failed: %s", exc)
        return {}, {}

    if sc is None or sc.empty:
        return {}, {}

    name_col = "player_name" if "player_name" in sc.columns else None

    batter_rows_out: list[dict] = []
    for batter_id, grp in sc.groupby("batter"):
        bbe_mask = grp["bb_type"].notna()
        n_bbe = bbe_mask.sum()
        if n_bbe < 5:
            continue
        bbe = grp[bbe_mask]
        n_barrel = (bbe["launch_speed_angle"] == 6).sum() if "launch_speed_angle" in bbe.columns else 0
        avg_ev = (
            float(bbe["launch_speed"].mean())
            if "launch_speed" in bbe.columns and bbe["launch_speed"].notna().any()
            else float("nan")
        )
        name = ""
        if name_col:
            names = grp[name_col].dropna()
            if len(names) > 0:
                name = _reverse_statcast_name(str(names.iloc[0]))
        norm = _normalize_name(name) if name else ""
        if not norm:
            continue
        batter_rows_out.append({
            "role": "batter",
            "norm_name": norm,
            "_pitcher_id": float("nan"),
            "rolling_brl_pct": float(n_barrel / n_bbe) * 100 if n_bbe > 0 else float("nan"),
            "rolling_avg_ev": avg_ev,
            "rolling_pitcher_hr9": float("nan"),
            "rolling_pitcher_gb_pct": float("nan"),
        })

    pitcher_rows_out: list[dict] = []
    for pitcher_id, grp in sc.groupby("pitcher"):
        bbe_mask = grp["bb_type"].notna()
        n_bbe = bbe_mask.sum()
        n_gb = (grp.loc[bbe_mask, "bb_type"] == "ground_ball").sum() if n_bbe > 0 else 0
        n_hr = (grp["events"] == "home_run").sum()
        n_pitches = len(grp)
        ip_approx = n_pitches / 15.0
        pitcher_rows_out.append({
            "role": "pitcher",
            "norm_name": f"_pid_{pitcher_id}",
            "_pitcher_id": float(pitcher_id),
            "rolling_brl_pct": float("nan"),
            "rolling_avg_ev": float("nan"),
            "rolling_pitcher_hr9": float(n_hr / ip_approx * 9) if ip_approx > 0 else float("nan"),
            "rolling_pitcher_gb_pct": float(n_gb / n_bbe) if n_bbe > 0 else float("nan"),
        })

    all_rows = batter_rows_out + pitcher_rows_out
    if all_rows:
        pd.DataFrame(all_rows).to_parquet(cache_path, index=False)

    batter_rolling = {
        r["norm_name"]: {
            "rolling_brl_pct": r["rolling_brl_pct"],
            "rolling_avg_ev": r["rolling_avg_ev"],
        }
        for r in batter_rows_out
    }
    pitcher_rolling_by_id = {
        int(r["_pitcher_id"]): {
            "rolling_pitcher_hr9": r["rolling_pitcher_hr9"],
            "rolling_pitcher_gb_pct": r["rolling_pitcher_gb_pct"],
        }
        for r in pitcher_rows_out
    }
    return batter_rolling, pitcher_rolling_by_id


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


def _get_opposing_starter_info(
    row: pd.Series, starters: dict
) -> dict | None:
    """Return the starters entry for the opposing team, or None."""
    game = str(row.get("game", ""))
    batter_team = str(row.get("team", ""))
    if " @ " not in game or not batter_team:
        return None
    away_name, home_name = game.split(" @ ", 1)
    away_abbrev = TEAM_NAME_TO_ABBREV.get(away_name.strip(), "")
    home_abbrev = TEAM_NAME_TO_ABBREV.get(home_name.strip(), "")
    batter_team_norm = _normalize_team_abbrev(batter_team)
    opposing_abbrev = away_abbrev if batter_team_norm == home_abbrev else (
        home_abbrev if batter_team_norm == away_abbrev else ""
    )
    return starters.get(opposing_abbrev)


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


def add_simulation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append simulation columns to final_df and return it.

    Added columns:
        sim_prob    — model-derived P(HR today), clipped to [0.01, 0.35]
        sim_edge    — sim_prob - pinnacle_prob (positive = sim more bullish)
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

        # Season stats (daily cached)
        all_seasons = [2024, 2025, 2026]
        batter_dfs = {s: _fetch_batter_stats(s) for s in all_seasons}
        pitcher_dfs = {s: _fetch_pitcher_stats(s) for s in all_seasons}

        # Probable starters + batter handedness (best-effort)
        starters = _fetch_probable_starters(today)
        batter_hands = _fetch_batter_hands()

        # Park factors
        with PARK_FACTORS_PATH.open(encoding="utf-8") as f:
            park_factors = json.load(f)

        # Model
        model = _get_or_train_model()

        # Bat speed sidecar
        bat_speed_lookup = _load_bat_speed_lookup()

        sim_probs: list[float | None] = []
        for _, row in df.iterrows():
            stats = _get_weighted_batter_stats(row["player_name"], batter_dfs)
            if stats is None:
                sim_probs.append(None)
                continue

            norm_name = _normalize_name(row["player_name"])
            park_factor = _get_park_factor(row.get("game", ""), park_factors)
            _pitcher_name, pitcher_hand, pitcher_hr9 = _get_pitcher_info(
                row, starters, pitcher_dfs
            )
            batter_hand = batter_hands.get(norm_name, "")
            same_hand = int(
                bool(batter_hand and pitcher_hand and batter_hand != "S" and batter_hand == pitcher_hand)
            )
            bat_speed = bat_speed_lookup.get(norm_name, LEAGUE_MEAN_BAT_SPEED)

            features = {
                "brl_percent":   float(stats["brl_percent"]),
                "avg_hit_speed": float(stats["avg_hit_speed"]),
                "ev95percent":   float(stats["ev95percent"]),
                "iso":           float(stats["iso"]),
                "bat_speed":     float(bat_speed),
                "park_factor":   float(park_factor),
                "same_hand":     float(same_hand),
                "pitcher_hr9":   float(pitcher_hr9),
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


def validate_simulation(df: pd.DataFrame) -> list[str]:
    """Slate-level sanity checks on simulation output. Returns warning strings.

    Checks:
      1. Coverage — fewer than 50% of players matched to sim data
      2. Systematic bias — mean sim_prob is <60% or >140% of mean Pinnacle prob
      3. Extreme divergences — >25% of matched players have |sim_edge| > 15pp
      4. Model age — model file older than 7 days should be retrained
    """
    warnings: list[str] = []

    if "sim_prob" not in df.columns:
        warnings.append("[sim-validate] sim_prob column missing — simulation not applied")
        return warnings

    total = len(df)
    sim = df["sim_prob"].dropna()
    covered = len(sim)

    if covered == 0:
        warnings.append("[sim-validate] No players have sim_prob — simulation produced no output")
        return warnings

    coverage_pct = covered / total * 100 if total else 0
    if coverage_pct < 50:
        warnings.append(
            f"[sim-validate] Low coverage: {covered}/{total} ({coverage_pct:.0f}%) "
            "players matched — check sim_unmatched.log"
        )

    if "pinnacle_prob" in df.columns:
        pin = pd.to_numeric(df.loc[df["sim_prob"].notna(), "pinnacle_prob"], errors="coerce")
        mean_sim = float(sim.mean())
        mean_pin = float(pin.mean()) if len(pin) else 0.0
        if mean_pin > 0:
            ratio = mean_sim / mean_pin
            if ratio < 0.60:
                warnings.append(
                    f"[sim-validate] Systematic bearish bias: mean sim={mean_sim:.3f} "
                    f"vs mean Pinnacle={mean_pin:.3f} (ratio={ratio:.2f}) — "
                    "model under-predicting HR rates slate-wide"
                )
            elif ratio > 1.40:
                warnings.append(
                    f"[sim-validate] Systematic bullish bias: mean sim={mean_sim:.3f} "
                    f"vs mean Pinnacle={mean_pin:.3f} (ratio={ratio:.2f}) — "
                    "model over-predicting HR rates slate-wide"
                )

    if "sim_edge" in df.columns and covered > 0:
        extreme = df["sim_edge"].dropna().abs() > 0.15
        n_extreme = int(extreme.sum())
        if n_extreme > 0 and n_extreme / covered > 0.25:
            warnings.append(
                f"[sim-validate] {n_extreme}/{covered} players have |sim_edge| > 15pp — "
                "large divergence between sim and Pinnacle; review model calibration"
            )

    if MODEL_PATH.exists():
        age_days = (datetime.now() - datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)).days
        if age_days >= MODEL_MAX_AGE_DAYS:
            warnings.append(
                f"[sim-validate] Model is {age_days} days old — "
                "delete data/sim_model.pkl to force retrain on next run"
            )
    else:
        warnings.append(
            "[sim-validate] No trained model found — "
            "run 'python -m agents.sim_build_training_data' then run the dashboard"
        )

    return warnings
