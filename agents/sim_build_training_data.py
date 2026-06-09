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
    LEAGUE_MEAN_BAT_SPEED,
    GAME_FEATURES,
    _normalize_team_abbrev,
    _reverse_statcast_name,
)

TRAINING_SEASONS = [2022, 2023, 2024, 2025]
CACHE_PATH = Path("data/sim_training_cache.parquet")
BAT_SPEED_PATH = Path("data/batter_bat_speed.parquet")
BATTER_SPLITS_PATH = Path("data/batter_splits.parquet")
PARK_FACTORS_PATH = Path("data/park_factors.json")
CHECKPOINT_DIR = Path("data/sim_cache")

_STATCAST_BASE_COLS = [
    "batter", "pitcher", "game_pk", "game_date",
    "home_team", "p_throws", "stand", "events",
    "bb_type", "bat_order", "launch_speed", "barrel",
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
        bat_order_vals = g["bat_order"].dropna() if "bat_order" in g.columns else pd.Series([], dtype=float)
        return pd.Series({
            "hit_hr": int((g["events"] == "home_run").any()),
            "home_team": g["home_team"].iloc[0],
            "stand": g["stand"].mode()[0] if not g["stand"].isna().all() else "",
            "p_throws": g["p_throws"].mode()[0] if not g["p_throws"].isna().all() else "",
            "opp_pitcher_id": int(opp_pitcher[0]) if len(opp_pitcher) > 0 else pd.NA,
            "bat_speed_mean": float(g["bat_speed"].mean())
            if has_bat_speed and g["bat_speed"].notna().any()
            else float("nan"),
            "lineup_slot": float(bat_order_vals.median()) if len(bat_order_vals) > 0 else float("nan"),
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

    if "mlbID" not in df.columns:
        return pd.DataFrame(columns=["player_id", "pitcher_hr9"])

    df = df.copy()
    df["player_id"] = pd.to_numeric(df["mlbID"], errors="coerce")
    df["pitcher_hr9"] = df.apply(
        lambda r: float(r["HR"]) / (float(r["IP"]) / 9.0)
        if pd.notna(r["IP"]) and float(r["IP"]) > 0 and pd.notna(r["HR"])
        else PITCHER_LEAGUE_HR9,
        axis=1,
    )
    return df[["player_id", "pitcher_hr9"]].dropna(subset=["player_id"]).copy()


def _fetch_batter_splits(sc: pd.DataFrame, season: int) -> pd.DataFrame:
    """
    Compute per-batter split stats vs LHP and RHP from pitch-level Statcast.

    Returns DataFrame: (player_id, season, vs_hand, brl_pct, iso, fb_pct, hr_fb)
    Requires sc to have: batter, p_throws, bb_type, events, barrel, launch_speed columns.
    Minimum 20 batted-ball events per split to be included.
    """
    required = {"batter", "p_throws", "bb_type", "events"}
    if not required.issubset(sc.columns):
        return pd.DataFrame(columns=["player_id", "season", "vs_hand", "brl_pct", "iso", "fb_pct", "hr_fb"])

    df = sc[sc["p_throws"].isin(["L", "R"])].copy()
    rows = []

    for (player_id, vs_hand), grp in df.groupby(["batter", "p_throws"]):
        bbe_mask = grp["bb_type"].notna()
        n_bbe = bbe_mask.sum()
        if n_bbe < 20:
            continue

        bbe = grp[bbe_mask]
        n_fb = (bbe["bb_type"] == "fly_ball").sum()
        n_hr = (grp["events"] == "home_run").sum()
        fb_pct = float(n_fb / n_bbe) if n_bbe > 0 else float("nan")
        hr_fb = float(n_hr / n_fb) if n_fb > 0 else float("nan")

        if "barrel" in bbe.columns:
            n_barrel = bbe["barrel"].fillna(0).sum()
            brl_pct = float(n_barrel / n_bbe) * 100
        else:
            brl_pct = float("nan")

        # ISO from events: (total_bases - hits) / AB
        ev = grp["events"].dropna()
        single = (ev == "single").sum()
        double_ = (ev == "double").sum()
        triple = (ev == "triple").sum()
        hr = (ev == "home_run").sum()
        ab_events = {"single", "double", "triple", "home_run", "strikeout",
                     "field_out", "grounded_into_double_play", "force_out",
                     "fielders_choice_out", "fielders_choice", "sac_fly"}
        ab = ev.isin(ab_events).sum()
        if ab > 0:
            hits = single + double_ + triple + hr
            total_bases = single + 2 * double_ + 3 * triple + 4 * hr
            iso = float((total_bases - hits) / ab)
        else:
            iso = float("nan")

        rows.append({
            "player_id": int(player_id),
            "season": season,
            "vs_hand": vs_hand,
            "brl_pct": brl_pct,
            "iso": iso,
            "fb_pct": fb_pct,
            "hr_fb": hr_fb,
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["player_id", "season", "vs_hand", "brl_pct", "iso", "fb_pct", "hr_fb"]
    )


def _build_season(season: int, park_factors: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Pull Statcast for `season`, aggregate to player-game level, join all features.

    Checkpoints to CHECKPOINT_DIR/training_{season}.parquet so re-runs are fast.

    Returns:
        (player_game_df, bat_speed_sidecar_df, splits_df)
        player_game_df columns: GAME_FEATURES + [hit_hr, season, player_id, game_pk, game_date]
        bat_speed_sidecar_df columns: [player_id, season, Name, avg_bat_speed]
        splits_df columns: [player_id, season, vs_hand, brl_pct, iso, fb_pct, hr_fb]
    """
    checkpoint = CHECKPOINT_DIR / f"training_{season}.parquet"
    bs_checkpoint = CHECKPOINT_DIR / f"bat_speed_{season}.parquet"
    splits_checkpoint = CHECKPOINT_DIR / f"splits_{season}.parquet"

    if checkpoint.exists() and bs_checkpoint.exists() and splits_checkpoint.exists():
        print(f"  [season {season}] Loading checkpoint...")
        return pd.read_parquet(checkpoint), pd.read_parquet(bs_checkpoint), pd.read_parquet(splits_checkpoint)

    import pybaseball  # only imported when a live pull is needed

    print(f"  [season {season}] Pulling Statcast for {season} (~20-30 min)...", flush=True)
    sc = pybaseball.statcast(f"{season}-04-01", f"{season}-11-30", verbose=False)

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

    # Batter splits sidecar (vs LHP / RHP)
    print(f"  [season {season}] Computing batter splits...")
    splits_df = _fetch_batter_splits(sc, season)

    # Join batter season stats (by MLBAM numeric ID — no name matching)
    print(f"  [season {season}] Joining batter season stats...")
    batter_stats = _fetch_batter_season_stats(season)
    pg = pg.merge(
        batter_stats[["player_id", "Name", "brl_percent", "avg_hit_speed", "ev95percent", "iso"]],
        on="player_id",
        how="inner",
    )

    # Add Names to splits sidecar
    batter_names = batter_stats[["player_id", "Name"]].copy()
    splits_df = splits_df.merge(batter_names, on="player_id", how="left")

    # Bat speed sidecar with Names for inference-time lookup
    bs_df = bs_raw.merge(batter_stats[["player_id", "Name"]], on="player_id", how="left")

    # Bat speed on player-game rows: use per-batter season mean (consistent with inference)
    bs_lookup = dict(zip(bs_raw["player_id"], bs_raw["avg_bat_speed"]))
    pg["bat_speed"] = pg["player_id"].map(bs_lookup).fillna(LEAGUE_MEAN_BAT_SPEED)

    # lineup_slot: already captured in _aggregate_to_player_game
    if "lineup_slot" not in pg.columns:
        pg["lineup_slot"] = 4.5
    pg["lineup_slot"] = pg["lineup_slot"].fillna(4.5)

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
    tmp_pg = checkpoint.with_suffix(".tmp.parquet")
    tmp_bs = bs_checkpoint.with_suffix(".tmp.parquet")
    tmp_splits = splits_checkpoint.with_suffix(".tmp.parquet")
    pg.to_parquet(tmp_pg, index=False)
    bs_df.to_parquet(tmp_bs, index=False)
    splits_df.to_parquet(tmp_splits, index=False)
    tmp_pg.replace(checkpoint)
    tmp_bs.replace(bs_checkpoint)
    tmp_splits.replace(splits_checkpoint)
    print(f"  [season {season}] Done — {len(pg):,} player-game rows. Checkpoint saved.")
    return pg, bs_df, splits_df


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
    all_splits: list[pd.DataFrame] = []

    for season in TRAINING_SEASONS:
        print(f"\n=== Season {season} ===")
        pg_df, bs_df, splits_df = _build_season(season, park_factors)
        if not pg_df.empty:
            all_pg.append(pg_df)
        if not bs_df.empty:
            all_bs.append(bs_df)
        if not splits_df.empty:
            all_splits.append(splits_df)

    if not all_pg:
        print("ERROR: No training data built. Check pybaseball connectivity.")
        sys.exit(1)

    training_df = pd.concat(all_pg, ignore_index=True)
    bat_speed_df = pd.concat(all_bs, ignore_index=True)

    training_df.to_parquet(CACHE_PATH, index=False)
    bat_speed_df.to_parquet(BAT_SPEED_PATH, index=False)

    if all_splits:
        splits_combined = pd.concat(all_splits, ignore_index=True)
        # Keep most recent season per (player_id, vs_hand)
        splits_combined = (
            splits_combined
            .sort_values("season")
            .groupby(["player_id", "vs_hand"], as_index=False)
            .last()
        )
        splits_combined.to_parquet(BATTER_SPLITS_PATH, index=False)
        print(f"Batter splits sidecar: {BATTER_SPLITS_PATH} ({len(splits_combined):,} rows)")

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
