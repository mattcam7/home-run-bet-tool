# Simulation Model v2 — Feature Enhancement Design

## Goal

Upgrade the HR simulation model from 10 to 14 features by adding LHP/RHP batter splits, rolling recent-form windows for batters and pitchers, pitcher ground-ball rate, and batting order position. Retrain the logistic regression on updated training data and validate accuracy improvement against the current baseline.

## Motivation

The v1 model uses season-average stats for all features. This causes systematic mispricing in three cases:
1. **Platoon matchups** — a switch-hitter batting RHH vs a LHP has materially different power numbers than their overall season stats; the model ignores this.
2. **Recent form** — a batter in a 10-game hot streak or a pitcher who has allowed 5 HRs in his last 3 starts is not distinguished from their season average.
3. **Expected plate appearances** — a cleanup hitter gets ~4.4 PA/game vs ~4.0 for a 9-hole hitter; batting order slot captures this without needing lineup-depth modeling.

## Feature Set

### Replaced Features (season overall → matchup-specific)

| Old Feature | New Feature | Source |
|-------------|-------------|--------|
| `brl_percent` | `brl_pct_vs_hand` | Statcast filtered by `p_throws` |
| `iso` | `iso_vs_hand` | Statcast expected stats filtered by `p_throws` |
| `fb_pct` | `fb_pct_vs_hand` | Statcast pitch-level `bb_type`, filtered by `p_throws` |
| `hr_fb` | `hr_fb_vs_hand` | Statcast pitch-level `bb_type` + `events`, filtered by `p_throws` |
| `pitcher_hr9` | `rolling_pitcher_hr9` | Last 5 starts pitcher game logs |

### New Features

| Feature | Description | Fallback |
|---------|-------------|----------|
| `rolling_brl_pct` | Batter barrel rate over last 10 games | Season `brl_percent` |
| `rolling_avg_ev` | Batter avg exit velocity over last 10 games | Season `avg_hit_speed` |
| `pitcher_gb_pct` | Pitcher ground ball % allowed over last 5 starts | League mean 0.44 |
| `lineup_slot` | Confirmed batting order position (1–9) | 4.5 (midpoint) |

### Retained Features (unchanged)

`avg_hit_speed`, `ev95percent`, `bat_speed`, `park_factor`, `same_hand`

### Complete GAME_FEATURES (14 total)

```python
GAME_FEATURES = [
    "brl_pct_vs_hand", "iso_vs_hand", "fb_pct_vs_hand", "hr_fb_vs_hand",
    "avg_hit_speed", "ev95percent", "bat_speed",
    "park_factor", "same_hand",
    "rolling_brl_pct", "rolling_avg_ev",
    "rolling_pitcher_hr9", "pitcher_gb_pct",
    "lineup_slot",
]
```

### Fallback Behavior

All new features degrade gracefully:
- Split features: when pitcher hand is unknown, use season-overall equivalent
- Rolling batter features: when fewer than 5 games available in window, use season stat
- Rolling pitcher features: when fewer than 3 starts available, use season HR/9 / league-mean GB%
- `lineup_slot`: when lineup not yet posted by MLB Stats API, use 4.5

---

## Architecture

### Files Modified

**`agents/sim_build_training_data.py`**
- Add `bat_order` and `gb_type` to `_STATCAST_BASE_COLS` pulled from Statcast
- In `_aggregate_to_player_game()`: capture median `bat_order` per player-game → `lineup_slot`
- New function `_compute_rolling_batter_features(pg_df)`: for each player-game row, compute barrel% and avg EV from the preceding 10 games (pandas GroupBy + shift/rolling on chronologically sorted data)
- New function `_compute_rolling_pitcher_features(pg_df)`: compute HR/9 and GB% over the preceding 5 starts per pitcher
- New function `_fetch_batter_splits(season)`: returns `(player_id, vs_hand, brl_pct, iso, fb_pct, hr_fb)` — computed by re-aggregating Statcast filtered by `p_throws`; produces `data/batter_splits.parquet`
- In `_build_season()`: join split features and rolling features onto player-game rows; compute `pitcher_gb_pct` from Statcast `bb_type == "ground_ball"` aggregated per pitcher

**`agents/simulation.py`**
- Update `GAME_FEATURES` constant to 14-feature list
- New constant `LEAGUE_MEAN_GB_PCT = 0.44`
- New constant `BATTER_SPLITS_PATH = Path("data/batter_splits.parquet")`
- New function `_load_batter_splits_lookup()`: reads `data/batter_splits.parquet`, returns `{(normalized_name, hand): {brl_pct, iso, fb_pct, hr_fb}}`
- New function `_fetch_rolling_window(days=30)`: single bulk Statcast pull for last 30 calendar days; computes per-player rolling 10-game batter stats and per-pitcher rolling 5-start stats; daily-cached to `data/sim_cache/rolling_{date}.parquet`
- Modify `_fetch_probable_starters()`: add `lineups` to the `hydrate` parameter; extract `batting_order` for each player
- Modify `add_simulation()`: assemble all 14 features per player, using splits and rolling stats with fallbacks

**`agents/sim_validate_model.py`** (new file)
- Temporal split: train on 2022–2024 rows from `sim_training_cache.parquet`, test on 2025 rows
- Trains both the current 10-feature model and the new 14-feature model on the same split
- Reports per model: AUC-ROC, Brier score, log-loss, calibration curve (10 probability buckets)
- Prints side-by-side comparison table
- Exit code 0 if new model AUC >= old model AUC, else exit code 1

### New Data Files

| File | Produced by | Used by | Contents |
|------|-------------|---------|----------|
| `data/batter_splits.parquet` | `sim_build_training_data.py` | `simulation.py` | `(player_id, Name, season, vs_hand, brl_pct, iso, fb_pct, hr_fb)` |
| `data/sim_cache/rolling_{date}.parquet` | `simulation.py` at runtime | `simulation.py` | Per-player rolling stats for today, daily cache |

`data/batter_splits.parquet` is committed to the repo (like `batter_bat_speed.parquet`) so GitHub Actions has it without rebuilding. The rolling cache is ephemeral and excluded from git.

---

## Training Data Changes

The full Statcast pull for 2022–2025 (~2–3 hours) must be re-run because the player-game feature set changes. Existing season checkpoints (`data/sim_cache/training_{season}.parquet`) must be deleted before rebuilding so the new columns are included.

Season-level split stats are derived from the same Statcast pull — no additional API calls required for the LHP/RHP split sidecar.

Rolling features are computed in-memory during `_build_season()` using a chronological sort + `groupby().rolling()` on the player-game DataFrame. No additional data fetching needed for training.

---

## Validation

`python -m agents.sim_validate_model` reports:

```
Model Comparison — 2025 holdout (N=XXXX player-game rows)
                    Baseline (10f)   v2 (14f)   Delta
AUC-ROC             0.XXX            0.XXX      +X.XXX
Brier Score         0.XXX            0.XXX      -X.XXX
Log-Loss            X.XXX            X.XXX      -X.XXX

Calibration (v2):
  Predicted 0–5%:   actual X.X%  (N=XXXX)
  Predicted 5–10%:  actual X.X%  (N=XXXX)
  Predicted 10–15%: actual X.X%  (N=XXXX)
  Predicted 15–20%: actual X.X%  (N=XXXX)
  Predicted 20%+:   actual X.X%  (N=XXXX)
```

Target: AUC-ROC improvement of ≥ 0.005 over baseline. Brier and log-loss should both decrease.

---

## Retrain Process

```bash
# 1. Clear stale training checkpoints (features changed)
rm data/sim_cache/training_*.parquet
rm data/sim_cache/bat_speed_*.parquet

# 2. Rebuild training data (~2-3 hours)
python -m agents.sim_build_training_data

# 3. Validate accuracy improvement
python -m agents.sim_validate_model

# 4. Commit new model and batter_splits sidecar
git add data/sim_model.pkl data/batter_splits.parquet
git commit -m "feat(sim): v2 model — splits, rolling form, pitcher GB%, lineup slot"
```

---

## Graceful Degradation (Inference)

All new data fetches are wrapped in try/except. Failure modes:
- Rolling window fetch fails → fall back to season stats for all rolling features
- Batter splits sidecar missing → fall back to season-overall stats
- Lineup not posted → `lineup_slot = 4.5` for all players
- Pitcher GB% unavailable → use `LEAGUE_MEAN_GB_PCT`

The existing outer try/except in `add_simulation()` remains; any unhandled exception returns `df` unchanged.

---

## Out of Scope (deferred to v3)

- Weather / wind (already tracked as highest-value remaining feature in project memory)
- Career park splits per batter
- Pitcher stuff+ or pitch-mix changes
- Opener / bullpen game detection
