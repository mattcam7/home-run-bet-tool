# Simulation Model Rebuild — Game-Level Logistic Regression Design

**Date:** 2026-06-07
**Status:** Approved
**Goal:** Replace the ad-hoc multiplier-based HR simulation model with a logistic regression trained on 2022–2025 game-level Statcast data where all coefficients—park, platoon, pitcher—are learned from actual binary HR outcomes.

---

## Problem

The current simulation model (`HRRateModel` in `agents/simulation.py`) has three structural defects:

1. **Wrong model target.** It predicts `hr_per_game` as a continuous rate using season-aggregate stats. This is not what we're trying to price — we want P(hit ≥1 HR in this specific game).

2. **Arbitrary multipliers.** Park, pitcher, and platoon effects are hand-coded post-hoc multipliers (±5% platoon, pitcher_hr9/1.30 ratio). None of these magnitudes are derived from actual HR game outcomes. The model says "18% base rate × 0.95 platoon penalty" — but the actual LvL platoon effect may be 0.98 or 0.91. We don't know.

3. **Disconnected starters lookup (now fixed).** The `_fetch_probable_starters()` bug was patched in the session immediately prior — it was returning `{}` on every call because the MLB API schedule endpoint returns team `name`, not `abbreviation`. This means the platoon and pitcher multipliers were always 1.0 since the model was built.

---

## Solution

Rebuild as `HRClassifier` — a `LogisticRegression(C=1.0)` trained on binary player-game outcomes from 2022–2025 Statcast data. All contextual features (park, platoon, pitcher) enter as training features so the model learns their actual effect sizes from historical HR outcomes rather than assuming them.

---

## Architecture

### Two distinct jobs

**One-time training data build** (`python -m agents.sim_build_training_data`):
- Pulls Statcast pitch-level data for 2022–2025 (~700k rows/year, ~2 hours total)
- Aggregates to player-game rows: one row per (batter, game_pk, date) with `hit_hr` binary label
- Joins all features by MLBAM numeric ID — no name matching during training
- Saves `data/sim_training_cache.parquet` (~400k rows) and `data/batter_bat_speed.parquet` (bat speed sidecar)
- Checkpoints by season so it can resume if interrupted

**Daily inference** (`add_simulation(df)` — same public interface, same columns):
- Loads `data/sim_model.pkl` if < 30 days old; otherwise trains HRClassifier from parquet cache and saves
- If parquet cache doesn't exist: logs warning, returns df unchanged (graceful degradation)
- Assembles 8-feature vector per player and calls `model.predict_proba()` for sim_prob

---

## Training Features (8, all data-derived)

| Feature | Source | Notes |
|---------|--------|-------|
| `brl_percent` | Batter season Statcast (exitvelo_barrels) | Barrel rate — strongest HR predictor |
| `avg_hit_speed` | Batter season Statcast (exitvelo_barrels) | Average exit velocity |
| `ev95percent` | Batter season Statcast (exitvelo_barrels) | Hard hit rate (95+ mph) |
| `iso` | Batter season Statcast (expected_stats: slg-ba) | Isolated power |
| `bat_speed` | Per-batter Statcast swing data (2024+); league mean (68.9 mph) for 2022–2023 | Strong predictor on HR events |
| `park_factor` | `data/park_factors.json` | Float; 1.0 = neutral, Coors ~1.20 |
| `same_hand` | `stand == p_throws` from game starter data | 1 if platoon disadvantage, 0 otherwise |
| `pitcher_hr9` | Starting pitcher's season HR/9 from bref | League mean 1.30 when missing |

**Excluded features and why:**
- `launch_speed`, `launch_angle`, `events` sub-fields — data leakage (PA outcomes)
- `age_bat` — collinear with power metrics
- Pitcher `spin_rate`, `release_speed`, `arm_angle` — near-zero correlation with HR rate (~0.03-0.09)
- `n_thruorder_pitcher` — difficult to compute consistently at inference time

**Model:** `Pipeline(StandardScaler, LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs'))`

**Target:** `hit_hr` — binary 1 if batter hit ≥1 HR in the game, 0 otherwise

**Training set:** 2022–2025 player-game rows. Expected HR rate ~11–12% (realistic game-level base rate). Expected ~400k rows after joining all features.

---

## Inference Pipeline (daily)

```
1. _get_weighted_batter_stats(player_name, batter_dfs)
   → Bayesian-weighted average of brl_percent, avg_hit_speed, ev95percent, iso
     across 2024 (10%), 2025 (30%), 2026 (60%)
     with current-season shrinkage when G_2026 < 100

2. _load_bat_speed_lookup()
   → {normalized_name: avg_bat_speed} from data/batter_bat_speed.parquet
     (most recent season; league mean 68.9 if missing)

3. _get_park_factor(game, park_factors)
   → float from data/park_factors.json by home team abbrev

4. _get_pitcher_info(row, starters, pitcher_dfs)
   → (pitcher_name, pitcher_hand, pitcher_hr9) from today's starters + pitcher season stats

5. same_hand = 1 if batter_hand == pitcher_hand else 0

6. features = {brl_percent, avg_hit_speed, ev95percent, iso, bat_speed,
               park_factor, same_hand, pitcher_hr9}

7. sim_prob = model.predict_proba([features])[0, 1]
   → clipped to [0.01, 0.35]
```

---

## Key Design Decisions

**Why not apply_correction()?** Correction factors from `ml_retrain.py` were an additional hand-coded layer that would be redundant (and potentially counterproductive) on top of a model already trained on actual outcomes. Removed.

**Why remove the multipliers?** `base_prob × park_factor × pitcher_factor × platoon_factor` is equivalent to assuming independence and multiplicativity of all effects — which isn't necessarily true. The logistic regression learns the actual contribution of each feature jointly. Park and pitcher and platoon interact.

**Why keep Bayesian shrinkage for batter stats?** Small-sample 2026 data should still shrink toward stable prior-season estimates. This is a correct inference procedure at the player level and doesn't conflict with the model training — the model sees weighted features, not raw features.

**Why 30-day model refresh?** The model is trained on 2022–2025 historical data. Adding a few more 2026 games doesn't materially change the trained coefficients. A monthly refresh (or "refresh when you run the build script") is sufficient.

**bat_speed handled separately from weighted average:** bat_speed is only in Statcast 2024+. Rather than complicating the weighted-average logic with nullable bat_speed, it's looked up from the sidecar (most recent season, league mean fallback). This is simpler and correct.

---

## Files Changed

| File | Action | Purpose |
|------|--------|---------|
| `agents/simulation.py` | Modify | Replace HRRateModel with HRClassifier; update constants, _get_or_train_model(), add_simulation() |
| `agents/sim_build_training_data.py` | Create | One-time training data builder |
| `tests/test_simulation.py` | Modify | Update model tests; remove multiplier and correction tests |
| `tests/test_sim_build_training_data.py` | Create | Tests for training data builder helpers |

**Public interface unchanged:** `add_simulation(df) → df` with `sim_prob`, `sim_edge`, `convergence` columns. Callers in `run.py` need no changes.

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Training data rows | ≥ 300k player-game rows (2022–2025) |
| Feature join rate | ≥ 80% of player-games get all 8 features |
| Model AUC-ROC (hold-out 2022) | ≥ 0.60 (better than league-mean baseline of 0.50) |
| sim_prob vs Pinnacle ratio | 0.65–1.25 (not systematically biased) |
| Coverage on daily slate | ≥ 75% of players get sim_prob |
| Pipeline runtime overhead | < 10s on cache hit |
