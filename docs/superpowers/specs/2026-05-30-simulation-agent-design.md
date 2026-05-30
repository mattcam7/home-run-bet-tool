# Agent 6 — Game-Level HR Simulation Model Design

**Date:** 2026-05-30  
**Status:** Approved  
**Goal:** Add a statistically calibrated game-level HR probability model (sim_prob) that supplements Pinnacle's de-vigged line and surfaces conviction plays where both signals agree.

---

## Architecture Overview

`agents/simulation.py` runs as part of the existing `run.py` pipeline — after `calculate_ev()` but before the dashboard is generated. It computes `sim_prob` for each player in `final_df` and appends simulation columns. The dashboard generator renders a dedicated **Simulation Analysis** section below the main EV table.

**Key invariant:** `sim_prob` supplements Pinnacle's `pin_prob` — it never overrides it. Pinnacle's line remains the EV anchor. The simulation is a second opinion.

**Graceful degradation:** if pybaseball data is unavailable or a player can't be matched, the dashboard shows "data unavailable" in the simulation section without crashing the pipeline.

---

## Data Pipeline

All sources fetched via `pybaseball`, cached daily to `data/sim_cache/` to avoid slow re-fetches.

### Batter Stats
- Source: `pybaseball.batting_stats(season, qual=50)` for 2024, 2025, 2026
- Features: `barrel_pct`, `iso`, `fb_pct`, `hard_hit_pct`, `avg_exit_velocity`, `hr_per_pa`
- Seasonal weighting: **10% 2024 / 30% 2025 / 60% 2026** (favors recency, smooths small 2026 samples)
- Cache file: `data/sim_cache/batter_YYYY-MM-DD.csv`

### Pitcher Stats
- Source: `pybaseball.pitching_stats(season, qual=1)` for 2024, 2025, 2026
- Features used: `HR/9`, `HR/FB%`, `xFIP`
- Today's probable starters: MLB Stats API (already used in `run.py`)
- Cache file: `data/sim_cache/pitcher_YYYY-MM-DD.csv`

### Park Factors
- Source: `data/park_factors.json` — hardcoded dict of stadium HR factors for all 30 parks
- Values centered at 1.0 (neutral). Examples: Coors Field ~1.18, Oracle Park ~0.82
- Sourced from FanGraphs park factors, updated manually once per season
- Keyed by home team abbreviation (matched from game string already in `final_df`)

### Name Matching (Known Risk)
OddsAPI and FanGraphs both use string names with no shared numeric ID. Mitigation:
1. Normalize both sides: title case, strip accents, remove suffixes (Jr., III)
2. Attempt exact match
3. Fall back to fuzzy match (token sort ratio ≥ 85)
4. Log every miss to `data/sim_unmatched.log` — never silently drop players
5. Document as known limitation per CLAUDE.md

---

## Model Design

### Training (one-time, refreshed when model is >7 days old)

The logistic regression learns a **neutral-context** batter quality signal: *given a player's contact profile, what baseline HR rate should we expect?* Park and pitcher effects average out at season level, so they are excluded from training and applied as post-prediction multipliers instead.

- **Dataset:** FanGraphs qualified hitters, 2024–2025 (~500 players × 2 seasons = ~1,000 rows)
- **Features:** `barrel_pct`, `iso`, `fb_pct`, `hard_hit_pct`, `avg_exit_velocity`
- **Label:** `hr_per_game` = season HR / games played (observed game-level HR rate)
- **Algorithm:** Logistic Regression (scikit-learn), with L2 regularization
- **Saved to:** `data/sim_model.pkl`

### Prediction (daily, per player)

```
1. Fetch player's weighted stats (10/30/60 across 2024/2025/2026)
2. base_sim_prob = model.predict_proba(batter_features)
3. × park_factor      (home team stadium HR index, centered at 1.0)
4. × pitcher_factor   (opponent starter HR/9 ÷ league avg HR/9, capped 0.5–2.0)
5. × platoon_factor   (1.05 favorable / 0.95 unfavorable / 1.0 neutral)
6. → sim_prob (clipped to 0.01–0.60)
```

**Note on v2 upgrade:** When upgraded to game-log Statcast training, park and pitcher factors move *into* the training features (one row per player-game). The `SimulationModel.fit()` / `predict()` interface stays unchanged — only the training dataset changes.

### Output Columns Added to `final_df`

| Column | Description |
|--------|-------------|
| `sim_prob` | Model-derived P(HR today), 0–1 |
| `pin_prob` | Pinnacle de-vigged prob (already exists as `pinnacle_prob`) |
| `sim_edge` | `sim_prob − pin_prob` (positive = sim bullish vs sharp market) |
| `convergence` | `AGREE` if `\|sim_edge\| < 0.03`, else `DIVERGE` |

---

## Dashboard Integration

### Simulation Analysis Section (below main EV table)

**Summary bar — three callout boxes:**
- 🟢 **Convergence plays** — count of +EV plays where `|sim_edge| < 3%`. Highest conviction.
- 🔵 **Sim bullish** — count where `sim_prob > pin_prob` by >3%. Model sees more edge than sharp market.
- 🔴 **Sim bearish** — count where `sim_prob < pin_prob` by >3%. Model disagrees — proceed cautiously.

**Sortable table columns:**

| Player | Team | Game | Sim % | Pin % | Sim Edge | Best Retail | EV% | Stake |

**Row color coding:**
- **Green** — sim bullish (+3%) AND +EV (model and market both see edge)
- **Yellow** — convergence within 3%, +EV (agreement play)
- **Gray** — sim bearish (-3%) (de-emphasized, still shown)

**Default sort:** `sim_edge` descending. Filterable by same EV% slider as main table.

### `run.py` Change
One new call after `calculate_ev()`:
```python
from agents.simulation import add_simulation
final_df = add_simulation(final_df)
```
The rest of the pipeline is unchanged.

---

## File Structure

### New Files
| Path | Purpose |
|------|---------|
| `agents/simulation.py` | Core module: fetch, cache, feature engineering, model train/load, predict. Public interface: `add_simulation(df) → df` |
| `data/park_factors.json` | Hardcoded stadium HR factors, all 30 parks |

### Generated at Runtime (gitignored)
| Path | Purpose |
|------|---------|
| `data/sim_cache/batter_YYYY-MM-DD.csv` | Daily cached FanGraphs batter stats |
| `data/sim_cache/pitcher_YYYY-MM-DD.csv` | Daily cached FanGraphs pitcher stats |
| `data/sim_model.pkl` | Trained logistic regression model |
| `data/sim_unmatched.log` | Players that failed name matching |

### Modified Files
| Path | Change |
|------|--------|
| `run.py` | Add `add_simulation(final_df)` call |
| `dashboard/generator.py` | Add Simulation Analysis section HTML/JS |
| `.gitignore` | Add sim_cache/, sim_model.pkl, sim_unmatched.log |

---

## KPIs / Success Criteria

| Metric | Target |
|--------|--------|
| Player match rate | ≥ 85% of `final_df` players matched to FanGraphs |
| Model calibration | Sim prob within ±5% of Pinnacle for >60% of players |
| Convergence play win rate | Track vs non-convergence plays over 30+ days |
| Pipeline overhead | Simulation step adds <10s to total run time (cache hit) |
| Graceful degradation | Zero dashboard crashes when sim data unavailable |

---

## v2 Upgrade Path

1. Replace season-aggregate training data with per-game Statcast logs (binary HR outcome per player-game)
2. Add park_factor and pitcher_factor as training features (now meaningful at game level)
3. Swap logistic regression for XGBoost — interface unchanged, only `SimulationModel.fit()` changes
4. Add recent form feature: HR rate last 7/14 days (rolling window from game logs)
