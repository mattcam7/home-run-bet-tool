# HR Bet Tool — WAT Workflow

**Framework:** Workflow → Agents → Tools  
**Last updated:** 2026-06-02

---

## Overview

Daily pipeline for HR prop betting. Three execution phases map to three time windows:

| Phase | Steps | When | Entry point |
|-------|-------|------|-------------|
| 1 — Acquisition | 1–5 | Afternoon ET (after Pinnacle posts lines) | `python pipeline.py` |
| 2 — Closing | 6–7 | Every 10 min, 5–11 PM ET | `python capture_closing.py` |
| 3 — Outcomes + ML | 8–10 | Next morning | `python -m agents.outcome_tracker --backfill` |

---

## Workflow Map

```
[1 Acquisition] → [2 Raw Validation] → [3 Dataframes] → [4 EV + Score] → [5 CLV Log]
                                                                              ↓
[6 Closing Capture] → [7 CLV Validation] ────────────────────────────────────┘
                              ↓
[8 Outcome Acquisition] → [9 Outcome Validation] → [10 Storage + ML Retrain]
```

Bad rows at any step → `data/quarantine.jsonl`. Pipeline continues with clean data.  
Halt only when retail_df or anchor_df is empty after Step 3.

---

## Agents

### Agent 1: Odds Scraper (`run.fetch_odds`, `agents/odds_scraper.py`)
Fetches dual-market (standard + alternate) HR props per event from OddsAPI.  
Merges standard-market books (Pinnacle, BetOnline, Caesars) with alternate-market retail books (DraftKings, FanDuel, BetMGM).

### Agent 2: Sharp Anchor (`agents/pinnacle_scraper.py`)
Extracts Pinnacle two-sided (over+under) HR lines and de-vigs to produce `pinnacle_prob`.  
Falls back to BetOnline when Pinnacle does not have a line for a player.  
Tags each line: `sharp_anchor = "pinnacle" | "betonlineag"`.

### Agent 3: EV Calculator (`agents/ev_calculator.py`)
Computes `ev_pct = (pinnacle_prob × best_retail_decimal) − 1`.  
Assigns Quarter-Kelly stake sizing (capped at 3u, 1u = $25).  
Hard rule: `over_only = True → kelly_units = 0, stake_usd = 0`.

### Agent 4: Validator (`agents/validation.py`)
Per-step validation with quarantine. Runs at Steps 2, 4, 7, 9.

### Agent 5: Scorer (`agents/scoring.py`)
Computes `bet_score` (0–100) and `bet_grade` using anchor-weighted z-score sigmoid.  
Score ≥ 80 = Strong. Runs after Step 4 EV calculation.

### Agent 6: Simulation (`agents/simulation.py`)
Predicts `sim_prob` from Statcast/bref batter + pitcher stats.  
Applies correction factor from `models/correction_factors.json` if available.

### Agent 7: CLV Logger (`agents/clv_log.py`)
Phase 1: logs open plays (bet-time price + EV) to `data/clv_log.csv`.  
Phase 2: captures Pinnacle closing line + CLV + lineup confirmation.

### Agent 8: Outcome Tracker (`agents/outcome_tracker.py`)
Fetches MLB box scores via Stats API, records HR outcomes per player in `data/hr_outcomes.db`.

### Agent 9: ML Retrain (`agents/ml_retrain.py`)
Computes per-player correction factors from outcome DB vs Pinnacle open probabilities.  
Triggers only when ≥ 50 settled outcomes exist. Saves to `models/correction_factors.json`.

---

## Tools

| Tool | Purpose | Credential |
|------|---------|-----------|
| OddsAPI v4 | HR prop odds (standard + alternate markets) | `ODDS_API_KEY` in `.env` |
| Pinnacle scraper | Sharp anchor — devigged two-sided lines | Public (scraper) |
| BetOnline scraper | Sharp anchor fallback | Public (scraper) |
| MLB Stats API | Player teams, lineups, box scores | Public |
| Statcast / pybaseball | Batter Barrel%, exit velo, ISO for simulation | Public |

---

## Step Specifications

### Step 1: Data Acquisition
- **Agent:** `run.fetch_odds()`
- **Tools:** OddsAPI v4 (standard + alternate markets), MLB Stats API (player teams)
- **Input:** `ODDS_API_KEY`, current UTC datetime
- **Output:** `list[dict]` — raw event objects with bookmaker odds
- **Halt condition:** API key missing → crash immediately

### Step 2: Raw Validation
- **Agent:** `agents/validation.validate_raw_odds()`
- **Tools:** none
- **Input:** raw events list
- **Output:** `StepResult` — clean events list
- **Quarantine:** events missing `id` or `bookmakers` key
- **Halt condition:** zero clean events after filtering

### Step 3: Dataframe Build
- **Agent:** `agents/odds_scraper.extract_retail_odds()`, `agents/pinnacle_scraper.extract_sharp_anchor()`
- **Tools:** none (operates on in-memory events)
- **Input:** clean events list
- **Output:** `retail_df`, `anchor_df`
- **Halt condition:** either dataframe is empty

### Step 4: EV Calculation + Validation + Scoring
- **Agent:** `agents/ev_calculator.calculate_ev()`, `agents/validation.validate_ev_output()`, `agents/scoring.compute_bet_score()`
- **Tools:** none
- **Input:** `retail_df`, `anchor_df`
- **Output:** `final_df` with `ev_pct`, `kelly_units`, `stake_usd`, `anchor_quality`, `bet_score`, `bet_grade`
- **Quarantine:** `|ev_pct| > 2.0` (impossible EV)
- **Validation warning:** slate > 150 plays; > 30% above +600 odds
- **Hard rule:** `over_only = True → kelly_units = 0, stake_usd = 0` (enforced in ev_calculator AND re-checked in validator)

### Step 5: CLV Log — Open Plays
- **Agent:** `agents/clv_log.log_open_plays()`
- **Tools:** `data/clv_log.csv` (upsert by game_date + game + player_name)
- **Input:** `final_df`
- **Output:** updated `data/clv_log.csv`
- **Upsert key:** `(game_date, game, player_name)` — re-runs same day refresh open side, never overwrite closing columns

### Step 6: Closing Line Capture
- **Agent:** `agents/clv_log.capture_closing()`
- **Tools:** OddsAPI (Pinnacle closing line), MLB Stats API (lineups)
- **Input:** `data/clv_log.csv` rows with `closing_pinnacle_prob = NaN` and first pitch within 30 min
- **Output:** updated `clv_pct`, `closing_pinnacle_prob`, `in_lineup` columns
- **Idempotent:** safe to run repeatedly

### Step 7: CLV Validation
- **Agent:** `agents/validation.validate_clv_log()`
- **Tools:** `data/clv_log.csv`
- **Input:** full CLV log DataFrame
- **Quarantine:** `|clv_pct| > 0.50`
- **Note:** quarantine here is informational — entries are flagged but not removed from the log

### Step 8: Outcome Acquisition
- **Agent:** `agents/outcome_tracker.get_all_hr_hitters()`
- **Tools:** MLB Stats API `/schedule` + `/game/{pk}/boxscore`
- **Input:** date string (YYYY-MM-DD) for yesterday
- **Output:** `dict[player_name → {hrs_hit, at_bats}]`

### Step 9: Outcome Validation
- **Agent:** `agents/validation.validate_outcomes()`
- **Tools:** none
- **Input:** outcomes dict, date string
- **Validation warning:** date-level HR rate > 50%
- **Halt condition:** empty outcomes dict (games may not be complete — skip, retry tomorrow)

### Step 10: Outcome Storage + ML Retrain
- **Agent:** `agents/outcome_tracker.update_for_date()`, `agents/ml_retrain.retrain_if_ready()`
- **Tools:** `data/hr_outcomes.db` (SQLite), `models/correction_factors.json`
- **Input:** validated outcomes dict, CLV log
- **Output:** updated outcomes DB, optionally updated correction factors
- **ML trigger:** ≥ 50 settled outcomes in DB

---

## Error Playbook

| Symptom | Likely Cause | Action |
|---------|-------------|--------|
| Step 1 halts — no events | OddsAPI key expired or Pinnacle not yet posted | Re-run after 2 PM ET |
| Step 3 halts — anchor empty | Pinnacle and BetOnline both offline or no HR props today | Check `anchor_df` manually |
| Step 4 warning — >150 plays | Alternate market contamination | Check `data/quarantine.jsonl` for `impossible_ev_over_200pct` |
| Step 7 quarantine — CLV > 50% | Stale retail odds captured near closing | Review flagged entries in `data/quarantine.jsonl` |
| Step 9 halts — empty outcomes | Games not finished yet | Re-run `--date YYYY-MM-DD` the following morning |
| ML retrain skips | < 50 outcomes | Normal early in season; no action needed |

---

## Data Contracts

### `retail_df` (Step 3 output)
```
player_name, game, commence_time, bookmaker, american_odds, implied_prob
```

### `anchor_df` (Step 3 output)
```
player_name, game, commence_time, pinnacle_odds, pinnacle_prob,
sharp_anchor, over_only
```

### `final_df` (Step 4 output)
```
player_name, team, game, commence_time,
pinnacle_odds, pinnacle_prob, sharp_anchor, over_only,
best_retail_odds, best_retail_decimal, best_retail_book,
ev_pct, composite_score, composite_z,
kelly_units, stake_usd, anchor_quality,
bet_score, bet_grade,
sim_prob, sim_edge, convergence
```

### `data/quarantine.jsonl`
```json
{"ts": "ISO8601", "step": "step_name", "reason": "reason_code",
 "player": "Name", "game_date": "YYYY-MM-DD", ...row-specific fields...}
```

### `models/correction_factors.json`
```json
{
  "Matt Olson": {
    "factor": 1.23,
    "n": 44,
    "actual_rate": 0.227,
    "predicted_rate": 0.185
  }
}
```

---

## Quarantine Reference

**Location:** `data/quarantine.jsonl`  
**Format:** one JSON object per line, append-only  
**How to read:**
```bash
python -c "
import json
with open('data/quarantine.jsonl') as f:
    for line in f:
        print(json.loads(line))
"
```

**How to clear (after review):**
```bash
# Archive first:
move data\quarantine.jsonl data\quarantine_YYYYMMDD.jsonl
```

**Common reason codes:**

| Code | Step | Meaning |
|------|------|---------|
| `missing_bookmakers` | raw_validation | API event had no bookmakers |
| `impossible_ev_over_200pct` | ev_calculation | EV > 200% — data error |
| `clv_exceeds_50pct` | clv_validation | CLV spike — possible pricing anomaly |
