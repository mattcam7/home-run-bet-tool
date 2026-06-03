# Discord Bot + Whop MVP Design

**Date:** 2026-06-03  
**Status:** Approved  
**Supersedes:** `2026-06-01-mvp-discord-bot-design.md`

---

## Goal

Deliver daily +EV HR picks to paying subscribers via a private Discord server, fully automated end-to-end with zero manual intervention. Billing via Whop at $15/month.

---

## Architecture

Webhook-based Discord posting — no persistent bot process. GitHub Actions runs all scheduled jobs in the cloud (machine-independent). Supabase hosts all persistent data. Discord `#system-status` provides owner-only health visibility.

```
GitHub Actions (cloud scheduler)
    ├── 11 AM ET daily      → run.py --no-browser  → post_picks()     → #picks
    ├── hourly 11 AM–9 PM   → monitor.py           → post_alert()     → #picks (on movement)
    └── 10 AM ET daily      → post_results.py      → post_results()   → #results
                                                   → post_recap()     → #weekly-recap (Sunday)

All scripts write status to #system-status (owner-only).
GitHub Actions emails owner on workflow failure.

Data layer: Supabase Postgres
    ├── clv_log table       (replaces data/clv_log.csv)
    └── hr_outcomes table   (replaces data/hr_outcomes.db)
```

**Tech stack:** Python, `requests` (Discord webhooks), GitHub Actions (cron), Supabase (Postgres via `psycopg2` or `supabase-py`)

---

## Featured Bet Definition

A play is a **featured bet** when ALL of the following are true:

- `kelly_units >= 0.5` — at least 0.5 unit Kelly conviction
- `ev_pct >= 0.10` — at least 10% edge over anchor's implied probability

Anchor quality is **not** a hard gate. It surfaces in `bet_grade` (Pinnacle-anchored plays score higher than BetOnline-anchored plays via `compute_bet_score`). Subscribers see the grade on every pick.

The `featured_bet` flag is written to Supabase at log time. All ROI/CLV track record metrics are computed on featured bets only.

---

## Discord Channels

| Channel | Audience | Purpose |
|---------|----------|---------|
| `#picks` | Subscribers | Daily picks + line movement alerts |
| `#results` | Subscribers | Next-morning settled outcomes |
| `#weekly-recap` | Subscribers | Sunday ROI/CLV summary |
| `#system-status` | Owner only | Automated health confirmations and failure alerts |

---

## Discord Message Formats

### `#picks` (11 AM ET)

```
⚾ HR Picks — Tue Jun 3

🔴 Strong  Aaron Judge · DK +320 · EV +18.2% · 1.5u · PIN · Bet by 6:35 PM ET
🟡 Solid   Jarren Duran · FD +280 · EV +12.1% · 1.0u · BOL · Bet by 7:05 PM ET
🟡 Solid   Kyle Tucker · DK +350 · EV +10.8% · 0.5u · PIN · Bet by 7:10 PM ET

3 plays · 1u = $25 · Running ROI: +19.4% · CLV: +8.5%
```

- Grade emoji: 🔴 Strong (≥80), 🟡 Solid (60–79), 🟠 Marginal (40–59)
- Sorted descending by `kelly_units` then `ev_pct`
- "Bet by" = first pitch time minus 30 minutes
- Anchor tag: `PIN` = Pinnacle, `BOL` = BetOnline
- Games starting within 90 minutes of post time are excluded
- If zero featured plays: `No featured plays today.`

### Line movement alerts (hourly monitor, posted to `#picks`)

Movement trigger — either condition:
- Retail line moves > 15 points (e.g., +300 → +280 is noise; +300 → +240 is a signal)
- EV drops from ≥10% to below 5%

Withdrawal trigger:
- EV goes negative (play is no longer +EV)

```
⚠️ Line alert — Aaron Judge: DK +320 → +240 · EV +18.2% → +6.1% · edge reduced
❌ Withdrawal — Jarren Duran: FD +280 → +190 · EV now -3.2% · skip this play
```

No human judgment required — thresholds fire automatically.

### `#results` (10 AM ET)

```
📋 Results — Mon Jun 2

✅ Aaron Judge · DK +320 · HIT · +$37.50
❌ Jarren Duran · FD +280 · miss · -$25.00
✅ Kyle Tucker · DK +350 · HIT · +$12.50

Day: +$25.00 · Week: +$62.50 · Running ROI: +19.4%
```

- Scratched/no-result: `Player · scratched — no result`
- Skip post entirely if no featured bets were logged that day

### `#weekly-recap` (Sunday only, posted after results)

```
📊 Weekly Recap · Jun 2 – Jun 8
This week: 12 plays · 5 hits · 41.7% · P&L: +$87.50
All-time:  148 plays · 19.4% ROI · +8.5% CLV · 38.5% hit rate
Anchor: Pinnacle/BOL devig · 1u = $25 · Kelly ¼ sizing
```

### `#system-status` (owner-only, every run)

| Event | Message |
|-------|---------|
| Picks posted | `✅ Picks posted — 5 plays · Jun 3 11:00 AM ET` |
| Picks posted, 0 plays | `⚠️ Picks ran — 0 featured plays found · check pipeline` |
| Picks failed | `❌ Picks FAILED — Jun 3 11:00 AM ET: {error}` |
| Monitor run (no movement) | `✅ Monitor — no significant movement · Jun 3 1:00 PM ET` |
| Monitor run (alert sent) | `⚠️ Monitor — line alert posted: Judge +320→+240 · Jun 3 1:00 PM ET` |
| Results posted | `✅ Results posted — Jun 2 · 3 settled` |
| Results failed | `❌ Results FAILED — Jun 2: {error}` |

GitHub Actions also sends an email to the owner on any workflow failure (built-in).

---

## Data Layer: Supabase

Replaces local `data/clv_log.csv` and `data/hr_outcomes.db`.

### `clv_log` table (mirrors existing CSV columns + new ones)

| Column | Type | Notes |
|--------|------|-------|
| id | serial PK | |
| game_date | date | |
| player_name | text | |
| best_retail_book | text | |
| best_retail_odds | integer | |
| pinnacle_over_odds | integer | |
| anchor_quality | text | pinnacle / betonlineag / unknown / pinnacle_over_only |
| ev_pct | numeric | |
| kelly_units | numeric | |
| stake_usd | numeric | |
| bet_score | integer | |
| bet_grade | text | |
| featured_bet | boolean | |
| closing_pinnacle_prob | numeric | nullable until captured |
| clv_pct | numeric | nullable until captured |
| logged_at | timestamptz | |

### `hr_outcomes` table (mirrors existing SQLite schema)

| Column | Type | Notes |
|--------|------|-------|
| id | serial PK | |
| game_date | date | |
| player_name | text | |
| hit_hr | boolean | nullable until settled |
| settled_at | timestamptz | nullable |

### Access pattern

All scripts connect via `SUPABASE_URL` + `SUPABASE_KEY` from environment variables (GitHub Actions secrets, local `.env`). Use `supabase-py` client for insert/select; fall back to `psycopg2` for bulk operations.

---

## GitHub Actions Workflows

### `.github/workflows/daily_picks.yml`
- Schedule: `0 15 * * *` (15:00 UTC = 11:00 AM ET)
- Command: `python run.py --no-browser`
- Secrets: `ODDS_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, `DISCORD_PICKS_WEBHOOK`, `DISCORD_STATUS_WEBHOOK`

### `.github/workflows/line_monitor.yml`
- Schedule: `0 15-1 * * *` (every hour 15:00–01:00 UTC = 11 AM–9 PM ET)
- Command: `python monitor.py`
- Secrets: same as above + `DISCORD_PICKS_WEBHOOK`

### `.github/workflows/post_results.yml`
- Schedule: `0 14 * * *` (14:00 UTC = 10:00 AM ET)
- Command: `python post_results.py`
- Secrets: `SUPABASE_URL`, `SUPABASE_KEY`, `DISCORD_RESULTS_WEBHOOK`, `DISCORD_RECAP_WEBHOOK`, `DISCORD_STATUS_WEBHOOK`

---

## Files

### New files

| File | Purpose |
|------|---------|
| `agents/discord_bot.py` | `post_picks()`, `post_results()`, `post_weekly_recap()`, `post_status()`, `post_alert()` |
| `agents/supabase_client.py` | Thin wrapper: `insert_clv_rows()`, `fetch_clv_log()`, `fetch_outcomes()`, `upsert_outcome()` |
| `monitor.py` | Hourly script: re-scrapes picks logged today, compares to current odds, fires alerts on threshold breach |
| `post_results.py` | Morning script: settle outcomes → post results → Sunday recap |
| `.github/workflows/daily_picks.yml` | GitHub Actions: 11 AM ET picks |
| `.github/workflows/line_monitor.yml` | GitHub Actions: hourly monitor |
| `.github/workflows/post_results.yml` | GitHub Actions: 10 AM ET results |

### Modified files

| File | Change |
|------|--------|
| `run.py` | Add `--no-browser` flag; call `discord_bot.post_picks()` when set; replace CSV write with `supabase_client.insert_clv_rows()` |
| `agents/clv_log.py` | Add `featured_bet` column; write to Supabase instead of CSV |
| `agents/outcome_tracker.py` | Add `featured_only=False` to `compute_roi_metrics()`; read/write Supabase instead of SQLite |
| `agents/capture_closing.py` | Write closing captures to Supabase `clv_log` instead of CSV |

---

## Environment Variables

### Local `.env` (never committed)
```
ODDS_API_KEY=...
SUPABASE_URL=https://<project>.supabase.co
SUPABASE_KEY=<service_role_key>
DISCORD_PICKS_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_RESULTS_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_RECAP_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_STATUS_WEBHOOK=https://discord.com/api/webhooks/...
```

### GitHub Actions secrets (same keys, set in repo Settings → Secrets)

`discord_bot.py` raises `EnvironmentError` at import if any Discord webhook vars are missing.

---

## `agents/discord_bot.py` — Function Specs

### `post_picks(final_df, now=None)`

1. Filter to `featured_bet == True`; exclude games starting within 90 min of `now`.
2. Sort by `kelly_units` desc, `ev_pct` desc.
3. If empty: post `"No featured plays today."` to picks webhook + status webhook.
4. Format each pick with grade emoji, player, book+odds, EV%, kelly units, anchor tag, "Bet by" time.
5. Footer: `{n} plays · 1u = $25 · Running ROI: {roi}% · CLV: {clv}%` from `compute_roi_metrics(featured_only=True)`.
6. POST to `DISCORD_PICKS_WEBHOOK`.
7. POST status to `DISCORD_STATUS_WEBHOOK`.
8. Catch all exceptions → log to `data/discord.log` → post failure to status webhook → do not re-raise.

### `post_alert(player_name, old_odds, new_odds, old_ev, new_ev, alert_type)`

- `alert_type`: `"movement"` or `"withdrawal"`
- Movement: `⚠️ Line alert — {player}: {book} {old_odds} → {new_odds} · EV {old_ev}% → {new_ev}% · edge reduced`
- Withdrawal: `❌ Withdrawal — {player}: {book} {old_odds} → {new_odds} · EV now {new_ev}% · skip this play`
- POST to `DISCORD_PICKS_WEBHOOK`.

### `post_results(date_str, now=None)`

1. Fetch CLV log from Supabase: `game_date == date_str AND featured_bet == True`.
2. Join with outcomes table on `(game_date, player_name)`.
3. If no featured bets: return without posting.
4. Format settled picks: hit → `✅`, miss → `❌`, NULL → scratched.
5. Footer: day P&L, week P&L, running ROI.
6. POST to `DISCORD_RESULTS_WEBHOOK` + status to `DISCORD_STATUS_WEBHOOK`.
7. Catch all exceptions → log → post failure to status → do not re-raise.

### `post_weekly_recap(now=None)`

1. Pull `compute_roi_metrics(featured_only=True)`.
2. Filter this-week picks: `game_date >= last Monday`.
3. Format and POST to `DISCORD_RECAP_WEBHOOK`.
4. Catch all exceptions → log → post failure to status → do not re-raise.

### `post_status(message)`

POST plain text to `DISCORD_STATUS_WEBHOOK`. Never raises — if this fails, silently pass (avoid infinite error loop).

---

## `monitor.py` — Script Spec

Runs hourly 11 AM–9 PM ET.

```
1. Fetch today's featured bets from Supabase clv_log
2. For each pick where game has not yet started:
   a. Re-scrape current retail odds via odds_scraper
   b. Re-compute EV against stored pinnacle_open_prob
   c. If retail line moved > 15 points OR EV dropped below 5%: call post_alert(..., "movement")
   d. If EV < 0: call post_alert(..., "withdrawal"); mark play as withdrawn in Supabase
3. Post status summary to #system-status
```

Tracks which alerts have already been sent (stored in Supabase or a local state file) to avoid re-alerting on the same move every hour.

---

## `post_results.py` — Script Spec

```
1. Run outcome_tracker.update_for_date(yesterday)  → writes to Supabase hr_outcomes
2. Call discord_bot.post_results(yesterday)
3. If today is Sunday: call discord_bot.post_weekly_recap()
4. Log completion
```

---

## `agents/supabase_client.py` — Function Specs

```python
def insert_clv_rows(rows: list[dict]) -> None
def fetch_clv_log(game_date=None, featured_only=False) -> pd.DataFrame
def upsert_outcome(game_date, player_name, hit_hr) -> None
def fetch_outcomes(game_date=None) -> pd.DataFrame
def mark_withdrawn(game_date, player_name) -> None
```

Connects via `SUPABASE_URL` + `SUPABASE_KEY`. Raises `EnvironmentError` if either is missing.

---

## `agents/clv_log.py` Changes

- `featured_bet` computed at log time: `kelly_units >= 0.5 AND ev_pct >= 0.10`
- Writes via `supabase_client.insert_clv_rows()` instead of appending to CSV
- Local CSV write retained as fallback if `SUPABASE_KEY` not set (dev/offline mode)

---

## `agents/outcome_tracker.py` Changes

```python
def compute_roi_metrics(featured_only: bool = False):
    clv = supabase_client.fetch_clv_log(featured_only=featured_only)
    outcomes = supabase_client.fetch_outcomes()
    # existing contamination filter still applied
```

---

## One-Time Manual Setup (outside codebase)

1. Create Supabase project → create `clv_log` and `hr_outcomes` tables with schemas above
2. Migrate existing `data/clv_log.csv` and `data/hr_outcomes.db` into Supabase
3. Create private Discord server → 4 channels (`#picks`, `#results`, `#weekly-recap`, `#system-status`)
4. Set `#system-status` to owner-only permissions; set other 3 to subscriber role + owner
5. Generate 4 webhook URLs → add to `.env` and GitHub Actions secrets
6. Add all other secrets to GitHub Actions (repo Settings → Secrets and variables → Actions)
7. Create Whop product at $15/month → connect to Discord server (auto-assigns subscriber role)
8. Keep Discord server invite-only (no public discovery link)

---

## Error Handling

- All Discord posting functions catch exceptions, log to `data/discord.log`, post failure to `#system-status`, and never re-raise
- GitHub Actions emails owner on workflow failure (built-in)
- `discord_bot.py` raises `EnvironmentError` at import if webhook vars missing — misconfiguration caught immediately, not silently at post time
- `supabase_client.py` raises `EnvironmentError` if connection vars missing
- `monitor.py` tracks sent alerts in Supabase to avoid duplicate alerts per hour

---

## Out of Scope (V1)

- Persistent Discord bot (slash commands, DMs)
- Public track record webpage
- DFS integration
- Email delivery
- Multiple pricing tiers
- Line movement history chart
- Mobile push notifications
