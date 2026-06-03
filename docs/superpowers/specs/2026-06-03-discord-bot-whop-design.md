# Discord Bot + Whop MVP Design

**Date:** 2026-06-03  
**Status:** Approved  
**Supersedes:** `2026-06-01-mvp-discord-bot-design.md`

---

## Goal

Deliver daily +EV HR picks to paying subscribers via a private Discord server, automated end-to-end with no manual intervention. Billing via Whop at $15/month.

## Architecture

Webhook-based Discord posting — no persistent bot process. Two Windows scheduled tasks drive the full loop.

| Time | Task | What happens |
|------|------|-------------|
| 2 PM ET daily | `HR_DailyPicks` | `run.py --no-browser` → EV calc → `discord_bot.post_picks()` |
| 10 AM ET daily | `HR_PostResults` | `post_results.py` → settle outcomes → `discord_bot.post_results()` → Sunday: `discord_bot.post_weekly_recap()` |

**Tech stack:** Python, `requests` (Discord webhooks), Windows Task Scheduler (XML), existing pipeline (`clv_log.csv`, `hr_outcomes.db`, `compute_roi_metrics`)

---

## Featured Bet Definition

A play is a **featured bet** when ALL of the following are true:

- `kelly_units >= 0.5` — at least 0.5 unit Kelly conviction
- `ev_pct >= 0.10` — at least 10% edge over anchor's implied probability

Anchor quality is **not** a hard gate. It is reflected in `bet_grade` (which already weights Pinnacle at 1.00 and BetOnline at 0.75 via `compute_bet_score`). A Strong-graded play signals a Pinnacle-anchored edge; Solid/Marginal plays signal a weaker anchor. Subscribers see the grade on every pick.

The `featured_bet` flag is written to the CLV log at log time. All ROI/CLV track record metrics are computed on featured bets only.

---

## Discord Message Formats

### `#picks` (2 PM ET)

```
⚾ HR Picks — Tue Jun 3

🔴 Strong  Aaron Judge · DK +320 · EV +18.2% · 1.5u · PIN
🟡 Solid   Jarren Duran · FD +280 · EV +12.1% · 1.0u · BOL
🟡 Solid   Kyle Tucker · DK +350 · EV +10.8% · 0.5u · PIN

3 plays · 1u = $25 · Running ROI: +19.4% · CLV: +8.5%
```

Grade emoji: 🔴 Strong (≥80), 🟡 Solid (60–79), 🟠 Marginal (40–59). Sorted descending by `kelly_units` then `ev_pct`.

Anchor tag: `PIN` = Pinnacle, `BOL` = BetOnline.

If zero featured plays: `No featured plays today.`

### `#results` (10 AM ET)

```
📋 Results — Mon Jun 2

✅ Aaron Judge · DK +320 · HIT · +$37.50
❌ Jarren Duran · FD +280 · miss · -$25.00
✅ Kyle Tucker · DK +350 · HIT · +$12.50

Day: +$25.00 · Week: +$62.50 · Running ROI: +19.4%
```

Scratched/no-result players: `Player · scratched — no result`

Skip post entirely if no featured bets were logged that day.

### `#weekly-recap` (Sunday only)

```
📊 Weekly Recap · Jun 2 – Jun 8
This week: 12 plays · 5 hits · 41.7% · P&L: +$87.50
All-time:  148 plays · 19.4% ROI · +8.5% CLV · 38.5% hit rate
Anchor: Pinnacle/BOL devig · 1u = $25 · Kelly ¼ sizing
```

---

## Files

### New files

- `agents/discord_bot.py` — `post_picks(df)`, `post_results(date_str)`, `post_weekly_recap()`
- `post_results.py` — morning entry point: runs outcome tracker then posts results + Sunday recap
- `tasks/HR_DailyPicks.xml` — scheduled task: 2 PM ET daily, `run.py --no-browser`
- `tasks/HR_PostResults.xml` — scheduled task: 10 AM ET daily, `post_results.py`

### Modified files

| File | Change |
|------|--------|
| `run.py` | Add `--no-browser` argparse flag; call `discord_bot.post_picks(final_df)` when flag is set |
| `agents/clv_log.py` | Add `featured_bet` column to `OPEN_COLS`; compute and store flag in `log_open_plays()` |
| `agents/outcome_tracker.py` | Add `featured_only: bool = False` to `compute_roi_metrics()`; filter on `featured_bet == True` when set |

---

## Environment Variables (`.env` — never committed)

```
DISCORD_PICKS_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_RESULTS_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_RECAP_WEBHOOK=https://discord.com/api/webhooks/...
```

`discord_bot.py` raises `EnvironmentError` at import if any are missing — failures are loud, not silent.

---

## `agents/discord_bot.py` — Function Specs

### `post_picks(final_df, now=None)`

1. Filter `final_df` to `featured_bet == True`; sort by `kelly_units` desc, `ev_pct` desc.
2. If empty: post `"No featured plays today."` and return.
3. Format each play: `{grade_emoji} {grade}  {player} · {book} {odds} · EV +{ev}% · {kelly}u · {anchor_tag}`
4. Footer: `{n} plays · 1u = $25 · Running ROI: {roi}% · CLV: {clv}%` (pulled from `compute_roi_metrics(featured_only=True)`)
5. POST plain text to `DISCORD_PICKS_WEBHOOK`.
6. Catch all exceptions → log to `data/discord.log` → do not re-raise.

Grade emoji map: Strong → `🔴`, Solid → `🟡`, Marginal → `🟠`, Skip → omitted (skip plays never reach featured).

Anchor tag: `anchor_quality == "pinnacle"` → `PIN`, else → `BOL`.

### `post_results(date_str, now=None)`

1. Load CLV log; filter to `game_date == date_str AND featured_bet == True`.
2. Join with outcomes DB on `(game_date, player_name)`.
3. If no featured bets: return without posting.
4. Format each settled pick: hit → `✅ {player} · {book} {odds} · HIT · +${pnl}`, miss → `❌ {player} · {book} {odds} · miss · -${pnl}`, NULL → `{player} · scratched — no result`
5. Footer: `Day: {day_pnl} · Week: {week_pnl} · Running ROI: {roi}%`
6. POST to `DISCORD_RESULTS_WEBHOOK`.
7. Catch all exceptions → log to `data/discord.log` → do not re-raise.

### `post_weekly_recap(now=None)`

1. Pull `compute_roi_metrics(featured_only=True)`.
2. Compute this-week P&L: filter settled picks to `game_date >= last Monday`.
3. Format and POST to `DISCORD_RECAP_WEBHOOK`.
4. Catch all exceptions → log to `data/discord.log` → do not re-raise.

---

## `post_results.py` — Script Spec

```
1. Run outcome_tracker.update_for_date(yesterday)
2. Call discord_bot.post_results(yesterday)
3. If today is Sunday: call discord_bot.post_weekly_recap()
4. Log completion to data/discord.log
```

Runs in under 30 seconds. No browser. No dashboard regeneration.

---

## `run.py` Changes

Add `argparse` with a single flag:
- `--no-browser`: skip `webbrowser.open()`, call `discord_bot.post_picks(final_df)` instead.

Manual run (no flag): opens dashboard, does NOT post to Discord.  
Scheduled run (`--no-browser`): no browser, posts to Discord.

---

## `agents/clv_log.py` Changes

Add `featured_bet` to `OPEN_COLS` after `anchor_quality`.

In `log_open_plays()`, compute before appending:
```python
featured = (
    float(r.get("kelly_units", 0)) >= 0.5
    and float(r.get("ev_pct", 0)) >= 0.10
)
```

Retroactive rows (pre-flag): `featured_bet` reads as NaN — treated as `False` everywhere downstream.

---

## `agents/outcome_tracker.py` Changes

```python
def compute_roi_metrics(featured_only: bool = False):
    ...
    if featured_only and "featured_bet" in clv.columns:
        clv = clv[clv["featured_bet"].astype(str) == "True"].copy()
    # existing pinnacle_over_only filter still applied after
```

---

## Scheduled Tasks

### `HR_DailyPicks`
- Trigger: daily 2:00 PM ET
- Command: `python run.py --no-browser >> data\picks.log 2>&1`
- Working dir: `C:\Users\mattc\home_run_bet_tool`

### `HR_PostResults`
- Trigger: daily 10:00 AM ET
- Command: `python post_results.py >> data\results.log 2>&1`
- Working dir: `C:\Users\mattc\home_run_bet_tool`
- Replaces existing `HR_OutcomeTracker` task (delete the old one after registering this)

---

## One-Time Manual Setup (outside codebase)

1. Create a private Discord server
2. Create channels: `#picks`, `#results`, `#weekly-recap`
3. Create a subscriber role; restrict each channel to that role + owner
4. Generate 3 webhook URLs (channel Settings → Integrations → Webhooks) → add to `.env`
5. Create Whop product at $15/month → connect to Discord server → Whop auto-assigns subscriber role on payment
6. Keep server invite-only (no public discovery link)

---

## Error Handling

All Discord posting functions catch exceptions and write to `data/discord.log` with timestamp. The pipeline (run.py, post_results.py) never crashes due to Discord failures. Webhook env vars validated at import time so misconfiguration is caught immediately on first run, not silently at post time.

---

## Out of Scope (V1)

- Persistent Discord bot (slash commands, DMs)
- Public track record webpage
- DFS integration
- Email delivery
- Line movement alerts
- Multiple pricing tiers
