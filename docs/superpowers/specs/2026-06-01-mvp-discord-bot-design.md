# MVP Discord Bot Design

**Goal:** Deliver a paying subscriber product via Discord + Whop: daily +EV HR picks, morning results, and weekly ROI recap — all automated, no manual intervention.

**Architecture:** Webhook-based Discord posting (no persistent bot process). Two scheduled tasks drive the full loop. Backend stores all plays; only Pinnacle-anchored featured bets count toward the advertised track record.

**Tech Stack:** Python, `requests` (Discord webhooks), Windows Task Scheduler (XML), existing pipeline (CLV log, outcomes DB, `compute_roi_metrics`)

---

## Product Decisions

- **Delivery:** Private Discord server, gated by Whop subscriber role
- **Billing:** Whop at $49/month (handles role assignment automatically)
- **DFS:** Excluded from V1
- **Channels:** `#picks`, `#results`, `#weekly-recap`

## Featured Bet Definition

A play is a **featured bet** when ALL of the following are true:
- `anchor_quality == "pinnacle"` — Pinnacle devigged probability (most reliable anchor; BetOnline-anchored plays are posted as informational only)
- `kelly_units >= 0.5` — at least 0.5 unit Kelly conviction
- `ev_pct >= 0.10` — at least 10% edge over Pinnacle's implied probability

The `featured_bet` flag is written to the CLV log at log time (`log_open_plays`). All plays are stored in the DB. ROI/CLV success metrics are computed **only on featured bets**.

BetOnline-anchored plays appear in `#picks` under a clearly labeled "Informational" section but are never included in track record metrics.

---

## Files

### New files
- `agents/discord_bot.py` — three posting functions (picks, results, weekly recap)
- `post_results.py` — morning script: runs outcome tracker then posts results; posts weekly recap on Sundays
- `tasks/HR_DailyPicks.xml` — scheduled task: 2 PM ET daily, runs `run.py --no-browser`
- `tasks/HR_PostResults.xml` — scheduled task: 10 AM ET daily, runs `post_results.py` (replaces direct `outcome_tracker` call)

### Modified files
- `run.py` — add `--no-browser` CLI flag; call `discord_bot.post_picks(final_df)` after EV calculation
- `agents/clv_log.py` — add `featured_bet` column to `OPEN_COLS`; compute and store flag in `log_open_plays()`
- `agents/outcome_tracker.py` — add `featured_only: bool = False` parameter to `compute_roi_metrics()`; filter on `featured_bet == True` when set
- `.env` — add three webhook URL variables (not committed to git)

---

## Environment Variables

```
DISCORD_PICKS_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_RESULTS_WEBHOOK=https://discord.com/api/webhooks/...
DISCORD_RECAP_WEBHOOK=https://discord.com/api/webhooks/...
```

All three are required. `discord_bot.py` raises `EnvironmentError` at import if any are missing, so failures are loud and immediate rather than silent.

---

## `agents/discord_bot.py` — Function Specs

### `post_picks(final_df, now=None)`

Called from `run.py` after `calculate_ev()`. Posts to `DISCORD_PICKS_WEBHOOK`.

**Logic:**
1. Split `final_df` into featured bets (`featured_bet == True`) and informational (`featured_bet == False AND ev_pct > 0`).
2. Sort each group descending by `kelly_units`, then `ev_pct`.
3. If zero featured bets: post "No featured plays today — {n} informational plays below."
4. Format each featured play: `Player · Book +Odds · EV +X.X% · Xu · PIN`
5. Format each informational play: `Player · Book +Odds · EV +X.X% · BOL` (no Kelly shown — not a sized recommendation)
6. Footer: `{n_featured} featured · {n_info} informational · 1u = $25 · Running ROI: {roi}% · CLV: {clv}%`
7. Running ROI/CLV pulled from `compute_roi_metrics(featured_only=True)`.
8. POST as a single Discord message (plain text, no embeds — works on mobile).

**Error handling:** Catch all exceptions, log to `data/discord.log`, do not re-raise (pipeline must not crash on Discord failure).

### `post_results(date_str, now=None)`

Called from `post_results.py` each morning. Posts to `DISCORD_RESULTS_WEBHOOK`.

**Logic:**
1. Load CLV log; filter to `game_date == date_str AND featured_bet == True`.
2. Join with outcomes DB on `(game_date, player_name)`.
3. Skip post if no featured bets that day (bot ran but no featured plays were logged).
4. Format each settled pick: `Player +Odds Book · HIT ✓ · +$XX.XX` or `Player +Odds Book · miss · -$XX.XX`
5. Scratched players (hit_hr is NULL): `Player · scratched — no result`
6. Footer: `Day P&L: +/-$X · Week P&L: +/-$X · Running ROI: +X.X% · {n} settled`
7. POST to results webhook.

### `post_weekly_recap(now=None)`

Called from `post_results.py` on Sundays only (`datetime.now().weekday() == 6`). Posts to `DISCORD_RECAP_WEBHOOK`.

**Logic:**
1. Pull `compute_roi_metrics(featured_only=True)`.
2. Compute this-week P&L: filter settled picks to `game_date >= last Monday`.
3. Format:
   ```
   📊 Weekly Recap · Mon DD – Sun DD
   This week: {n} plays · {n_hr} hits · {hit_rate}% · P&L: +/-$X
   All-time:  {n_total} plays · {roi}% ROI · {clv}% CLV · {hit_rate}% hit rate
   Anchor: Pinnacle devig · 1u = $25 · Kelly ¼ sizing
   ```
4. POST to recap webhook.

---

## `post_results.py` — Script Spec

Entry point for the 10 AM ET scheduled task.

```
1. Run outcome_tracker.update_for_date(yesterday)
2. Call discord_bot.post_results(yesterday)
3. If today is Sunday: call discord_bot.post_weekly_recap()
4. Log completion to data/discord.log
```

Does not open a browser. Does not regenerate the dashboard. Runs in under 30 seconds.

---

## `run.py` Changes

Add `argparse` with a single flag:
- `--no-browser`: skip `webbrowser.open()` and call `discord_bot.post_picks(final_df)` instead.

When `--no-browser` is absent (manual run): existing behavior unchanged — opens dashboard, does NOT post to Discord (avoids double-posting when user runs manually).

When `--no-browser` is present (scheduled run): no browser, posts to Discord.

---

## `agents/clv_log.py` Changes

Add `featured_bet` to `OPEN_COLS` (after `anchor_quality`):

```python
OPEN_COLS = [
    ...,
    "anchor_quality",
    "featured_bet",   # True = Pinnacle anchor + kelly>=0.5 + ev>=10%
]
```

In `log_open_plays()`, compute the flag before appending:

```python
featured = (
    str(r.get("anchor_quality", "")) == "pinnacle"
    and float(r.get("kelly_units", 0)) >= 0.5
    and float(r.get("ev_pct", 0)) >= 0.10
)
rows.append({
    ...,
    "featured_bet": featured,
})
```

Retroactive rows (pre-flag): `featured_bet` reads back as NaN/blank — treated as `False` in all downstream logic.

---

## `agents/outcome_tracker.py` Changes

`compute_roi_metrics(featured_only: bool = False)`:

```python
if featured_only and "featured_bet" in clv.columns:
    clv = clv[clv["featured_bet"].astype(str) == "True"].copy()
```

Applied after the existing `anchor_quality != "pinnacle_over_only"` contamination filter. The two filters compose cleanly.

---

## Scheduled Tasks

### `HR_DailyPicks` (new)
- Trigger: daily at 2:00 PM ET
- Command: `python run.py --no-browser >> data\picks.log 2>&1`
- Working dir: `C:\Users\mattc\home_run_bet_tool`

### `HR_PostResults` (replaces direct outcome tracker invocation)
- Trigger: daily at 10:00 AM ET (same time as existing `HR_OutcomeTracker`)
- Command: `python post_results.py >> data\results.log 2>&1`
- Working dir: `C:\Users\mattc\home_run_bet_tool`
- The existing `HR_OutcomeTracker` task is **deleted** — `post_results.py` runs the tracker internally.

---

## Discord Channel Setup (manual, one-time)

Before the bot can post, set up in Discord:
1. Create a private Discord server
2. Create three text channels: `#picks`, `#results`, `#weekly-recap`
3. Create a subscriber role (Whop assigns this automatically on payment)
4. Set channel permissions: only the subscriber role (and you) can read
5. Generate webhook URLs: channel Settings → Integrations → Webhooks → New Webhook → Copy URL
6. Add the three webhook URLs to `.env`
7. Set up Whop product page at whop.com, connect to the Discord server, set price to $49/month

---

## Out of Scope (V1)

- Persistent Discord bot (slash commands, DMs, member count tracking)
- Public track record webpage
- DFS integration
- Email delivery
- BetOnline-anchored plays in the featured track record
- Line movement alerts
