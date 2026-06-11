# Home Run Bet Tool — Claude Instructions

## Project Context

Home run betting prediction tool. Stack TBD as project develops.

The core value of this project is a demonstrably sharp prediction model. Model accuracy and calibration are always the top priority.

When two fix options exist, always implement the correct one even if it requires retraining. Speed of implementation is never a valid reason to choose a technically inferior approach.

We follow a deliberate validation cycle: **pinpoint major issues first → fix them → validate → repeat.** Do not propose incremental tweaks while known structural issues remain unresolved.

Always tackle the hardest item first. Easy wins before hard problems creates false progress — the hard problem still exists and now there's less time to solve it.

## Live Test After Every Change

After implementing any code, framework, or process change — especially anything that touches GitHub Actions, scheduled workflows, or cloud infrastructure — immediately run a live end-to-end test to catch errors before assuming it works. Never declare a change "done" based on code review alone.

**Required checks after any GitHub Actions change:**
1. Trigger the workflow manually via `workflow_dispatch` and watch it complete
2. Confirm the expected output appeared (Discord message, Supabase row, log entry)
3. Only then consider the task complete

This rule exists because local environments differ from CI (missing packages, different Python versions, no local files) and these gaps cause silent production failures.

## Always Match on IDs, Never Strings

When joining, matching, or looking up players (or any entity) across data sources, always use numeric IDs rather than name strings. Name-string matching silently breaks on duplicate names, Unicode differences, and nicknames.

- Player lookups: use numeric IDs from the data source — never player name strings
- If a data source doesn't expose an ID, document the limitation explicitly and apply a plausibility guard rather than silently writing bad data
- When adding new data pipeline joins, the first question to ask is: "what ID am I joining on?" — if the answer is a name string, flag it as a known risk and add a guard

## Known Limitation: Simulation Agent Name Matching

`agents/simulation.py` joins player data across OddsAPI (player props) and FanGraphs/pybaseball (batter/pitcher stats). Neither source exposes a shared numeric player ID for props markets, making string matching unavoidable.

**Mitigations in place:**
- Names normalized both sides: title case, accent stripped, suffixes (Jr./III) removed
- Exact match attempted first; fuzzy (rapidfuzz token_sort_ratio ≥ 85) only on miss
- Every unmatched player logged to `data/sim_unmatched.log` with timestamp — never silently dropped
- Players with no stats match are excluded from sim predictions (shown as None in sim_prob)

**Residual risk:** Duplicate last names or unusual Unicode variants could produce wrong matches. Monitor `data/sim_unmatched.log` after each run. This limitation will be resolved in v2 when game-log Statcast data is used (Statcast exposes MLBAM numeric IDs).

## Never Assume — Always Verify

Before referencing any file, script, command, endpoint, or function name, confirm it exists first using Glob or Grep. Never tell the user to run a file, call a function, or use a command that hasn't been verified to exist in the codebase. Assumptions about file names, script entry points, or CLI flags that turn out to be wrong waste the user's time and erode trust.

## Research Before Asking

Before asking the user a context question, exhaust all self-serve options first:
- Read relevant source files and grep the codebase
- Query databases directly via available credentials in `.env`
- Search the web for MLB stats or player data if needed

Only ask the user if the answer cannot be found through code, DB queries, or web research.

## Back-Test Before Suggesting Model Changes

Before proposing any model fix, validate the hypothesis against actual data first:
- Confirm the suspected root cause exists at the scale claimed
- Estimate the expected improvement before implementing — if the affected rows are <5% of the training set, the impact will likely be negligible
- If you can simulate the fix locally without retraining, do so and report the predicted impact
- Never recommend a retrain for a fix that hasn't been pre-validated to move the needle

## Using CLAUDE plugins

- Whenever able, use the superpower plugin and brainstorm our best course of action before enacting any changes
- Provide concrete and detailed plans clearly identifying what new deployments aim to fix
- Do not make assumptions and validate all data and workflows
- Use Playwright whenever able to backtest and scrape sites for testing and data validation

## Trigger Keyphrase

Typing `run the hr dashboard` in this Claude Code chat session triggers execution of `python run.py` from the repo root. This fetches today's MLB HR prop odds, computes EV vs Pinnacle's lines, and opens the interactive HTML dashboard in the default browser.

## Sports Analytics Context

- DFS (GPP/cash) value is DISTINCT from betting EV/CLV. Never conflate them: a player can be a fade for outright bets but a strong DFS play due to ownership leverage, salary, and ceiling.
- For betting tools: always track CLV (closing line value) and validate model outputs before recommending bets; auto-suspend markets that fail sanity checks.
- For DFS tools: surface cut%, leverage score, ownership projection, and salary value alongside raw projections.

## Betting vs DFS

Distinguish betting EV from DFS GPP value: a player can be a bad bet (-EV) but a great DFS play (high GPP leverage), and vice versa. Never conflate the two when giving recommendations.

## Data Quality & Pipeline Validation

Correct data is more important than any model improvement. A bad data pipeline silently corrupts every downstream metric — ROI, CLV, EV — and produces false signals that are worse than having no data at all.

**Anchor market mismatch is a first-class bug.** In the HR pipeline, `batter_home_runs_alternate` retail props (DraftKings/FanDuel) cover different events than Pinnacle's standard `batter_home_runs` line (e.g., "HR vs RHP only", "HR in first 5 innings"). Comparing alternate-market retail odds to a standard-market Pinnacle anchor produces false +EV and bad Kelly sizing. Always verify the anchor and retail book are pricing the same market before computing EV.

**Before trusting any model output, run a distribution sanity check:**
- Play count per date: flag any date with >2× the rolling average (signals alternate market contamination or a scraping bug)
- Odds distribution: if >30% of a slate's plays are above +600, inspect for alternate market leakage
- EV rate: if >80% of plays show +EV on a given date, the anchor is likely wrong or mismatched

**Pipeline validation rules:**
- After any change to odds scraping, EV calculation, or market selection logic: run the pipeline, inspect the raw play count and odds histogram before shipping
- After any change affecting which plays enter the CLV log: backfill and diff the ROI summary against the prior baseline
- When adding a new data source or market type: write a validation step that confirms the anchor and retail book are covering the same event before that source goes live
- Never trust aggregated metrics (ROI, CLV%, hit rate) without first validating the underlying play-level data is clean

**Alternate market guard (required for HR pipeline):**
- Any HR prop priced above +600 at retail where Pinnacle does not have a direct line for that player must be excluded or flagged
- Log excluded plays to a separate file so they can be reviewed, not silently dropped

## Windows Task Scheduling

- Do NOT use `schtasks /SC MINUTE` as a one-shot — verify recurrence with `schtasks /Query /TN <name> /V /FO LIST` after creating any scheduled task.
- Prefer creating tasks via XML definition or PowerShell `Register-ScheduledTask` with explicit trigger repetition intervals.
- After scheduling, always confirm the next 2-3 run times are populated before considering the task 'done'.
