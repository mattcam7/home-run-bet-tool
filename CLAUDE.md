# Home Run Bet Tool — Claude Instructions

## Project Context

Home run betting prediction tool. Stack TBD as project develops.

The core value of this project is a demonstrably sharp prediction model. Model accuracy and calibration are always the top priority.

When two fix options exist, always implement the correct one even if it requires retraining. Speed of implementation is never a valid reason to choose a technically inferior approach.

We follow a deliberate validation cycle: **pinpoint major issues first → fix them → validate → repeat.** Do not propose incremental tweaks while known structural issues remain unresolved.

Always tackle the hardest item first. Easy wins before hard problems creates false progress — the hard problem still exists and now there's less time to solve it.

## Always Match on IDs, Never Strings

When joining, matching, or looking up players (or any entity) across data sources, always use numeric IDs rather than name strings. Name-string matching silently breaks on duplicate names, Unicode differences, and nicknames.

- Player lookups: use numeric IDs from the data source — never player name strings
- If a data source doesn't expose an ID, document the limitation explicitly and apply a plausibility guard rather than silently writing bad data
- When adding new data pipeline joins, the first question to ask is: "what ID am I joining on?" — if the answer is a name string, flag it as a known risk and add a guard

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

## Insights Suggestions 5/22 

Add under a ## Domain Rules or ## Betting vs DFS section in CLAUDE.md\n\nDistinguish betting EV from DFS GPP value: a player can be a bad bet (-EV) but a great DFS play (high GPP leverage), and vice versa. Never conflate the two when giving recommendations.

Add under a ## Scheduling / Automation section in CLAUDE.md\n\nFor Windows scheduled tasks, never use 'schtasks /SC MINUTE' alone (one-shot trigger); use /SC MINUTE with /MO and verify recurrence, or prefer a robust scheduler. Always validate the task actually recurs before relying on it.

Add near the top under ## Project Overview and a ## Validation subsection\n\nThis is a Python sports analytics codebase (golf DFS, home run betting, NHL props). Always run model validation and suspend/flag bad markets after changing pricing or props logic.
