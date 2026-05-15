# Home Run Bet Tool — Agents

Agents defined here are scheduled or triggered automations that run against this repo.

---

## Agent Template

```
### [Agent Name]

**Purpose:** What this agent does and why it exists.
**Schedule:** Cron expression or trigger event (e.g. `0 9 * * *` = daily 9am ET)
**Inputs:** Data sources, APIs, or files it reads
**Outputs:** Where results go (DB, Slack, file, etc.)
**Notes:** Any constraints, dependencies, or edge cases
```

---

## Agents

AGENT 1 - Odds API Scraper
**Purpose:** To scrape the odds API for today's baseball games being played and return ALL AVAILABLE HOME RUN MARKETS and the odds behind them
**Schedule:** When invoked within this chat
**Inputs:** OddsAPI — key loaded from `$ODDS_API_KEY` in `.env`
**Outputs:** Should be output as a pandas dataframe and something I can easily copy into google sheets - also would be good to express the odds in percent form
**Notes:** Ensure we are only scraping for today's unplayed games (not future games or games that have already begun)

AGENT 2 - PINNACLE ODDS SCRAPER
**Purpose:** To scrape the same home run odds but through the Pinnacle sportsbook.
**Schedule:** When invoked within this chat
**Inputs:** We can use the same oddsapi key but if it's not availble we should use an alternative source (webscraping)
**Outputs:** Should be output as a pandas dataframe and someething I can easily copy into google sheets - should also be converted into percent form
**Notes:** This is important to ensure if odds are available at retail sportsbook, then we also have the odds availanble at pinnacle - do not pull player odds if they aren't available at both retail and pinnacle books

AGENT 3 - EV Agent
**Purpose:** To show the difference between retail odds and pinnacle odds for a player to hit a homerun
**Schedule:** When invoked within this chat
**Inputs:** Returned data from the previous 2 agents
**Outputs:** All finalized data merged into a single dataframe, including the ev we gain (Pinnacle odds are sharper so if a player has better odds at pinnacle then retail this would be positive expected value)
**Notes:** We should show all ev calculations but also account for players who have high ev solely based on lower odds.  EX - A player who is showing a 5% chance at pinnacle but a 1% chance at retail is true 4% EV but it's still an extremely low likelihood event.  There is likely more value in a player with 18% at pinnacle but 16% at retail.  Technically lower EV but much more likely to hit.

## WHEN TO RUN ##
**We should run through all agent proccesses through a keyphrase when called, and ensure all data is collected properly through the CLAUDE.md rules**
