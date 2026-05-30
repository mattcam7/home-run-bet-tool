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

AGENT 4 - Audit Agent
**Purpose:** To run through the current version of our project and audit any key dependencies missing and how it can be improved through the lens of an expert sports bettor.
**Schedule:** When invoked but if we could run this on a daily basis for continual improvement and performance tracking that would be optimal.
**Inputs:** The audit agent should use the finalized project version as the primary input, in order to see where improvements can be made.
**Outputs:** This agent should return a finalized report that shows how the current version of our project can be improved with actionable feedback, clear paths, and defined KPIs / goals to try and hit
**Notes:** This agent should always work through the lens of an expert sports bettor.  Ensure any suggestions come through this lens and can actually provide value.  There is a world where we may want to sell a finalized version of this product. 

AGENT 5 - KPI Tracking Agent
**Purpose:** Similar to the audit agent, this should run through all the key dependencies on the project but also look through the chat and look for better ways to implement ideas and tracking of bets.  This should also identify any key context missing that should be added to our tool.
**Schedule:** When invoked and please note this should NOT be a typical agent called when we run our HR betting dashboard scrapes
**Inputs:** The KPI Tracking Agent should assess the chat and betting results (as well as looking at ways to be more efficient via the /insights command.)- Also use brainstorming superpowers to ingest all data and come up with ideas to advance the project.
**Outputs:** The agent should return a finalized analysis and report to the main agent.  I would like it in HTML format and easily readible with solid visuals and clear callouts.
**Notes:** This agent should work through the lens of "How can we make this tool better to truly create high EV bets as well as productionalize something to sell to the public".  Scrape external sites and use their visuals if needed.  We should be presenting EV on a percentage basis.

AGENT 6 - Simulation Agent
**Purpose:** The simulation agent is a new agent that should scrape MLB statistics as best as possible and create or propose modeling techniques that help simulate the likelihood of MLB players hitting home runs in their respective games.
**Schedule:** This agent should be an addition to the typical HR betting dashboard scraper.  We should be using sharp lines, statistics, and linear models to simulate outcomes on an AT BAT scale and provide the likelihood of a player hitting a home run.
**Inputs:** This agent should use external sites such as baseball reference or other sites (potentially fangraphs) to take in adcvanced data, run linear regressions (or something the agent thinks is worthwhile), to figure out which variables are best to assess home run likelihood in a single game.
**Outputs:** This agent should present a report to the main agent that shows our simulation likelihood result, expected value of betting lines presented, and grades / weights around specific variables to solidify and prove the simulation is in good standing.
**Notes:** I realize this is likely and intensive agent process that will require a solid amout of thinking, use the model you see fit and prompt me for context if you need it.  


## WHEN TO RUN ##
**We should run through all agent proccesses through a keyphrase when called, and ensure all data is collected properly through the CLAUDE.md rules**
