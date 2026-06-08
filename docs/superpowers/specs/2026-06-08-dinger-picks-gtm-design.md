# Dinger Picks — Go-to-Market Design Spec

**Date:** 2026-06-08
**Status:** Approved

---

## Goal

Launch a $15/month subscription home run picks service ("Dinger Picks") targeting recreational bettors. Picks are model-driven, Pinnacle-anchored, and delivered daily to a private Discord server. The product launches immediately while CLV data accumulates in the background to support long-term credibility.

---

## 1. Product Overview

**Name:** Dinger Picks

**Value proposition:** A machine learning model trained on 200,000+ Statcast game logs, anchored to Pinnacle's sharpest closing lines. Every pick posted is +EV — or it doesn't get posted.

**Price:** $15/month (beta pricing, cancel anytime)

**Delivery channel:** Private Discord server, gated by Whop subscription

**Primary audience:** Recreational bettors who want a simple, trustworthy picks service without needing to understand the model

**Secondary audience:** Analytics-minded bettors who want access to the raw model output (EV%, sim prob, Kelly sizing)

---

## 2. Landing Page

### Hosting
GitHub Pages, deployed from the `main` branch of this repository under a `docs/` or `gh-pages` branch. Custom domain: `dingerpicks.com` (or similar — ~$10/yr on Namecheap/Cloudflare). HTTPS via GitHub Pages automatic TLS.

Deployment: GitHub Actions workflow pushes the landing page static files to GitHub Pages on every push to `main`.

### Visual Direction
**Bold/Sports aesthetic** — deep navy and red, sports-forward. Familiar to casual bettors; lower barrier to entry than a quant-terminal aesthetic.

Color palette:
- Background: `#111` / `#0d0d1a`
- Primary accent: `#e94560` (red)
- Secondary: `#16213e`, `#0f3460` (navy shades)
- Text muted: `#a8b2d8`
- Body copy muted: `#64748b`

### Page Structure (top to bottom)

1. **Nav** — Logo ("⚾ DINGER PICKS"), "Join $15/mo" CTA button (top right)
2. **Hero** — Tag line ("⚾ DAILY MLB HOME RUN PICKS"), H1 ("Daily Home Run Picks That Win."), sub-copy explaining model, primary CTA ("GET PICKS — $15/MO →"), sub-note ("Cancel anytime · Delivered to Discord daily")
3. **Stats bar** — 4 stats: 200k+ Statcast game logs, 0.68 AUC (2022 holdout), Daily picks posted, $15/mo
4. **How It Works** — 3 cards: (1) Sharp Lines, (2) Statcast Sim, (3) +EV Picks Only
5. **Discord Channels** — 3 channel cards: #picks, #data, #results
6. **Example Pick** — Sample pick card: player name, game, book+odds (green), EV% (green), sim prob, unit size
7. **Pricing** — Single card with "BETA PRICE" badge, $15/mo, feature list, "JOIN NOW →" Whop CTA
8. **Footer** — Legal disclaimer ("Picks are for informational purposes only · 21+ · Gamble responsibly"), copyright

### Stats Bar — Placeholder Policy
The "Daily" picks stat and "$15" price stat are not numeric gaps that need filling. The CLV beat-close rate is intentionally omitted from the stats bar until ≥20 closing lines are captured. Once that gate is passed (via the `validate-models` skill), add a fifth stat: "XX% CLV Beat Rate" sourced from the CLV report.

### Copy Guardrails
- Do not claim specific ROI, win rate, or profit unless directly supported by CLV log data
- All picks framed as "+EV" not "guaranteed winners"
- Footer disclaimer required on every page

---

## 3. Discord Server

### Structure
Private invite-only server. No public Discord discovery link ever published.

| Channel | Audience | Content | Posted by |
|---|---|---|---|
| `#picks` | All subscribers | Simple daily picks: player, book, odds, unit size | Pipeline automation |
| `#data` | All subscribers | Full model output: EV%, sim prob, Pinnacle line, Kelly sizing | Pipeline automation |
| `#results` | All subscribers | Morning outcome recap, running ROI | Pipeline automation |
| `#system-status` | Owner only | Pipeline health, run errors, exclusion log summaries | Pipeline automation |

### Pick Format — `#picks`

```
⚾ TODAY'S HR PICKS — [Date]

🎯 Aaron Judge — NYY vs BAL
   DraftKings: +450 | 1.5u
   +EV vs Pinnacle's line

🎯 [Player] — [Matchup]
   [Book]: [Odds] | [Units]u
   +EV vs Pinnacle's line

---
Posted by Dinger Picks | bet responsibly
```

### Pick Format — `#data`

```
📊 MODEL DATA — [Date]

Aaron Judge — NYY vs BAL
  Pinnacle: +380 (de-vig: 20.8%)
  Sim prob: 16.2%
  EV: +11.2%
  Kelly: 1.5u

[Player] — [Matchup]
  Pinnacle: ...
  ...
```

### Access Gating
Whop payment gateway auto-assigns `subscriber` Discord role on successful $15/mo payment. Role removed on cancellation or payment failure. Invite link is embedded in Whop product confirmation page only.

---

## 4. Monetization — Whop

- Product: "Dinger Picks" at $15/month, cancel anytime
- Whop checkout CTA on landing page pricing section and nav button
- On successful payment: Whop auto-assigns Discord `subscriber` role
- On cancellation/failure: role auto-removed
- No free tier, no trial — pick a single clean price point to start

---

## 5. Acquisition Strategy

**Primary channel: Social media / media outreach**

- Twitter/X: daily pick posts referencing the landing page; model results as credibility content once CLV accumulates
- Reddit: r/sportsbook, r/baseballbetting — participate in existing threads with value, not spam; reference picks only in appropriate contexts
- No paid ads at launch

**What NOT to do at launch:**
- Do not ask friends to pay; organic/media acquisition only
- Do not post public Discord invite links

**Credibility content cadence (once CLV data exists):**
- Weekly "how the model did this week" posts on Twitter with actual CLV numbers
- Monthly recap of beat-close rate vs theoretical edge

---

## 6. Infrastructure — 5 Pending Manual Steps

The delivery pipeline (Discord webhooks, CLV logging, pick formatting) is code-complete. The following one-time manual steps remain before go-live:

| # | Step | Notes |
|---|---|---|
| 1 | **Supabase project setup** | Create project, run schema migrations, add `SUPABASE_URL` and `SUPABASE_KEY` to `.env` |
| 2 | **Discord server creation** | Create server, add 4 channels, generate webhooks for each, add 4 webhook URLs to `.env` |
| 3 | **GitHub Actions secrets** | Add all secrets to repo: `ODDS_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, 4 Discord webhook URLs |
| 4 | **Whop product creation** | Create "Dinger Picks" product at $15/mo, connect Discord bot, configure role assignment |
| 5 | **Smoke test** | Run pipeline end-to-end: picks post to Discord, CLV log writes, `#system-status` health message posts |

**Security constraints (non-negotiable):**
- `ODDS_API_KEY`, `SUPABASE_URL`, `SUPABASE_KEY`, all Discord webhook URLs: never committed to git. Local `.env` only, GitHub Actions secrets only.
- Discord server kept invite-only. No public discovery link.

---

## 7. CLV Accumulation

CLV tracking runs automatically each time the pipeline executes. The `data/clv_log.csv` file records pick odds and closing Pinnacle odds for every captured play.

**Gates before promoting CLV data on landing page:**
- ≥ 20 closing lines captured
- Run `validate-models` skill to assess beat-close rate and mean CLV%
- If +EV picks beat-close rate ≥ 55% and mean CLV ≥ +1.5%: update stats bar with live CLV beat rate
- If below gates: continue accumulating, do not update stats bar

**Target timeline:** 4–6 weeks of daily pipeline runs to reach the 20-capture gate (typical MLB slate has 8–15 tracked plays per day).

---

## 8. Launch Sequence

### Phase 1 — Infrastructure (days 1–2)
1. Register `dingerpicks.com` (or best available variant)
2. Complete the 5 pending Discord/Whop infra steps
3. Build landing page HTML/CSS from approved mockup, test locally
4. Configure GitHub Actions deployment to GitHub Pages with custom domain
5. Verify `dingerpicks.com` resolves to landing page with HTTPS

### Phase 2 — Soft Launch (day 3)
6. Run pipeline once manually; verify all 4 Discord channels receive correct messages
7. Verify Whop checkout flow → Discord role assignment end-to-end
8. Post landing page link on Twitter/X with a simple description

### Phase 3 — Track Record (weeks 1–6)
9. Pipeline runs daily; CLV log accumulates
10. Post daily picks on social media referencing `dingerpicks.com`
11. At ≥20 CLV captures, run `validate-models`; if gates pass, update stats bar
12. Begin weekly CLV recap posts on social media

---

## 9. Out of Scope (deferred)

- Paid advertising
- Free trial or free tier
- Email list / newsletter
- Mobile app
- DFS lineup delivery (separate product concept)
- Weather/wind features in the sim model (tracked in `project_sim_weather_todo.md`)
- May 22 ROI contamination fix (separate bug, tracked in `project_roi_contamination.md`)

---

## Appendix — Landing Page Source

Approved mockup: `.superpowers/brainstorm/1203-1780940124/content/landing-mockup.html`

This is the canonical design reference. The production HTML/CSS is built directly from this mockup.
