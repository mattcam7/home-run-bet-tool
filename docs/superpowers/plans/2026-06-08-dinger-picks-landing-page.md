# Dinger Picks Landing Page + GitHub Pages Deployment Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deploy the approved Dinger Picks landing page mockup as a production static site on GitHub Pages with custom domain support.

**Architecture:** Single-file static HTML/CSS at `docs/index.html` — no build step, no JS framework. GitHub Actions deploys on every push to `main`. `docs/CNAME` enables custom domain once registered. All Whop CTA links use a placeholder constant so one edit updates every CTA.

**Tech Stack:** Static HTML/CSS, GitHub Actions (`actions/deploy-pages@v4`), GitHub Pages

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `docs/index.html` | Create | Production landing page (full HTML/CSS from approved mockup) |
| `docs/CNAME` | Create | Custom domain — set to `dingerpicks.com` once registered |
| `.github/workflows/pages.yml` | Create | GitHub Actions: deploy `docs/` to GitHub Pages on push to `main` |

No test files — static HTML is verified visually via `file://` preview before committing.

---

## Task 1: Production landing page

**Files:**
- Create: `docs/index.html`

- [ ] **Step 1: Create `docs/` directory and `index.html`**

Full production HTML below. Key differences from the mockup:
- All CTA links point to `https://whop.com/dinger-picks/` (update when you have the real Whop URL)
- `rel="noopener noreferrer"` on all external links
- `target="_blank"` on Whop CTA only — keep nav/section CTAs same-page for cleaner mobile UX
- Mobile viewport meta and basic responsive tweaks applied

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dinger Picks — Daily MLB Home Run Picks</title>
<meta name="description" content="Daily +EV MLB home run picks anchored to Pinnacle's sharpest lines. ML model trained on 200k+ Statcast game logs. $15/month, delivered to Discord.">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #111; color: #fff; }
  a { text-decoration: none; }

  /* NAV */
  nav { background: #0d0d1a; padding: 14px 40px; display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid #1e2a4a; position: sticky; top: 0; z-index: 100; }
  .logo { font-size: 16px; font-weight: 800; letter-spacing: 1px; color: #e94560; }
  .nav-cta { background: #e94560; color: #fff; padding: 8px 18px; border-radius: 4px; font-size: 13px; font-weight: 700; }
  .nav-cta:hover { background: #c73652; }

  /* HERO */
  .hero { background: linear-gradient(135deg, #0d0d1a 0%, #16213e 60%, #0f3460 100%); padding: 80px 40px 70px; text-align: center; }
  .hero-tag { color: #e94560; font-size: 12px; letter-spacing: 3px; font-weight: 700; margin-bottom: 16px; }
  .hero h1 { font-size: 48px; font-weight: 900; line-height: 1.1; margin-bottom: 18px; }
  .hero h1 span { color: #e94560; }
  .hero p { color: #a8b2d8; font-size: 17px; max-width: 520px; margin: 0 auto 32px; line-height: 1.6; }
  .hero-cta { background: #e94560; color: #fff; padding: 16px 36px; border-radius: 6px; font-size: 16px; font-weight: 800; display: inline-block; margin-bottom: 14px; letter-spacing: 0.5px; }
  .hero-cta:hover { background: #c73652; }
  .hero-sub { color: #64748b; font-size: 12px; }

  /* STATS BAR */
  .stats { background: #0d0d1a; padding: 24px 40px; display: flex; justify-content: center; gap: 60px; border-bottom: 1px solid #1e2a4a; flex-wrap: wrap; }
  .stat { text-align: center; }
  .stat-num { font-size: 26px; font-weight: 900; color: #e94560; }
  .stat-label { font-size: 11px; color: #64748b; letter-spacing: 1px; margin-top: 3px; }

  /* SECTION */
  section { padding: 60px 40px; max-width: 860px; margin: 0 auto; }
  .section-tag { color: #e94560; font-size: 11px; letter-spacing: 3px; font-weight: 700; margin-bottom: 10px; }
  section h2 { font-size: 30px; font-weight: 800; margin-bottom: 30px; }

  /* HOW IT WORKS */
  .steps { display: flex; gap: 24px; flex-wrap: wrap; }
  .step { flex: 1; min-width: 200px; background: #16213e; border: 1px solid #1e2a4a; border-radius: 8px; padding: 24px; }
  .step-num { background: #e94560; color: #fff; width: 28px; height: 28px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 800; margin-bottom: 12px; }
  .step h3 { font-size: 15px; font-weight: 700; margin-bottom: 8px; }
  .step p { font-size: 13px; color: #a8b2d8; line-height: 1.6; }

  /* CHANNELS */
  .channels { display: flex; flex-direction: column; gap: 14px; }
  .channel { background: #16213e; border: 1px solid #1e2a4a; border-radius: 8px; padding: 18px 22px; display: flex; align-items: flex-start; gap: 16px; }
  .channel-icon { font-size: 20px; margin-top: 2px; }
  .channel h3 { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
  .channel p { font-size: 13px; color: #a8b2d8; line-height: 1.5; }
  .channel-tag { display: inline-block; background: #0f3460; color: #a8b2d8; font-size: 10px; padding: 2px 8px; border-radius: 10px; margin-top: 6px; }

  /* EXAMPLE PICK */
  .pick-card { background: #16213e; border: 1px solid #e9456033; border-radius: 8px; padding: 20px 24px; max-width: 480px; }
  .pick-header { font-size: 11px; color: #64748b; margin-bottom: 10px; letter-spacing: 1px; }
  .pick-player { font-size: 22px; font-weight: 800; margin-bottom: 4px; }
  .pick-detail { font-size: 13px; color: #a8b2d8; margin-bottom: 14px; }
  .pick-row { display: flex; gap: 10px; flex-wrap: wrap; }
  .pick-pill { background: #0f3460; color: #a8b2d8; font-size: 12px; padding: 4px 12px; border-radius: 12px; }
  .pick-pill.green { background: #0a2e1a; color: #00c853; }

  /* PRICING */
  .price-card { background: linear-gradient(135deg, #16213e, #0f3460); border: 2px solid #e94560; border-radius: 12px; padding: 40px; text-align: center; max-width: 360px; margin: 0 auto; }
  .price-badge { background: #e94560; color: #fff; font-size: 11px; padding: 4px 12px; border-radius: 10px; letter-spacing: 1px; font-weight: 700; display: inline-block; margin-bottom: 16px; }
  .price-amount { font-size: 52px; font-weight: 900; margin-bottom: 4px; }
  .price-period { color: #a8b2d8; font-size: 14px; margin-bottom: 24px; }
  .price-features { list-style: none; margin-bottom: 28px; }
  .price-features li { font-size: 14px; color: #a8b2d8; padding: 7px 0; border-bottom: 1px solid #1e2a4a; }
  .price-features li:last-child { border: none; }
  .price-features li::before { content: "✓ "; color: #e94560; font-weight: 700; }
  .price-cta { background: #e94560; color: #fff; padding: 14px 32px; border-radius: 6px; font-size: 15px; font-weight: 800; display: inline-block; width: 100%; }
  .price-cta:hover { background: #c73652; }

  /* FOOTER */
  footer { background: #0d0d1a; padding: 24px 40px; text-align: center; color: #374151; font-size: 12px; border-top: 1px solid #1e2a4a; }

  .note { background: #1a1a2e; border-left: 3px solid #e94560; padding: 10px 16px; border-radius: 0 6px 6px 0; font-size: 12px; color: #64748b; margin-top: 20px; font-style: italic; }

  /* RESPONSIVE */
  @media (max-width: 600px) {
    nav { padding: 12px 20px; }
    .hero { padding: 60px 20px 50px; }
    .hero h1 { font-size: 32px; }
    .stats { gap: 30px; padding: 20px; }
    section { padding: 40px 20px; }
  }
</style>
</head>
<body>

<nav>
  <div class="logo">⚾ DINGER PICKS</div>
  <a href="https://whop.com/dinger-picks/" class="nav-cta" target="_blank" rel="noopener noreferrer">Join $15/mo</a>
</nav>

<div class="hero">
  <div class="hero-tag">⚾ DAILY MLB HOME RUN PICKS</div>
  <h1>Daily Home Run<br>Picks <span>That Win.</span></h1>
  <p>A machine learning model trained on 200,000+ Statcast game logs, anchored to Pinnacle's sharpest lines. Every pick is +EV or it doesn't get posted.</p>
  <a href="https://whop.com/dinger-picks/" class="hero-cta" target="_blank" rel="noopener noreferrer">GET PICKS — $15/MO →</a>
  <div class="hero-sub">Cancel anytime · Delivered to Discord daily</div>
</div>

<div class="stats">
  <div class="stat">
    <div class="stat-num">200k+</div>
    <div class="stat-label">STATCAST GAME LOGS</div>
  </div>
  <div class="stat">
    <div class="stat-num">0.68</div>
    <div class="stat-label">MODEL AUC (2022 HOLDOUT)</div>
  </div>
  <div class="stat">
    <div class="stat-num">Daily</div>
    <div class="stat-label">PICKS POSTED</div>
  </div>
  <div class="stat">
    <div class="stat-num">$15</div>
    <div class="stat-label">PER MONTH</div>
  </div>
</div>

<section>
  <div class="section-tag">THE MODEL</div>
  <h2>How It Works</h2>
  <div class="steps">
    <div class="step">
      <div class="step-num">1</div>
      <h3>Sharp Lines</h3>
      <p>We pull Pinnacle's closing lines — the sharpest market in the world — and use them as our probability anchor.</p>
    </div>
    <div class="step">
      <div class="step-num">2</div>
      <h3>Statcast Sim</h3>
      <p>Our model runs barrel rate, bat speed, fly ball %, park factor, and pitcher HR/9 through a logistic regression trained on 4 years of real game data.</p>
    </div>
    <div class="step">
      <div class="step-num">3</div>
      <h3>+EV Picks Only</h3>
      <p>We compare retail odds at DraftKings, FanDuel, and BetMGM to our model. Only plays with positive expected value get posted.</p>
    </div>
  </div>
</section>

<section style="padding-top:0;">
  <div class="section-tag">DISCORD CHANNELS</div>
  <h2>What You Get</h2>
  <div class="channels">
    <div class="channel">
      <div class="channel-icon">🎯</div>
      <div>
        <h3>#picks</h3>
        <p>Daily HR picks with player, book, odds, and unit size. Posted each afternoon before first pitch.</p>
        <span class="channel-tag">MAIN CHANNEL</span>
      </div>
    </div>
    <div class="channel">
      <div class="channel-icon">📊</div>
      <div>
        <h3>#data</h3>
        <p>Full model output — EV%, simulation probability, Kelly sizing, Pinnacle line. For subscribers who want to dig into the numbers.</p>
        <span class="channel-tag">ANALYTICS</span>
      </div>
    </div>
    <div class="channel">
      <div class="channel-icon">☀️</div>
      <div>
        <h3>#results</h3>
        <p>Morning recap of yesterday's picks. Outcome tracking and running ROI updated daily.</p>
        <span class="channel-tag">RESULTS</span>
      </div>
    </div>
  </div>
</section>

<section style="padding-top:0;">
  <div class="section-tag">EXAMPLE</div>
  <h2>What a Pick Looks Like</h2>
  <div class="pick-card">
    <div class="pick-header">TODAY'S PICK · DraftKings · 1.5u</div>
    <div class="pick-player">Aaron Judge</div>
    <div class="pick-detail">NYY vs BAL · Over 0.5 HR</div>
    <div class="pick-row">
      <span class="pick-pill green">+450 DraftKings</span>
      <span class="pick-pill green">+11.2% EV</span>
      <span class="pick-pill">16% sim prob</span>
      <span class="pick-pill">1.5 units</span>
    </div>
  </div>
  <div class="note">Picks are for informational purposes. Always bet responsibly.</div>
</section>

<section style="padding-top:0;">
  <div class="section-tag">PRICING</div>
  <h2>Simple, No B.S. Pricing</h2>
  <div class="price-card">
    <div class="price-badge">BETA PRICE</div>
    <div class="price-amount">$15</div>
    <div class="price-period">per month · cancel anytime</div>
    <ul class="price-features">
      <li>Daily +EV HR picks</li>
      <li>Full model data &amp; analytics</li>
      <li>Morning results recap</li>
      <li>Private Discord access</li>
    </ul>
    <a href="https://whop.com/dinger-picks/" class="price-cta" target="_blank" rel="noopener noreferrer">JOIN NOW →</a>
  </div>
</section>

<footer>
  &copy; 2026 Dinger Picks &nbsp;·&nbsp; Picks are for informational purposes only &nbsp;·&nbsp; 21+ &nbsp;·&nbsp; Gamble responsibly
</footer>

</body>
</html>
```

- [ ] **Step 2: Preview locally**

Open the file directly in a browser:
```
start docs/index.html    # Windows
open docs/index.html     # Mac
```

Verify: nav sticky, hero gradient, stats bar wraps on mobile (resize browser to 375px), pricing card centered, all 3 CTA links go to correct Whop URL.

- [ ] **Step 3: Commit**

```bash
git add docs/index.html
git commit -m "feat: add Dinger Picks production landing page"
```

---

## Task 2: GitHub Pages deployment workflow

**Files:**
- Create: `.github/workflows/pages.yml`

- [ ] **Step 1: Create the workflow file**

```yaml
name: Deploy to GitHub Pages

on:
  push:
    branches: [main]
    paths: ["docs/**"]
  workflow_dispatch:

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: pages
  cancel-in-progress: true

jobs:
  deploy:
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/configure-pages@v4
      - uses: actions/upload-pages-artifact@v3
        with:
          path: docs/
      - id: deployment
        uses: actions/deploy-pages@v4
```

- [ ] **Step 2: Verify workflow syntax**

```bash
# No local validation needed for simple YAML — GitHub Actions validates on push.
# Confirm file exists:
ls .github/workflows/pages.yml
```

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/pages.yml
git commit -m "ci: add GitHub Pages deployment workflow"
```

---

## Task 3: CNAME for custom domain

**Files:**
- Create: `docs/CNAME`

- [ ] **Step 1: Create CNAME file**

```
dingerpicks.com
```

(One line, no trailing newline. Update to exact domain once registered.)

- [ ] **Step 2: Commit**

```bash
git add docs/CNAME
git commit -m "chore: add CNAME for custom domain"
```

---

## Post-Deploy Manual Steps

After these 3 tasks are committed and pushed to `main`:

1. **Enable GitHub Pages** in repo Settings → Pages → Source → "GitHub Actions"
2. **Register domain** (Namecheap/Cloudflare) → add CNAME DNS record pointing to `<username>.github.io`
3. **Verify HTTPS** at `https://dingerpicks.com` (GitHub Pages auto-provisions TLS within ~15 min)
4. **Update Whop CTA URL** once Whop product is created: find all `https://whop.com/dinger-picks/` in `docs/index.html` and replace with the real checkout URL
