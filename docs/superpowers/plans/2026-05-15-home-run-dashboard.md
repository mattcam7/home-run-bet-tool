# Home Run Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a locally-run Python pipeline that fetches today's MLB HR prop odds, computes +EV plays vs Pinnacle's sharp lines, and renders an interactive HTML dashboard with a manual parlay builder.

**Architecture:** Three agent modules (retail scraper, Pinnacle scraper, EV calculator) feed into an HTML dashboard generator, all chained by a single `run.py` orchestrator. One OddsAPI call fetches all bookmakers; Pinnacle is split client-side. Dashboard is a static HTML file with vanilla JS for sorting, filtering, and live parlay calculations.

**Tech Stack:** Python 3.9+, pandas, requests, python-dotenv, tzdata, pytest

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `requirements.txt` | Create | Python dependencies |
| `.env.example` | Create | API key template |
| `.gitignore` | Create | Exclude .env and generated HTML |
| `agents/__init__.py` | Create | Package marker |
| `agents/utils.py` | Create | Shared `american_to_decimal` helper |
| `agents/odds_scraper.py` | Create | Agent 1: retail odds extraction |
| `agents/pinnacle_scraper.py` | Create | Agent 2: Pinnacle odds extraction |
| `agents/ev_calculator.py` | Create | Agent 3: EV%, composite z-score |
| `dashboard/__init__.py` | Create | Package marker |
| `dashboard/generator.py` | Create | HTML generation + browser open |
| `run.py` | Create | Orchestrator / pipeline entry point |
| `tests/__init__.py` | Create | Package marker |
| `tests/conftest.py` | Create | Shared fixture OddsAPI payload |
| `tests/test_utils.py` | Create | Tests for american_to_decimal |
| `tests/test_odds_scraper.py` | Create | Tests for retail odds extraction |
| `tests/test_pinnacle_scraper.py` | Create | Tests for Pinnacle odds extraction |
| `tests/test_ev_calculator.py` | Create | Tests for EV + z-score logic |
| `tests/test_generator.py` | Create | Tests for HTML output |
| `tests/test_run.py` | Create | Integration test for orchestrator |
| `CLAUDE.md` | Modify | Add keyphrase trigger section |
| `AGENTS.md` | Modify | Replace plaintext API key with `$ODDS_API_KEY` |

---

## Task 1: Scaffold

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `agents/__init__.py`
- Create: `dashboard/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Create requirements.txt**

```
requests
pandas
python-dotenv
tzdata
pytest
```

- [ ] **Step 2: Create .env.example**

```
ODDS_API_KEY=your_key_here
```

- [ ] **Step 3: Create .gitignore**

```
.env
hr_dashboard.html
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 4: Create package markers**

Create three empty files: `agents/__init__.py`, `dashboard/__init__.py`, `tests/__init__.py`

- [ ] **Step 5: Create tests/conftest.py**

```python
from datetime import datetime, timezone

FIXTURE_NOW = datetime(2026, 5, 15, 20, 0, 0, tzinfo=timezone.utc)

FIXTURE_PAYLOAD = [
    {
        "id": "game1",
        "sport_key": "baseball_mlb",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": "2026-05-15T23:05:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Aaron Judge", "price": 450},
                    {"name": "Rafael Devers", "price": 600},
                ]}],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Aaron Judge", "price": 420},
                    {"name": "Rafael Devers", "price": 580},
                ]}],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Aaron Judge", "price": 380},
                    {"name": "Rafael Devers", "price": 520},
                ]}],
            },
        ],
    },
    {
        "id": "game2",
        "sport_key": "baseball_mlb",
        "home_team": "Chicago Cubs",
        "away_team": "Los Angeles Dodgers",
        "commence_time": "2026-05-15T17:00:00Z",  # already started
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Shohei Ohtani", "price": 350},
                ]}],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Shohei Ohtani", "price": 320},
                ]}],
            },
        ],
    },
]
```

- [ ] **Step 6: Install dependencies**

```
pip install -r requirements.txt
```

- [ ] **Step 7: Commit**

```bash
git add requirements.txt .env.example .gitignore agents/__init__.py dashboard/__init__.py tests/__init__.py tests/conftest.py
git commit -m "chore: scaffold project structure and shared test fixture"
```

---

## Task 2: Shared Utility

**Files:**
- Create: `agents/utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_utils.py
import pytest
from agents.utils import american_to_decimal

def test_positive_odds():
    assert american_to_decimal(450) == 5.5

def test_negative_odds():
    assert american_to_decimal(-110) == pytest.approx(1 + 100/110, abs=1e-6)

def test_plus_100():
    assert american_to_decimal(100) == 2.0

def test_minus_100():
    assert american_to_decimal(-100) == 2.0
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_utils.py -v
```
Expected: `ModuleNotFoundError` or `ImportError`

- [ ] **Step 3: Implement**

```python
# agents/utils.py
def american_to_decimal(odds: int) -> float:
    if odds > 0:
        return (odds / 100) + 1
    return (100 / abs(odds)) + 1
```

- [ ] **Step 4: Run to confirm pass**

```
pytest tests/test_utils.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add agents/utils.py tests/test_utils.py
git commit -m "feat: add american_to_decimal utility"
```

---

## Task 3: Agent 1 — Retail Odds Scraper

**Files:**
- Create: `agents/odds_scraper.py`
- Create: `tests/test_odds_scraper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_odds_scraper.py
import copy
from tests.conftest import FIXTURE_PAYLOAD, FIXTURE_NOW
from agents.odds_scraper import extract_retail_odds

def test_excludes_pinnacle():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "Pinnacle" not in df["bookmaker"].values

def test_excludes_started_games():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "Shohei Ohtani" not in df["player_name"].values

def test_returns_expected_columns():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    for col in ["player_name", "game", "commence_time", "bookmaker", "american_odds", "implied_prob"]:
        assert col in df.columns

def test_normalizes_player_names():
    modified = copy.deepcopy(FIXTURE_PAYLOAD)
    modified[0]["bookmakers"][0]["markets"][0]["outcomes"][0]["name"] = "aaron judge"
    df = extract_retail_odds(modified, FIXTURE_NOW)
    assert "Aaron Judge" in df["player_name"].values

def test_implied_prob_positive_odds():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    row = df[(df["player_name"] == "Aaron Judge") & (df["bookmaker"] == "DraftKings")].iloc[0]
    assert abs(row["implied_prob"] - (1 / 5.5)) < 0.001

def test_game_format():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "New York Yankees @ Boston Red Sox" in df["game"].values
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_odds_scraper.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement**

```python
# agents/odds_scraper.py
import pandas as pd
from datetime import datetime
from agents.utils import american_to_decimal

def extract_retail_odds(raw_payload: list, now: datetime) -> pd.DataFrame:
    rows = []
    for event in raw_payload:
        commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if commence_time <= now:
            continue
        game = f"{event['away_team']} @ {event['home_team']}"
        for bookmaker in event["bookmakers"]:
            if bookmaker["key"] == "pinnacle":
                continue
            for market in bookmaker["markets"]:
                if market["key"] != "batter_home_runs":
                    continue
                for outcome in market["outcomes"]:
                    american_odds = outcome["price"]
                    rows.append({
                        "player_name": outcome["name"].strip().title(),
                        "game": game,
                        "commence_time": commence_time,
                        "bookmaker": bookmaker["title"],
                        "american_odds": american_odds,
                        "implied_prob": 1 / american_to_decimal(american_odds),
                    })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to confirm pass**

```
pytest tests/test_odds_scraper.py -v
```
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add agents/odds_scraper.py tests/test_odds_scraper.py
git commit -m "feat: add Agent 1 retail odds scraper"
```

---

## Task 4: Agent 2 — Pinnacle Scraper

**Files:**
- Create: `agents/pinnacle_scraper.py`
- Create: `tests/test_pinnacle_scraper.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pinnacle_scraper.py
from tests.conftest import FIXTURE_PAYLOAD, FIXTURE_NOW
from agents.pinnacle_scraper import extract_pinnacle_odds

def test_returns_only_unplayed_pinnacle_players():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert len(df) == 2  # Aaron Judge + Rafael Devers from game1 only

def test_excludes_started_games():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "Shohei Ohtani" not in df["player_name"].values

def test_returns_expected_columns():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    for col in ["player_name", "game", "commence_time", "pinnacle_odds", "pinnacle_prob"]:
        assert col in df.columns

def test_implied_prob_calculation():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    # +380 -> decimal 4.8 -> prob = 1/4.8
    assert abs(judge["pinnacle_prob"] - (1 / 4.8)) < 0.001
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_pinnacle_scraper.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement**

```python
# agents/pinnacle_scraper.py
import pandas as pd
from datetime import datetime
from agents.utils import american_to_decimal

def extract_pinnacle_odds(raw_payload: list, now: datetime) -> pd.DataFrame:
    rows = []
    for event in raw_payload:
        commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if commence_time <= now:
            continue
        game = f"{event['away_team']} @ {event['home_team']}"
        for bookmaker in event["bookmakers"]:
            if bookmaker["key"] != "pinnacle":
                continue
            for market in bookmaker["markets"]:
                if market["key"] != "batter_home_runs":
                    continue
                for outcome in market["outcomes"]:
                    american_odds = outcome["price"]
                    rows.append({
                        "player_name": outcome["name"].strip().title(),
                        "game": game,
                        "commence_time": commence_time,
                        "pinnacle_odds": american_odds,
                        "pinnacle_prob": 1 / american_to_decimal(american_odds),
                    })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run to confirm pass**

```
pytest tests/test_pinnacle_scraper.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add agents/pinnacle_scraper.py tests/test_pinnacle_scraper.py
git commit -m "feat: add Agent 2 Pinnacle scraper"
```

---

## Task 5: Agent 3 — EV Calculator

**Files:**
- Create: `agents/ev_calculator.py`
- Create: `tests/test_ev_calculator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ev_calculator.py
import pytest
import pandas as pd
from datetime import datetime, timezone
from agents.ev_calculator import calculate_ev

COMMENCE = datetime(2026, 5, 15, 23, 5, tzinfo=timezone.utc)
GAME = "New York Yankees @ Boston Red Sox"

RETAIL_DF = pd.DataFrame([
    {"player_name": "Aaron Judge",   "game": GAME, "commence_time": COMMENCE, "bookmaker": "DraftKings", "american_odds": 450, "implied_prob": 1/5.5},
    {"player_name": "Aaron Judge",   "game": GAME, "commence_time": COMMENCE, "bookmaker": "FanDuel",    "american_odds": 420, "implied_prob": 1/5.2},
    {"player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE, "bookmaker": "DraftKings", "american_odds": 600, "implied_prob": 1/7.0},
    {"player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE, "bookmaker": "FanDuel",    "american_odds": 580, "implied_prob": 1/6.8},
])

PINNACLE_DF = pd.DataFrame([
    {"player_name": "Aaron Judge",   "game": GAME, "commence_time": COMMENCE, "pinnacle_odds": 380, "pinnacle_prob": 1/4.8},
    {"player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE, "pinnacle_odds": 520, "pinnacle_prob": 1/6.2},
])

def test_one_row_per_player():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    assert len(df) == 2

def test_excludes_players_not_at_pinnacle():
    extra = pd.concat([RETAIL_DF, pd.DataFrame([{
        "player_name": "Ghost Player", "game": GAME, "commence_time": COMMENCE,
        "bookmaker": "DraftKings", "american_odds": 800, "implied_prob": 0.11,
    }])])
    df = calculate_ev(extra, PINNACLE_DF)
    assert "Ghost Player" not in df["player_name"].values

def test_ev_formula():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    expected = (1/4.8 * 5.5) - 1
    assert abs(judge["ev_pct"] - expected) < 0.001

def test_best_retail_selects_highest_decimal():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    assert judge["best_retail_odds"] == 450  # DK +450 beats FD +420

def test_composite_score():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    assert abs(judge["composite_score"] - (judge["ev_pct"] * judge["pinnacle_prob"])) < 0.0001

def test_composite_z_mean_is_zero():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    assert abs(df["composite_z"].mean()) < 0.0001

def test_sorted_by_composite_z_descending():
    df = calculate_ev(RETAIL_DF, PINNACLE_DF)
    assert df["composite_z"].iloc[0] >= df["composite_z"].iloc[-1]
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_ev_calculator.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement**

```python
# agents/ev_calculator.py
import pandas as pd
from agents.utils import american_to_decimal

def calculate_ev(retail_df: pd.DataFrame, pinnacle_df: pd.DataFrame) -> pd.DataFrame:
    pivot = retail_df.pivot_table(
        index=["player_name", "game", "commence_time"],
        columns="bookmaker",
        values="american_odds",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    merged = pivot.merge(
        pinnacle_df[["player_name", "game", "pinnacle_odds", "pinnacle_prob"]],
        on=["player_name", "game"],
        how="inner",
    )

    meta_cols = {"player_name", "game", "commence_time", "pinnacle_odds", "pinnacle_prob"}
    book_cols = [c for c in merged.columns if c not in meta_cols]

    def _best_retail(row):
        best_dec, best_odds = -float("inf"), None
        for col in book_cols:
            val = row[col]
            if pd.isna(val):
                continue
            dec = american_to_decimal(int(val))
            if dec > best_dec:
                best_dec, best_odds = dec, int(val)
        return pd.Series({"best_retail_odds": best_odds, "best_retail_decimal": best_dec})

    merged[["best_retail_odds", "best_retail_decimal"]] = merged.apply(_best_retail, axis=1)
    merged["ev_pct"] = (merged["pinnacle_prob"] * merged["best_retail_decimal"]) - 1
    merged["composite_score"] = merged["ev_pct"] * merged["pinnacle_prob"]

    mean_c = merged["composite_score"].mean()
    std_c = merged["composite_score"].std(ddof=0)
    merged["composite_z"] = (merged["composite_score"] - mean_c) / std_c

    return merged.sort_values("composite_z", ascending=False).reset_index(drop=True)
```

- [ ] **Step 4: Run to confirm pass**

```
pytest tests/test_ev_calculator.py -v
```
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add agents/ev_calculator.py tests/test_ev_calculator.py
git commit -m "feat: add Agent 3 EV calculator with composite z-score"
```

---

## Task 6: Dashboard Generator

**Files:**
- Create: `dashboard/generator.py`
- Create: `tests/test_generator.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_generator.py
import os
import pandas as pd
import pytest
from datetime import datetime, timezone
from dashboard.generator import generate_dashboard

COMMENCE = datetime(2026, 5, 15, 23, 5, tzinfo=timezone.utc)
GAME = "New York Yankees @ Boston Red Sox"

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
            "pinnacle_odds": 380, "pinnacle_prob": 1/4.8,
            "DraftKings": 450, "FanDuel": 420,
            "best_retail_odds": 450, "best_retail_decimal": 5.5,
            "ev_pct": (1/4.8 * 5.5) - 1,
            "composite_score": ((1/4.8 * 5.5) - 1) * (1/4.8),
            "composite_z": 1.0,
        },
        {
            "player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE,
            "pinnacle_odds": 520, "pinnacle_prob": 1/6.2,
            "DraftKings": 600, "FanDuel": 580,
            "best_retail_odds": 600, "best_retail_decimal": 7.0,
            "ev_pct": (1/6.2 * 7.0) - 1,
            "composite_score": ((1/6.2 * 7.0) - 1) * (1/6.2),
            "composite_z": -1.0,
        },
    ])

def test_creates_html_file(tmp_path, sample_df):
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    assert os.path.exists(output)

def test_html_contains_player_names(tmp_path, sample_df):
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Aaron Judge" in content
    assert "Rafael Devers" in content

def test_html_contains_parlay_section(tmp_path, sample_df):
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "parlay" in content.lower()

def test_opens_browser(tmp_path, sample_df, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=True)
    assert len(opened) == 1
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_generator.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement dashboard/generator.py**

```python
# dashboard/generator.py
import json
import os
import webbrowser
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")

META_COLS = {
    "player_name", "game", "commence_time",
    "pinnacle_odds", "pinnacle_prob",
    "best_retail_odds", "best_retail_decimal",
    "ev_pct", "composite_score", "composite_z",
}

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HR Dashboard</title>
  <style>
    body{font-family:system-ui,sans-serif;padding:20px;background:#f8f9fa;margin:0}
    h1{color:#222;margin-bottom:4px}
    .meta{color:#666;font-size:.9em;margin-bottom:16px}
    .controls{margin-bottom:12px;display:flex;align-items:center;gap:12px}
    table{border-collapse:collapse;width:100%;background:#fff;box-shadow:0 1px 4px rgba(0,0,0,.1);font-size:.9em}
    th{background:#343a40;color:#fff;padding:10px 8px;cursor:pointer;text-align:left;white-space:nowrap;user-select:none}
    th:hover{background:#495057}
    td{padding:8px;border-bottom:1px solid #dee2e6;white-space:nowrap}
    tr.positive-ev{background:#d4edda}
    tr.strong-play{background:#28a745!important;color:#fff;font-weight:700}
    tr.negative-ev td{color:#aaa}
    tr.hidden{display:none}
    #parlay-builder{margin-top:32px;background:#fff;padding:20px 24px;box-shadow:0 1px 4px rgba(0,0,0,.1);border-radius:4px}
    #parlay-builder h2{margin-top:0}
    .parlay-leg{font-size:.95em;margin:2px 0}
    #parlay-stats{margin-top:12px;line-height:1.8}
    .stat-label{font-weight:600}
  </style>
</head>
<body>
  <h1>Home Run Dashboard</h1>
  <p class="meta">__N_PLAYERS__ players &nbsp;|&nbsp; __N_POSITIVE__ +EV plays &nbsp;|&nbsp; __TIMESTAMP__</p>
  <div class="controls">
    <label>Min EV%: <input type="range" id="ev-filter" min="-100" max="100" value="-100" step="1" oninput="applyFilter(+this.value)"></label>
    <span id="ev-label">-100%</span>
  </div>
  <table id="player-table">
    <thead><tr>
      <th></th>
      <th onclick="sortBy('player')">Player</th>
      <th onclick="sortBy('game')">Game</th>
      <th onclick="sortBy('time_sort')">Time (ET)</th>
      <th onclick="sortBy('pinnacle_pct')">Pin %</th>
      __BOOK_HEADERS__
      <th onclick="sortBy('best_retail_odds')">Best Retail</th>
      <th onclick="sortBy('ev_pct')">EV%</th>
      <th onclick="sortBy('composite_z')">Composite Z</th>
    </tr></thead>
    <tbody id="table-body"></tbody>
  </table>
  <div id="parlay-builder">
    <h2>Parlay Builder</h2>
    <div id="parlay-legs"><em style="color:#999">Select players above to build a parlay</em></div>
    <div id="parlay-stats"></div>
  </div>
  <script>
    const DATA=__DATA__;
    const BOOKS=__BOOK_NAMES__;
    let sortKey='composite_z',sortDir=-1,minEv=-100;
    const legs={};
    function fmtOdds(v){return v==null?'--':v>0?'+'+v:''+v}
    function fmtPct(v){return(v>=0?'+':'')+v.toFixed(2)+'%'}
    function fmtZ(v){return(v>=0?'+':'')+v.toFixed(2)}
    function rowCls(r){
      if(r.composite_z>=1.5)return 'strong-play';
      if(r.ev_pct>0)return 'positive-ev';
      return 'negative-ev';
    }
    function renderTable(){
      const sorted=[...DATA].sort((a,b)=>{
        const av=a[sortKey],bv=b[sortKey];
        if(typeof av==='string')return sortDir*av.localeCompare(bv);
        return sortDir*((av??-Infinity)-(bv??-Infinity));
      });
      document.getElementById('table-body').innerHTML=sorted.map(r=>`
        <tr class="${rowCls(r)}${r.ev_pct<minEv?' hidden':''}">
          <td><input type="checkbox" ${legs[r.player+'|'+r.game]?'checked':''} onchange="toggleLeg('${r.player}','${r.game}',this)"></td>
          <td>${r.player}</td><td>${r.game}</td><td>${r.time}</td>
          <td>${r.pinnacle_pct.toFixed(1)}%</td>
          ${BOOKS.map(b=>`<td>${fmtOdds(r[b])}</td>`).join('')}
          <td>${fmtOdds(r.best_retail_odds)}</td>
          <td>${fmtPct(r.ev_pct)}</td>
          <td>${fmtZ(r.composite_z)}</td>
        </tr>`).join('');
    }
    function sortBy(k){if(sortKey===k)sortDir*=-1;else{sortKey=k;sortDir=-1;}renderTable();}
    function applyFilter(v){minEv=v;document.getElementById('ev-label').textContent=v+'%';renderTable();}
    function americanToDecimal(o){return o>0?(o/100)+1:(100/Math.abs(o))+1}
    function decimalToAmerican(d){return d>=2?'+'+(Math.round((d-1)*100)):''+Math.round(-100/(d-1))}
    function toggleLeg(player,game,cb){
      const key=player+'|'+game;
      if(cb.checked)legs[key]=DATA.find(d=>d.player===player&&d.game===game);
      else delete legs[key];
      updateParlay();
    }
    function updateParlay(){
      const sel=Object.values(legs);
      const legsDiv=document.getElementById('parlay-legs');
      const statsDiv=document.getElementById('parlay-stats');
      if(!sel.length){
        legsDiv.innerHTML='<em style="color:#999">Select players above to build a parlay</em>';
        statsDiv.innerHTML='';return;
      }
      legsDiv.innerHTML=sel.map(l=>`<div class="parlay-leg">${l.player} &mdash; ${fmtOdds(l.best_retail_odds)}</div>`).join('');
      const pDec=sel.reduce((a,l)=>a*americanToDecimal(l.best_retail_odds),1);
      const pProb=sel.reduce((a,l)=>a*(l.pinnacle_pct/100),1);
      const pEv=(pProb*pDec)-1;
      const pComp=pEv*pProb;
      const comps=DATA.map(d=>d.ev_pct/100*(d.pinnacle_pct/100));
      const mean=comps.reduce((a,b)=>a+b,0)/comps.length;
      const std=Math.sqrt(comps.map(x=>(x-mean)**2).reduce((a,b)=>a+b,0)/comps.length);
      const pZ=std>0?fmtZ((pComp-mean)/std):'--';
      statsDiv.innerHTML=`
        <div><span class="stat-label">Legs:</span> ${sel.length}</div>
        <div><span class="stat-label">Combined Probability:</span> ${(pProb*100).toFixed(2)}%</div>
        <div><span class="stat-label">Combined Odds:</span> ${decimalToAmerican(pDec)}</div>
        <div><span class="stat-label">Combined EV%:</span> ${fmtPct(pEv*100)}</div>
        <div><span class="stat-label">Composite Z:</span> ${pZ}</div>`;
    }
    renderTable();
  </script>
</body>
</html>"""


def generate_dashboard(
    final_df: pd.DataFrame,
    output_path: str = "hr_dashboard.html",
    open_browser: bool = True,
) -> None:
    book_cols = [c for c in final_df.columns if c not in META_COLS]

    records = []
    for _, row in final_df.iterrows():
        record = {
            "player": row["player_name"],
            "game": row["game"],
            "time": row["commence_time"].astimezone(ET).strftime("%I:%M %p ET"),
            "time_sort": row["commence_time"].timestamp(),
            "pinnacle_pct": round(row["pinnacle_prob"] * 100, 2),
            "best_retail_odds": int(row["best_retail_odds"]),
            "ev_pct": round(row["ev_pct"] * 100, 2),
            "composite_z": round(row["composite_z"], 2),
        }
        for col in book_cols:
            val = row.get(col)
            record[col] = int(val) if pd.notna(val) else None
        records.append(record)

    timestamp = datetime.now(ET).strftime("%Y-%m-%d %I:%M %p ET")
    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
    book_headers = "".join(f'<th onclick="sortBy(\'{b}\')">{b}</th>' for b in book_cols)

    html = (
        HTML_TEMPLATE
        .replace("__DATA__", json.dumps(records))
        .replace("__BOOK_NAMES__", json.dumps(book_cols))
        .replace("__BOOK_HEADERS__", book_headers)
        .replace("__TIMESTAMP__", timestamp)
        .replace("__N_PLAYERS__", str(n_players))
        .replace("__N_POSITIVE__", str(n_positive))
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    if open_browser:
        abs_path = os.path.abspath(output_path).replace("\\", "/")
        webbrowser.open(f"file:///{abs_path}")
```

- [ ] **Step 4: Run to confirm pass**

```
pytest tests/test_generator.py -v
```
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add dashboard/generator.py tests/test_generator.py
git commit -m "feat: add HTML dashboard generator with parlay builder"
```

---

## Task 7: Orchestrator

**Files:**
- Create: `run.py`
- Create: `tests/test_run.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_run.py
from unittest.mock import MagicMock
from tests.conftest import FIXTURE_PAYLOAD
from run import fetch_odds, main

def test_fetch_odds_calls_correct_endpoint(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.json.return_value = []
    mock_resp.raise_for_status = lambda: None
    calls = []
    def fake_get(url, params, timeout):
        calls.append((url, params))
        return mock_resp
    monkeypatch.setattr("requests.get", fake_get)
    result = fetch_odds("test_key")
    assert calls[0][0] == "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds"
    assert calls[0][1]["markets"] == "batter_home_runs"
    assert calls[0][1]["apiKey"] == "test_key"
    assert result == []

def test_main_runs_full_pipeline(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test_key")
    monkeypatch.setattr("run.fetch_odds", lambda key: FIXTURE_PAYLOAD)
    monkeypatch.setattr("run.generate_dashboard", lambda df, **kwargs: None)
    main()  # should not raise
```

- [ ] **Step 2: Run to confirm failure**

```
pytest tests/test_run.py -v
```
Expected: `ImportError`

- [ ] **Step 3: Implement run.py**

```python
# run.py
import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from agents.ev_calculator import calculate_ev
from agents.odds_scraper import extract_retail_odds
from agents.pinnacle_scraper import extract_pinnacle_odds
from dashboard.generator import generate_dashboard


def fetch_odds(api_key: str) -> list:
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        params={
            "apiKey": api_key,
            "regions": "us",
            "markets": "batter_home_runs",
            "oddsFormat": "american",
        },
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    load_dotenv()
    api_key = os.environ["ODDS_API_KEY"]

    print("Fetching odds from OddsAPI...")
    raw = fetch_odds(api_key)
    now = datetime.now(timezone.utc)

    print("Extracting retail odds...")
    retail_df = extract_retail_odds(raw, now)

    print("Extracting Pinnacle odds...")
    pinnacle_df = extract_pinnacle_odds(raw, now)

    print("Calculating EV...")
    final_df = calculate_ev(retail_df, pinnacle_df)

    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
    print(f"Found {n_players} players | {n_positive} +EV plays")

    generate_dashboard(final_df)
    print("Dashboard opened in browser.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to confirm pass**

```
pytest tests/test_run.py -v
```
Expected: 2 passed

- [ ] **Step 5: Run full test suite**

```
pytest tests/ -v
```
Expected: all tests pass

- [ ] **Step 6: Commit**

```bash
git add run.py tests/test_run.py
git commit -m "feat: add run.py orchestrator"
```

---

## Task 8: Housekeeping

**Files:**
- Modify: `CLAUDE.md`
- Modify: `AGENTS.md`

- [ ] **Step 1: Add keyphrase trigger to CLAUDE.md**

Append this section to `CLAUDE.md`:

```markdown
## Trigger Keyphrase

Typing `run the hr dashboard` in this Claude Code chat session triggers execution of `python run.py` from the repo root. This fetches today's MLB HR prop odds, computes EV vs Pinnacle's lines, and opens the interactive HTML dashboard in the default browser.
```

- [ ] **Step 2: Remove plaintext API key from AGENTS.md**

In `AGENTS.md`, replace the line:
```
**Inputs:** Read through the available odds at the oddsapi using my API Key (d581db06a0c0f59513e1a6dc018eeab7)
```
With:
```
**Inputs:** OddsAPI — key loaded from `$ODDS_API_KEY` in `.env`
```

- [ ] **Step 3: Create .env with real key**

Create `.env` (not committed):
```
ODDS_API_KEY=d581db06a0c0f59513e1a6dc018eeab7
```

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md AGENTS.md
git commit -m "chore: add keyphrase docs and remove plaintext API key"
```
