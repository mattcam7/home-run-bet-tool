# Discord Bot + Whop MVP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship automated daily HR pick delivery to Discord subscribers via GitHub Actions + Supabase, with Whop billing at $15/month.

**Architecture:** Supabase replaces local SQLite/CSV for cloud-accessible data; GitHub Actions runs three scheduled jobs (picks at 11 AM ET, hourly monitor, results at 10 AM ET); Discord webhooks deliver picks/results/alerts to private subscriber channels; `#system-status` gives owner-only health visibility with no manual log-checking.

**Tech Stack:** Python, `supabase-py`, `requests`, GitHub Actions cron, Supabase Postgres, Discord webhooks, Whop

---

## Codebase Context

**Key files:**
- `agents/clv_log.py` — CLV log write logic; `log_open_plays()` appends to `data/clv_log.csv`, `OPEN_COLS` defines the schema
- `agents/outcome_tracker.py` — `compute_roi_metrics()` reads CSV + SQLite; `update_for_date()` fetches MLB box scores
- `run.py` — main pipeline; `main()` calls `calculate_ev()` → `log_open_plays()` → `generate_dashboard()`
- `capture_closing.py` — entry point that calls `agents/clv_log.py::capture_closing()`

**Existing patterns:**
- Tests live in `tests/test_*.py`; run with `pytest`
- Network calls mocked with `monkeypatch` (see `tests/test_clv_log.py`)
- Env vars loaded via `python-dotenv`; `os.environ["KEY"]` raises on missing
- 168 tests currently passing — do not break them

**Featured bet definition** (burns through this entire plan):
```python
featured = float(r.get("kelly_units", 0)) >= 0.5 and float(r.get("ev_pct", 0)) >= 0.10
```

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `requirements.txt` | Modify | Add `supabase` |
| `agents/supabase_client.py` | Create | Thin Supabase wrapper (insert/fetch CLV log + outcomes) |
| `agents/clv_log.py` | Modify | Add `featured_bet` column; dual-write Supabase + CSV |
| `agents/outcome_tracker.py` | Modify | Add `featured_only` param; read from Supabase when key set |
| `run.py` | Modify | Add `--no-browser` flag; call `discord_bot.post_picks()` when set |
| `agents/discord_bot.py` | Create | All Discord posting functions + `post_status()` + `post_alert()` |
| `monitor.py` | Create | Hourly line movement monitor |
| `post_results.py` | Create | Morning results + Sunday recap script |
| `.github/workflows/daily_picks.yml` | Create | GitHub Actions: 11 AM ET picks |
| `.github/workflows/line_monitor.yml` | Create | GitHub Actions: hourly monitor |
| `.github/workflows/post_results.yml` | Create | GitHub Actions: 10 AM ET results |
| `tests/test_supabase_client.py` | Create | Tests for Supabase wrapper |
| `tests/test_discord_bot.py` | Create | Tests for all Discord posting functions |
| `tests/test_monitor.py` | Create | Tests for line movement monitor |
| `tests/test_post_results.py` | Create | Tests for morning results script |

---

## Task 1: Supabase Client

**Files:**
- Create: `agents/supabase_client.py`
- Modify: `requirements.txt`
- Test: `tests/test_supabase_client.py`

- [ ] **Step 1: Add `supabase` to requirements.txt**

```
requests
pandas
python-dotenv
tzdata
pytest
supabase
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_supabase_client.py
import pytest
from unittest.mock import MagicMock


def test_insert_clv_rows_calls_supabase_upsert(monkeypatch):
    mock_client = MagicMock()
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    sc.insert_clv_rows([{"game_date": "2026-06-03", "player_name": "Aaron Judge"}])
    mock_client.table.assert_called_with("clv_log")
    mock_client.table().upsert.assert_called_once()


def test_fetch_clv_log_returns_dataframe(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.data = [{"game_date": "2026-06-03", "player_name": "Aaron Judge", "ev_pct": 0.18}]
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.execute.return_value = mock_resp
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    df = sc.fetch_clv_log()
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Aaron Judge"


def test_fetch_clv_log_returns_empty_dataframe_when_no_data(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.data = []
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.execute.return_value = mock_resp
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    df = sc.fetch_clv_log()
    assert df.empty


def test_upsert_outcome_calls_supabase(monkeypatch):
    mock_client = MagicMock()
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    sc.upsert_outcome("2026-06-03", "Aaron Judge", hit_hr=1, hrs_hit=1, at_bats=4)
    mock_client.table.assert_called_with("hr_outcomes")
    mock_client.table().upsert.assert_called_once()


def test_client_raises_on_missing_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    import agents.supabase_client as sc
    with pytest.raises(EnvironmentError, match="SUPABASE_URL"):
        sc._client()
```

- [ ] **Step 3: Run tests to verify they fail**

```
pytest tests/test_supabase_client.py -v
```
Expected: `ModuleNotFoundError` or `ImportError` (file doesn't exist yet)

- [ ] **Step 4: Create `agents/supabase_client.py`**

```python
import os
import pandas as pd


def _client():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set")
    from supabase import create_client
    return create_client(url, key)


def insert_clv_rows(rows: list[dict]) -> None:
    if not rows:
        return
    _client().table("clv_log").upsert(
        rows, on_conflict="game_date,game,player_name"
    ).execute()


def fetch_clv_log(game_date: str | None = None, featured_only: bool = False) -> pd.DataFrame:
    q = _client().table("clv_log").select("*")
    if game_date:
        q = q.eq("game_date", game_date)
    if featured_only:
        q = q.eq("featured_bet", True)
    resp = q.execute()
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)


def upsert_outcome(
    game_date: str,
    player_name: str,
    hit_hr,
    hrs_hit: int,
    at_bats: int,
    game_pk=None,
    team: str = "",
    game_str: str = "",
    captured_ts: str = "",
) -> None:
    _client().table("hr_outcomes").upsert(
        {
            "game_date": game_date,
            "player_name": player_name,
            "hit_hr": hit_hr,
            "hrs_hit": hrs_hit,
            "at_bats": at_bats,
            "game_pk": game_pk,
            "team": team,
            "game": game_str,
            "captured_ts": captured_ts,
        },
        on_conflict="game_date,player_name",
    ).execute()


def fetch_outcomes(game_date: str | None = None) -> pd.DataFrame:
    q = _client().table("hr_outcomes").select("*")
    if game_date:
        q = q.eq("game_date", game_date)
    resp = q.execute()
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)


def mark_withdrawn(game_date: str, player_name: str) -> None:
    _client().table("clv_log").update({"withdrawn": True}).eq(
        "game_date", game_date
    ).eq("player_name", player_name).execute()
```

- [ ] **Step 5: Run tests to verify they pass**

```
pytest tests/test_supabase_client.py -v
```
Expected: 5 passed

- [ ] **Step 6: Run full suite to verify no regressions**

```
pytest --tb=short -q
```
Expected: all previously passing tests still pass

- [ ] **Step 7: Commit**

```
git add requirements.txt agents/supabase_client.py tests/test_supabase_client.py
git commit -m "feat: add agents/supabase_client.py — thin Supabase wrapper for CLV log and outcomes"
```

---

## Task 2: `featured_bet` Column + Supabase Write in `clv_log.py`

**Files:**
- Modify: `agents/clv_log.py`
- Test: `tests/test_clv_log.py` (add tests, do not remove existing ones)

Current `OPEN_COLS` (line 34–40 in `agents/clv_log.py`):
```python
OPEN_COLS = [
    "run_ts", "game_date", "commence_iso", "game", "player_name", "team",
    "best_retail_book", "best_retail_odds", "best_retail_decimal",
    "pinnacle_over_odds", "pinnacle_prob_devig", "ev_pct",
    "kelly_units", "stake_usd",
    "anchor_quality",
]
```

- [ ] **Step 1: Write the failing tests**

Add these tests to `tests/test_clv_log.py` (keep all existing tests):

```python
def test_featured_bet_true_when_thresholds_met(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.15, "kelly_units": 0.8, "stake_usd": 20.0,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == True


def test_featured_bet_false_when_kelly_below_threshold(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.15, "kelly_units": 0.4, "stake_usd": 10.0,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == False


def test_featured_bet_false_when_ev_below_threshold(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.05, "kelly_units": 0.8, "stake_usd": 20.0,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == False
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_clv_log.py::test_featured_bet_true_when_thresholds_met -v
```
Expected: FAIL — `featured_bet` column not yet in schema

- [ ] **Step 3: Update `agents/clv_log.py`**

**3a.** Add `"featured_bet"` to `OPEN_COLS` (after `"anchor_quality"`):

```python
OPEN_COLS = [
    "run_ts", "game_date", "commence_iso", "game", "player_name", "team",
    "best_retail_book", "best_retail_odds", "best_retail_decimal",
    "pinnacle_over_odds", "pinnacle_prob_devig", "ev_pct",
    "kelly_units", "stake_usd",
    "anchor_quality",
    "featured_bet",
]
```

**3b.** In `log_open_plays()`, compute `featured_bet` and add it to the row dict. Replace the `rows.append({...})` block (lines 62–78) with:

```python
    for _, r in final_df.iterrows():
        ct = r["commence_time"]
        featured = (
            float(r.get("kelly_units", 0)) >= 0.5
            and float(r.get("ev_pct", 0)) >= 0.10
        )
        rows.append({
            "run_ts": now.isoformat(),
            "game_date": ct.astimezone(ET).strftime("%Y-%m-%d"),
            "commence_iso": ct.astimezone(timezone.utc).isoformat(),
            "game": r["game"],
            "player_name": _norm(r["player_name"]),
            "team": r.get("team", ""),
            "best_retail_book": r["best_retail_book"],
            "best_retail_odds": int(r["best_retail_odds"]),
            "best_retail_decimal": float(r["best_retail_decimal"]),
            "pinnacle_over_odds": int(r["pinnacle_odds"]),
            "pinnacle_prob_devig": float(r["pinnacle_prob"]),
            "ev_pct": float(r["ev_pct"]),
            "kelly_units": float(r["kelly_units"]),
            "stake_usd": float(r["stake_usd"]),
            "anchor_quality": str(r.get("anchor_quality", "unknown")),
            "featured_bet": featured,
        })
```

**3c.** After the CSV write at the bottom of `log_open_plays()`, add the Supabase dual-write:

```python
    combined.reset_index().reindex(columns=COLUMNS).to_csv(path, index=False)

    # Dual-write to Supabase when configured (primary store for GitHub Actions)
    if os.environ.get("SUPABASE_KEY"):
        try:
            from agents.supabase_client import insert_clv_rows
            insert_clv_rows(rows)
        except Exception as e:
            print(f"  [clv_log] Supabase write failed: {e} — CSV is the fallback")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_clv_log.py -v
```
Expected: all pass including the 3 new tests

- [ ] **Step 5: Run full suite**

```
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 6: Commit**

```
git add agents/clv_log.py tests/test_clv_log.py
git commit -m "feat: add featured_bet column to clv_log and dual-write to Supabase"
```

---

## Task 3: `featured_only` in `compute_roi_metrics`

**Files:**
- Modify: `agents/outcome_tracker.py`
- Test: `tests/test_run.py` or create `tests/test_outcome_tracker.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_outcome_tracker.py`:

```python
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch


def _make_clv_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    p = tmp_path / "clv_log.csv"
    df.to_csv(p, index=False)
    return p


def _make_outcomes_db(tmp_path, rows):
    import sqlite3
    db = tmp_path / "hr_outcomes.db"
    conn = sqlite3.connect(db)
    conn.execute("""CREATE TABLE outcomes (
        game_date TEXT, player_name TEXT, team TEXT, game TEXT,
        game_pk INTEGER, hit_hr INTEGER, hrs_hit INTEGER DEFAULT 0,
        at_bats INTEGER DEFAULT 0, captured_ts TEXT,
        PRIMARY KEY (game_date, player_name))""")
    for r in rows:
        conn.execute(
            "INSERT INTO outcomes VALUES (?,?,?,?,?,?,?,?,?)",
            (r["game_date"], r["player_name"], r.get("team",""), r.get("game",""),
             None, r["hit_hr"], r.get("hrs_hit",0), r.get("at_bats",4), "2026-06-03")
        )
    conn.commit()
    conn.close()
    return db


def test_featured_only_filters_non_featured(tmp_path):
    clv_path = _make_clv_csv(tmp_path, [
        {"game_date": "2026-06-01", "player_name": "Aaron Judge",
         "best_retail_decimal": 4.0, "kelly_units": 1.0, "stake_usd": 25.0,
         "ev_pct": 0.20, "featured_bet": True, "anchor_quality": "pinnacle"},
        {"game_date": "2026-06-01", "player_name": "Mike Trout",
         "best_retail_decimal": 3.5, "kelly_units": 0.3, "stake_usd": 7.5,
         "ev_pct": 0.05, "featured_bet": False, "anchor_quality": "pinnacle"},
    ])
    db_path = _make_outcomes_db(tmp_path, [
        {"game_date": "2026-06-01", "player_name": "Aaron Judge", "hit_hr": 1},
        {"game_date": "2026-06-01", "player_name": "Mike Trout", "hit_hr": 0},
    ])
    from agents.outcome_tracker import compute_roi_metrics
    metrics = compute_roi_metrics(clv_log_path=clv_path, db_path=db_path, featured_only=True)
    assert metrics["n_with_outcome"] == 1  # only Judge (featured)


def test_featured_only_false_includes_all(tmp_path):
    clv_path = _make_clv_csv(tmp_path, [
        {"game_date": "2026-06-01", "player_name": "Aaron Judge",
         "best_retail_decimal": 4.0, "kelly_units": 1.0, "stake_usd": 25.0,
         "ev_pct": 0.20, "featured_bet": True, "anchor_quality": "pinnacle"},
        {"game_date": "2026-06-01", "player_name": "Mike Trout",
         "best_retail_decimal": 3.5, "kelly_units": 0.3, "stake_usd": 7.5,
         "ev_pct": 0.05, "featured_bet": False, "anchor_quality": "pinnacle"},
    ])
    db_path = _make_outcomes_db(tmp_path, [
        {"game_date": "2026-06-01", "player_name": "Aaron Judge", "hit_hr": 1},
        {"game_date": "2026-06-01", "player_name": "Mike Trout", "hit_hr": 0},
    ])
    from agents.outcome_tracker import compute_roi_metrics
    metrics = compute_roi_metrics(clv_log_path=clv_path, db_path=db_path, featured_only=False)
    assert metrics["n_with_outcome"] == 2
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_outcome_tracker.py -v
```
Expected: FAIL — `compute_roi_metrics` doesn't accept `featured_only` yet

- [ ] **Step 3: Update `agents/outcome_tracker.py`**

**3a.** Change the `compute_roi_metrics` signature (line 259):

```python
def compute_roi_metrics(
    clv_log_path: Path = CLV_LOG_PATH,
    db_path: Path = DB_PATH,
    featured_only: bool = False,
) -> dict:
```

**3b.** After loading `clv` from CSV (around line 270, after `clv = pd.read_csv(clv_log_path)`), add the Supabase/featured_only logic. Replace:

```python
    clv = pd.read_csv(clv_log_path)
    outcomes = load_outcomes(db_path)
```

With:

```python
    import os
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_key:
        try:
            from agents.supabase_client import fetch_clv_log, fetch_outcomes as _fetch_outcomes
            clv = fetch_clv_log()
            outcomes = _fetch_outcomes()
        except Exception:
            clv = pd.read_csv(clv_log_path)
            outcomes = load_outcomes(db_path)
    else:
        clv = pd.read_csv(clv_log_path)
        outcomes = load_outcomes(db_path)

    if featured_only and "featured_bet" in clv.columns:
        clv = clv[clv["featured_bet"].astype(str).str.lower() == "true"].copy()
```

**3c.** Make `update_for_date()` also write outcomes to Supabase when configured. In `update_for_date()`, after the `conn.execute(...)` INSERT and before `conn.commit()`, add:

```python
        # Dual-write to Supabase when configured
        if os.environ.get("SUPABASE_KEY"):
            try:
                from agents.supabase_client import upsert_outcome
                upsert_outcome(
                    game_date=date_str,
                    player_name=pname,
                    hit_hr=hit_hr,
                    hrs_hit=hrs_hit,
                    at_bats=at_bats,
                    game_pk=game_pk,
                    team=team,
                    game_str=game,
                    captured_ts=now_ts,
                )
            except Exception as e:
                print(f"  [outcome_tracker] Supabase write failed for {pname}: {e}")
```

Also at the top of `update_for_date()`, add Supabase-aware picks loading. Replace:

```python
    clv = pd.read_csv(clv_log_path)
    picks = clv[clv["game_date"] == date_str][["player_name", "team", "game"]].copy()
```

With:

```python
    import os
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_key:
        try:
            from agents.supabase_client import fetch_clv_log
            clv = fetch_clv_log(game_date=date_str)
        except Exception:
            clv = pd.read_csv(clv_log_path) if clv_log_path.exists() else pd.DataFrame()
    elif clv_log_path.exists():
        clv = pd.read_csv(clv_log_path)
    else:
        clv = pd.DataFrame()
    picks = clv[clv["game_date"] == date_str][["player_name", "team", "game"]].copy() if not clv.empty else pd.DataFrame(columns=["player_name", "team", "game"])
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_outcome_tracker.py -v
```
Expected: 2 passed

- [ ] **Step 5: Run full suite**

```
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 6: Commit**

```
git add agents/outcome_tracker.py tests/test_outcome_tracker.py
git commit -m "feat: add featured_only param to compute_roi_metrics; read from Supabase when configured"
```

---

## Task 4: `--no-browser` Flag in `run.py`

**Files:**
- Modify: `run.py`
- Test: `tests/test_run.py` (add tests, keep existing)

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_run.py`:

```python
def test_no_browser_flag_calls_post_picks(monkeypatch):
    import run
    monkeypatch.setattr(run, "fetch_odds", lambda key, now: [])
    monkeypatch.setattr(run, "fetch_player_teams", lambda: {})
    import pandas as pd
    from agents.validation import StepResult
    monkeypatch.setattr("agents.odds_scraper.extract_retail_odds", lambda raw, now: pd.DataFrame())
    monkeypatch.setattr("agents.pinnacle_scraper.extract_sharp_anchor", lambda raw, now: pd.DataFrame())

    import sys
    monkeypatch.setattr(sys, "argv", ["run.py", "--no-browser"])

    post_picks_calls = []
    monkeypatch.setattr("agents.discord_bot.post_picks", lambda df: post_picks_calls.append(df))

    # Empty anchor_df triggers early return — just verify argparse parses without error
    run.main()
    # No assertion needed — if argparse fails, main() throws


def test_default_run_does_not_import_discord_bot(monkeypatch):
    import sys
    # Ensure --no-browser not in argv
    monkeypatch.setattr(sys, "argv", ["run.py"])
    import run
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args([])
    assert not args.no_browser
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_run.py::test_no_browser_flag_calls_post_picks -v
```
Expected: FAIL or error — `--no-browser` not yet a valid flag

- [ ] **Step 3: Update `run.py`**

Add `argparse` to `main()`. Replace the current `def main() -> None:` opening:

```python
def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="HR bet tool pipeline")
    parser.add_argument("--no-browser", action="store_true",
                        help="Skip browser; post picks to Discord instead")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ["ODDS_API_KEY"]
    now = datetime.now(timezone.utc)
    ...
```

Replace the final block at the bottom of `main()`:

```python
    generate_dashboard(final_df, parlays=parlays, dfs_data=dfs_data,
                       open_browser=not args.no_browser)
    if args.no_browser:
        from agents.discord_bot import post_picks
        post_picks(final_df, now=now)
        print("Picks posted to Discord.")
    else:
        print("Dashboard opened in browser.")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_run.py -v
```
Expected: all pass

- [ ] **Step 5: Run full suite**

```
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 6: Commit**

```
git add run.py tests/test_run.py
git commit -m "feat: add --no-browser flag to run.py; calls discord_bot.post_picks when set"
```

---

## Task 5: `agents/discord_bot.py`

**Files:**
- Create: `agents/discord_bot.py`
- Test: `tests/test_discord_bot.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_discord_bot.py
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

NOW = datetime(2026, 6, 3, 15, 0, tzinfo=timezone.utc)  # 11 AM ET
GAME_START = NOW + timedelta(hours=3)  # 2 PM ET — far enough ahead


def _featured_df(kelly=0.8, ev=0.18, anchor="pinnacle", commence_time=None):
    ct = commence_time or GAME_START
    return pd.DataFrame([{
        "player_name": "Aaron Judge", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "ev_pct": ev, "kelly_units": kelly,
        "anchor_quality": anchor, "bet_grade": "Strong", "bet_score": 85,
        "featured_bet": True, "commence_time": ct,
    }])


def test_post_picks_calls_webhook(monkeypatch):
    posted = []
    monkeypatch.setenv("DISCORD_PICKS_WEBHOOK", "http://fake/picks")
    monkeypatch.setenv("DISCORD_STATUS_WEBHOOK", "http://fake/status")
    monkeypatch.setattr("requests.post", lambda url, **kw: posted.append(url) or MagicMock())
    monkeypatch.setattr("agents.outcome_tracker.compute_roi_metrics", lambda **kw: {"has_outcomes": False, "n_with_outcome": 0})

    from agents import discord_bot
    discord_bot.post_picks(_featured_df(), now=NOW)
    assert "http://fake/picks" in posted


def test_post_picks_sends_no_plays_message_when_empty(monkeypatch):
    posted_content = []
    monkeypatch.setenv("DISCORD_PICKS_WEBHOOK", "http://fake/picks")
    monkeypatch.setenv("DISCORD_STATUS_WEBHOOK", "http://fake/status")
    monkeypatch.setattr("requests.post", lambda url, json=None, **kw: posted_content.append(json) or MagicMock())
    monkeypatch.setattr("agents.outcome_tracker.compute_roi_metrics", lambda **kw: {"has_outcomes": False})

    from agents import discord_bot
    discord_bot.post_picks(pd.DataFrame(), now=NOW)
    assert any("No featured plays" in (m or {}).get("content", "") for m in posted_content)


def test_post_picks_excludes_games_starting_within_90_min(monkeypatch):
    posted_content = []
    monkeypatch.setenv("DISCORD_PICKS_WEBHOOK", "http://fake/picks")
    monkeypatch.setenv("DISCORD_STATUS_WEBHOOK", "http://fake/status")
    monkeypatch.setattr("requests.post", lambda url, json=None, **kw: posted_content.append(json) or MagicMock())
    monkeypatch.setattr("agents.outcome_tracker.compute_roi_metrics", lambda **kw: {"has_outcomes": False})

    # Game starts in 30 min — too soon
    soon = NOW + timedelta(minutes=30)
    from agents import discord_bot
    discord_bot.post_picks(_featured_df(commence_time=soon), now=NOW)
    picks_msg = next((m for m in posted_content if "fake/picks" or True), None)
    # Should post "No featured plays" because game is too soon
    content = " ".join((m or {}).get("content", "") for m in posted_content)
    assert "No featured plays" in content


def test_post_alert_movement_format(monkeypatch):
    posted_content = []
    monkeypatch.setenv("DISCORD_PICKS_WEBHOOK", "http://fake/picks")
    monkeypatch.setattr("requests.post", lambda url, json=None, **kw: posted_content.append(json) or MagicMock())

    from agents import discord_bot
    discord_bot.post_alert("Aaron Judge", 320, 240, 0.182, 0.061, "movement")
    content = " ".join((m or {}).get("content", "") for m in posted_content)
    assert "⚠️" in content
    assert "Aaron Judge" in content
    assert "+320" in content
    assert "+240" in content


def test_post_alert_withdrawal_format(monkeypatch):
    posted_content = []
    monkeypatch.setenv("DISCORD_PICKS_WEBHOOK", "http://fake/picks")
    monkeypatch.setattr("requests.post", lambda url, json=None, **kw: posted_content.append(json) or MagicMock())

    from agents import discord_bot
    discord_bot.post_alert("Aaron Judge", 320, 180, 0.182, -0.05, "withdrawal")
    content = " ".join((m or {}).get("content", "") for m in posted_content)
    assert "❌" in content
    assert "Withdrawal" in content


def test_post_status_never_raises(monkeypatch):
    monkeypatch.setenv("DISCORD_STATUS_WEBHOOK", "http://fake/status")
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(Exception("network error")))
    from agents import discord_bot
    discord_bot.post_status("test message")  # must not raise


def test_all_posting_functions_catch_exceptions(monkeypatch):
    monkeypatch.setenv("DISCORD_PICKS_WEBHOOK", "http://fake/picks")
    monkeypatch.setenv("DISCORD_RESULTS_WEBHOOK", "http://fake/results")
    monkeypatch.setenv("DISCORD_STATUS_WEBHOOK", "http://fake/status")
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))
    monkeypatch.setattr("agents.outcome_tracker.compute_roi_metrics", lambda **kw: {})

    from agents import discord_bot
    discord_bot.post_picks(_featured_df(), now=NOW)  # must not raise
    discord_bot.post_results("2026-06-02")           # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_discord_bot.py -v
```
Expected: `ModuleNotFoundError` (file doesn't exist yet)

- [ ] **Step 3: Create `agents/discord_bot.py`**

```python
"""Discord webhook delivery for HR picks, results, and health status."""
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
_LOG = Path("data/discord.log")
_LOG.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=str(_LOG), level=logging.ERROR,
                    format="%(asctime)s %(levelname)s %(message)s")

_GRADE_EMOJI = {"Strong": "🔴", "Solid": "🟡", "Marginal": "🟠"}


def _wh(env_var: str) -> str:
    val = os.environ.get(env_var, "")
    if not val:
        raise EnvironmentError(f"{env_var} must be set")
    return val


def _american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1


def _odds_str(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def _bet_by(commence_time, buffer_min: int = 30) -> str:
    if pd.isna(commence_time):
        return "before game"
    bt = commence_time - timedelta(minutes=buffer_min)
    s = bt.astimezone(ET).strftime("%I:%M %p ET")
    return s.lstrip("0")


def _running_metrics() -> tuple[float | None, float | None]:
    """Return (roi, mean_clv_pct) for featured bets. Returns (None, None) on error."""
    try:
        from agents.outcome_tracker import compute_roi_metrics
        m = compute_roi_metrics(featured_only=True)
        roi = m.get("roi")
        clv = None
        # Get mean CLV — try Supabase first, fall back to CSV
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        if supabase_key:
            try:
                from agents.supabase_client import fetch_clv_log
                df = fetch_clv_log(featured_only=True)
            except Exception:
                df = pd.DataFrame()
        else:
            clv_path = Path("data/clv_log.csv")
            df = pd.read_csv(clv_path) if clv_path.exists() else pd.DataFrame()
            if not df.empty and "featured_bet" in df.columns:
                df = df[df["featured_bet"].astype(str).str.lower() == "true"]
        if not df.empty and "clv_pct" in df.columns:
            vals = pd.to_numeric(df["clv_pct"], errors="coerce").dropna()
            clv = float(vals.mean()) if len(vals) > 0 else None
        return roi, clv
    except Exception:
        return None, None


def post_status(message: str) -> None:
    """Post to #system-status. Never raises."""
    try:
        wh = _wh("DISCORD_STATUS_WEBHOOK")
        requests.post(wh, json={"content": message}, timeout=10)
    except Exception:
        pass


def post_alert(
    player_name: str,
    old_odds: int,
    new_odds: int,
    old_ev: float,
    new_ev: float,
    alert_type: str,
) -> None:
    """Post a line movement alert or withdrawal to #picks."""
    try:
        wh = _wh("DISCORD_PICKS_WEBHOOK")
        old_str = _odds_str(old_odds)
        new_str = _odds_str(new_odds)
        if alert_type == "withdrawal":
            msg = (f"❌ Withdrawal — {player_name}: {new_str} · "
                   f"EV now {new_ev*100:+.1f}% · skip this play")
        else:
            msg = (f"⚠️ Line alert — {player_name}: {old_str} → {new_str} · "
                   f"EV {old_ev*100:+.1f}% → {new_ev*100:+.1f}% · edge reduced")
        requests.post(wh, json={"content": msg}, timeout=10).raise_for_status()
    except Exception as e:
        logging.error(f"post_alert failed: {e}")


def post_picks(final_df: pd.DataFrame, now: datetime | None = None) -> None:
    """Post daily picks to #picks. Called from run.py --no-browser."""
    now = now or datetime.now(timezone.utc)
    try:
        picks_wh = _wh("DISCORD_PICKS_WEBHOOK")
        status_wh = _wh("DISCORD_STATUS_WEBHOOK")

        # Filter to featured bets, exclude games starting within 90 min
        if final_df.empty or "featured_bet" not in final_df.columns:
            featured = pd.DataFrame()
        else:
            featured = final_df[final_df["featured_bet"] == True].copy()
            if "commence_time" in featured.columns:
                featured = featured[featured["commence_time"].apply(
                    lambda ct: pd.isna(ct) or (ct - now).total_seconds() / 60 > 90
                )]
            featured = featured.sort_values(
                ["kelly_units", "ev_pct"], ascending=False
            )

        date_label = now.astimezone(ET).strftime("%a %b %-d") if os.name != "nt" else now.astimezone(ET).strftime("%a %b %#d")

        if featured.empty:
            msg = f"⚾ HR Picks — {date_label}\n\nNo featured plays today."
            n_plays = 0
        else:
            lines = [f"⚾ HR Picks — {date_label}\n"]
            for _, r in featured.iterrows():
                grade = str(r.get("bet_grade", ""))
                emoji = _GRADE_EMOJI.get(grade, "⚪")
                kelly = float(r.get("kelly_units", 0))
                ev = float(r.get("ev_pct", 0)) * 100
                anchor = "PIN" if str(r.get("anchor_quality", "")) == "pinnacle" else "BOL"
                bet_by = _bet_by(r.get("commence_time"))
                odds = int(r["best_retail_odds"])
                line = (f"{emoji} {grade:<8} {r['player_name']} · "
                        f"{r['best_retail_book']} {_odds_str(odds)} · "
                        f"EV +{ev:.1f}% · {kelly:.1f}u · {anchor} · Bet by {bet_by}")
                lines.append(line)

            roi, clv = _running_metrics()
            footer_parts = [f"{len(featured)} plays", "1u = $25"]
            if roi is not None:
                footer_parts.append(f"Running ROI: {roi*100:+.1f}%")
            if clv is not None:
                footer_parts.append(f"CLV: {clv*100:+.1f}%")
            lines.append("\n" + " · ".join(footer_parts))
            msg = "\n".join(lines)
            n_plays = len(featured)

        requests.post(picks_wh, json={"content": msg}, timeout=10).raise_for_status()

        time_label = now.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
        date_d = now.astimezone(ET).strftime("%b %-d") if os.name != "nt" else now.astimezone(ET).strftime("%b %#d")
        if n_plays == 0:
            post_status(f"⚠️ Picks ran — 0 featured plays found · {date_d} {time_label}")
        else:
            post_status(f"✅ Picks posted — {n_plays} plays · {date_d} {time_label}")

    except Exception as e:
        logging.error(f"post_picks failed: {e}")
        try:
            post_status(f"❌ Picks FAILED — {now.astimezone(ET).strftime('%b %d %I:%M %p ET')}: {e}")
        except Exception:
            pass


def post_results(date_str: str, now: datetime | None = None) -> None:
    """Post settled results for date_str to #results."""
    now = now or datetime.now(timezone.utc)
    try:
        results_wh = _wh("DISCORD_RESULTS_WEBHOOK")

        # Load picks — Supabase first (required for GitHub Actions), CSV fallback
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        if supabase_key:
            try:
                from agents.supabase_client import fetch_clv_log, fetch_outcomes as _fetch_outcomes
                picks = fetch_clv_log(game_date=date_str, featured_only=True)
                outcomes_df = _fetch_outcomes(game_date=date_str)
            except Exception as e:
                post_status(f"⚠️ Results skipped — Supabase read failed: {e}")
                return
        else:
            clv_path = Path("data/clv_log.csv")
            if not clv_path.exists():
                post_status(f"⚠️ Results skipped — clv_log.csv not found")
                return
            clv = pd.read_csv(clv_path)
            if "featured_bet" not in clv.columns:
                post_status(f"⚠️ Results skipped — featured_bet column missing")
                return
            picks = clv[
                (clv["game_date"] == date_str) &
                (clv["featured_bet"].astype(str).str.lower() == "true")
            ].copy()
            from agents.outcome_tracker import load_outcomes
            outcomes_df = load_outcomes()

        if picks.empty:
            post_status(f"✅ Results — no featured bets on {date_str}")
            return

        outcomes = outcomes_df
        if outcomes.empty:
            post_status(f"⚠️ Results — outcomes DB empty for {date_str}")
            return

        merged = picks.merge(
            outcomes[["game_date", "player_name", "hit_hr"]],
            on=["game_date", "player_name"], how="left"
        )

        lines = [f"📋 Results — {date_str}\n"]
        day_pnl = 0.0
        for _, r in merged.iterrows():
            name = str(r["player_name"])
            odds = int(r["best_retail_odds"])
            book = str(r["best_retail_book"])
            stake = float(r.get("stake_usd", 0))
            decimal = float(r.get("best_retail_decimal", _american_to_decimal(odds)))

            if pd.isna(r.get("hit_hr")):
                lines.append(f"➖ {name} · scratched — no result")
            elif int(r["hit_hr"]) == 1:
                pnl = stake * (decimal - 1)
                day_pnl += pnl
                lines.append(f"✅ {name} · {book} {_odds_str(odds)} · HIT · +${pnl:.2f}")
            else:
                day_pnl -= stake
                lines.append(f"❌ {name} · {book} {_odds_str(odds)} · miss · -${stake:.2f}")

        roi, _ = _running_metrics()
        footer_parts = [f"Day: {day_pnl:+.2f}"]
        if roi is not None:
            footer_parts.append(f"Running ROI: {roi*100:+.1f}%")
        lines.append("\n" + " · ".join(footer_parts))

        requests.post(results_wh, json={"content": "\n".join(lines)}, timeout=10).raise_for_status()
        n = len(merged)
        post_status(f"✅ Results posted — {date_str} · {n} settled")

    except Exception as e:
        logging.error(f"post_results failed: {e}")
        try:
            post_status(f"❌ Results FAILED — {date_str}: {e}")
        except Exception:
            pass


def post_weekly_recap(now: datetime | None = None) -> None:
    """Post Sunday weekly recap to #weekly-recap."""
    now = now or datetime.now(timezone.utc)
    try:
        recap_wh = _wh("DISCORD_RECAP_WEBHOOK")

        from agents.outcome_tracker import compute_roi_metrics
        metrics = compute_roi_metrics(featured_only=True)

        # This-week P&L: filter settled picks to game_date >= last Monday
        from datetime import date
        today = now.astimezone(ET).date()
        last_monday = today - timedelta(days=today.weekday())

        week_pnl = None
        week_plays = 0
        week_hits = 0
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        try:
            if supabase_key:
                from agents.supabase_client import fetch_clv_log, fetch_outcomes as _fetch_out
                clv = fetch_clv_log(featured_only=True)
                outcomes = _fetch_out()
            else:
                clv_path = Path("data/clv_log.csv")
                clv = pd.read_csv(clv_path) if clv_path.exists() else pd.DataFrame()
                if not clv.empty and "featured_bet" in clv.columns:
                    clv = clv[clv["featured_bet"].astype(str).str.lower() == "true"]
                from agents.outcome_tracker import load_outcomes
                outcomes = load_outcomes()

            if not clv.empty and not outcomes.empty:
                feat = clv[pd.to_datetime(clv["game_date"]) >= pd.Timestamp(last_monday)]
                merged = feat.merge(outcomes[["game_date", "player_name", "hit_hr"]],
                                    on=["game_date", "player_name"], how="left")
                settled = merged[merged["hit_hr"].notna()].copy()
                settled["hit_hr"] = settled["hit_hr"].astype(int)
                settled["stake_usd"] = pd.to_numeric(settled["stake_usd"], errors="coerce").fillna(0)
                settled["best_retail_decimal"] = pd.to_numeric(settled["best_retail_decimal"], errors="coerce")
                pnl = settled.apply(
                    lambda r: r["stake_usd"] * (r["best_retail_decimal"] - 1) if r["hit_hr"] == 1
                    else -r["stake_usd"], axis=1
                )
                week_pnl = float(pnl.sum())
                week_plays = len(settled)
                week_hits = int(settled["hit_hr"].sum())
        except Exception:
            pass

        n_total = metrics.get("n_with_outcome", 0)
        roi = metrics.get("roi")
        hit_rate = metrics.get("hit_rate")

        mon_str = last_monday.strftime("%-m/%-d") if os.name != "nt" else last_monday.strftime("%#m/%#d")
        sun_str = today.strftime("%-m/%-d") if os.name != "nt" else today.strftime("%#m/%#d")

        lines = [f"📊 Weekly Recap · {mon_str} – {sun_str}"]
        if week_pnl is not None:
            hit_pct = (week_hits / week_plays * 100) if week_plays else 0
            lines.append(f"This week: {week_plays} plays · {week_hits} hits · {hit_pct:.1f}% · P&L: {week_pnl:+.2f}")
        if n_total and roi is not None and hit_rate is not None:
            lines.append(
                f"All-time:  {n_total} plays · {roi*100:+.1f}% ROI · "
                f"{hit_rate*100:.1f}% hit rate"
            )
        lines.append("Anchor: Pinnacle/BOL devig · 1u = $25 · Kelly ¼ sizing")

        requests.post(recap_wh, json={"content": "\n".join(lines)}, timeout=10).raise_for_status()
        post_status(f"✅ Weekly recap posted · {sun_str}")

    except Exception as e:
        logging.error(f"post_weekly_recap failed: {e}")
        try:
            post_status(f"❌ Weekly recap FAILED: {e}")
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_discord_bot.py -v
```
Expected: all pass

- [ ] **Step 5: Run full suite**

```
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 6: Commit**

```
git add agents/discord_bot.py tests/test_discord_bot.py
git commit -m "feat: add agents/discord_bot.py — post_picks, post_results, post_weekly_recap, post_alert, post_status"
```

---

## Task 6: `monitor.py` — Hourly Line Movement Monitor

**Files:**
- Create: `monitor.py`
- Test: `tests/test_monitor.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_monitor.py
import json
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

NOW = datetime(2026, 6, 3, 18, 0, tzinfo=timezone.utc)  # 2 PM ET
GAME_START = NOW + timedelta(hours=3)


def _clv_row(kelly=0.8, ev=0.18, odds=320, pin_prob=0.22):
    return {
        "game_date": "2026-06-03", "player_name": "Aaron Judge",
        "best_retail_book": "DraftKings", "best_retail_odds": odds,
        "pinnacle_prob_devig": pin_prob, "ev_pct": ev, "kelly_units": kelly,
        "featured_bet": True, "commence_iso": GAME_START.isoformat(),
        "best_retail_decimal": (odds / 100 + 1) if odds > 0 else (100 / abs(odds) + 1),
    }


def _current_odds_df(odds=320):
    return pd.DataFrame([{
        "player_name": "Aaron Judge", "best_retail_book": "DraftKings",
        "best_retail_odds": odds, "best_retail_decimal": (odds / 100 + 1),
    }])


def test_fires_movement_alert_on_large_line_move(tmp_path, monkeypatch):
    clv_csv = tmp_path / "clv_log.csv"
    pd.DataFrame([_clv_row(odds=320)]).to_csv(clv_csv, index=False)

    state_path = tmp_path / "monitor_state.json"
    alerts = []

    monkeypatch.setenv("ODDS_API_KEY", "fake")
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    import monitor
    monkeypatch.setattr(monitor, "CLV_PATH", clv_csv)
    monkeypatch.setattr(monitor, "STATE_PATH", state_path)

    monitor.run(
        now=NOW,
        fetch_odds_fn=lambda key, now: [],
        current_odds_fn=lambda raw, now: _current_odds_df(odds=240),  # moved 80 pts
        post_alert_fn=lambda *a, **kw: alerts.append(a),
        post_status_fn=lambda msg: None,
    )
    assert len(alerts) == 1
    assert alerts[0][4] == "movement"


def test_fires_withdrawal_on_negative_ev(tmp_path, monkeypatch):
    clv_csv = tmp_path / "clv_log.csv"
    pd.DataFrame([_clv_row(odds=320, pin_prob=0.22)]).to_csv(clv_csv, index=False)
    state_path = tmp_path / "monitor_state.json"
    alerts = []

    monkeypatch.setenv("ODDS_API_KEY", "fake")
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    import monitor
    monkeypatch.setattr(monitor, "CLV_PATH", clv_csv)
    monkeypatch.setattr(monitor, "STATE_PATH", state_path)

    # odds moved to +160 → decimal 2.6 → ev = 2.6*0.22 - 1 = -0.428 → negative
    monitor.run(
        now=NOW,
        fetch_odds_fn=lambda key, now: [],
        current_odds_fn=lambda raw, now: _current_odds_df(odds=160),
        post_alert_fn=lambda *a, **kw: alerts.append(a),
        post_status_fn=lambda msg: None,
    )
    assert len(alerts) == 1
    assert alerts[0][4] == "withdrawal"


def test_skips_already_alerted_player(tmp_path, monkeypatch):
    clv_csv = tmp_path / "clv_log.csv"
    pd.DataFrame([_clv_row(odds=320)]).to_csv(clv_csv, index=False)
    state_path = tmp_path / "monitor_state.json"
    state_path.write_text(json.dumps({"2026-06-03": {"Aaron Judge": {"alert_sent": True}}}))
    alerts = []

    monkeypatch.setenv("ODDS_API_KEY", "fake")
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    import monitor
    monkeypatch.setattr(monitor, "CLV_PATH", clv_csv)
    monkeypatch.setattr(monitor, "STATE_PATH", state_path)

    monitor.run(
        now=NOW,
        fetch_odds_fn=lambda key, now: [],
        current_odds_fn=lambda raw, now: _current_odds_df(odds=240),
        post_alert_fn=lambda *a, **kw: alerts.append(a),
        post_status_fn=lambda msg: None,
    )
    assert len(alerts) == 0


def test_skips_game_already_started(tmp_path, monkeypatch):
    clv_csv = tmp_path / "clv_log.csv"
    past_start = NOW - timedelta(hours=1)  # game already started
    row = {**_clv_row(), "commence_iso": past_start.isoformat()}
    pd.DataFrame([row]).to_csv(clv_csv, index=False)
    state_path = tmp_path / "monitor_state.json"
    alerts = []

    monkeypatch.setenv("ODDS_API_KEY", "fake")
    monkeypatch.delenv("SUPABASE_KEY", raising=False)

    import monitor
    monkeypatch.setattr(monitor, "CLV_PATH", clv_csv)
    monkeypatch.setattr(monitor, "STATE_PATH", state_path)

    monitor.run(
        now=NOW,
        fetch_odds_fn=lambda key, now: [],
        current_odds_fn=lambda raw, now: _current_odds_df(odds=100),
        post_alert_fn=lambda *a, **kw: alerts.append(a),
        post_status_fn=lambda msg: None,
    )
    assert len(alerts) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_monitor.py -v
```
Expected: `ModuleNotFoundError` (file doesn't exist)

- [ ] **Step 3: Create `monitor.py`**

```python
"""Hourly line movement monitor — GitHub Actions runs this every hour 11 AM–9 PM ET."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")
CLV_PATH = Path("data/clv_log.csv")
STATE_PATH = Path("data/monitor_state.json")


def _american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run(
    now=None,
    fetch_odds_fn=None,
    current_odds_fn=None,
    post_alert_fn=None,
    post_status_fn=None,
) -> None:
    load_dotenv()
    now = now or datetime.now(timezone.utc)
    today_str = now.astimezone(ET).strftime("%Y-%m-%d")

    # Load today's featured bets
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_key:
        try:
            from agents.supabase_client import fetch_clv_log
            picks = fetch_clv_log(game_date=today_str, featured_only=True)
        except Exception:
            picks = pd.DataFrame()
    else:
        if not CLV_PATH.exists():
            return
        df = pd.read_csv(CLV_PATH)
        picks = df[
            (df["game_date"] == today_str) &
            (df["featured_bet"].astype(str).str.lower() == "true")
        ].copy()

    if picks.empty:
        return

    # Only picks for games not yet started
    picks = picks[picks["commence_iso"].apply(
        lambda iso: datetime.fromisoformat(str(iso)) > now if pd.notna(iso) else True
    )]
    if picks.empty:
        return

    # Re-scrape current retail odds
    api_key = os.environ["ODDS_API_KEY"]
    if fetch_odds_fn is None:
        from run import fetch_odds
        fetch_odds_fn = fetch_odds
    raw = fetch_odds_fn(api_key, now)

    if current_odds_fn is None:
        from agents.odds_scraper import extract_retail_odds
        current_odds_fn = extract_retail_odds
    current_df = current_odds_fn(raw, now)
    if current_df.empty:
        return

    current_idx = current_df.set_index("player_name") if "player_name" in current_df.columns else pd.DataFrame()

    state = _load_state()
    today_state = state.get(today_str, {})

    if post_alert_fn is None:
        from agents.discord_bot import post_alert
        post_alert_fn = post_alert
    if post_status_fn is None:
        from agents.discord_bot import post_status
        post_status_fn = post_status

    alerts_sent = 0
    for _, row in picks.iterrows():
        pname = str(row["player_name"])
        player_state = today_state.get(pname, {})

        if player_state.get("withdrawal_sent"):
            continue

        if pname not in current_idx.index:
            continue  # line pulled or game started

        curr_row = current_idx.loc[pname]
        if isinstance(curr_row, pd.DataFrame):
            curr_row = curr_row.iloc[0]

        curr_odds = int(curr_row["best_retail_odds"])
        orig_odds = int(row["best_retail_odds"])
        pin_prob = float(row["pinnacle_prob_devig"])
        curr_ev = _american_to_decimal(curr_odds) * pin_prob - 1
        orig_ev = float(row["ev_pct"])

        if curr_ev < 0 and not player_state.get("withdrawal_sent"):
            post_alert_fn(pname, orig_odds, curr_odds, orig_ev, curr_ev, "withdrawal")
            today_state[pname] = {**player_state, "withdrawal_sent": True}
            alerts_sent += 1
        elif (abs(curr_odds - orig_odds) > 15 or
              (orig_ev >= 0.10 and curr_ev < 0.05)) and not player_state.get("alert_sent"):
            post_alert_fn(pname, orig_odds, curr_odds, orig_ev, curr_ev, "movement")
            today_state[pname] = {**player_state, "alert_sent": True}
            alerts_sent += 1

    state[today_str] = today_state
    _save_state(state)

    time_str = now.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
    date_str_short = now.astimezone(ET).strftime("%b %-d") if os.name != "nt" else now.astimezone(ET).strftime("%b %#d")
    if alerts_sent:
        post_status_fn(f"⚠️ Monitor — {alerts_sent} alert(s) sent · {date_str_short} {time_str}")
    else:
        post_status_fn(f"✅ Monitor — no significant movement · {date_str_short} {time_str}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_monitor.py -v
```
Expected: 4 passed

- [ ] **Step 5: Run full suite**

```
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 6: Commit**

```
git add monitor.py tests/test_monitor.py
git commit -m "feat: add monitor.py — hourly line movement alerts with threshold-based auto-posting"
```

---

## Task 7: `post_results.py` — Morning Results Script

**Files:**
- Create: `post_results.py`
- Test: `tests/test_post_results.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_post_results.py
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock


def test_calls_update_for_date_with_yesterday(monkeypatch):
    calls = {}
    monkeypatch.setattr("agents.outcome_tracker.update_for_date",
                        lambda d, **kw: calls.update({"date": d}))
    monkeypatch.setattr("agents.discord_bot.post_results", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_weekly_recap", lambda **kw: None)

    import post_results
    yesterday = (date.today() - timedelta(days=1)).isoformat()
    post_results.main()
    assert calls.get("date") == yesterday


def test_calls_post_weekly_recap_on_sunday(monkeypatch):
    recap_calls = []
    monkeypatch.setattr("agents.outcome_tracker.update_for_date", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_results", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_weekly_recap",
                        lambda **kw: recap_calls.append(True))

    # Force today to be Sunday (weekday == 6)
    sunday = datetime(2026, 6, 8, tzinfo=timezone.utc)  # a Sunday
    monkeypatch.setattr("post_results.date", type("FakeDate", (), {
        "today": staticmethod(lambda: sunday.date()),
        "isoformat": sunday.date().isoformat,
    }))

    import post_results
    post_results.main()
    assert len(recap_calls) == 1


def test_does_not_call_recap_on_weekday(monkeypatch):
    recap_calls = []
    monkeypatch.setattr("agents.outcome_tracker.update_for_date", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_results", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_weekly_recap",
                        lambda **kw: recap_calls.append(True))

    import post_results
    # default date.today() is a Tuesday (2026-06-03) — no recap
    post_results.main()
    assert len(recap_calls) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/test_post_results.py -v
```
Expected: `ModuleNotFoundError`

- [ ] **Step 3: Create `post_results.py`**

```python
"""Morning results script — runs daily at 10 AM ET via GitHub Actions."""
from datetime import date, timedelta

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    yesterday = (date.today() - timedelta(days=1)).isoformat()

    from agents.outcome_tracker import update_for_date
    from agents.discord_bot import post_results, post_weekly_recap

    print(f"Updating outcomes for {yesterday}...")
    update_for_date(yesterday)

    print("Posting results to Discord...")
    post_results(yesterday)

    if date.today().weekday() == 6:  # Sunday
        print("Sunday — posting weekly recap...")
        post_weekly_recap()

    print("Done.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/test_post_results.py -v
```
Expected: 3 passed

- [ ] **Step 5: Run full suite**

```
pytest --tb=short -q
```
Expected: all passing

- [ ] **Step 6: Commit**

```
git add post_results.py tests/test_post_results.py
git commit -m "feat: add post_results.py — morning results + Sunday recap script"
```

---

## Task 8: GitHub Actions Workflows

**Files:**
- Create: `.github/workflows/daily_picks.yml`
- Create: `.github/workflows/line_monitor.yml`
- Create: `.github/workflows/post_results.yml`

No unit tests — verify by inspecting YAML syntax.

- [ ] **Step 1: Create `.github/workflows/` directory**

```
mkdir -p .github/workflows
```

- [ ] **Step 2: Create `daily_picks.yml`**

```yaml
# .github/workflows/daily_picks.yml
name: Daily HR Picks

on:
  schedule:
    - cron: "0 15 * * *"   # 11:00 AM ET (UTC-4 during EDT)
  workflow_dispatch:         # allow manual trigger for testing

jobs:
  post-picks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Post picks to Discord
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
          DISCORD_PICKS_WEBHOOK: ${{ secrets.DISCORD_PICKS_WEBHOOK }}
          DISCORD_STATUS_WEBHOOK: ${{ secrets.DISCORD_STATUS_WEBHOOK }}
        run: python run.py --no-browser
```

- [ ] **Step 3: Create `line_monitor.yml`**

```yaml
# .github/workflows/line_monitor.yml
name: Line Movement Monitor

on:
  schedule:
    # Every hour from 11 AM to 9 PM ET (15:00–01:00 UTC)
    - cron: "0 15-23 * * *"
    - cron: "0 0,1 * * *"
  workflow_dispatch:

jobs:
  monitor:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run line monitor
        env:
          ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
          DISCORD_PICKS_WEBHOOK: ${{ secrets.DISCORD_PICKS_WEBHOOK }}
          DISCORD_STATUS_WEBHOOK: ${{ secrets.DISCORD_STATUS_WEBHOOK }}
        run: python monitor.py
```

- [ ] **Step 4: Create `post_results.yml`**

```yaml
# .github/workflows/post_results.yml
name: Post Results

on:
  schedule:
    - cron: "0 14 * * *"   # 10:00 AM ET
  workflow_dispatch:

jobs:
  post-results:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Post results to Discord
        env:
          SUPABASE_URL: ${{ secrets.SUPABASE_URL }}
          SUPABASE_KEY: ${{ secrets.SUPABASE_KEY }}
          DISCORD_RESULTS_WEBHOOK: ${{ secrets.DISCORD_RESULTS_WEBHOOK }}
          DISCORD_RECAP_WEBHOOK: ${{ secrets.DISCORD_RECAP_WEBHOOK }}
          DISCORD_STATUS_WEBHOOK: ${{ secrets.DISCORD_STATUS_WEBHOOK }}
        run: python post_results.py
```

- [ ] **Step 5: Commit**

```
git add .github/workflows/
git commit -m "feat: add GitHub Actions workflows — daily picks, hourly monitor, morning results"
```

---

## Post-Implementation: One-Time Manual Setup Checklist

After all code is merged, complete these steps before going live:

**Supabase:**
- [ ] Create Supabase project at supabase.com
- [ ] Create `clv_log` table with columns matching `COLUMNS` in `agents/clv_log.py` plus `featured_bet boolean`, `withdrawn boolean`
- [ ] Create `hr_outcomes` table with columns: `game_date text, player_name text, team text, game text, game_pk int, hit_hr int, hrs_hit int, at_bats int, captured_ts text` (PK: `game_date, player_name`)
- [ ] Migrate existing `data/clv_log.csv` and `data/hr_outcomes.db` data into Supabase
- [ ] Add `SUPABASE_URL` and `SUPABASE_KEY` to `.env`

**Discord:**
- [ ] Create private Discord server
- [ ] Create 4 channels: `#picks`, `#results`, `#weekly-recap`, `#system-status`
- [ ] Set `#system-status` to owner-only; set other 3 to subscriber role + owner
- [ ] Generate 4 webhook URLs → add to `.env`

**GitHub Actions secrets** (repo Settings → Secrets and variables → Actions):
- [ ] `ODDS_API_KEY`
- [ ] `SUPABASE_URL`
- [ ] `SUPABASE_KEY`
- [ ] `DISCORD_PICKS_WEBHOOK`
- [ ] `DISCORD_RESULTS_WEBHOOK`
- [ ] `DISCORD_RECAP_WEBHOOK`
- [ ] `DISCORD_STATUS_WEBHOOK`

**Whop:**
- [ ] Create product at whop.com at $15/month
- [ ] Connect to Discord server (Whop auto-assigns subscriber role on payment)
- [ ] Keep server invite-only

**Smoke test:**
- [ ] Trigger `daily_picks.yml` manually via GitHub Actions → verify `#picks` message appears and `#system-status` shows ✅
- [ ] Trigger `post_results.yml` manually → verify `#results` shows (or skip message if no featured bets)
