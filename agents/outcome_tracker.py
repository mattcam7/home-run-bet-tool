"""agents/outcome_tracker.py — Tracks actual HR outcomes vs CLV log picks.

Queries the MLB Stats API box scores for completed games and records whether
each logged pick resulted in the player hitting a home run.

Database: data/hr_outcomes.db (SQLite)

Schema — outcomes table:
    game_date       TEXT  (YYYY-MM-DD)
    player_name     TEXT  (normalized title-case)
    team            TEXT
    game            TEXT  (e.g. "Texas Rangers @ New York Yankees")
    game_pk         INTEGER  (MLB internal game ID)
    hit_hr          INTEGER  NULL=didn't play / no AB, 0=played no HR, 1=hit HR
    hrs_hit         INTEGER  (0, 1, 2 ...)
    at_bats         INTEGER
    captured_ts     TEXT  (ISO timestamp)

Usage:
    python -m agents.outcome_tracker              # update yesterday
    python -m agents.outcome_tracker --backfill   # fill all CLV log dates
    python -m agents.outcome_tracker --date 2026-05-22
"""
from __future__ import annotations

import sqlite3
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import requests

CLV_LOG_PATH = Path("data/clv_log.csv")
DB_PATH = Path("data/hr_outcomes.db")
MLB_API = "https://statsapi.mlb.com/api/v1"
REQUEST_DELAY = 0.25  # seconds between MLB API calls — be polite


def _norm(name: str) -> str:
    return str(name).strip().title()


# ---------------------------------------------------------------------------
# DB bootstrap
# ---------------------------------------------------------------------------

def _get_conn(db_path: Path = DB_PATH) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS outcomes (
            game_date    TEXT NOT NULL,
            player_name  TEXT NOT NULL,
            team         TEXT,
            game         TEXT,
            game_pk      INTEGER,
            hit_hr       INTEGER,
            hrs_hit      INTEGER DEFAULT 0,
            at_bats      INTEGER DEFAULT 0,
            captured_ts  TEXT,
            PRIMARY KEY (game_date, player_name)
        )
    """)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# MLB Stats API helpers
# ---------------------------------------------------------------------------

def _get(path: str, params: dict | None = None) -> dict:
    resp = requests.get(f"{MLB_API}{path}", params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def fetch_completed_game_pks(date_str: str) -> list[int]:
    """Return gamePk list for regular-season games that are Final on date_str."""
    data = _get("/schedule", {"sportId": 1, "date": date_str, "gameType": "R"})
    pks = []
    for day in data.get("dates", []):
        for game in day.get("games", []):
            state = game.get("status", {}).get("codedGameState", "")
            if state == "F":  # Final
                pks.append(game["gamePk"])
    return pks


def fetch_batter_outcomes(game_pk: int) -> dict[str, dict]:
    """Return {normalized_player_name: {hrs_hit, at_bats}} for all batters."""
    data = _get(f"/game/{game_pk}/boxscore")
    result: dict[str, dict] = {}
    for side in ("home", "away"):
        team = data.get("teams", {}).get(side, {})
        for pid, pdata in team.get("players", {}).items():
            name = _norm(pdata.get("person", {}).get("fullName", ""))
            if not name:
                continue
            batting = pdata.get("stats", {}).get("batting", {})
            ab = int(batting.get("atBats", 0))
            hr = int(batting.get("homeRuns", 0))
            result[name] = {"hrs_hit": hr, "at_bats": ab}
    return result


def get_all_hr_hitters(date_str: str) -> dict[str, dict]:
    """Return all batter outcomes across all completed games on date_str."""
    pks = fetch_completed_game_pks(date_str)
    merged: dict[str, dict] = {}
    for pk in pks:
        time.sleep(REQUEST_DELAY)
        try:
            outcomes = fetch_batter_outcomes(pk)
            for name, stats in outcomes.items():
                # If player appears in multiple games (DH), accumulate
                if name in merged:
                    merged[name]["hrs_hit"] += stats["hrs_hit"]
                    merged[name]["at_bats"] += stats["at_bats"]
                    merged[name]["game_pks"].append(pk)
                else:
                    merged[name] = {**stats, "game_pks": [pk]}
        except Exception as e:
            print(f"  [outcome_tracker] Warning: gamePk {pk} fetch failed — {e}")
    return merged


# ---------------------------------------------------------------------------
# Core update logic
# ---------------------------------------------------------------------------

def update_for_date(
    date_str: str,
    db_path: Path = DB_PATH,
    clv_log_path: Path = CLV_LOG_PATH,
    verbose: bool = True,
) -> dict:
    """Fetch MLB outcomes for date_str and write results for CLV log picks.

    Returns summary dict with keys: date, n_picks, n_matched, n_hr_hits, n_no_hr, n_no_ab.
    """
    import os
    import pandas as pd

    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_key:
        try:
            from agents.supabase_client import fetch_clv_log
            clv = fetch_clv_log(game_date=date_str)
        except Exception:
            clv = pd.read_csv(clv_log_path) if Path(clv_log_path).exists() else pd.DataFrame()
    elif Path(clv_log_path).exists():
        clv = pd.read_csv(clv_log_path)
    else:
        clv = pd.DataFrame()
    if clv.empty:
        picks = pd.DataFrame(columns=["player_name", "team", "game"])
    else:
        picks = clv[clv["game_date"] == date_str][["player_name", "team", "game"]].copy()
    picks = picks.drop_duplicates("player_name")

    if picks.empty:
        if verbose:
            print(f"  {date_str}: no picks in CLV log — skipped")
        return {"date": date_str, "n_picks": 0, "n_matched": 0, "n_hr_hits": 0, "n_no_hr": 0, "n_no_ab": 0}

    if verbose:
        print(f"  {date_str}: fetching MLB box scores for {len(picks)} picks...")

    all_outcomes = get_all_hr_hitters(date_str)
    now_ts = datetime.now(timezone.utc).isoformat()

    conn = _get_conn(db_path)
    n_matched = n_hr = n_no_hr = n_no_ab = 0

    for _, row in picks.iterrows():
        pname = _norm(str(row["player_name"]))
        team = str(row.get("team", ""))
        game = str(row.get("game", ""))

        if pname in all_outcomes:
            stats = all_outcomes[pname]
            hit_hr = 1 if stats["hrs_hit"] > 0 else 0
            hrs_hit = stats["hrs_hit"]
            at_bats = stats["at_bats"]
            game_pk = stats["game_pks"][0] if stats.get("game_pks") else None
            n_matched += 1
            if hit_hr:
                n_hr += 1
            else:
                n_no_hr += 1
        else:
            # Player not found in any box score → didn't play / scratched
            hit_hr = None
            hrs_hit = 0
            at_bats = 0
            game_pk = None
            n_no_ab += 1

        conn.execute("""
            INSERT INTO outcomes
                (game_date, player_name, team, game, game_pk, hit_hr, hrs_hit, at_bats, captured_ts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(game_date, player_name) DO UPDATE SET
                hit_hr=excluded.hit_hr,
                hrs_hit=excluded.hrs_hit,
                at_bats=excluded.at_bats,
                game_pk=excluded.game_pk,
                captured_ts=excluded.captured_ts
        """, (date_str, pname, team, game, game_pk, hit_hr, hrs_hit, at_bats, now_ts))

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

    conn.commit()
    conn.close()

    if verbose:
        print(f"  {date_str}: {n_matched}/{len(picks)} matched -> {n_hr} HR hits, {n_no_hr} no HR, {n_no_ab} no AB")

    return {
        "date": date_str,
        "n_picks": len(picks),
        "n_matched": n_matched,
        "n_hr_hits": n_hr,
        "n_no_hr": n_no_hr,
        "n_no_ab": n_no_ab,
    }


def backfill(
    db_path: Path = DB_PATH,
    clv_log_path: Path = CLV_LOG_PATH,
    skip_today: bool = True,
) -> list[dict]:
    """Update outcomes for every date present in the CLV log."""
    import pandas as pd

    clv = pd.read_csv(clv_log_path)
    dates = sorted(clv["game_date"].dropna().unique())
    today_str = date.today().isoformat()

    results = []
    for d in dates:
        if skip_today and d == today_str:
            print(f"  {d}: skipping today (games may not be complete)")
            continue
        results.append(update_for_date(d, db_path=db_path, clv_log_path=clv_log_path))

    return results


# ---------------------------------------------------------------------------
# Query helpers for reports
# ---------------------------------------------------------------------------

def load_outcomes(db_path: Path = DB_PATH) -> "pd.DataFrame":
    """Return all outcomes as a DataFrame. Returns empty DF if DB not found."""
    import pandas as pd

    if not db_path.exists():
        return pd.DataFrame(columns=[
            "game_date", "player_name", "team", "game",
            "game_pk", "hit_hr", "hrs_hit", "at_bats", "captured_ts",
        ])
    conn = _get_conn(db_path)
    df = pd.read_sql("SELECT * FROM outcomes", conn)
    conn.close()
    return df


def compute_roi_metrics(
    clv_log_path: Path = CLV_LOG_PATH,
    db_path: Path = DB_PATH,
    featured_only: bool = False,
) -> dict:
    """Join CLV log with outcomes and compute ROI, hit rate, and CLV correlation.

    Only rows where hit_hr IS NOT NULL (player actually took the field) are
    included in win/loss calculations — scratched players are excluded.

    featured_only: when True, restrict CLV log to rows where featured_bet is True.
    """
    import os
    import pandas as pd

    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_key:
        try:
            from agents.supabase_client import fetch_clv_log, fetch_outcomes as _fetch_outcomes
            clv = fetch_clv_log()
            outcomes = _fetch_outcomes()
        except Exception:
            clv = pd.read_csv(clv_log_path) if Path(clv_log_path).exists() else pd.DataFrame()
            outcomes = load_outcomes(db_path)
    else:
        clv = pd.read_csv(clv_log_path) if Path(clv_log_path).exists() else pd.DataFrame()
        outcomes = load_outcomes(db_path)

    if featured_only and "featured_bet" in clv.columns:
        clv = clv[clv["featured_bet"].astype(str).str.lower() == "true"].copy()

    if outcomes.empty:
        return {"has_outcomes": False, "n_with_outcome": 0}

    clv["player_name"] = clv["player_name"].apply(_norm)
    outcomes["player_name"] = outcomes["player_name"].apply(_norm)

    # Exclude contaminated rows (over-only anchor produced fabricated EV/Kelly)
    if "anchor_quality" in clv.columns:
        n_contaminated = int((clv["anchor_quality"] == "pinnacle_over_only").sum())
        if n_contaminated:
            clv = clv[clv["anchor_quality"] != "pinnacle_over_only"].copy()

    merged = clv.merge(
        outcomes[["game_date", "player_name", "hit_hr", "hrs_hit", "at_bats"]],
        on=["game_date", "player_name"],
        how="left",
    )

    # Only rows with a definitive outcome (player took the field)
    settled = merged[merged["hit_hr"].notna()].copy()
    settled["hit_hr"] = settled["hit_hr"].astype(int)
    settled["best_retail_decimal"] = pd.to_numeric(settled["best_retail_decimal"], errors="coerce")
    settled["kelly_units"] = pd.to_numeric(settled["kelly_units"], errors="coerce").fillna(0)
    settled["stake_usd"] = pd.to_numeric(settled["stake_usd"], errors="coerce").fillna(0)
    settled["ev_pct"] = pd.to_numeric(settled["ev_pct"], errors="coerce")

    n_total = len(settled)
    n_hr = int(settled["hit_hr"].sum())
    hit_rate = n_hr / n_total if n_total else 0.0

    # P&L: win → stake × (decimal - 1), loss → -stake
    settled["pnl"] = settled.apply(
        lambda r: r["stake_usd"] * (r["best_retail_decimal"] - 1)
        if r["hit_hr"] == 1 else -r["stake_usd"],
        axis=1,
    )
    total_staked = settled["stake_usd"].sum()
    total_pnl = settled["pnl"].sum()
    roi = total_pnl / total_staked if total_staked > 0 else None

    # Split by EV tier
    pos_ev = settled[settled["ev_pct"] > 0]
    neg_ev = settled[settled["ev_pct"] <= 0]

    def _seg(df: pd.DataFrame) -> dict:
        if df.empty:
            return {"n": 0, "n_hr": 0, "hit_rate": None, "roi": None, "total_pnl": 0}
        stake = df["stake_usd"].sum()
        pnl = df["pnl"].sum()
        return {
            "n": len(df),
            "n_hr": int(df["hit_hr"].sum()),
            "hit_rate": float(df["hit_hr"].mean()),
            "roi": float(pnl / stake) if stake > 0 else None,
            "total_pnl": float(pnl),
        }

    # CLV correlation: among settled plays that also have CLV captured
    has_clv = settled[settled["clv_pct"].notna()].copy() if "clv_pct" in settled.columns else pd.DataFrame()
    clv_corr = None
    if len(has_clv) >= 10:
        has_clv["clv_pct"] = pd.to_numeric(has_clv["clv_pct"], errors="coerce")
        clv_corr = float(has_clv["clv_pct"].corr(has_clv["hit_hr"].astype(float)))

    return {
        "has_outcomes": True,
        "n_with_outcome": n_total,
        "n_scratched": int((merged["hit_hr"].isna()).sum()),
        "n_hr_hits": n_hr,
        "hit_rate": hit_rate,
        "total_staked": float(total_staked),
        "total_pnl": float(total_pnl),
        "roi": roi,
        "positive_ev": _seg(pos_ev),
        "negative_ev": _seg(neg_ev),
        "clv_outcome_correlation": clv_corr,
        # By date
        "by_date": (
            settled.groupby("game_date")
            .apply(lambda g: {
                "n": len(g),
                "n_hr": int(g["hit_hr"].sum()),
                "pnl": float(g["pnl"].sum()),
            })
            .to_dict()
        ),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Track HR outcomes against CLV log")
    parser.add_argument("--backfill", action="store_true", help="Fill all CLV log dates")
    parser.add_argument("--date", help="Fill a specific date (YYYY-MM-DD)")
    parser.add_argument("--yesterday", action="store_true", help="Fill yesterday (default)")
    parser.add_argument("--report", action="store_true", help="Print ROI summary after update")
    args = parser.parse_args()

    if args.backfill:
        print("Backfilling all CLV log dates...")
        results = backfill()
        total_hr = sum(r["n_hr_hits"] for r in results)
        total_picks = sum(r["n_picks"] for r in results)
        print(f"\nBackfill complete: {total_hr} HR hits across {total_picks} settled picks")
    elif args.date:
        update_for_date(args.date)
    else:
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        update_for_date(yesterday)

    if args.report or args.backfill:
        print("\n--- ROI Summary ---")
        metrics = compute_roi_metrics()
        if not metrics["has_outcomes"]:
            print("No outcomes recorded yet.")
            return
        m = metrics
        print(f"Settled picks : {m['n_with_outcome']}  (scratched: {m['n_scratched']})")
        print(f"HR hit rate   : {m['hit_rate']*100:.1f}%  ({m['n_hr_hits']} hits)")
        print(f"Total staked  : ${m['total_staked']:,.0f}")
        print(f"Total P&L     : ${m['total_pnl']:+,.2f}")
        if m["roi"] is not None:
            print(f"ROI           : {m['roi']*100:+.2f}%")
        pev = m["positive_ev"]
        print(f"\n+EV picks     : n={pev['n']}  hit%={pev['hit_rate']*100:.1f}%  ROI={pev['roi']*100:+.2f}%" if pev["n"] else "\n+EV picks     : none")
        if m["clv_outcome_correlation"] is not None:
            print(f"CLV<->Hit corr : {m['clv_outcome_correlation']:+.3f}")


if __name__ == "__main__":
    main()
