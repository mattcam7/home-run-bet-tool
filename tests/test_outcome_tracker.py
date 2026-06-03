import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch
import sqlite3


def _make_clv_csv(tmp_path, rows):
    df = pd.DataFrame(rows)
    p = tmp_path / "clv_log.csv"
    df.to_csv(p, index=False)
    return p


def _make_outcomes_db(tmp_path, rows):
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
