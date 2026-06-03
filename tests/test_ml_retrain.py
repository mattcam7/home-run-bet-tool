import json
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from agents.ml_retrain import (
    apply_correction,
    compute_correction_factors,
    load_correction_factors,
    retrain_if_ready,
)


def _make_outcome_db(path: Path, rows: list[dict]) -> None:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE outcomes (
            game_date TEXT, player_name TEXT, hit_hr INTEGER,
            hrs_hit INTEGER DEFAULT 0, at_bats INTEGER DEFAULT 0,
            captured_ts TEXT
        )
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO outcomes VALUES (?,?,?,?,?,?)",
            (r["game_date"], r["player_name"], r.get("hit_hr", 0),
             r.get("hrs_hit", 0), r.get("at_bats", 4), "2026-06-01T00:00:00+00:00"),
        )
    conn.commit()
    conn.close()


def _make_clv_log(path: Path, rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_retrain_if_ready_skips_below_threshold(tmp_path):
    db = tmp_path / "outcomes.db"
    _make_outcome_db(db, [
        {"game_date": "2026-05-30", "player_name": "Matt Olson",
         "hit_hr": 1, "hrs_hit": 1, "at_bats": 4}
    ])
    result = retrain_if_ready(db_path=db, n_threshold=50,
                              correction_path=tmp_path / "cf.json",
                              log_path=tmp_path / "log.json")
    assert result["ran"] is False
    assert "only 1 outcomes" in result["reason"]


def test_retrain_if_ready_runs_with_enough_outcomes(tmp_path):
    db = tmp_path / "outcomes.db"
    rows = []
    for day in range(11):
        for p in range(5):
            rows.append({
                "game_date": f"2026-05-{17+day:02d}",
                "player_name": f"Player {p}",
                "hit_hr": 1 if p == 0 else 0,
                "hrs_hit": 1 if p == 0 else 0,
                "at_bats": 4,
            })
    _make_outcome_db(db, rows)

    clv = tmp_path / "clv.csv"
    clv_rows = []
    for day in range(11):
        for p in range(5):
            clv_rows.append({
                "game_date": f"2026-05-{17+day:02d}",
                "player_name": f"Player {p}",
                "pinnacle_prob_devig": 0.15,
            })
    _make_clv_log(clv, clv_rows)

    result = retrain_if_ready(
        db_path=db,
        clv_log_path=clv,
        n_threshold=50,
        correction_path=tmp_path / "cf.json",
        log_path=tmp_path / "log.json",
    )
    assert result["ran"] is True
    assert result["n_players_updated"] > 0
    assert (tmp_path / "cf.json").exists()


def test_apply_correction_returns_base_when_no_factors(tmp_path):
    result = apply_correction("Matt Olson", 0.18, correction_path=tmp_path / "cf.json")
    assert result == pytest.approx(0.18)


def test_apply_correction_blends_factor(tmp_path):
    cf_path = tmp_path / "cf.json"
    cf_path.write_text(json.dumps({
        "Matt Olson": {"factor": 2.0, "n": 200, "actual_rate": 0.36, "predicted_rate": 0.18}
    }))
    corrected = apply_correction("Matt Olson", 0.18, correction_path=cf_path)
    # alpha = min(200/200, 0.40) = 0.40; corrected = 0.18 * (0.40*2.0 + 0.60) = 0.18 * 1.40 = 0.252
    assert corrected == pytest.approx(0.252, rel=0.01)


def test_apply_correction_is_capped_at_max_blend(tmp_path):
    cf_path = tmp_path / "cf.json"
    cf_path.write_text(json.dumps({
        "Big Swinger": {"factor": 5.0, "n": 10000, "actual_rate": 0.50, "predicted_rate": 0.10}
    }))
    base = 0.10
    corrected = apply_correction("Big Swinger", base, correction_path=cf_path)
    # alpha capped at 0.40; corrected = 0.10 * (0.40*5.0 + 0.60) = 0.10 * 2.60 = 0.26
    assert corrected == pytest.approx(0.26, rel=0.01)
