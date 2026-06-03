import json
import os
import tempfile

import pandas as pd
import pytest

from agents.validation import (
    StepResult,
    append_quarantine,
    validate_clv_log,
    validate_ev_output,
    validate_outcomes,
    validate_raw_odds,
)


# ── validate_raw_odds ─────────────────────────────────────────────────────────

def test_validate_raw_odds_empty_list_returns_not_ok():
    result = validate_raw_odds([])
    assert result.ok is False

def test_validate_raw_odds_drops_event_missing_bookmakers():
    events = [{"id": "abc"}]  # no bookmakers key
    result = validate_raw_odds(events)
    assert len(result.clean) == 0
    assert len(result.quarantined) == 1
    assert result.quarantined[0]["reason"] == "missing_bookmakers"

def test_validate_raw_odds_passes_valid_event():
    events = [{"id": "abc", "bookmakers": [{"key": "draftkings"}]}]
    result = validate_raw_odds(events)
    assert result.ok is True
    assert len(result.clean) == 1
    assert result.quarantined == []


# ── validate_ev_output ────────────────────────────────────────────────────────

def _ev_row(**kwargs):
    base = {
        "player_name": "Test Player",
        "ev_pct": 0.05,
        "pinnacle_prob": 0.18,
        "kelly_units": 0.5,
        "stake_usd": 12.5,
        "best_retail_odds": 450,
        "anchor_quality": "pinnacle",
    }
    base.update(kwargs)
    return base

def test_validate_ev_output_quarantines_impossible_ev():
    df = pd.DataFrame([_ev_row(ev_pct=3.0)])  # 300% EV — impossible
    result = validate_ev_output(df)
    assert len(result.clean) == 0
    assert any(r["reason"] == "impossible_ev_over_200pct" for r in result.quarantined)

def test_validate_ev_output_passes_normal_row():
    df = pd.DataFrame([_ev_row()])
    result = validate_ev_output(df)
    assert len(result.clean) == 1
    assert result.quarantined == []

def test_validate_ev_output_warns_on_large_slate():
    rows = [_ev_row(player_name=f"Player {i}") for i in range(160)]
    df = pd.DataFrame(rows)
    result = validate_ev_output(df)
    assert any("large slate" in w.lower() for w in result.warnings)

def test_validate_ev_output_zeroes_over_only_kelly_if_violated():
    df = pd.DataFrame([_ev_row(anchor_quality="pinnacle_over_only",
                                over_only=True, kelly_units=1.0, stake_usd=25.0)])
    result = validate_ev_output(df)
    assert result.clean.iloc[0]["kelly_units"] == 0.0
    assert result.clean.iloc[0]["stake_usd"] == 0.0


# ── validate_clv_log ──────────────────────────────────────────────────────────

def test_validate_clv_log_quarantines_extreme_clv():
    df = pd.DataFrame([{
        "closing_pinnacle_prob": 0.18,
        "clv_pct": 0.75,  # 75% CLV — anomaly
        "player_name": "Anomaly Player",
        "game_date": "2026-05-22",
    }])
    result = validate_clv_log(df)
    assert any(r["reason"] == "clv_exceeds_50pct" for r in result.quarantined)

def test_validate_clv_log_passes_normal_clv():
    df = pd.DataFrame([{
        "closing_pinnacle_prob": 0.18,
        "clv_pct": 0.08,
        "player_name": "Normal Player",
        "game_date": "2026-06-01",
    }])
    result = validate_clv_log(df)
    assert result.quarantined == []


# ── validate_outcomes ─────────────────────────────────────────────────────────

def test_validate_outcomes_empty_returns_not_ok():
    result = validate_outcomes({}, "2026-06-01")
    assert result.ok is False

def test_validate_outcomes_warns_on_high_hr_rate():
    # 10 of 12 players hit HRs — 83% is anomalous
    outcomes = {f"Player {i}": {"hrs_hit": 1, "at_bats": 4} for i in range(10)}
    outcomes.update({f"No HR {i}": {"hrs_hit": 0, "at_bats": 4} for i in range(2)})
    result = validate_outcomes(outcomes, "2026-06-01")
    assert any("anomalous" in w.lower() for w in result.warnings)

def test_validate_outcomes_normal_data_passes():
    outcomes = {f"Player {i}": {"hrs_hit": 0, "at_bats": 4} for i in range(10)}
    outcomes["HR Guy"] = {"hrs_hit": 1, "at_bats": 4}
    result = validate_outcomes(outcomes, "2026-06-01")
    assert result.ok is True
    assert result.warnings == []


# ── append_quarantine ─────────────────────────────────────────────────────────

def test_append_quarantine_writes_jsonl():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        rows = [{"step": "test", "reason": "bad_data", "player": "X"}]
        append_quarantine(rows, path=path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 1
        parsed = json.loads(lines[0])
        assert parsed["reason"] == "bad_data"
        assert "ts" in parsed
    finally:
        os.unlink(path)

def test_validate_ev_output_handles_missing_ev_pct_column():
    df = pd.DataFrame([{"player_name": "X", "kelly_units": 0.5, "stake_usd": 12.5,
                         "best_retail_odds": 400, "anchor_quality": "pinnacle"}])
    result = validate_ev_output(df)  # must not crash
    assert result.ok is True

def test_validate_clv_log_handles_missing_closing_pinnacle_prob_column():
    df = pd.DataFrame([{"player_name": "X", "clv_pct": 0.05, "game_date": "2026-06-01"}])
    result = validate_clv_log(df)  # must not crash
    assert "closing_pinnacle_prob" in result.warnings[0]

def test_validate_clv_log_handles_missing_clv_pct_column():
    df = pd.DataFrame([{"player_name": "X", "closing_pinnacle_prob": 0.18, "game_date": "2026-06-01"}])
    result = validate_clv_log(df)  # must not crash
    assert "clv_pct" in result.warnings[0]

def test_validate_ev_output_empty_after_quarantine_sets_ok_false():
    df = pd.DataFrame([{"player_name": "X", "ev_pct": 3.0, "kelly_units": 0,
                         "stake_usd": 0, "best_retail_odds": 400, "anchor_quality": "pinnacle"}])
    result = validate_ev_output(df)
    assert result.ok is False  # all rows quarantined → ok=False


def test_append_quarantine_appends_not_overwrites():
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False, mode="w") as f:
        path = f.name
    try:
        append_quarantine([{"step": "a", "reason": "first"}], path=path)
        append_quarantine([{"step": "b", "reason": "second"}], path=path)
        with open(path) as f:
            lines = f.readlines()
        assert len(lines) == 2
    finally:
        os.unlink(path)
