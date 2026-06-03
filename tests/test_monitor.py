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
