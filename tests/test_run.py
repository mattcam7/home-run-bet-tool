from datetime import datetime, timezone
from unittest.mock import MagicMock
from tests.conftest import FIXTURE_PAYLOAD, FIXTURE_NOW
from run import fetch_odds, main


def test_fetch_odds_fetches_event_list_then_props(monkeypatch):
    mock_h2h = MagicMock()
    mock_h2h.raise_for_status = lambda: None
    mock_h2h.json.return_value = [
        {"id": "ev1", "commence_time": "2026-05-15T23:00:00Z", "away_team": "NYY", "home_team": "BOS"},
        {"id": "ev2", "commence_time": "2026-05-15T17:00:00Z", "away_team": "LAD", "home_team": "CHC"},  # already started
    ]

    mock_props = MagicMock()
    mock_props.raise_for_status = lambda: None
    mock_props.json.return_value = {"id": "ev1", "bookmakers": []}

    calls = []

    def fake_get(url, params, timeout):
        calls.append((url, params))
        if "events" in url:
            return mock_props
        return mock_h2h

    monkeypatch.setattr("requests.get", fake_get)
    now = datetime(2026, 5, 15, 20, 0, 0, tzinfo=timezone.utc)
    result = fetch_odds("test_key", now)

    # First call fetches event list
    assert "baseball_mlb/odds" in calls[0][0]
    assert calls[0][1]["apiKey"] == "test_key"
    # Second call fetches props for the ONE unplayed event (ev2 is skipped)
    assert len(calls) == 2
    assert "events/ev1/odds" in calls[1][0]
    assert calls[1][1]["markets"] == "batter_home_runs"
    assert calls[1][1]["regions"] == "us,eu"
    assert result == [{"id": "ev1", "bookmakers": []}]


def test_main_runs_full_pipeline(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test_key")
    monkeypatch.setattr("run.fetch_odds", lambda key, now: FIXTURE_PAYLOAD)
    monkeypatch.setattr("run.generate_dashboard", lambda df, **kwargs: None)
    main()  # should not raise
