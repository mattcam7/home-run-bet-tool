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
