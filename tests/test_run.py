from datetime import datetime, timezone
from unittest.mock import MagicMock
from tests.conftest import FIXTURE_PAYLOAD, FIXTURE_NOW
from run import fetch_event_odds, fetch_odds, main


def test_fetch_event_odds_merges_standard_and_alternate(monkeypatch):
    std_bk = {"key": "pinnacle", "title": "Pinnacle", "markets": [{"key": "batter_home_runs", "outcomes": []}]}
    alt_bk = {"key": "draftkings", "title": "DraftKings", "markets": [{"key": "batter_home_runs_alternate", "outcomes": []}]}

    mock_std = MagicMock()
    mock_std.raise_for_status = lambda: None
    mock_std.json.return_value = {"id": "ev1", "bookmakers": [std_bk]}

    mock_alt = MagicMock()
    mock_alt.raise_for_status = lambda: None
    mock_alt.json.return_value = {"id": "ev1", "bookmakers": [alt_bk]}

    calls = []
    def fake_get(url, params, timeout):
        calls.append(params.get("markets"))
        return mock_std if params.get("markets") == "batter_home_runs" else mock_alt

    monkeypatch.setattr("requests.get", fake_get)
    result = fetch_event_odds("test_key", "ev1")

    book_keys = {bk["key"] for bk in result["bookmakers"]}
    assert "pinnacle" in book_keys
    assert "draftkings" in book_keys
    # Alternate market key should be normalized
    dk = next(bk for bk in result["bookmakers"] if bk["key"] == "draftkings")
    assert dk["markets"][0]["key"] == "batter_home_runs"


def test_fetch_event_odds_deduplicates_books(monkeypatch):
    shared_bk = {"key": "espnbet", "title": "ESPN Bet", "markets": [{"key": "batter_home_runs", "outcomes": []}]}

    mock_std = MagicMock()
    mock_std.raise_for_status = lambda: None
    mock_std.json.return_value = {"id": "ev1", "bookmakers": [shared_bk]}

    mock_alt = MagicMock()
    mock_alt.raise_for_status = lambda: None
    mock_alt.json.return_value = {"id": "ev1", "bookmakers": [shared_bk]}

    monkeypatch.setattr("requests.get", lambda url, params, timeout: mock_std if params.get("markets") == "batter_home_runs" else mock_alt)
    result = fetch_event_odds("test_key", "ev1")
    assert len(result["bookmakers"]) == 1  # deduped, not doubled


def test_fetch_odds_skips_started_games(monkeypatch):
    mock_h2h = MagicMock()
    mock_h2h.raise_for_status = lambda: None
    mock_h2h.json.return_value = [
        {"id": "ev1", "commence_time": "2026-05-15T23:00:00Z", "away_team": "NYY", "home_team": "BOS"},
        {"id": "ev2", "commence_time": "2026-05-15T17:00:00Z", "away_team": "LAD", "home_team": "CHC"},
    ]

    mock_props = MagicMock()
    mock_props.raise_for_status = lambda: None
    mock_props.json.return_value = {"id": "ev1", "bookmakers": []}

    event_calls = []
    def fake_get(url, params, timeout):
        if "events" in url:
            event_calls.append(url)
            return mock_props
        return mock_h2h

    monkeypatch.setattr("requests.get", fake_get)
    now = datetime(2026, 5, 15, 20, 0, 0, tzinfo=timezone.utc)
    result = fetch_odds("test_key", now)

    # Only ev1 (unplayed) should trigger event-level calls; ev2 skipped
    assert len(result) == 1
    assert all("ev1" in url for url in event_calls)
    assert not any("ev2" in url for url in event_calls)


def test_main_runs_full_pipeline(monkeypatch):
    monkeypatch.setenv("ODDS_API_KEY", "test_key")
    monkeypatch.setattr("run.fetch_odds", lambda key, now: FIXTURE_PAYLOAD)
    monkeypatch.setattr("run.generate_dashboard", lambda df, **kwargs: None)
    main()  # should not raise
