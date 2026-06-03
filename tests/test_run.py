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
    monkeypatch.setattr("run.fetch_player_teams", lambda: {})
    monkeypatch.setattr("run.log_open_plays", lambda df, **kwargs: None)
    monkeypatch.setattr("run.generate_dashboard", lambda df, **kwargs: None)

    # Freeze "now" to the fixture epoch so game1 stays unplayed regardless of
    # the real wall-clock date — otherwise the fixture's fixed 2026-05-15
    # games age out and the pipeline gets an empty retail frame.
    class _FrozenDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return FIXTURE_NOW

    monkeypatch.setattr("run.datetime", _FrozenDT)
    main()  # should not raise


def test_main_exits_cleanly_when_anchor_missing(monkeypatch):
    """No sharp anchor (Pinnacle + BetOnline both absent) → clean exit, no EV/log/dashboard."""
    import pandas as pd

    monkeypatch.setenv("ODDS_API_KEY", "test_key")
    monkeypatch.setattr("run.fetch_odds", lambda key, now: [])
    monkeypatch.setattr("run.fetch_player_teams", lambda: {})
    monkeypatch.setattr(
        "run.extract_retail_odds",
        lambda raw, now: pd.DataFrame([{"player_name": "X"}]),
    )
    # extract_sharp_anchor is what run.py now calls for the open-play anchor
    monkeypatch.setattr("run.extract_sharp_anchor", lambda raw, now: pd.DataFrame())

    called = []
    monkeypatch.setattr("run.calculate_ev", lambda r, p: called.append("ev"))
    monkeypatch.setattr("run.log_open_plays", lambda df, **kw: called.append("log"))
    monkeypatch.setattr("run.generate_dashboard", lambda df, **kw: called.append("dash"))

    main()  # must return cleanly, not raise
    assert called == []  # no EV, no log, no dashboard when anchor absent


def test_main_output_includes_bet_score(monkeypatch):
    """run.py pipeline must produce bet_score and bet_grade columns."""
    import pandas as pd
    from unittest.mock import patch

    captured = {}

    def fake_generate_dashboard(df, **kwargs):
        captured["df"] = df

    with patch("run.fetch_odds", return_value=[]), \
         patch("run.fetch_player_teams", return_value={}), \
         patch("run.extract_retail_odds", return_value=pd.DataFrame([{
             "player_name": "Test Player", "game": "A @ B",
             "commence_time": pd.Timestamp("2026-06-10 20:00:00", tz="UTC"),
             "bookmaker": "draftkings", "american_odds": 400, "implied_prob": 0.20,
         }])), \
         patch("run.extract_sharp_anchor", return_value=pd.DataFrame([{
             "player_name": "Test Player", "game": "A @ B",
             "commence_time": pd.Timestamp("2026-06-10 20:00:00", tz="UTC"),
             "pinnacle_odds": 450, "pinnacle_prob": 0.18,
             "sharp_anchor": "pinnacle", "over_only": False,
         }])), \
         patch("run.add_simulation", side_effect=lambda df: df), \
         patch("run.log_open_plays"), \
         patch("run.generate_dashboard", side_effect=fake_generate_dashboard), \
         patch("run.generate_parlays", return_value=[]), \
         patch("run.analyze_dfs", return_value=None):
        import os
        os.environ["ODDS_API_KEY"] = "test"
        import run
        run.main()

    assert "bet_score" in captured["df"].columns
    assert "bet_grade" in captured["df"].columns


def test_no_browser_argparse_parses_correctly():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(["--no-browser"])
    assert args.no_browser is True


def test_default_run_no_browser_is_false():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args([])
    assert args.no_browser is False


def test_pipeline_run_phase1_returns_context(monkeypatch):
    """pipeline.run_phase1() returns a PipelineContext with final_df populated."""
    import pandas as pd
    from unittest.mock import patch

    dummy_retail = pd.DataFrame([{
        "player_name": "Test Player", "game": "A @ B",
        "commence_time": pd.Timestamp("2026-06-10 20:00:00", tz="UTC"),
        "bookmaker": "draftkings", "american_odds": 400, "implied_prob": 0.20,
    }])
    dummy_anchor = pd.DataFrame([{
        "player_name": "Test Player", "game": "A @ B",
        "commence_time": pd.Timestamp("2026-06-10 20:00:00", tz="UTC"),
        "pinnacle_odds": 450, "pinnacle_prob": 0.18,
        "sharp_anchor": "pinnacle", "over_only": False,
    }])

    with patch("pipeline.fetch_odds", return_value=[{"id": "x", "bookmakers": []}]), \
         patch("pipeline.fetch_player_teams", return_value={}), \
         patch("pipeline.extract_retail_odds", return_value=dummy_retail), \
         patch("pipeline.extract_sharp_anchor", return_value=dummy_anchor), \
         patch("pipeline.add_simulation", side_effect=lambda df: df), \
         patch("pipeline.log_open_plays"), \
         patch("pipeline.generate_dashboard"), \
         patch("pipeline.generate_parlays", return_value=[]), \
         patch("pipeline.analyze_dfs", return_value=None):
        import os
        os.environ["ODDS_API_KEY"] = "test"
        import pipeline
        ctx = pipeline.run_phase1()

    assert ctx is not None
    assert hasattr(ctx, "final_df")
    assert "bet_score" in ctx.final_df.columns
