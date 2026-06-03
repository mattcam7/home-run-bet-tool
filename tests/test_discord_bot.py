# tests/test_discord_bot.py
import pandas as pd
import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

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
    monkeypatch.setenv("DISCORD_RECAP_WEBHOOK", "http://fake/recap")
    monkeypatch.setenv("DISCORD_STATUS_WEBHOOK", "http://fake/status")
    monkeypatch.setattr("requests.post", lambda *a, **kw: (_ for _ in ()).throw(Exception("fail")))
    monkeypatch.setattr("agents.outcome_tracker.compute_roi_metrics", lambda **kw: {})

    from agents import discord_bot
    discord_bot.post_picks(_featured_df(), now=NOW)      # must not raise
    discord_bot.post_results("2026-06-02")               # must not raise
    discord_bot.post_alert("Judge", 320, 240, 0.18, 0.06, "movement")  # must not raise
    discord_bot.post_weekly_recap(now=NOW)               # must not raise
