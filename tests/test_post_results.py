# tests/test_post_results.py
from datetime import date, timedelta


def test_calls_update_for_date_with_yesterday(monkeypatch):
    calls = {}
    monkeypatch.setattr("agents.outcome_tracker.update_for_date",
                        lambda d, **kw: calls.update({"date": d}))
    monkeypatch.setattr("agents.discord_bot.post_results", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_weekly_recap", lambda **kw: None)

    import post_results
    post_results.main()

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    assert calls.get("date") == yesterday


def test_calls_post_weekly_recap_on_sunday(monkeypatch):
    recap_calls = []
    monkeypatch.setattr("agents.outcome_tracker.update_for_date", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_results", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_weekly_recap",
                        lambda **kw: recap_calls.append(True))

    import post_results

    # FakeDate inherits from date so timedelta subtraction works correctly
    class FakeDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 7)  # a Sunday (weekday == 6)

    monkeypatch.setattr(post_results, "date", FakeDate)

    post_results.main()
    assert len(recap_calls) == 1


def test_does_not_call_recap_on_weekday(monkeypatch):
    recap_calls = []
    monkeypatch.setattr("agents.outcome_tracker.update_for_date", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_results", lambda d, **kw: None)
    monkeypatch.setattr("agents.discord_bot.post_weekly_recap",
                        lambda **kw: recap_calls.append(True))

    import post_results

    # 2026-06-03 is a Tuesday — no recap
    class FakeDate(date):
        @classmethod
        def today(cls):
            return cls(2026, 6, 3)  # Tuesday (weekday == 1)

    monkeypatch.setattr(post_results, "date", FakeDate)

    post_results.main()
    assert len(recap_calls) == 0
