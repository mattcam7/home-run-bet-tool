# tests/test_supabase_client.py
import pytest
from unittest.mock import MagicMock


def test_insert_clv_rows_calls_supabase_upsert(monkeypatch):
    mock_client = MagicMock()
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    sc.insert_clv_rows([{"game_date": "2026-06-03", "player_name": "Aaron Judge"}])
    mock_client.table.assert_called_with("clv_log")
    mock_client.table().upsert.assert_called_once()


def test_fetch_clv_log_returns_dataframe(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.data = [{"game_date": "2026-06-03", "player_name": "Aaron Judge", "ev_pct": 0.18}]
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.execute.return_value = mock_resp
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    df = sc.fetch_clv_log()
    assert len(df) == 1
    assert df.iloc[0]["player_name"] == "Aaron Judge"


def test_fetch_clv_log_returns_empty_dataframe_when_no_data(monkeypatch):
    mock_resp = MagicMock()
    mock_resp.data = []
    mock_client = MagicMock()
    mock_client.table.return_value.select.return_value.execute.return_value = mock_resp
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    df = sc.fetch_clv_log()
    assert df.empty


def test_upsert_outcome_calls_supabase(monkeypatch):
    mock_client = MagicMock()
    import agents.supabase_client as sc
    monkeypatch.setattr(sc, "_client", lambda: mock_client)
    sc.upsert_outcome("2026-06-03", "Aaron Judge", hit_hr=1, hrs_hit=1, at_bats=4)
    mock_client.table.assert_called_with("hr_outcomes")
    mock_client.table().upsert.assert_called_once()


def test_client_raises_on_missing_env(monkeypatch):
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_KEY", raising=False)
    import agents.supabase_client as sc
    with pytest.raises(EnvironmentError, match="SUPABASE_URL"):
        sc._client()
