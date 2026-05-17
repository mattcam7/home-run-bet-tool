# tests/test_generator.py
import os
import pandas as pd
import pytest
from datetime import datetime, timezone
from dashboard.generator import generate_dashboard

COMMENCE = datetime(2026, 5, 15, 23, 5, tzinfo=timezone.utc)
GAME = "New York Yankees @ Boston Red Sox"

@pytest.fixture
def sample_df():
    return pd.DataFrame([
        {
            "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
            "pinnacle_odds": 380, "pinnacle_prob": 1/4.8,
            "DraftKings": 450, "FanDuel": 420,
            "best_retail_odds": 450, "best_retail_decimal": 5.5,
            "ev_pct": (1/4.8 * 5.5) - 1,
            "composite_score": ((1/4.8 * 5.5) - 1) * (1/4.8),
            "composite_z": 1.0,
            "kelly_units": 1.0, "stake_usd": 25.0,
        },
        {
            "player_name": "Rafael Devers", "game": GAME, "commence_time": COMMENCE,
            "pinnacle_odds": 520, "pinnacle_prob": 1/6.2,
            "DraftKings": 600, "FanDuel": 580,
            "best_retail_odds": 600, "best_retail_decimal": 7.0,
            "ev_pct": (1/6.2 * 7.0) - 1,
            "composite_score": ((1/6.2 * 7.0) - 1) * (1/6.2),
            "composite_z": -1.0,
            "kelly_units": 0.0, "stake_usd": 0.0,
        },
    ])

def test_creates_html_file(tmp_path, sample_df):
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    assert os.path.exists(output)

def test_html_contains_player_names(tmp_path, sample_df):
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Aaron Judge" in content
    assert "Rafael Devers" in content

def test_html_contains_parlay_section(tmp_path, sample_df):
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "parlay" in content.lower()

def test_opens_browser(tmp_path, sample_df, monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url: opened.append(url))
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=True)
    assert len(opened) == 1
