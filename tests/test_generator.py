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


from datetime import datetime, timezone

GAME_SIM = "Texas Rangers @ New York Yankees"
COMMENCE_SIM = datetime(2026, 5, 15, 23, 5, tzinfo=timezone.utc)


@pytest.fixture
def sample_df_with_sim():
    return pd.DataFrame([
        {
            "player_name": "Aaron Judge", "team": "NYY", "game": GAME_SIM,
            "commence_time": COMMENCE_SIM,
            "pinnacle_odds": 380, "pinnacle_prob": 0.18,
            "DraftKings": 450, "FanDuel": 420,
            "best_retail_odds": 450, "best_retail_decimal": 5.5,
            "best_retail_book": "DraftKings",
            "sharp_anchor": "pinnacle",
            "ev_pct": 0.05, "composite_score": 0.009, "composite_z": 1.2,
            "kelly_units": 0.5, "stake_usd": 12.5,
            "sim_prob": 0.21, "sim_edge": 0.03, "convergence": "AGREE",
        },
        {
            "player_name": "Rafael Devers", "team": "BOS", "game": GAME_SIM,
            "commence_time": COMMENCE_SIM,
            "pinnacle_odds": 520, "pinnacle_prob": 0.14,
            "DraftKings": 600, "FanDuel": 580,
            "best_retail_odds": 600, "best_retail_decimal": 7.0,
            "best_retail_book": "DraftKings",
            "sharp_anchor": "pinnacle",
            "ev_pct": -0.02, "composite_score": -0.003, "composite_z": -0.8,
            "kelly_units": 0.0, "stake_usd": 0.0,
            "sim_prob": 0.10, "sim_edge": -0.04, "convergence": "DIVERGE",
        },
    ])


def test_sim_columns_not_treated_as_book_columns(tmp_path, sample_df_with_sim):
    """sim_prob, sim_edge, convergence must NOT appear in the BOOKS JS variable."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df_with_sim, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    # Extract the BOOKS= JS assignment and confirm sim columns are not in it
    books_section = content.split("const BOOKS=")[1].split(";")[0]
    assert "sim_prob" not in books_section
    assert "sim_edge" not in books_section
    assert "convergence" not in books_section


def test_sim_section_rendered_when_data_present(tmp_path, sample_df_with_sim):
    """Simulation Analysis section is in the HTML when sim columns present."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df_with_sim, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Simulation Analysis" in content
    assert "sim-section" in content


def test_sim_section_unavailable_message_when_no_sim_data(tmp_path, sample_df):
    """Shows unavailable message when sim columns absent."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Simulation Analysis" in content
    assert "unavailable" in content.lower()


def test_sim_data_injected_into_js(tmp_path, sample_df_with_sim):
    """SIM_DATA JS variable is populated with player records."""
    output = str(tmp_path / "test.html")
    generate_dashboard(sample_df_with_sim, output, open_browser=False)
    content = open(output, encoding="utf-8").read()
    assert "Aaron Judge" in content
    assert "SIM_DATA" in content
