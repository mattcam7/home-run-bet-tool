import copy
from tests.conftest import FIXTURE_PAYLOAD, FIXTURE_NOW
from agents.odds_scraper import extract_retail_odds

def test_excludes_pinnacle():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "Pinnacle" not in df["bookmaker"].values

def test_excludes_started_games():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "Shohei Ohtani" not in df["player_name"].values

def test_returns_expected_columns():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    for col in ["player_name", "game", "commence_time", "bookmaker", "american_odds", "implied_prob"]:
        assert col in df.columns

def test_normalizes_player_names():
    modified = copy.deepcopy(FIXTURE_PAYLOAD)
    modified[0]["bookmakers"][0]["markets"][0]["outcomes"][0]["name"] = "aaron judge"
    df = extract_retail_odds(modified, FIXTURE_NOW)
    assert "Aaron Judge" in df["player_name"].values

def test_implied_prob_positive_odds():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    row = df[(df["player_name"] == "Aaron Judge") & (df["bookmaker"] == "DraftKings")].iloc[0]
    assert abs(row["implied_prob"] - (1 / 5.5)) < 0.001

def test_game_format():
    df = extract_retail_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "New York Yankees @ Boston Red Sox" in df["game"].values
