from tests.conftest import FIXTURE_PAYLOAD, FIXTURE_NOW
from agents.pinnacle_scraper import extract_pinnacle_odds


def test_returns_only_unplayed_pinnacle_players():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert len(df) == 2  # Aaron Judge + Rafael Devers from game1 only


def test_excludes_started_games():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert "Shohei Ohtani" not in df["player_name"].values


def test_returns_expected_columns():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    for col in ["player_name", "game", "commence_time", "pinnacle_odds", "pinnacle_prob"]:
        assert col in df.columns


def test_implied_prob_calculation():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    # +380 -> decimal 4.8 -> prob = 1/4.8
    assert abs(judge["pinnacle_prob"] - (1 / 4.8)) < 0.001
