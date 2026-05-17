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


def test_pinnacle_prob_is_devigged():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    # Over +380 -> dec 4.8 ; Under -550 -> dec 1.181818
    # No-vig prob = over_imp / (over_imp + under_imp)
    over_imp = 1 / 4.8
    under_imp = 1 / (100 / 550 + 1)
    expected = over_imp / (over_imp + under_imp)
    assert abs(judge["pinnacle_prob"] - expected) < 1e-6
    # Sanity: de-vigged prob is strictly below the raw vig-inclusive prob
    assert judge["pinnacle_prob"] < over_imp


def test_pinnacle_odds_column_still_holds_over_price():
    df = extract_pinnacle_odds(FIXTURE_PAYLOAD, FIXTURE_NOW)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    assert judge["pinnacle_odds"] == 380


def test_devig_falls_back_when_under_missing():
    payload = [{
        "id": "g", "home_team": "Boston Red Sox", "away_team": "New York Yankees",
        "commence_time": "2026-05-15T23:05:00Z",
        "bookmakers": [{
            "key": "pinnacle", "title": "Pinnacle",
            "markets": [{"key": "batter_home_runs", "outcomes": [
                {"name": "Over", "description": "Solo Over", "price": 400, "point": 0.5},
            ]}],
        }],
    }]
    df = extract_pinnacle_odds(payload, FIXTURE_NOW)
    row = df[df["player_name"] == "Solo Over"].iloc[0]
    assert abs(row["pinnacle_prob"] - (1 / (400 / 100 + 1))) < 1e-6
