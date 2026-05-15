from datetime import datetime, timezone

FIXTURE_NOW = datetime(2026, 5, 15, 20, 0, 0, tzinfo=timezone.utc)

FIXTURE_PAYLOAD = [
    {
        "id": "game1",
        "sport_key": "baseball_mlb",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": "2026-05-15T23:05:00Z",
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Aaron Judge", "price": 450},
                    {"name": "Rafael Devers", "price": 600},
                ]}],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Aaron Judge", "price": 420},
                    {"name": "Rafael Devers", "price": 580},
                ]}],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Aaron Judge", "price": 380},
                    {"name": "Rafael Devers", "price": 520},
                ]}],
            },
        ],
    },
    {
        "id": "game2",
        "sport_key": "baseball_mlb",
        "home_team": "Chicago Cubs",
        "away_team": "Los Angeles Dodgers",
        "commence_time": "2026-05-15T17:00:00Z",  # already started
        "bookmakers": [
            {
                "key": "draftkings",
                "title": "DraftKings",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Shohei Ohtani", "price": 350},
                ]}],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Shohei Ohtani", "price": 320},
                ]}],
            },
        ],
    },
]
