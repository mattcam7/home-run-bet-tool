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
                    {"name": "Over", "description": "Aaron Judge", "price": 450, "point": 0.5},
                    {"name": "Under", "description": "Aaron Judge", "price": -700, "point": 0.5},
                    {"name": "Over", "description": "Rafael Devers", "price": 600, "point": 0.5},
                    {"name": "Under", "description": "Rafael Devers", "price": -1200, "point": 0.5},
                ]}],
            },
            {
                "key": "fanduel",
                "title": "FanDuel",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Over", "description": "Aaron Judge", "price": 420, "point": 0.5},
                    {"name": "Under", "description": "Aaron Judge", "price": -650, "point": 0.5},
                    {"name": "Over", "description": "Rafael Devers", "price": 580, "point": 0.5},
                    {"name": "Under", "description": "Rafael Devers", "price": -1100, "point": 0.5},
                ]}],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Over", "description": "Aaron Judge", "price": 380, "point": 0.5},
                    {"name": "Under", "description": "Aaron Judge", "price": -550, "point": 0.5},
                    {"name": "Over", "description": "Rafael Devers", "price": 520, "point": 0.5},
                    {"name": "Under", "description": "Rafael Devers", "price": -900, "point": 0.5},
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
                    {"name": "Over", "description": "Shohei Ohtani", "price": 350, "point": 0.5},
                    {"name": "Under", "description": "Shohei Ohtani", "price": -550, "point": 0.5},
                ]}],
            },
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Over", "description": "Shohei Ohtani", "price": 320, "point": 0.5},
                    {"name": "Under", "description": "Shohei Ohtani", "price": -500, "point": 0.5},
                ]}],
            },
        ],
    },
]
