import pandas as pd
from datetime import datetime
from agents.utils import american_to_decimal

RETAIL_BOOKS = {"draftkings", "betmgm", "fanduel", "williamhill_us", "fanatics"}

def extract_retail_odds(raw_payload: list, now: datetime) -> pd.DataFrame:
    rows = []
    for event in raw_payload:
        commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if commence_time <= now:
            continue
        game = f"{event['away_team']} @ {event['home_team']}"
        for bookmaker in event["bookmakers"]:
            if bookmaker["key"] not in RETAIL_BOOKS:
                continue
            for market in bookmaker["markets"]:
                if market["key"] != "batter_home_runs":
                    continue
                for outcome in market["outcomes"]:
                    if outcome.get("name") != "Over" or outcome.get("point") != 0.5:
                        continue
                    american_odds = outcome["price"]
                    rows.append({
                        "player_name": outcome["description"].strip().title(),
                        "game": game,
                        "commence_time": commence_time,
                        "bookmaker": bookmaker["title"],
                        "american_odds": american_odds,
                        "implied_prob": 1 / american_to_decimal(american_odds),
                    })
    return pd.DataFrame(rows)
