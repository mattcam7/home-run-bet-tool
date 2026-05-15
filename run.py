import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from agents.ev_calculator import calculate_ev
from agents.odds_scraper import extract_retail_odds
from agents.pinnacle_scraper import extract_pinnacle_odds
from dashboard.generator import generate_dashboard


def fetch_odds(api_key: str, now: datetime) -> list:
    # Step 1: get event list (h2h is cheapest market, gives us IDs + commence times)
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json()

    # Step 2: per-event batter_home_runs fetch (player props not available on bulk endpoint)
    result = []
    for event in events:
        commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if commence_time <= now:
            continue
        props_resp = requests.get(
            f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event['id']}/odds",
            params={"apiKey": api_key, "regions": "us,eu", "markets": "batter_home_runs", "oddsFormat": "american"},
            timeout=15,
        )
        props_resp.raise_for_status()
        result.append(props_resp.json())
    return result


def main() -> None:
    load_dotenv()
    api_key = os.environ["ODDS_API_KEY"]
    now = datetime.now(timezone.utc)

    print("Fetching odds from OddsAPI...")
    raw = fetch_odds(api_key, now)

    print("Extracting retail odds...")
    retail_df = extract_retail_odds(raw, now)

    print("Extracting Pinnacle odds...")
    pinnacle_df = extract_pinnacle_odds(raw, now)

    print("Calculating EV...")
    final_df = calculate_ev(retail_df, pinnacle_df)

    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
    print(f"Found {n_players} players | {n_positive} +EV plays")

    generate_dashboard(final_df)
    print("Dashboard opened in browser.")


if __name__ == "__main__":
    main()
