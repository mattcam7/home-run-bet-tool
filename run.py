import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from agents.ev_calculator import calculate_ev
from agents.odds_scraper import extract_retail_odds
from agents.pinnacle_scraper import extract_pinnacle_odds
from dashboard.generator import generate_dashboard


def fetch_event_odds(api_key: str, event_id: str) -> dict:
    """Fetch batter_home_runs from both standard and alternate markets and merge.

    Sharp books (Pinnacle, BetOnline, Caesars) use 'batter_home_runs'.
    US retail books (DraftKings, FanDuel, BetMGM) use 'batter_home_runs_alternate'.
    Both are fetched and merged into one event object so the scrapers see all books.
    """
    std_resp = requests.get(
        f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds",
        params={
            "apiKey": api_key,
            "regions": "us,us2,eu",
            "markets": "batter_home_runs",
            "oddsFormat": "american",
        },
        timeout=15,
    )
    alt_resp = requests.get(
        f"https://api.the-odds-api.com/v4/sports/baseball_mlb/events/{event_id}/odds",
        params={
            "apiKey": api_key,
            "regions": "us,us2",
            "markets": "batter_home_runs_alternate",
            "oddsFormat": "american",
        },
        timeout=15,
    )
    std_resp.raise_for_status()
    alt_resp.raise_for_status()

    std_data = std_resp.json()
    alt_data = alt_resp.json()

    # Merge: standard books take priority; add alternate books not already present.
    # Normalize alternate market key so scrapers treat it identically.
    std_book_keys = {bk["key"] for bk in std_data.get("bookmakers", [])}
    merged = list(std_data.get("bookmakers", []))
    for bk in alt_data.get("bookmakers", []):
        if bk["key"] in std_book_keys:
            continue
        for market in bk["markets"]:
            if market["key"] == "batter_home_runs_alternate":
                market["key"] = "batter_home_runs"
        merged.append(bk)

    std_data["bookmakers"] = merged
    return std_data


def fetch_odds(api_key: str, now: datetime) -> list:
    # Step 1: lightweight h2h call to get event list and filter to unplayed games
    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/baseball_mlb/odds",
        params={"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"},
        timeout=15,
    )
    resp.raise_for_status()
    events = resp.json()

    # Step 2: per-event dual-market fetch (player props not available on bulk endpoint)
    result = []
    for event in events:
        commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if commence_time <= now:
            continue
        result.append(fetch_event_odds(api_key, event["id"]))
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
