import os
from datetime import datetime, timezone

import requests
from dotenv import load_dotenv

from agents.clv_log import log_open_plays
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


def fetch_player_teams() -> dict:
    """Returns {normalized_player_name: team_abbreviation} via MLB Stats API.

    Two calls: teams (id→abbrev) then players (name→team_id). Name matching
    is string-based — a known limitation since OddsAPI exposes no numeric
    player IDs for props markets. Mismatches result in an empty team string.
    """
    teams_resp = requests.get(
        "https://statsapi.mlb.com/api/v1/teams",
        params={"sportId": 1, "season": datetime.now(timezone.utc).year},
        timeout=15,
    )
    teams_resp.raise_for_status()
    team_abbrev = {t["id"]: t["abbreviation"] for t in teams_resp.json().get("teams", [])}

    players_resp = requests.get(
        "https://statsapi.mlb.com/api/v1/sports/1/players",
        params={"season": datetime.now(timezone.utc).year, "gameType": "R"},
        timeout=15,
    )
    players_resp.raise_for_status()
    result = {}
    for player in players_resp.json().get("people", []):
        name = player["fullName"].strip().title()
        team_id = player.get("currentTeam", {}).get("id")
        if team_id and team_id in team_abbrev:
            result[name] = team_abbrev[team_id]
    return result


def main() -> None:
    load_dotenv()
    api_key = os.environ["ODDS_API_KEY"]
    now = datetime.now(timezone.utc)

    print("Fetching odds from OddsAPI...")
    raw = fetch_odds(api_key, now)

    print("Fetching player team data...")
    player_teams = fetch_player_teams()

    print("Extracting retail odds...")
    retail_df = extract_retail_odds(raw, now)

    print("Extracting Pinnacle odds...")
    pinnacle_df = extract_pinnacle_odds(raw, now)

    if retail_df.empty or pinnacle_df.empty:
        missing = "Pinnacle" if pinnacle_df.empty else "retail"
        print(f"\nNo {missing} HR props available yet for today's games.")
        if pinnacle_df.empty:
            print(
                "Pinnacle posts MLB HR props in the afternoon ET - the sharp "
                "line is the EV anchor, so nothing can be computed until it's up."
            )
            print("Re-run after ~2 PM ET (closer to first pitch is sharper).")
        return

    print("Calculating EV...")
    final_df = calculate_ev(retail_df, pinnacle_df)
    final_df["team"] = final_df["player_name"].map(player_teams).fillna("")

    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
    print(f"Found {n_players} players | {n_positive} +EV plays")

    log_open_plays(final_df, now=now)
    print("Logged open plays to CLV log.")

    generate_dashboard(final_df)
    print("Dashboard opened in browser.")


if __name__ == "__main__":
    main()
