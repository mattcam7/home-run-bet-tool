import os
from datetime import datetime, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

from agents.clv_log import log_open_plays
from agents.dfs import analyze_dfs
from agents.ev_calculator import calculate_ev, validate_slate
from agents.odds_scraper import extract_retail_odds
from agents.parlay import format_parlays, generate_parlays
from agents.pinnacle_scraper import extract_sharp_anchor
from agents.scoring import compute_bet_score
from agents.simulation import add_simulation, validate_simulation
from agents.validation import StepResult, append_quarantine, validate_ev_output
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

    # Merge alternate into standard, book by book.
    # Books only in alternate: rename market key and append.
    # Books in both: merge alternate outcomes into the standard batter_home_runs
    # market so players exclusive to the alternate market aren't dropped.
    std_bk_by_key = {bk["key"]: bk for bk in std_data.get("bookmakers", [])}
    for bk in alt_data.get("bookmakers", []):
        alt_outcomes = []
        for market in bk["markets"]:
            if market["key"] == "batter_home_runs_alternate":
                alt_outcomes = market["outcomes"]
                break
        if not alt_outcomes:
            continue

        if bk["key"] in std_bk_by_key:
            # Book exists in standard data — merge in any players missing there.
            std_bk = std_bk_by_key[bk["key"]]
            std_market = next(
                (m for m in std_bk["markets"] if m["key"] == "batter_home_runs"),
                None,
            )
            if std_market is None:
                std_bk["markets"].append({"key": "batter_home_runs", "outcomes": alt_outcomes})
            else:
                existing_players = {o.get("description", "").strip().lower() for o in std_market["outcomes"]}
                for outcome in alt_outcomes:
                    if outcome.get("description", "").strip().lower() not in existing_players:
                        std_market["outcomes"].append(outcome)
        else:
            # Book only in alternate — rename key and add wholesale.
            std_data.setdefault("bookmakers", []).append({
                **bk,
                "markets": [{"key": "batter_home_runs", "outcomes": alt_outcomes}],
            })

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
    import argparse
    parser = argparse.ArgumentParser(description="HR bet tool pipeline")
    parser.add_argument("--no-browser", action="store_true",
                        help="Skip browser; post picks to Discord instead")
    args, _ = parser.parse_known_args()

    load_dotenv()
    api_key = os.environ["ODDS_API_KEY"]
    now = datetime.now(timezone.utc)

    print("Fetching odds from OddsAPI...")
    raw = fetch_odds(api_key, now)

    print("Fetching player team data...")
    player_teams = fetch_player_teams()

    print("Extracting retail odds...")
    retail_df = extract_retail_odds(raw, now)

    print("Extracting sharp anchor odds (Pinnacle + BetOnline fallback)...")
    anchor_df = extract_sharp_anchor(raw, now)

    if retail_df.empty or anchor_df.empty:
        missing = "sharp anchor (Pinnacle/BetOnline)" if anchor_df.empty else "retail"
        print(f"\nNo {missing} HR props available yet for today's games.")
        if anchor_df.empty:
            print(
                "Pinnacle and BetOnline post MLB HR props in the afternoon ET - "
                "the sharp line is the EV anchor, so nothing can be computed until it's up."
            )
            print("Re-run after ~2 PM ET (closer to first pitch is sharper).")
        if args.no_browser:
            from agents.discord_bot import post_status
            post_status(
                f"⚾ No HR lines yet — {missing} props not posted. "
                f"Pipeline ran at {now.strftime('%H:%M UTC')}."
            )
        return

    n_pin = int((anchor_df.get("sharp_anchor", "") == "pinnacle").sum()) if "sharp_anchor" in anchor_df.columns else len(anchor_df)
    n_bol = len(anchor_df) - n_pin
    print(f"Anchor coverage: {n_pin} Pinnacle + {n_bol} BetOnline fallback = {len(anchor_df)} total")

    print("Calculating EV...")
    final_df = calculate_ev(retail_df, anchor_df)
    validate_slate(final_df)
    final_df["team"] = final_df["player_name"].map(player_teams).fillna("")

    # Validate EV output and quarantine bad rows
    ev_result = validate_ev_output(final_df)
    if ev_result.quarantined:
        append_quarantine(ev_result.quarantined)
    for w in ev_result.warnings:
        print(f"  [validation] {w}")
    final_df = ev_result.clean

    print("Running simulation model...")
    final_df = add_simulation(final_df)
    for w in validate_simulation(final_df):
        print(f"  {w}")

    # Compute bet quality score (0-100) — must run after simulation so sim_prob
    # is available for the divergence dampener in compute_bet_score.
    final_df = compute_bet_score(final_df)

    # featured_bet: plays that meet the kelly + EV threshold for posting to Discord.
    # Must be set here so post_picks can filter on it (log_open_plays computes it
    # internally but does not write it back to final_df).
    final_df["featured_bet"] = (
        (pd.to_numeric(final_df["kelly_units"], errors="coerce").fillna(0) >= 0.5) &
        (pd.to_numeric(final_df["ev_pct"], errors="coerce").fillna(0) >= 0.10)
    )

    n_players = len(final_df)
    n_positive = int((final_df["ev_pct"] > 0).sum())
    print(f"Found {n_players} players | {n_positive} +EV plays")

    log_open_plays(final_df, now=now)
    print("Logged open plays to CLV log.")

    parlays = generate_parlays(final_df)
    if parlays:
        print(f"\n{format_parlays(parlays)}")

    dfs_data = analyze_dfs("data/dfs_projections.csv", final_df)
    if dfs_data:
        meta = dfs_data["meta"]
        print(
            f"DFS projections loaded: {meta['active_hitters']} active hitters, "
            f"{meta['hr_matches']} HR prop matches, "
            f"{len(dfs_data['convergences'])} convergence plays."
        )
    else:
        print("No DFS projections found at data/dfs_projections.csv — DFS tab will be empty.")

    generate_dashboard(final_df, parlays=parlays, dfs_data=dfs_data,
                       open_browser=not args.no_browser)
    if args.no_browser:
        from agents.discord_bot import post_picks
        post_picks(final_df, now=now)
        print("Picks posted to Discord.")
    else:
        print("Dashboard opened in browser.")


if __name__ == "__main__":
    main()
