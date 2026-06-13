"""Hourly line movement monitor — GitHub Actions runs this every hour 11 AM–9 PM ET."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from dotenv import load_dotenv

ET = ZoneInfo("America/New_York")
CLV_PATH = Path("data/clv_log.csv")
STATE_PATH = Path("data/monitor_state.json")


def _american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1


def _load_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_state(state: dict) -> None:
    STATE_PATH.parent.mkdir(exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2))


def run(
    now=None,
    fetch_odds_fn=None,
    current_odds_fn=None,
    post_alert_fn=None,
    post_status_fn=None,
) -> None:
    load_dotenv()
    now = now or datetime.now(timezone.utc)
    today_str = now.astimezone(ET).strftime("%Y-%m-%d")

    # Load today's featured bets
    supabase_key = os.environ.get("SUPABASE_KEY", "")
    if supabase_key:
        try:
            from agents.supabase_client import fetch_clv_log
            picks = fetch_clv_log(game_date=today_str, featured_only=True)
        except Exception:
            picks = pd.DataFrame()
    else:
        if not CLV_PATH.exists():
            return
        df = pd.read_csv(CLV_PATH)
        picks = df[
            (df["game_date"] == today_str) &
            (df["featured_bet"].astype(str).str.lower() == "true")
        ].copy()

    if picks.empty:
        return

    # Only picks for games not yet started
    def _game_not_started(iso):
        if pd.isna(iso):
            return True
        ct = datetime.fromisoformat(str(iso))
        # Make tz-aware if naive
        ct = ct.replace(tzinfo=timezone.utc) if ct.tzinfo is None else ct
        return ct > now

    picks = picks[picks["commence_iso"].apply(_game_not_started)]
    if picks.empty:
        return

    # Re-scrape current retail odds
    api_key = os.environ["ODDS_API_KEY"]
    if fetch_odds_fn is None:
        from run import fetch_odds
        fetch_odds_fn = fetch_odds
    raw = fetch_odds_fn(api_key, now)

    if current_odds_fn is None:
        from agents.odds_scraper import extract_retail_odds
        current_odds_fn = extract_retail_odds
    current_df = current_odds_fn(raw, now)
    if current_df.empty:
        return

    # Aggregate to best (highest) odds per player across all retail books
    best_current = (
        current_df.groupby("player_name")["american_odds"]
        .max()
        .reset_index()
        .rename(columns={"american_odds": "best_retail_odds"})
    )
    current_idx = best_current.set_index("player_name")

    state = _load_state()
    today_state = state.get(today_str, {})

    if post_alert_fn is None:
        from agents.discord_bot import post_alert
        post_alert_fn = post_alert
    if post_status_fn is None:
        from agents.discord_bot import post_status
        post_status_fn = post_status

    alerts_sent = 0
    for _, row in picks.iterrows():
        pname = str(row["player_name"])
        player_state = today_state.get(pname, {})

        # Skip only if withdrawal already sent — nothing more to do for this player
        if player_state.get("withdrawal_sent"):
            continue

        if pname not in current_idx.index:
            continue  # line pulled or game started

        curr_row = current_idx.loc[pname]
        if isinstance(curr_row, pd.DataFrame):
            curr_row = curr_row.iloc[0]

        curr_odds = int(curr_row["best_retail_odds"])
        orig_odds = int(row["best_retail_odds"])
        pin_prob = float(row["pinnacle_prob_devig"])
        orig_ev = float(row["ev_pct"])
        curr_ev = _american_to_decimal(curr_odds) * pin_prob - 1

        if curr_ev < 0 and not player_state.get("withdrawal_sent"):
            try:
                post_alert_fn(pname, orig_odds, curr_odds, orig_ev, "withdrawal")
                today_state[pname] = {**player_state, "withdrawal_sent": True}
                alerts_sent += 1
            except Exception as e:
                import logging
                logging.error(f"monitor: withdrawal alert failed for {pname}: {e}")
        elif (abs(curr_odds - orig_odds) > 15 or
              (orig_ev >= 0.10 and curr_ev < 0.05)) and not player_state.get("alert_sent"):
            try:
                post_alert_fn(pname, orig_odds, curr_odds, orig_ev, "movement")
                today_state[pname] = {**player_state, "alert_sent": True}
                alerts_sent += 1
            except Exception as e:
                import logging
                logging.error(f"monitor: movement alert failed for {pname}: {e}")

    state[today_str] = today_state
    _save_state(state)

    if os.name == "nt":
        time_str = now.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
        date_str_short = now.astimezone(ET).strftime("%b %#d")
    else:
        time_str = now.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
        date_str_short = now.astimezone(ET).strftime("%b %-d")

    if alerts_sent:
        post_status_fn(f"⚠️ Monitor — {alerts_sent} alert(s) sent · {date_str_short} {time_str}")
    else:
        post_status_fn(f"✅ Monitor — no significant movement · {date_str_short} {time_str}")


if __name__ == "__main__":
    run()
