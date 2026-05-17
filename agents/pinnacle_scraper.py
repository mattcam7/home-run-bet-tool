import pandas as pd
from datetime import datetime
from agents.utils import american_to_decimal


def extract_pinnacle_odds(raw_payload: list, now: datetime) -> pd.DataFrame:
    rows = []
    for event in raw_payload:
        commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if commence_time <= now:
            continue
        game = f"{event['away_team']} @ {event['home_team']}"
        for bookmaker in event["bookmakers"]:
            if bookmaker["key"] != "pinnacle":
                continue
            for market in bookmaker["markets"]:
                if market["key"] != "batter_home_runs":
                    continue
                # Pair the Over/Under at point 0.5 per player so we can strip
                # Pinnacle's vig. Raw 1/over_decimal overstates the true
                # probability by the book's hold (~4-6% on HR props).
                sides: dict[str, dict[str, int]] = {}
                for outcome in market["outcomes"]:
                    if outcome.get("point") != 0.5:
                        continue
                    side = outcome.get("name")
                    if side not in ("Over", "Under"):
                        continue
                    player = outcome["description"].strip().title()
                    sides.setdefault(player, {})[side] = outcome["price"]

                for player, prices in sides.items():
                    if "Over" not in prices:
                        continue
                    over_imp = 1 / american_to_decimal(prices["Over"])
                    if "Under" in prices:
                        under_imp = 1 / american_to_decimal(prices["Under"])
                        true_prob = over_imp / (over_imp + under_imp)
                    else:
                        # No Under posted — cannot strip vig. Fall back to the
                        # vig-inclusive implied prob (plausibility guard) so the
                        # player is still surfaced rather than silently dropped.
                        true_prob = over_imp
                    rows.append({
                        "player_name": player,
                        "game": game,
                        "commence_time": commence_time,
                        "pinnacle_odds": prices["Over"],
                        "pinnacle_prob": true_prob,
                    })
    return pd.DataFrame(rows)
