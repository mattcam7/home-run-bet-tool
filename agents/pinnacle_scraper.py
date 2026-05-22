import pandas as pd
from datetime import datetime
from agents.utils import american_to_decimal

# BetOnline posts the fullest HR menu among non-Pinnacle sharp books and
# accepts de-vig via Over/Under pairing. Used as fallback when Pinnacle
# doesn't price a player (typically chalk-only: 2-3 per game).
BETONLINE_KEY = "betonlineag"


def _extract_book_devig(raw_payload: list, now: datetime, book_key: str) -> list[dict]:
    """Extract de-vigged HR Over probabilities from a single sportsbook.

    Pairs the Over/Under at point 0.5 per player to strip vig.  Falls back
    to the vig-inclusive implied prob when no Under is posted (plausibility
    guard — player is surfaced rather than silently dropped).

    Returns a list of dicts (not a DataFrame) so callers can merge before
    constructing the frame.
    """
    rows: list[dict] = []
    for event in raw_payload:
        commence_time = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))
        if commence_time <= now:
            continue
        game = f"{event['away_team']} @ {event['home_team']}"
        for bookmaker in event["bookmakers"]:
            if bookmaker["key"] != book_key:
                continue
            for market in bookmaker["markets"]:
                if market["key"] != "batter_home_runs":
                    continue
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
    return rows


def extract_pinnacle_odds(raw_payload: list, now: datetime) -> pd.DataFrame:
    """Extract de-vigged Pinnacle HR Over probabilities.

    Used by capture_closing (CLV Phase 2) to get the true closing line.
    For the open-play anchor (Phase 1 / dashboard), use extract_sharp_anchor
    instead — it covers the full player menu via BetOnline fallback.
    """
    return pd.DataFrame(_extract_book_devig(raw_payload, now, "pinnacle"))


def extract_sharp_anchor(raw_payload: list, now: datetime) -> pd.DataFrame:
    """Pinnacle-first de-vigged anchor; BetOnline fallback for uncovered players.

    Pinnacle prices only ~2-3 HR props per game (chalk), leaving ~85 % of
    retail players without a sharp reference line.  BetOnline posts the
    fullest HR menu among remaining sharp books and accepts the same
    Over/Under de-vig.

    Returns a DataFrame merging both sources:
      - Pinnacle players tagged  sharp_anchor='pinnacle'
      - Additional BetOnline-only players tagged  sharp_anchor='betonlineag'

    Column schema matches extract_pinnacle_odds output plus sharp_anchor:
      player_name, game, commence_time, pinnacle_odds, pinnacle_prob, sharp_anchor

    Note: pinnacle_odds / pinnacle_prob hold BetOnline values for fallback
    rows — the naming is intentionally kept consistent so ev_calculator and
    clv_log require no structural changes.
    """
    pin_rows = _extract_book_devig(raw_payload, now, "pinnacle")
    pin_keys = {(r["player_name"], r["game"]) for r in pin_rows}
    for r in pin_rows:
        r["sharp_anchor"] = "pinnacle"

    bol_rows = _extract_book_devig(raw_payload, now, BETONLINE_KEY)
    fallback = [
        {**r, "sharp_anchor": BETONLINE_KEY}
        for r in bol_rows
        if (r["player_name"], r["game"]) not in pin_keys
    ]

    all_rows = pin_rows + fallback
    if not all_rows:
        return pd.DataFrame()
    return pd.DataFrame(all_rows)
