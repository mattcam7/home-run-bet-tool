"""agents/parlay.py — Longshot parlay generator.

Identifies +EV combinations of 3-5 legs where each leg falls in the
+500 to +1500 American odds band (decimal 6.0-16.0).

Same-game legs are excluded — HR outcomes in the same game share the same
pitcher, weather, and park factors, making positive correlation likely.
The independence assumption required for the combined-probability math only
holds across different games.

EV is computed assuming each leg is bet at the best available retail price:
    combined_decimal = product of best_retail_decimal per leg
    combined_prob    = product of sharp-anchor de-vigged prob per leg
    combined_ev_pct  = combined_decimal * combined_prob - 1

When all individual legs are +EV, the combined EV is always positive
(product of terms > 1), but the combined probability may be tiny so
the practical edge depends on the parlay multiplier vs. true prob.

Confidence tiers
----------------
  sharp_anchor == "pinnacle"    : Pinnacle de-vigged line — highest confidence
  sharp_anchor == "betonlineag" : BetOnline de-vigged line — good, but slightly
                                  more model uncertainty; flagged in output
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import pandas as pd

# Decimal-odds bounds for the longshot band
DEFAULT_MIN_DECIMAL: float = 6.0   # ~+500 American
DEFAULT_MAX_DECIMAL: float = 16.0  # ~+1500 American
DEFAULT_MIN_LEGS: int = 3
DEFAULT_MAX_LEGS: int = 5
DEFAULT_TOP_N: int = 10

# Trim candidates to this many before combinatorics so runtime stays <1 s
# even on a big slate.  C(50,5) = 2,118,760 — still fast in Python.
CANDIDATE_CAP: int = 50


def _american_from_decimal(d: float) -> str:
    """Convert a decimal multiplier to American odds string (e.g. 6.0 -> '+500')."""
    if d >= 2.0:
        return f"+{int(round((d - 1) * 100))}"
    return str(int(round(-100 / (d - 1))))


def generate_parlays(
    df: pd.DataFrame,
    min_legs: int = DEFAULT_MIN_LEGS,
    max_legs: int = DEFAULT_MAX_LEGS,
    min_leg_decimal: float = DEFAULT_MIN_DECIMAL,
    max_leg_decimal: float = DEFAULT_MAX_DECIMAL,
    top_n: int = DEFAULT_TOP_N,
) -> list[dict[str, Any]]:
    """Return top ``top_n`` +EV longshot parlay combinations.

    Parameters
    ----------
    df : DataFrame
        Output of ``calculate_ev``.  Required columns: player_name, game,
        best_retail_decimal, best_retail_odds, best_retail_book,
        pinnacle_prob, ev_pct.  Optional: sharp_anchor.
    min_legs / max_legs : int
        Inclusive leg-count range.
    min_leg_decimal / max_leg_decimal : float
        Decimal odds band per leg (default: 6.0 - 16.0, i.e. +500 to +1500).
    top_n : int
        Number of top combinations to return (ranked by combined_ev_pct desc).

    Returns
    -------
    list[dict]
        Each dict contains:
          legs              – player names
          books             – best retail book per leg
          leg_odds          – American odds per leg
          leg_ev_pcts       – individual EV% per leg (×100, rounded to 2dp)
          combined_decimal  – product of leg decimal odds
          combined_prob_pct – combined probability in % (product of probs)
          combined_ev_pct   – combined EV in % (combined_decimal×prob − 1)×100
          combined_american – combined odds as American string (e.g. "+33500")
          n_legs
          all_same_book     – True when every leg has the same best retail book
          has_betonline_anchor – True when any leg's anchor is BetOnline
    """
    # --- filter to +EV longshots ---
    cands = df[
        (df["ev_pct"] > 0)
        & (df["best_retail_decimal"] >= min_leg_decimal)
        & (df["best_retail_decimal"] <= max_leg_decimal)
    ].copy()

    if len(cands) < min_legs:
        return []

    # Safety cap — trim to highest-EV players to keep combinatorics fast
    if len(cands) > CANDIDATE_CAP:
        cands = cands.nlargest(CANDIDATE_CAP, "ev_pct")

    records = cands.to_dict("records")
    results: list[dict] = []

    for n_legs in range(min_legs, max_legs + 1):
        for combo in combinations(records, n_legs):
            # --- same-game exclusion ---
            games = [r["game"] for r in combo]
            if len(set(games)) < n_legs:
                continue  # at least two legs share a game

            combined_decimal = 1.0
            combined_prob = 1.0
            for r in combo:
                combined_decimal *= r["best_retail_decimal"]
                combined_prob *= r["pinnacle_prob"]

            combined_ev = combined_decimal * combined_prob - 1
            # Guard: should always be positive when all legs are +EV, but
            # floating-point can produce tiny negatives near zero — skip.
            if combined_ev <= 0:
                continue

            books = [r.get("best_retail_book") for r in combo]
            results.append({
                "legs": [r["player_name"] for r in combo],
                "books": books,
                "leg_odds": [int(r["best_retail_odds"]) for r in combo],
                "leg_ev_pcts": [round(r["ev_pct"] * 100, 2) for r in combo],
                "combined_decimal": round(combined_decimal, 2),
                "combined_prob_pct": round(combined_prob * 100, 4),
                "combined_ev_pct": round(combined_ev * 100, 2),
                "combined_american": _american_from_decimal(combined_decimal),
                "n_legs": n_legs,
                "all_same_book": len(set(books)) == 1,
                "has_betonline_anchor": any(
                    r.get("sharp_anchor") == "betonlineag" for r in combo
                ),
            })

    results.sort(key=lambda x: x["combined_ev_pct"], reverse=True)
    return results[:top_n]


def format_parlays(parlays: list[dict]) -> str:
    """Return a terminal-friendly summary of the top longshot parlays."""
    if not parlays:
        return (
            "No qualifying longshot parlays found "
            "(need 3-5 +EV legs at +500 to +1500, no same-game legs)."
        )

    lines = [
        f"Top {len(parlays)} Longshot Parlays  (+500 to +1500 per leg | no same-game)",
        "=" * 70,
    ]
    for rank, p in enumerate(parlays, 1):
        flags = []
        if p["has_betonline_anchor"]:
            flags.append("BOL anchor")
        if p["all_same_book"]:
            flags.append(f"all {p['books'][0]}")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"#{rank}  {p['n_legs']}-leg  "
            f"EV {p['combined_ev_pct']:+.2f}%  "
            f"Prob {p['combined_prob_pct']:.4f}%  "
            f"Odds {p['combined_american']}"
            f"{flag_str}"
        )
        for i, (leg, book, odds, ev) in enumerate(
            zip(p["legs"], p["books"], p["leg_odds"], p["leg_ev_pcts"]), 1
        ):
            odds_fmt = f"+{odds}" if odds > 0 else str(odds)
            lines.append(f"   {i}. {leg}  {odds_fmt} @ {book}  (leg EV {ev:+.2f}%)")
        lines.append("")

    return "\n".join(lines)
