"""agents/parlay.py — Same-book longshot parlay generator.

Identifies +EV combinations of 3-5 legs where every leg is available at
the same sportsbook and falls in the +500 to +1500 American odds band
(decimal 6.0–16.0).

Parlays MUST use a single sportsbook — you cannot combine legs from
different books because the whole parlay is one bet placed at one book.
Leg EV is therefore recalculated using the specific book's odds, not
the best-retail price across all books.

Same-game legs are excluded — HR outcomes in the same game share the same
pitcher, weather, and park factors, making positive correlation likely.
The independence assumption required for the combined-probability math only
holds across different games.

EV per leg (at a given book):
    leg_ev = book_decimal × pinnacle_prob − 1

Combined parlay EV:
    combined_decimal = product of book_decimal per leg
    combined_prob    = product of sharp-anchor de-vigged prob per leg
    combined_ev_pct  = (combined_decimal × combined_prob − 1) × 100

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

# Per-book trim before combinatorics so runtime stays <1 s on big slates.
# C(50,5) = 2,118,760 — still fast in Python.
CANDIDATE_CAP: int = 50

# Meta columns from calculate_ev output — everything else is a book column.
_META: frozenset[str] = frozenset({
    "player_name", "team", "game", "commence_time",
    "pinnacle_odds", "pinnacle_prob", "sharp_anchor",
    "best_retail_odds", "best_retail_decimal", "best_retail_book",
    "ev_pct", "composite_score", "composite_z",
    "kelly_units", "stake_usd",
})


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
    """Return top ``top_n`` +EV same-book longshot parlay combinations.

    All legs in each returned parlay are from the same sportsbook so the
    parlay can actually be placed.  Leg EV is recalculated using that
    book's specific odds (not best-retail), so every parlay is genuinely
    placeable at one book.

    Parameters
    ----------
    df : DataFrame
        Output of ``calculate_ev``.  Required columns: player_name, game,
        pinnacle_prob.  Per-book columns (e.g. "draftkings", "fanduel") hold
        American odds integers and are auto-detected as any column not in the
        meta set.  Optional: sharp_anchor.
    min_legs / max_legs : int
        Inclusive leg-count range.
    min_leg_decimal / max_leg_decimal : float
        Decimal odds band per leg (default: 6.0–16.0, i.e. +500 to +1500).
    top_n : int
        Number of top combinations returned (ranked by combined_ev_pct desc).

    Returns
    -------
    list[dict]
        Each dict:
          book              – sportsbook for all legs
          books             – [book × n_legs] (backward compat with dashboard JS)
          legs              – player names
          leg_odds          – American odds per leg at this book
          leg_ev_pcts       – per-leg EV% using this book's odds
          combined_decimal
          combined_prob_pct
          combined_ev_pct
          combined_american
          n_legs
          all_same_book     – always True; kept for dashboard backward compat
          has_betonline_anchor
    """
    from agents.utils import american_to_decimal as _a2d

    book_cols = [c for c in df.columns if c not in _META]

    # Build per-(player, book) candidate legs.
    # EV is recalculated using each book's actual odds, not best-retail.
    book_legs: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        prob = float(row["pinnacle_prob"])
        anchor = str(row.get("sharp_anchor", "pinnacle"))
        for book in book_cols:
            val = row.get(book)
            if pd.isna(val) or val is None:
                continue
            dec = _a2d(int(val))
            if dec < min_leg_decimal or dec > max_leg_decimal:
                continue
            ev = dec * prob - 1
            if ev <= 0:
                continue
            book_legs.setdefault(book, []).append({
                "player": str(row["player_name"]),
                "game": str(row["game"]),
                "decimal": dec,
                "american": int(val),
                "prob": prob,
                "ev_pct": round(ev * 100, 2),
                "is_bol_anchor": anchor == "betonlineag",
            })

    results: list[dict] = []

    for book, legs in book_legs.items():
        if len(legs) < min_legs:
            continue
        # Sort by per-book EV desc; cap to control combinatorial explosion
        legs = sorted(legs, key=lambda x: -x["ev_pct"])[:CANDIDATE_CAP]

        for n_legs in range(min_legs, max_legs + 1):
            if len(legs) < n_legs:
                continue
            for combo in combinations(legs, n_legs):
                # No same-game legs
                games = [c["game"] for c in combo]
                if len(set(games)) < n_legs:
                    continue

                combined_decimal = 1.0
                combined_prob = 1.0
                for c in combo:
                    combined_decimal *= c["decimal"]
                    combined_prob *= c["prob"]

                combined_ev = combined_decimal * combined_prob - 1
                # Guard: should always be positive when all legs are +EV, but
                # floating-point can produce tiny negatives near zero — skip.
                if combined_ev <= 0:
                    continue

                results.append({
                    "book": book,
                    "books": [book] * n_legs,  # backward compat with dashboard JS
                    "legs": [c["player"] for c in combo],
                    "leg_odds": [c["american"] for c in combo],
                    "leg_ev_pcts": [c["ev_pct"] for c in combo],
                    "combined_decimal": round(combined_decimal, 2),
                    "combined_prob_pct": round(combined_prob * 100, 4),
                    "combined_ev_pct": round(combined_ev * 100, 2),
                    "combined_american": _american_from_decimal(combined_decimal),
                    "n_legs": n_legs,
                    "all_same_book": True,  # always; kept for dashboard backward compat
                    "has_betonline_anchor": any(c["is_bol_anchor"] for c in combo),
                })

    results.sort(key=lambda x: x["combined_ev_pct"], reverse=True)
    return results[:top_n]


def format_parlays(parlays: list[dict]) -> str:
    """Return a terminal-friendly summary of the top same-book longshot parlays."""
    if not parlays:
        return (
            "No qualifying same-book longshot parlays found "
            "(need 3-5 +EV legs at +500 to +1500 within one sportsbook, no same-game legs)."
        )

    lines = [
        f"Top {len(parlays)} Same-Book Longshot Parlays  (+500 to +1500 per leg | no same-game)",
        "=" * 70,
    ]
    for rank, p in enumerate(parlays, 1):
        book = p.get("book") or (p["books"][0] if p.get("books") else "?")
        flags = []
        if p["has_betonline_anchor"]:
            flags.append("BOL anchor")
        flag_str = f"  [{', '.join(flags)}]" if flags else ""
        lines.append(
            f"#{rank}  {p['n_legs']}-leg  @ {book}  "
            f"EV {p['combined_ev_pct']:+.2f}%  "
            f"Prob {p['combined_prob_pct']:.4f}%  "
            f"Odds {p['combined_american']}"
            f"{flag_str}"
        )
        for i, (leg, odds, ev) in enumerate(
            zip(p["legs"], p["leg_odds"], p["leg_ev_pcts"]), 1
        ):
            odds_fmt = f"+{odds}" if odds > 0 else str(odds)
            lines.append(f"   {i}. {leg}  {odds_fmt}  (leg EV {ev:+.2f}%)")
        lines.append("")

    return "\n".join(lines)
