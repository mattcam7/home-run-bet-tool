"""Tests for the same-book longshot parlay generator (agents/parlay.py)."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from agents.parlay import (
    CANDIDATE_CAP,
    DEFAULT_MAX_DECIMAL,
    DEFAULT_MIN_DECIMAL,
    _american_from_decimal,
    format_parlays,
    generate_parlays,
)
from agents.utils import american_to_decimal

COMMENCE = datetime(2026, 5, 15, 23, 5, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers to build minimal DataFrames
# ---------------------------------------------------------------------------

def _make_player(name, game, decimal_odds, prob, ev_pct, book="draftkings", anchor="pinnacle"):
    """Return a single-row dict matching calculate_ev output schema.

    ``book`` becomes a column name holding the American odds integer,
    mirroring how calculate_ev produces per-book columns from the OddsAPI.
    Parlays are generated from these per-book columns — not from
    best_retail_decimal — so using a single book per player here gives
    fully deterministic test behaviour.
    """
    american = int(round((decimal_odds - 1) * 100))
    return {
        "player_name": name,
        "game": game,
        "commence_time": COMMENCE,
        "best_retail_decimal": decimal_odds,
        "best_retail_odds": american,
        "best_retail_book": book,
        "pinnacle_prob": prob,
        "ev_pct": ev_pct,
        "sharp_anchor": anchor,
        book: american,  # per-book column used by generate_parlays
    }


def _df(*rows):
    return pd.DataFrame(list(rows))


# Three players, different games — perfect 3-leg parlay candidates (all draftkings)
JUDGE     = _make_player("Aaron Judge",     "NYY @ BOS", 7.0,  0.18, 0.26)   # +600, +EV
DEVERS    = _make_player("Rafael Devers",   "NYY @ BOS", 9.0,  0.13, 0.17)   # +800, +EV same game as Judge
ALONSO    = _make_player("Pete Alonso",     "NYM @ PHI", 8.0,  0.15, 0.20)   # +700, +EV
BETTS     = _make_player("Mookie Betts",    "LAD @ SF",  6.5,  0.20, 0.30)   # +550, +EV
OHTANI    = _make_player("Shohei Ohtani",   "LAD @ SF",  8.5,  0.14, 0.19)   # +750, +EV same game as Betts
HARPER    = _make_player("Bryce Harper",    "CLE @ PHI", 3.5,  0.30, 0.05)   # +250, below min band
NEG_EV    = _make_player("Neg Ev Player",   "CHC @ STL", 7.0,  0.10, -0.03)  # -EV
BOL_LEG   = _make_player("BOL Player",      "MIA @ ATL", 7.5,  0.16, 0.20, anchor="betonlineag")


# ---------------------------------------------------------------------------
# _american_from_decimal
# ---------------------------------------------------------------------------

def test_american_from_decimal_positive():
    assert _american_from_decimal(6.0) == "+500"
    assert _american_from_decimal(2.0) == "+100"

def test_american_from_decimal_large():
    # 336.0 = 6×8×7 parlay
    assert _american_from_decimal(336.0) == "+33500"


# ---------------------------------------------------------------------------
# generate_parlays — basic behaviour
# ---------------------------------------------------------------------------

def test_returns_empty_when_fewer_than_min_legs_eligible():
    # Only 1 qualifying player in band
    df = _df(JUDGE)
    result = generate_parlays(df, min_legs=3)
    assert result == []


def test_excludes_same_game_legs():
    """Judge and Devers are in the same game — no 2-leg combo should contain both."""
    df = _df(JUDGE, DEVERS, ALONSO)
    # min_legs=2 to make the check easier
    result = generate_parlays(df, min_legs=2, max_legs=2)
    for p in result:
        # Each 2-leg combo must span 2 different games
        assert len(set(df[df["player_name"].isin(p["legs"])]["game"].values)) == 2


def test_excludes_negative_ev_legs():
    df = _df(JUDGE, ALONSO, BETTS, NEG_EV)
    result = generate_parlays(df, min_legs=3)
    for p in result:
        assert "Neg Ev Player" not in p["legs"]


def test_excludes_legs_outside_odds_band():
    """Harper at +250 (decimal 3.5) is below the default min +500 (6.0)."""
    df = _df(JUDGE, ALONSO, BETTS, HARPER)
    result = generate_parlays(df, min_legs=3)
    for p in result:
        assert "Bryce Harper" not in p["legs"]


def test_combined_ev_positive_when_all_legs_positive():
    """Product of +EV decimals × true probs - 1 must be > 0."""
    df = _df(JUDGE, ALONSO, BETTS)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result, "Expected at least one 3-leg parlay"
    for p in result:
        assert p["combined_ev_pct"] > 0


def test_combined_ev_formula():
    """combined_ev_pct = (prod_decimal × prod_prob - 1) × 100, rounded to 2dp.

    Uses each book's actual odds (not best-retail).  In the test data every
    player has a single book column so book odds == best_retail_decimal.
    """
    df = _df(JUDGE, ALONSO, BETTS)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result
    p = result[0]
    legs_map = {r["player_name"]: r for r in [JUDGE, ALONSO, BETTS]}
    prod_dec = 1.0
    prod_prob = 1.0
    for leg_name in p["legs"]:
        r = legs_map[leg_name]
        # Use the book-column odds (American integer) to derive decimal
        dec = american_to_decimal(r[r["best_retail_book"]])
        prod_dec *= dec
        prod_prob *= r["pinnacle_prob"]
    expected = round((prod_dec * prod_prob - 1) * 100, 2)
    assert abs(p["combined_ev_pct"] - expected) < 0.01


def test_sorted_by_combined_ev_descending():
    df = _df(JUDGE, ALONSO, BETTS, BOL_LEG)
    result = generate_parlays(df, min_legs=2, max_legs=4)
    evs = [p["combined_ev_pct"] for p in result]
    assert evs == sorted(evs, reverse=True)


def test_top_n_limits_output():
    df = _df(JUDGE, ALONSO, BETTS, BOL_LEG)
    result = generate_parlays(df, min_legs=2, max_legs=4, top_n=3)
    assert len(result) <= 3


def test_all_same_book_flag():
    """all_same_book is always True — parlays are always placed at one sportsbook."""
    df = _df(JUDGE, ALONSO, BETTS)  # all draftkings
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result
    for p in result:
        assert p["all_same_book"] is True
        assert p["book"] == "draftkings"


def test_cross_book_players_not_combined():
    """Players at different sportsbooks cannot form a same-book parlay together.

    JUDGE and ALONSO are only priced at draftkings (2 legs).
    MIXED_PLAYER is only priced at fanduel (1 leg).
    No book has ≥3 qualifying legs → no 3-leg parlay generated.
    """
    mixed_book = _make_player("Mixed Player", "MIA @ ATL", 7.0, 0.18, 0.26, book="fanduel")
    df = _df(JUDGE, ALONSO, mixed_book)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result == [], "Cross-book legs must not be combined into a single parlay"


def test_same_book_parlay_uses_book_odds_not_best_retail():
    """EV is computed from that book's odds, not from a different book's best price.

    Player A: fanduel=+600 (dec 7.0), draftkings=+550 (dec 6.5)
    Player B: fanduel=+700 (dec 8.0), draftkings=+650 (dec 7.5)
    Player C: fanduel=+550 (dec 6.5)  (no draftkings)

    A 3-leg fanduel parlay must use fanduel prices (+600, +700, +550),
    not best-retail (+600, +700, +650 from different books).
    """
    # Player A: available at both books
    row_a = {
        "player_name": "Player A", "game": "G1 @ H1", "commence_time": COMMENCE,
        "pinnacle_prob": 0.17, "ev_pct": 0.19, "sharp_anchor": "pinnacle",
        "best_retail_decimal": 7.0, "best_retail_odds": 600, "best_retail_book": "fanduel",
        "fanduel": 600, "draftkings": 550,
    }
    # Player B: available at both books
    row_b = {
        "player_name": "Player B", "game": "G2 @ H2", "commence_time": COMMENCE,
        "pinnacle_prob": 0.14, "ev_pct": 0.12, "sharp_anchor": "pinnacle",
        "best_retail_decimal": 8.0, "best_retail_odds": 700, "best_retail_book": "fanduel",
        "fanduel": 700, "draftkings": 650,
    }
    # Player C: fanduel only
    row_c = {
        "player_name": "Player C", "game": "G3 @ H3", "commence_time": COMMENCE,
        "pinnacle_prob": 0.19, "ev_pct": 0.23, "sharp_anchor": "pinnacle",
        "best_retail_decimal": 6.5, "best_retail_odds": 550, "best_retail_book": "fanduel",
        "fanduel": 550,
    }
    df = pd.DataFrame([row_a, row_b, row_c])
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result, "FanDuel has 3 qualifying legs — expect a parlay"

    fd_parlay = next((p for p in result if p["book"] == "fanduel"), None)
    assert fd_parlay is not None

    # Verify FanDuel odds are used, not DraftKings
    legs_map = {"Player A": 600, "Player B": 700, "Player C": 550}
    for leg, odds in zip(fd_parlay["legs"], fd_parlay["leg_odds"]):
        assert odds == legs_map[leg], f"{leg}: expected {legs_map[leg]}, got {odds}"

    # Verify combined decimal uses FanDuel prices
    expected_dec = american_to_decimal(600) * american_to_decimal(700) * american_to_decimal(550)
    assert abs(fd_parlay["combined_decimal"] - round(expected_dec, 2)) < 0.01


def test_has_betonline_anchor_flag():
    df = _df(JUDGE, ALONSO, BOL_LEG)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result
    # BOL_LEG is in the only valid 3-leg combo (all different games)
    p = result[0]
    if "BOL Player" in p["legs"]:
        assert p["has_betonline_anchor"] is True


def test_no_betonline_flag_when_all_pinnacle():
    df = _df(JUDGE, ALONSO, BETTS)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result
    assert result[0]["has_betonline_anchor"] is False


def test_combined_american_string_format():
    df = _df(JUDGE, ALONSO, BETTS)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result
    # Should start with '+'
    assert result[0]["combined_american"].startswith("+")


def test_parlay_n_legs_field():
    df = _df(JUDGE, ALONSO, BETTS, BOL_LEG)
    result = generate_parlays(df, min_legs=2, max_legs=4)
    for p in result:
        assert p["n_legs"] == len(p["legs"])


def test_no_duplicate_players_per_parlay():
    df = _df(JUDGE, ALONSO, BETTS)
    result = generate_parlays(df, min_legs=2, max_legs=3)
    for p in result:
        assert len(set(p["legs"])) == len(p["legs"])


def test_books_list_backward_compat():
    """books[] per-leg list must equal [book × n_legs] for dashboard JS compat."""
    df = _df(JUDGE, ALONSO, BETTS)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert result
    p = result[0]
    assert p["books"] == [p["book"]] * p["n_legs"]


# ---------------------------------------------------------------------------
# format_parlays
# ---------------------------------------------------------------------------

def test_format_parlays_empty():
    out = format_parlays([])
    assert "no qualifying" in out.lower()


def test_format_parlays_contains_player_names():
    df = _df(JUDGE, ALONSO, BETTS)
    parlays = generate_parlays(df, min_legs=3, max_legs=3)
    out = format_parlays(parlays)
    for name in ["Aaron Judge", "Pete Alonso", "Mookie Betts"]:
        assert name in out


def test_format_parlays_shows_book_in_header():
    """The sportsbook must appear at the parlay level, not buried per-leg."""
    df = _df(JUDGE, ALONSO, BETTS)
    parlays = generate_parlays(df, min_legs=3, max_legs=3)
    out = format_parlays(parlays)
    assert "@ draftkings" in out


def test_format_parlays_shows_ev():
    df = _df(JUDGE, ALONSO, BETTS)
    parlays = generate_parlays(df, min_legs=3, max_legs=3)
    out = format_parlays(parlays)
    assert "EV" in out


def test_format_parlays_bol_anchor_flagged():
    df = _df(JUDGE, ALONSO, BOL_LEG)
    parlays = generate_parlays(df, min_legs=3, max_legs=3)
    out = format_parlays(parlays)
    if any(p["has_betonline_anchor"] for p in parlays):
        assert "BOL anchor" in out


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_works_without_sharp_anchor_column():
    """Backward compat: DataFrame without sharp_anchor column should not raise."""
    rows = [
        {k: v for k, v in r.items() if k != "sharp_anchor"}
        for r in [JUDGE, ALONSO, BETTS]
    ]
    df = pd.DataFrame(rows)
    result = generate_parlays(df, min_legs=3, max_legs=3)
    assert isinstance(result, list)


def test_candidate_cap_applied(monkeypatch):
    """If more than CANDIDATE_CAP candidates per book, the generator trims before combinatorics."""
    # Build CANDIDATE_CAP + 5 unique players, each in its own game, all at draftkings
    rows = [
        _make_player(f"Player{i}", f"Home{i} @ Away{i}", 7.0, 0.18, 0.20)
        for i in range(CANDIDATE_CAP + 5)
    ]
    df = pd.DataFrame(rows)
    # Just verify it runs without error and returns <= top_n
    result = generate_parlays(df, min_legs=3, max_legs=3, top_n=5)
    assert len(result) <= 5
