"""Tests for extract_sharp_anchor (Path B: Pinnacle-first + BetOnline fallback)."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from agents.pinnacle_scraper import extract_pinnacle_odds, extract_sharp_anchor
from tests.conftest import FIXTURE_NOW, FIXTURE_PAYLOAD

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

GAME = "New York Yankees @ Boston Red Sox"
COMMENCE = "2026-05-15T23:05:00Z"

PAYLOAD_PIN_AND_BOL = [
    {
        "id": "g1",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": COMMENCE,
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Over",  "description": "Aaron Judge",   "price": 380, "point": 0.5},
                    {"name": "Under", "description": "Aaron Judge",   "price": -550, "point": 0.5},
                    # Pinnacle prices ONLY Judge — Devers is uncovered
                ]}],
            },
            {
                "key": "betonlineag",
                "title": "BetOnline.ag",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    # BetOnline prices both — Judge should be deduplicated
                    {"name": "Over",  "description": "Aaron Judge",   "price": 370, "point": 0.5},
                    {"name": "Under", "description": "Aaron Judge",   "price": -530, "point": 0.5},
                    {"name": "Over",  "description": "Rafael Devers", "price": 520, "point": 0.5},
                    {"name": "Under", "description": "Rafael Devers", "price": -900, "point": 0.5},
                ]}],
            },
        ],
    }
]

PAYLOAD_NO_BOL = [
    {
        "id": "g1",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": COMMENCE,
        "bookmakers": [
            {
                "key": "pinnacle",
                "title": "Pinnacle",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Over",  "description": "Aaron Judge", "price": 380, "point": 0.5},
                    {"name": "Under", "description": "Aaron Judge", "price": -550, "point": 0.5},
                ]}],
            },
        ],
    }
]

PAYLOAD_BOL_ONLY = [
    {
        "id": "g1",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": COMMENCE,
        "bookmakers": [
            {
                "key": "betonlineag",
                "title": "BetOnline.ag",
                "markets": [{"key": "batter_home_runs", "outcomes": [
                    {"name": "Over",  "description": "Rafael Devers", "price": 520, "point": 0.5},
                    {"name": "Under", "description": "Rafael Devers", "price": -900, "point": 0.5},
                ]}],
            },
        ],
    }
]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_pinnacle_players_tagged_correctly():
    df = extract_sharp_anchor(PAYLOAD_PIN_AND_BOL, FIXTURE_NOW)
    judge = df[df["player_name"] == "Aaron Judge"].iloc[0]
    assert judge["sharp_anchor"] == "pinnacle"


def test_betonline_fallback_fills_uncovered_players():
    df = extract_sharp_anchor(PAYLOAD_PIN_AND_BOL, FIXTURE_NOW)
    devers = df[df["player_name"] == "Rafael Devers"].iloc[0]
    assert devers["sharp_anchor"] == "betonlineag"


def test_pinnacle_takes_priority_over_betonline():
    """Judge is in both books; Pinnacle entry wins, BetOnline entry is dropped."""
    df = extract_sharp_anchor(PAYLOAD_PIN_AND_BOL, FIXTURE_NOW)
    judge_rows = df[df["player_name"] == "Aaron Judge"]
    assert len(judge_rows) == 1
    assert judge_rows.iloc[0]["sharp_anchor"] == "pinnacle"
    # Pinnacle odds (+380) not BetOnline's (+370)
    assert judge_rows.iloc[0]["pinnacle_odds"] == 380


def test_coverage_expands_with_betonline():
    """BetOnline adds Devers that Pinnacle doesn't price."""
    pin_only = extract_pinnacle_odds(PAYLOAD_PIN_AND_BOL, FIXTURE_NOW)
    combined = extract_sharp_anchor(PAYLOAD_PIN_AND_BOL, FIXTURE_NOW)
    assert len(combined) > len(pin_only)
    assert "Rafael Devers" in combined["player_name"].values
    assert "Rafael Devers" not in pin_only["player_name"].values


def test_returns_empty_when_neither_book_present():
    df = extract_sharp_anchor([], FIXTURE_NOW)
    assert df.empty


def test_betonline_devig_is_correct():
    """BetOnline de-vig uses same Over/Under pairing as Pinnacle."""
    df = extract_sharp_anchor(PAYLOAD_BOL_ONLY, FIXTURE_NOW)
    devers = df[df["player_name"] == "Rafael Devers"].iloc[0]
    # +520 -> dec 6.2 ; -900 -> dec 1.1111
    over_imp = 1 / 6.2
    under_imp = 1 / (100 / 900 + 1)
    expected = over_imp / (over_imp + under_imp)
    assert abs(devers["pinnacle_prob"] - expected) < 1e-6


def test_started_games_excluded():
    """Games already in progress are not surfaced from BetOnline."""
    payload = [{
        "id": "g",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": "2026-05-15T17:00:00Z",  # before FIXTURE_NOW (20:00Z)
        "bookmakers": [{"key": "betonlineag", "title": "BetOnline.ag", "markets": [
            {"key": "batter_home_runs", "outcomes": [
                {"name": "Over", "description": "Stale Player", "price": 500, "point": 0.5},
            ]},
        ]}],
    }]
    df = extract_sharp_anchor(payload, FIXTURE_NOW)
    assert df.empty


def test_schema_has_sharp_anchor_column():
    df = extract_sharp_anchor(PAYLOAD_NO_BOL, FIXTURE_NOW)
    assert "sharp_anchor" in df.columns


def test_fixture_payload_works_with_extract_sharp_anchor():
    """Smoke-test: the shared conftest fixture produces valid output."""
    df = extract_sharp_anchor(FIXTURE_PAYLOAD, FIXTURE_NOW)
    assert not df.empty
    assert set(df["sharp_anchor"].unique()).issubset({"pinnacle", "betonlineag"})


def test_betonline_over_only_is_excluded():
    """BetOnline Over-only lines (no matching Under) must be skipped.

    Over-only prices are one-sided novelty lines with no reliable de-vig
    signal — using vig-inclusive implied prob inflates the probability and
    generates phantom EV.  The player should be absent from the anchor df.
    """
    payload = [{
        "id": "g1",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": COMMENCE,
        "bookmakers": [{
            "key": "betonlineag",
            "title": "BetOnline.ag",
            "markets": [{"key": "batter_home_runs", "outcomes": [
                # Over only — no matching Under posted
                {"name": "Over", "description": "Bench Guy", "price": 700, "point": 0.5},
            ]}],
        }],
    }]
    df = extract_sharp_anchor(payload, FIXTURE_NOW)
    assert df.empty or "Bench Guy" not in df["player_name"].values


def test_betonline_two_sided_market_is_included():
    """BetOnline with both Over and Under should still produce an anchor row."""
    df = extract_sharp_anchor(PAYLOAD_BOL_ONLY, FIXTURE_NOW)
    # PAYLOAD_BOL_ONLY has Devers with Over +520 / Under -900
    assert "Rafael Devers" in df["player_name"].values


def test_pinnacle_over_only_still_surfaced_as_fallback():
    """Pinnacle Over-only uses the vig-inclusive fallback (not skipped).

    Pinnacle rarely omits an Under — when they do, the Over price is still
    a sharp signal and we prefer surfacing it over silently dropping it.
    """
    payload = [{
        "id": "g",
        "home_team": "Boston Red Sox",
        "away_team": "New York Yankees",
        "commence_time": COMMENCE,
        "bookmakers": [{
            "key": "pinnacle",
            "title": "Pinnacle",
            "markets": [{"key": "batter_home_runs", "outcomes": [
                {"name": "Over", "description": "Solo Over", "price": 400, "point": 0.5},
            ]}],
        }],
    }]
    df = extract_sharp_anchor(payload, FIXTURE_NOW)
    assert "Solo Over" in df["player_name"].values
    row = df[df["player_name"] == "Solo Over"].iloc[0]
    # Prob is vig-inclusive (1/5.0 = 0.20), not stripped
    assert abs(row["pinnacle_prob"] - (1 / (400 / 100 + 1))) < 1e-6
