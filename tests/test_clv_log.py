import os
from datetime import datetime, timedelta, timezone

import pandas as pd

from agents.clv_log import COLUMNS, capture_closing, log_open_plays

NOW = datetime(2026, 5, 15, 22, 50, tzinfo=timezone.utc)
COMMENCE = NOW + timedelta(minutes=15)  # within the 30-min closing window
GAME = "New York Yankees @ Boston Red Sox"


def _final_df(judge_odds=450, judge_dec=5.5):
    return pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": judge_odds, "best_retail_decimal": judge_dec,
        "pinnacle_odds": 380, "pinnacle_prob": 0.19,
        "ev_pct": 0.045, "kelly_units": 1.0, "stake_usd": 25.0,
    }])


def _closing_payload():
    return [{
        "id": "g1", "home_team": "Boston Red Sox", "away_team": "New York Yankees",
        "commence_time": COMMENCE.isoformat().replace("+00:00", "Z"),
        "bookmakers": [{
            "key": "pinnacle", "title": "Pinnacle",
            "markets": [{"key": "batter_home_runs", "outcomes": [
                {"name": "Over", "description": "Aaron Judge", "price": 300, "point": 0.5},
                {"name": "Under", "description": "Aaron Judge", "price": -450, "point": 0.5},
            ]}],
        }],
    }]


def test_log_open_plays_writes_expected_schema(tmp_path):
    path = str(tmp_path / "clv.csv")
    log_open_plays(_final_df(), path=path, now=NOW)
    df = pd.read_csv(path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["closing_pinnacle_prob"] != df.iloc[0]["closing_pinnacle_prob"]  # NaN


def test_log_open_plays_upserts_not_duplicates(tmp_path):
    path = str(tmp_path / "clv.csv")
    log_open_plays(_final_df(judge_odds=450), path=path, now=NOW)
    log_open_plays(_final_df(judge_odds=500), path=path, now=NOW)
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.iloc[0]["best_retail_odds"] == 500  # open side refreshed


def test_log_open_plays_preserves_existing_closing(tmp_path):
    path = str(tmp_path / "clv.csv")
    log_open_plays(_final_df(), path=path, now=NOW)
    df = pd.read_csv(path)
    df.loc[0, "closing_pinnacle_prob"] = 0.17
    df.loc[0, "clv_pct"] = -0.02
    df.to_csv(path, index=False)

    log_open_plays(_final_df(judge_odds=500), path=path, now=NOW)
    df = pd.read_csv(path)
    assert df.iloc[0]["best_retail_odds"] == 500       # open refreshed
    assert abs(df.iloc[0]["closing_pinnacle_prob"] - 0.17) < 1e-9  # closing kept


def test_capture_closing_fills_clv_and_lineup(tmp_path):
    path = str(tmp_path / "clv.csv")
    log_open_plays(_final_df(judge_dec=5.5), path=path, now=NOW)

    capture_closing(
        "k", NOW, path=path, window_min=30,
        fetch_odds_fn=lambda now: _closing_payload(),
        lineup_fn=lambda d: ({"Aaron Judge"}, True),
    )
    row = pd.read_csv(path).iloc[0]
    # Pinnacle close: Over +300 -> dec 4.0 ; Under -450 -> dec 1.2222
    over_imp = 1 / 4.0
    under_imp = 1 / (100 / 450 + 1)
    closing_prob = over_imp / (over_imp + under_imp)
    assert abs(row["closing_pinnacle_prob"] - closing_prob) < 1e-6
    assert abs(row["clv_pct"] - (5.5 * closing_prob - 1)) < 1e-6
    assert bool(row["in_lineup"]) is True


def test_capture_closing_skips_out_of_window(tmp_path):
    path = str(tmp_path / "clv.csv")
    far = datetime(2026, 5, 16, 23, 0, tzinfo=timezone.utc)
    df = _final_df()
    df.loc[0, "commence_time"] = far
    log_open_plays(df, path=path, now=NOW)

    capture_closing(
        "k", NOW, path=path, window_min=30,
        fetch_odds_fn=lambda now: _closing_payload(),
        lineup_fn=lambda d: ({"Aaron Judge"}, True),
    )
    row = pd.read_csv(path).iloc[0]
    assert row["closing_pinnacle_prob"] != row["closing_pinnacle_prob"]  # untouched NaN


def test_capture_closing_lineup_blank_when_not_posted(tmp_path):
    path = str(tmp_path / "clv.csv")
    log_open_plays(_final_df(), path=path, now=NOW)

    capture_closing(
        "k", NOW, path=path, window_min=30,
        fetch_odds_fn=lambda now: _closing_payload(),
        lineup_fn=lambda d: (set(), False),  # lineups not posted yet
    )
    row = pd.read_csv(path).iloc[0]
    assert row["closing_pinnacle_prob"] == row["closing_pinnacle_prob"]  # closing filled
    assert row["in_lineup"] != row["in_lineup"]  # in_lineup left blank (NaN)


def test_capture_closing_noop_when_no_log(tmp_path):
    path = str(tmp_path / "missing.csv")
    capture_closing("k", NOW, path=path, fetch_odds_fn=lambda now: _closing_payload())
    assert not os.path.exists(path)


def test_featured_bet_true_when_thresholds_met(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.15, "kelly_units": 0.8, "stake_usd": 20.0,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == True


def test_featured_bet_false_when_kelly_below_threshold(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.15, "kelly_units": 0.4, "stake_usd": 10.0,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == False


def test_featured_bet_false_when_ev_below_threshold(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.05, "kelly_units": 0.8, "stake_usd": 20.0,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == False


def test_featured_bet_true_at_kelly_boundary(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.15, "kelly_units": 0.5, "stake_usd": 12.5,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == True


def test_featured_bet_true_at_ev_boundary(tmp_path):
    path = str(tmp_path / "clv.csv")
    df = pd.DataFrame([{
        "player_name": "Aaron Judge", "game": GAME, "commence_time": COMMENCE,
        "team": "NYY", "best_retail_book": "DraftKings",
        "best_retail_odds": 320, "best_retail_decimal": 4.2,
        "pinnacle_odds": 300, "pinnacle_prob": 0.22,
        "ev_pct": 0.10, "kelly_units": 0.8, "stake_usd": 20.0,
        "anchor_quality": "pinnacle",
    }])
    log_open_plays(df, path=path, now=NOW)
    result = pd.read_csv(path)
    assert result.iloc[0]["featured_bet"] == True
