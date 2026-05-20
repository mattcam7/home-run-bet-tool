from datetime import datetime, timezone

import pandas as pd

from agents.bets import (
    BET_COLUMNS,
    add_bet,
    fetch_box_scores_from_payload,
    format_ledger,
    ledger,
    settle,
)

NOW = datetime(2026, 5, 19, 23, 0, tzinfo=timezone.utc)
GAME_DATE = "2026-05-19"
GAME = "Houston Astros @ Texas Rangers"


def _seed_clv_log(path, *, closing_prob=0.20, with_closing=True):
    cols = [
        "run_ts", "game_date", "commence_iso", "game", "player_name", "team",
        "best_retail_book", "best_retail_odds", "best_retail_decimal",
        "pinnacle_over_odds", "pinnacle_prob_devig", "ev_pct",
        "kelly_units", "stake_usd",
        "closing_ts", "closing_pinnacle_odds", "closing_pinnacle_prob",
        "clv_pct", "in_lineup",
    ]
    row = dict.fromkeys(cols)
    row.update({
        "run_ts": "2026-05-19T19:00:00+00:00",
        "game_date": GAME_DATE,
        "commence_iso": "2026-05-19T23:30:00+00:00",
        "game": GAME, "player_name": "Yordan Alvarez", "team": "HOU",
        "best_retail_book": "BetMGM",
        "best_retail_odds": 333, "best_retail_decimal": 4.33,
        "pinnacle_over_odds": 300, "pinnacle_prob_devig": 0.265,
        "ev_pct": 0.147, "kelly_units": 1.0, "stake_usd": 25.0,
    })
    if with_closing:
        row["closing_pinnacle_prob"] = closing_prob
    pd.DataFrame([row], columns=cols).to_csv(path, index=False)


def test_add_bet_snapshots_model_view(tmp_path):
    log_p = str(tmp_path / "clv.csv"); bets_p = str(tmp_path / "bets.csv")
    _seed_clv_log(log_p, with_closing=False)
    rec = add_bet("Yordan Alvarez", "BetMGM", 340, 1.0,
                  log_path=log_p, bets_path=bets_p, now=NOW)
    df = pd.read_csv(bets_p)
    assert list(df.columns) == BET_COLUMNS
    assert len(df) == 1
    r = df.iloc[0]
    assert r["player_name"] == "Yordan Alvarez"
    assert r["odds_american"] == 340
    assert abs(r["odds_decimal"] - 4.4) < 1e-9
    assert r["stake_usd"] == 25.0
    # Snapshot of what the model thought *at placement*:
    assert abs(r["model_pinnacle_prob_devig"] - 0.265) < 1e-9
    assert abs(r["model_ev_pct"] - 0.147) < 1e-9
    assert r["outcome"] == "PENDING"
    assert rec["bet_id"] == r["bet_id"]


def test_add_bet_rejects_unknown_player(tmp_path):
    log_p = str(tmp_path / "clv.csv"); bets_p = str(tmp_path / "bets.csv")
    _seed_clv_log(log_p)
    try:
        add_bet("Ghost Player", "BetMGM", 333, 1.0,
                log_path=log_p, bets_path=bets_p, now=NOW)
    except ValueError as e:
        assert "Ghost Player" in str(e)
    else:
        raise AssertionError("expected ValueError for unknown player")


def test_settle_auto_marks_win_and_joins_clv(tmp_path):
    log_p = str(tmp_path / "clv.csv"); bets_p = str(tmp_path / "bets.csv")
    _seed_clv_log(log_p, closing_prob=0.20)
    add_bet("Yordan Alvarez", "BetMGM", 340, 1.0,
            log_path=log_p, bets_path=bets_p, now=NOW)

    result = settle(
        GAME_DATE,
        fetch_box_fn=lambda d: {"Yordan Alvarez": 1},
        prompt_fn=lambda msg: "y",
        log_path=log_p, bets_path=bets_p, now=NOW,
    )
    assert result["updated"] == 1
    df = pd.read_csv(bets_p); r = df.iloc[0]
    assert r["outcome"] == "WIN"
    # Payout = stake * (decimal - 1). +340 -> dec 4.4 -> profit 3.4 * 25 = 85.
    assert abs(r["payout_usd"] - 85.0) < 1e-9
    assert abs(r["closing_pinnacle_prob"] - 0.20) < 1e-9
    # actual_clv_pct = odds_decimal * closing_prob - 1 = 4.4 * 0.20 - 1 = -0.12
    assert abs(r["actual_clv_pct"] - (-0.12)) < 1e-9


def test_settle_auto_marks_loss(tmp_path):
    log_p = str(tmp_path / "clv.csv"); bets_p = str(tmp_path / "bets.csv")
    _seed_clv_log(log_p)
    add_bet("Yordan Alvarez", "BetMGM", 333, 1.0,
            log_path=log_p, bets_path=bets_p, now=NOW)
    settle(GAME_DATE,
           fetch_box_fn=lambda d: {"Yordan Alvarez": 0},
           prompt_fn=lambda msg: "y",
           log_path=log_p, bets_path=bets_p, now=NOW)
    r = pd.read_csv(bets_p).iloc[0]
    assert r["outcome"] == "LOSS"
    assert r["payout_usd"] == -25.0


def test_settle_leaves_pending_when_player_not_in_box(tmp_path):
    """Plausibility guard: a name we can't find in the box score stays
    PENDING rather than being silently marked LOSS."""
    log_p = str(tmp_path / "clv.csv"); bets_p = str(tmp_path / "bets.csv")
    _seed_clv_log(log_p)
    add_bet("Yordan Alvarez", "BetMGM", 333, 1.0,
            log_path=log_p, bets_path=bets_p, now=NOW)
    settle(GAME_DATE,
           fetch_box_fn=lambda d: {},  # empty box
           prompt_fn=lambda msg: "",   # accept prefill
           log_path=log_p, bets_path=bets_p, now=NOW)
    r = pd.read_csv(bets_p).iloc[0]
    assert r["outcome"] == "PENDING"
    assert pd.isna(r["payout_usd"])


def test_settle_respects_user_override(tmp_path):
    log_p = str(tmp_path / "clv.csv"); bets_p = str(tmp_path / "bets.csv")
    _seed_clv_log(log_p)
    add_bet("Yordan Alvarez", "BetMGM", 333, 1.0,
            log_path=log_p, bets_path=bets_p, now=NOW)
    # Auto pre-fills WIN but user types 'void' (rain delay etc.)
    settle(GAME_DATE,
           fetch_box_fn=lambda d: {"Yordan Alvarez": 1},
           prompt_fn=lambda msg: "void",
           log_path=log_p, bets_path=bets_p, now=NOW)
    r = pd.read_csv(bets_p).iloc[0]
    assert r["outcome"] == "VOID"
    assert r["payout_usd"] == 0.0


def test_settle_is_idempotent(tmp_path):
    log_p = str(tmp_path / "clv.csv"); bets_p = str(tmp_path / "bets.csv")
    _seed_clv_log(log_p)
    add_bet("Yordan Alvarez", "BetMGM", 333, 1.0,
            log_path=log_p, bets_path=bets_p, now=NOW)
    settle(GAME_DATE, fetch_box_fn=lambda d: {"Yordan Alvarez": 1},
           prompt_fn=lambda msg: "y", log_path=log_p, bets_path=bets_p, now=NOW)
    # Second settle pass: nothing pending, no double-credit.
    res = settle(GAME_DATE, fetch_box_fn=lambda d: {"Yordan Alvarez": 1},
                 prompt_fn=lambda msg: "y", log_path=log_p, bets_path=bets_p, now=NOW)
    assert res["updated"] == 0


def test_ledger_metrics(tmp_path):
    bets_p = str(tmp_path / "bets.csv")
    rows = []
    cols = BET_COLUMNS
    def _row(outcome, stake, payout, model_ev, actual_clv):
        d = dict.fromkeys(cols)
        d.update({
            "bet_id":"x", "placed_ts":"t", "game_date":"d", "game":"g",
            "player_name":"P", "team":"T", "sportsbook":"B",
            "odds_american":200, "odds_decimal":3.0,
            "stake_units":1.0, "stake_usd":stake,
            "model_pinnacle_prob_devig":0.3, "model_ev_pct":model_ev,
            "model_stake_units":1.0,
            "outcome":outcome, "payout_usd":payout,
            "actual_clv_pct":actual_clv,
        })
        return d
    rows = [
        _row("WIN", 25, 50.0, 0.10, 0.08),
        _row("LOSS", 25, -25.0, 0.05, -0.02),
        _row("WIN", 25, 75.0, 0.15, 0.12),
        _row("PENDING", 25, None, 0.08, None),
    ]
    pd.DataFrame(rows, columns=cols).to_csv(bets_p, index=False)
    m = ledger(bets_p=bets_p)
    assert m["n_bets"] == 4
    assert m["n_settled"] == 3
    assert m["n_pending"] == 1
    assert abs(m["win_rate"] - (2/3)) < 1e-9
    assert m["total_stake"] == 75.0
    assert m["net_pnl"] == 100.0
    assert abs(m["roi_pct"] - (100/75)) < 1e-9
    assert m["small_sample"] is True
    # Mean actual CLV across the 3 settled with closing data
    assert abs(m["mean_actual_clv_pct"] - ((0.08 - 0.02 + 0.12)/3)) < 1e-9
    # Format must not crash
    assert "ROI" in format_ledger(bets_p=bets_p)


def test_format_ledger_handles_empty(tmp_path):
    bets_p = str(tmp_path / "bets.csv")
    pd.DataFrame(columns=BET_COLUMNS).to_csv(bets_p, index=False)
    out = format_ledger(bets_p=bets_p)
    assert "no bets" in out.lower()


def test_box_score_parser_extracts_hr_by_player():
    payload = {
        "dates": [{
            "games": [{
                "boxscore": {
                    "teams": {
                        "home": {"players": {
                            "ID660271": {
                                "person": {"fullName": "Shohei Ohtani"},
                                "stats": {"batting": {"homeRuns": 2}},
                            },
                            "ID12345": {
                                "person": {"fullName": "Mookie Betts"},
                                "stats": {"batting": {"homeRuns": 0}},
                            },
                        }},
                        "away": {"players": {
                            "ID111": {
                                "person": {"fullName": "Aaron Judge"},
                                "stats": {"batting": {"homeRuns": 1}},
                            },
                        }},
                    }
                }
            }]
        }]
    }
    box = fetch_box_scores_from_payload(payload)
    assert box["Shohei Ohtani"] == 2
    assert box["Mookie Betts"] == 0
    assert box["Aaron Judge"] == 1
