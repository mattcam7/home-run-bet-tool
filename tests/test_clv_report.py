import pandas as pd

from agents.clv_report import compute_metrics, format_report


def _df(rows):
    cols = [
        "player_name", "ev_pct", "kelly_units", "best_retail_book",
        "closing_pinnacle_prob", "clv_pct", "in_lineup",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_no_closing_rows_is_handled():
    df = _df([
        ["A", 0.05, 1.0, "DraftKings", None, None, None],
        ["B", -0.02, 0.0, "FanDuel", None, None, None],
    ])
    m = compute_metrics(df)
    assert m["n_logged"] == 2
    assert m["n_captured"] == 0
    assert m["n_pending"] == 2
    # Report must not crash and should say capture is pending.
    assert "no closing lines captured" in format_report(df).lower()


def test_beat_close_rate_and_segments():
    df = _df([
        # player, ev_pct, kelly, book, closing_prob, clv_pct, in_lineup
        ["A", 0.06, 1.0, "DraftKings", 0.20, 0.08, True],   # +EV pick, beat close
        ["B", 0.03, 0.5, "FanDuel", 0.15, -0.01, True],     # +EV pick, lost
        ["C", -0.04, 0.0, "FanDuel", 0.10, -0.05, False],   # negative-EV, lost
        ["D", 0.10, 2.0, "BetMGM", 0.25, 0.12, True],       # +EV pick, beat close
    ])
    m = compute_metrics(df)
    assert m["n_captured"] == 4
    assert abs(m["beat_close_rate"] - 0.5) < 1e-9          # 2 of 4 clv>0
    assert abs(m["mean_clv_pct"] - (0.08 - 0.01 - 0.05 + 0.12) / 4) < 1e-9
    # The sharpness test: +EV picks should be reported separately.
    pos = m["positive_ev"]
    assert pos["n"] == 3
    assert abs(pos["beat_close_rate"] - (2 / 3)) < 1e-9
    assert m["small_sample"] is True                        # n < 20


def test_small_sample_clears_with_enough_rows():
    rows = [["P%d" % i, 0.05, 1.0, "DraftKings", 0.2, 0.01, True] for i in range(25)]
    m = compute_metrics(_df(rows))
    assert m["n_captured"] == 25
    assert m["small_sample"] is False


def test_by_book_breakdown():
    df = _df([
        ["A", 0.06, 1.0, "DraftKings", 0.20, 0.10, True],
        ["B", 0.04, 1.0, "DraftKings", 0.15, 0.20, True],
        ["C", 0.02, 1.0, "FanDuel", 0.10, -0.05, True],
    ])
    by_book = compute_metrics(df)["by_book"]
    dk = next(b for b in by_book if b["book"] == "DraftKings")
    assert dk["n"] == 2
    assert abs(dk["mean_clv_pct"] - 0.15) < 1e-9
