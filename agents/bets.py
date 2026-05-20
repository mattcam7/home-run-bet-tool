# agents/bets.py
"""Bet tracking + settlement.

Closes the feedback loop: which plays you *actually* bet, at what real price
(captures slippage), settled hit/miss, your personal ROI and your actual CLV
versus what the model predicted. The data is independently valuable on either
model outcome -- the only piece of the actionable layer worth building before
the CLV gate closes.

Settlement is auto-with-manual-confirm: a single MLB Stats API call retrieves
HR counts per player for the date, pre-fills WIN/LOSS, and asks you to confirm.
A bet with no matching box-score entry stays PENDING (plausibility guard per
CLAUDE.md: never silently mark a LOSS on a name we couldn't resolve).
"""
import argparse
import os
import uuid
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from agents.clv_log import DEFAULT_PATH as CLV_DEFAULT_PATH
from agents.utils import american_to_decimal

ET = ZoneInfo("America/New_York")
DEFAULT_BETS_PATH = "data/bets.csv"
UNIT_USD = 25.0  # 1 unit = $25 (1% of a $2,500 bankroll, per agreed convention)

BET_COLUMNS = [
    "bet_id", "placed_ts", "game_date", "game", "player_name", "team",
    "sportsbook", "odds_american", "odds_decimal",
    "stake_units", "stake_usd",
    "model_pinnacle_prob_devig", "model_ev_pct", "model_stake_units",
    "settled_ts", "outcome", "payout_usd",
    "closing_pinnacle_prob", "actual_clv_pct",
]


def _norm(name: str) -> str:
    return str(name).strip().title()


_TEXT_COLS = (
    "bet_id", "placed_ts", "game_date", "game", "player_name", "team",
    "sportsbook", "settled_ts", "outcome",
)


def _load(path: str) -> pd.DataFrame:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        df = pd.read_csv(path).reindex(columns=BET_COLUMNS)
    else:
        df = pd.DataFrame(columns=BET_COLUMNS)
    # A freshly-seeded CSV may have an all-empty text column that read_csv
    # infers as float64; assigning a string into that later raises TypeError.
    # Object dtype holds mixed and round-trips correctly.
    for c in _TEXT_COLS:
        df[c] = df[c].astype(object)
    return df


def _save(df: pd.DataFrame, path: str) -> None:
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    df.reindex(columns=BET_COLUMNS).to_csv(path, index=False)


# ----------------------------------------------------------------------- add

def add_bet(player_name, sportsbook, odds_american, stake_units, *,
            log_path=CLV_DEFAULT_PATH, bets_path=DEFAULT_BETS_PATH, now=None):
    """Record a placed bet, snapshotting the model's view at this moment.

    Looks the player up in the CLV log (most recent run_ts wins) so the bet
    inherits the same game/team/closing-line lineage. Raises ValueError if the
    player isn't on today's logged board -- a typo guard.
    """
    now = now or datetime.now(timezone.utc)
    odds_american = int(odds_american)
    odds_decimal = american_to_decimal(odds_american)
    stake_units = float(stake_units)
    stake_usd = stake_units * UNIT_USD

    log = pd.read_csv(log_path)
    log = log.copy()
    log["_pn"] = log["player_name"].apply(_norm)
    pn = _norm(player_name)
    today_date = now.astimezone(ET).strftime("%Y-%m-%d")
    match = log[(log["game_date"] == today_date) & (log["_pn"] == pn)]
    if match.empty:
        # Fallback: any date (in case the user is recording for a later slate).
        match = log[log["_pn"] == pn]
    if match.empty:
        raise ValueError(f"No log entry for {player_name}")
    snap = match.sort_values("run_ts").iloc[-1]

    rec = {c: None for c in BET_COLUMNS}
    rec.update({
        "bet_id": uuid.uuid4().hex[:8],
        "placed_ts": now.isoformat(),
        "game_date": snap["game_date"],
        "game": snap["game"],
        "player_name": pn,
        "team": snap.get("team", "") if pd.notna(snap.get("team")) else "",
        "sportsbook": sportsbook,
        "odds_american": odds_american,
        "odds_decimal": odds_decimal,
        "stake_units": stake_units,
        "stake_usd": stake_usd,
        "model_pinnacle_prob_devig": float(snap["pinnacle_prob_devig"]),
        "model_ev_pct": float(snap["ev_pct"]),
        "model_stake_units": float(snap["kelly_units"]),
        "outcome": "PENDING",
    })

    df = _load(bets_path)
    df = pd.concat([df, pd.DataFrame([rec], columns=BET_COLUMNS)], ignore_index=True)
    _save(df, bets_path)
    return rec


# -------------------------------------------------------------------- settle

def _resolve_outcome(resp: str, prefill):
    """Map user input + box-score pre-fill to a final outcome (or None=skip)."""
    resp = (resp or "").strip().lower()
    if resp == "skip" or resp == "n":
        return None
    if resp in ("", "y"):
        return prefill  # may be None -> stays PENDING
    if resp == "win":
        return "WIN"
    if resp == "loss":
        return "LOSS"
    if resp == "void":
        return "VOID"
    return None


def _render_prompt(bet, prefill):
    pre = f"[prefill: {prefill}]" if prefill else "[no box-score match]"
    odds = int(bet["odds_american"])
    return (f"{bet['player_name']} @ {bet['sportsbook']} ({odds:+d}, "
            f"{bet['stake_units']}u) {pre} - win/loss/void/y/n/skip: ")


def settle(game_date, *, fetch_box_fn=None, prompt_fn=input,
           log_path=CLV_DEFAULT_PATH, bets_path=DEFAULT_BETS_PATH, now=None):
    """Settle pending bets for a date. Auto-pre-fills from MLB box scores,
    asks for confirmation, joins CLV log for closing line and actual CLV.

    Args:
        fetch_box_fn: callable(date_str) -> {normalized_name: hr_count}. Defaults
            to a live MLB Stats API call. Inject in tests / for --manual mode
            pass a lambda returning {}.
        prompt_fn: callable(message) -> user response string. Defaults to input.
    """
    now = now or datetime.now(timezone.utc)
    df = _load(bets_path)
    if df.empty:
        return {"updated": 0}

    pending_mask = (df["game_date"].astype(str) == str(game_date)) & \
                   (df["outcome"] == "PENDING")
    if not pending_mask.any():
        return {"updated": 0}

    if fetch_box_fn is None:
        fetch_box_fn = _fetch_box_scores_via_api
    box = {_norm(k): int(v) for k, v in (fetch_box_fn(game_date) or {}).items()}

    log = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
    if not log.empty:
        log = log.copy()
        log["_pn"] = log["player_name"].apply(_norm)

    updated = 0
    for i in df.index[pending_mask]:
        bet = df.loc[i]
        pn = _norm(bet["player_name"])
        prefill = None
        if pn in box:
            prefill = "WIN" if box[pn] >= 1 else "LOSS"

        resp = prompt_fn(_render_prompt(bet, prefill))
        outcome = _resolve_outcome(resp, prefill)
        if outcome is None:
            continue  # stays PENDING

        df.at[i, "outcome"] = outcome
        df.at[i, "settled_ts"] = now.isoformat()
        stake = float(bet["stake_usd"])
        dec = float(bet["odds_decimal"])
        if outcome == "WIN":
            df.at[i, "payout_usd"] = stake * (dec - 1)
        elif outcome == "LOSS":
            df.at[i, "payout_usd"] = -stake
        else:  # VOID
            df.at[i, "payout_usd"] = 0.0

        # Join the CLV log for closing line + actual personal CLV.
        if not log.empty:
            lm = log[(log["game_date"] == bet["game_date"]) &
                     (log["game"] == bet["game"]) &
                     (log["_pn"] == pn)]
            if not lm.empty:
                cp = lm.iloc[-1].get("closing_pinnacle_prob")
                if pd.notna(cp):
                    cp = float(cp)
                    df.at[i, "closing_pinnacle_prob"] = cp
                    df.at[i, "actual_clv_pct"] = dec * cp - 1
        updated += 1

    _save(df, bets_path)
    return {"updated": updated}


# ----------------------------------------------------------------- box scores

def fetch_box_scores_from_payload(payload: dict) -> dict:
    """Parse MLB Stats API schedule+boxscore JSON to {normalized_name: hr_count}.

    Players without a homeRuns field in the boxscore are skipped (not assumed
    zero) so we never spuriously mark a LOSS for someone whose stats line is
    incomplete.
    """
    hrs: dict[str, int] = {}
    for day in payload.get("dates", []) or []:
        for game in day.get("games", []) or []:
            box = game.get("boxscore") or {}
            teams = box.get("teams") or {}
            for side in ("home", "away"):
                t = teams.get(side) or {}
                for _, pdata in (t.get("players") or {}).items():
                    person = pdata.get("person") or {}
                    name = _norm(person.get("fullName") or "")
                    bat = (pdata.get("stats") or {}).get("batting") or {}
                    if "homeRuns" in bat and name:
                        hrs[name] = int(bat["homeRuns"])
    return hrs


def _fetch_box_scores_via_api(date_str: str) -> dict:
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": date_str, "hydrate": "boxscore"},
        timeout=15,
    )
    resp.raise_for_status()
    return fetch_box_scores_from_payload(resp.json())


# ----------------------------------------------------------------- ledger

def ledger(bets_p: str = DEFAULT_BETS_PATH) -> dict:
    """Headline metrics over the bets ledger. Small-sample guarded."""
    df = _load(bets_p)
    res = {"n_bets": int(len(df)), "n_settled": 0, "n_pending": 0,
           "small_sample": True}
    if df.empty:
        return res
    settled = df[df["outcome"].isin(["WIN", "LOSS"])]
    res["n_settled"] = int(len(settled))
    res["n_pending"] = int((df["outcome"] == "PENDING").sum())
    res["small_sample"] = res["n_settled"] < 20
    if settled.empty:
        return res

    stake = pd.to_numeric(settled["stake_usd"], errors="coerce").fillna(0)
    payout = pd.to_numeric(settled["payout_usd"], errors="coerce").fillna(0)
    total_stake = float(stake.sum())
    net_pnl = float(payout.sum())
    res["win_rate"] = float((settled["outcome"] == "WIN").mean())
    res["total_stake"] = total_stake
    res["net_pnl"] = net_pnl
    res["roi_pct"] = net_pnl / total_stake if total_stake else 0.0

    clv = pd.to_numeric(settled["actual_clv_pct"], errors="coerce").dropna()
    if len(clv):
        res["mean_actual_clv_pct"] = float(clv.mean())
        ev = pd.to_numeric(
            settled.loc[clv.index, "model_ev_pct"], errors="coerce"
        ).dropna()
        if len(ev):
            res["mean_model_ev_pct"] = float(ev.mean())
    return res


def format_ledger(bets_p: str = DEFAULT_BETS_PATH) -> str:
    m = ledger(bets_p=bets_p)
    if m["n_bets"] == 0:
        return "No bets logged yet."
    L = ["=" * 56, "BET LEDGER", "=" * 56,
         f"Total: {m['n_bets']}  Settled: {m['n_settled']}  "
         f"Pending: {m['n_pending']}"]
    if m["n_settled"] == 0:
        L += ["(no settled bets yet)", "=" * 56]
        return "\n".join(L)
    if m["small_sample"]:
        L += ["", "!! INSUFFICIENT SAMPLE (<20 settled) - directional only !!"]
    L += [
        "",
        f"Win rate : {m['win_rate'] * 100:.1f}%",
        f"Stake    : ${m['total_stake']:,.2f}",
        f"Net P&L  : ${m['net_pnl']:+,.2f}",
        f"ROI      : {m['roi_pct'] * 100:+.2f}%",
    ]
    if "mean_actual_clv_pct" in m:
        L += [
            "",
            f"Mean actual CLV : {m['mean_actual_clv_pct'] * 100:+.2f}%",
            f"Mean model EV%  : {m.get('mean_model_ev_pct', 0.0) * 100:+.2f}%",
        ]
    L.append("=" * 56)
    return "\n".join(L)


# --------------------------------------------------------------------- CLI

def _print_pending(bets_path: str = DEFAULT_BETS_PATH) -> None:
    df = _load(bets_path)
    pend = df[df["outcome"] == "PENDING"]
    if pend.empty:
        print("No pending bets.")
        return
    for _, r in pend.iterrows():
        print(f"  {r['placed_ts']}  {r['player_name']:<22s} "
              f"{r['sportsbook']:<12s} {int(r['odds_american']):+d}  "
              f"{r['stake_units']}u  bet_id={r['bet_id']}")


def _add_interactive(log_path: str = CLV_DEFAULT_PATH,
                     bets_path: str = DEFAULT_BETS_PATH) -> None:
    log = pd.read_csv(log_path)
    today = datetime.now(ET).strftime("%Y-%m-%d")
    today_plays = log[log["game_date"] == today].drop_duplicates("player_name")
    pos = today_plays[today_plays["ev_pct"] > 0].sort_values("ev_pct", ascending=False).head(20)
    if pos.empty:
        print(f"No +EV plays in the log for {today}. Run `python run.py` first.")
        return
    print(f"\nToday's +EV plays ({today}):\n")
    for i, (_, r) in enumerate(pos.iterrows(), 1):
        print(f"  {i:2d}. {r['player_name']:<22s} "
              f"{int(r['best_retail_odds']):+5d} {r['best_retail_book']:<12s}  "
              f"EV {r['ev_pct'] * 100:+.1f}%  Pin {r['pinnacle_prob_devig'] * 100:.1f}%  "
              f"rec {r['kelly_units']}u")
    sel = input("\nWhich one? (number or 'q'): ").strip().lower()
    if sel == "q" or not sel.isdigit():
        return
    n = int(sel)
    if not (1 <= n <= len(pos)):
        print("invalid selection")
        return
    play = pos.iloc[n - 1]
    book_default = play["best_retail_book"]
    book = input(f"Sportsbook [{book_default}]: ").strip() or book_default
    odds_default = int(play["best_retail_odds"])
    odds_in = input(f"Odds you got [{odds_default}]: ").strip()
    odds = int(odds_in) if odds_in else odds_default
    units_default = float(play["kelly_units"])
    units_in = input(f"Stake units [{units_default}]: ").strip()
    units = float(units_in) if units_in else units_default
    rec = add_bet(play["player_name"], book, odds, units,
                  log_path=log_path, bets_path=bets_path)
    print(f"\nOK  {rec['player_name']} {odds:+d} @ {book}, {units}u "
          f"(${rec['stake_usd']:.2f}). bet_id={rec['bet_id']}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="bets")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("add", help="interactively log a placed bet")
    sub.add_parser("pending", help="list pending bets")
    s = sub.add_parser("settle", help="settle pending bets for a date")
    s.add_argument("date", help="YYYY-MM-DD")
    s.add_argument("--manual", action="store_true",
                   help="skip MLB API; ask for each outcome manually")
    sub.add_parser("ledger", help="ROI / CLV summary")
    args = p.parse_args(argv)

    if args.cmd == "add":
        _add_interactive()
    elif args.cmd == "pending":
        _print_pending()
    elif args.cmd == "settle":
        fetch_box_fn = (lambda d: {}) if args.manual else None
        r = settle(args.date, fetch_box_fn=fetch_box_fn)
        print(f"settled: {r['updated']} bet(s).")
    elif args.cmd == "ledger":
        print(format_ledger())


if __name__ == "__main__":
    main()
