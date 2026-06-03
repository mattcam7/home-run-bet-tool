# agents/clv_log.py
"""Closing Line Value (CLV) logging.

Two-phase capture:
  Phase 1 (log_open_plays): called from run.py on every dashboard run. Appends
    one row per surfaced play with the price we could have bet, upserting by
    (game_date, game, player_name) so re-runs the same day refresh the open
    side without clobbering an already-captured closing line.
  Phase 2 (capture_closing): called from capture_closing.py near first pitch.
    Re-fetches Pinnacle, computes the no-vig closing prob, and writes back the
    CLV plus a confirmed-lineup flag.

CLV metric: clv_pct = best_retail_decimal * closing_pinnacle_prob - 1, i.e. the
EV of the price we got measured against the sharp *closing* line. Positive
means we beat the close.

ID-matching caveat (per CLAUDE.md): OddsAPI exposes no player IDs for props, so
the lineup check matches on name strings. A plausibility guard only sets
in_lineup once lineups are actually posted; ambiguous cases stay blank rather
than emitting a false negative.
"""
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from agents.pinnacle_scraper import extract_pinnacle_odds

ET = ZoneInfo("America/New_York")
DEFAULT_PATH = "data/clv_log.csv"

OPEN_COLS = [
    "run_ts", "game_date", "commence_iso", "game", "player_name", "team",
    "best_retail_book", "best_retail_odds", "best_retail_decimal",
    "pinnacle_over_odds", "pinnacle_prob_devig", "ev_pct",
    "kelly_units", "stake_usd",
    "anchor_quality",  # "pinnacle" | "pinnacle_over_only" | "betonlineag" | "unknown"
    "featured_bet",
]
CLOSING_COLS = [
    "closing_ts", "closing_pinnacle_odds", "closing_pinnacle_prob",
    "clv_pct", "in_lineup",
]
COLUMNS = OPEN_COLS + CLOSING_COLS
KEY = ["game_date", "game", "player_name"]
# Open columns that are not part of the upsert key (KEY lives in the index
# during the upsert, so it must not be referenced as a column there).
OPEN_UPDATE_COLS = [c for c in OPEN_COLS if c not in KEY]


def _norm(name: str) -> str:
    return str(name).strip().title()


def log_open_plays(final_df: pd.DataFrame, path: str = DEFAULT_PATH, now=None) -> None:
    """Upsert the open (bet-time) side of the CLV log, keyed by game/player."""
    now = now or datetime.now(timezone.utc)
    rows = []
    for _, r in final_df.iterrows():
        ct = r["commence_time"]
        featured = (
            float(r.get("kelly_units", 0)) >= 0.5
            and float(r.get("ev_pct", 0)) >= 0.10
        )
        rows.append({
            "run_ts": now.isoformat(),
            "game_date": ct.astimezone(ET).strftime("%Y-%m-%d"),
            "commence_iso": ct.astimezone(timezone.utc).isoformat(),
            "game": r["game"],
            "player_name": _norm(r["player_name"]),
            "team": r.get("team", ""),
            "best_retail_book": r["best_retail_book"],
            "best_retail_odds": int(r["best_retail_odds"]),
            "best_retail_decimal": float(r["best_retail_decimal"]),
            "pinnacle_over_odds": int(r["pinnacle_odds"]),
            "pinnacle_prob_devig": float(r["pinnacle_prob"]),
            "ev_pct": float(r["ev_pct"]),
            "kelly_units": float(r["kelly_units"]),
            "stake_usd": float(r["stake_usd"]),
            "anchor_quality": str(r.get("anchor_quality", "unknown")),
            "featured_bet": featured,
        })
    new_idx = pd.DataFrame(rows, columns=COLUMNS).set_index(KEY)
    # Defensive dedup: if final_df somehow produced two rows for the same
    # (game_date, game, player_name), keep the last (most recent within run).
    new_idx = new_idx[~new_idx.index.duplicated(keep="last")]

    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    if os.path.exists(path):
        # dtype=object: an all-empty column (e.g. team="") would otherwise read
        # back as float64 and reject the string upsert. CSV is dtype-agnostic
        # and capture_closing re-coerces with float()/int() where it matters.
        cur = pd.read_csv(path, dtype=object).reindex(columns=COLUMNS).set_index(KEY)
        # update() only writes non-NA values, so closing columns survive.
        cur.update(new_idx[OPEN_UPDATE_COLS])
        fresh = new_idx.loc[~new_idx.index.isin(cur.index)]
        combined = pd.concat([cur, fresh]) if not fresh.empty else cur
    else:
        combined = new_idx

    combined.reset_index().reindex(columns=COLUMNS).to_csv(path, index=False)

    # Dual-write to Supabase when configured (primary store for GitHub Actions)
    if os.environ.get("SUPABASE_KEY"):
        try:
            from agents.supabase_client import insert_clv_rows
            insert_clv_rows(rows)
        except Exception as e:
            print(f"  [clv_log] Supabase write failed: {e} — CSV is the fallback")


def fetch_confirmed_lineups(date_str: str):
    """Return (set_of_normalized_player_names, lineups_posted_bool) for a date.

    Uses MLB Stats API schedule with the lineups hydration. Returns posted=False
    (and an empty set) when no game on the date has a lineup yet, so callers
    leave in_lineup blank instead of marking everyone absent.
    """
    resp = requests.get(
        "https://statsapi.mlb.com/api/v1/schedule",
        params={"sportId": 1, "date": date_str, "hydrate": "lineups"},
        timeout=15,
    )
    resp.raise_for_status()
    names: set[str] = set()
    posted = False
    for day in resp.json().get("dates", []):
        for game in day.get("games", []):
            lineups = game.get("lineups", {})
            for side in ("homePlayers", "awayPlayers"):
                players = lineups.get(side) or []
                if players:
                    posted = True
                for p in players:
                    full = p.get("fullName")
                    if full:
                        names.add(_norm(full))
    return names, posted


def _default_fetch_odds(api_key: str):
    # Lazy import to avoid a run <-> clv_log import cycle.
    from run import fetch_odds

    return lambda now: fetch_odds(api_key, now)


def capture_closing(
    api_key: str,
    now: datetime,
    path: str = DEFAULT_PATH,
    window_min: int = 30,
    fetch_odds_fn=None,
    lineup_fn=None,
) -> None:
    """Fill closing line + CLV + lineup flag for plays within window_min of first pitch.

    Idempotent: only rows with no closing line yet and a first pitch in
    (now, now + window_min] are touched. Games already started are skipped —
    Pinnacle pulls props at first pitch, so no true closing line remains.
    """
    if not os.path.exists(path):
        return
    fetch_odds_fn = fetch_odds_fn or _default_fetch_odds(api_key)
    lineup_fn = lineup_fn or fetch_confirmed_lineups

    df = pd.read_csv(path).reindex(columns=COLUMNS)
    # An all-empty CSV column reads back as float64, which rejects the string
    # timestamp / bool lineup flag we write below. Object dtype holds mixed.
    df[CLOSING_COLS] = df[CLOSING_COLS].astype(object)
    pending = df[df["closing_pinnacle_prob"].isna()]
    if pending.empty:
        return

    def _in_window(iso: str) -> bool:
        ct = datetime.fromisoformat(iso)
        delta_min = (ct - now).total_seconds() / 60.0
        return 0 < delta_min <= window_min

    targets = pending[pending["commence_iso"].apply(_in_window)]
    if targets.empty:
        return

    raw = fetch_odds_fn(now)
    pin = extract_pinnacle_odds(raw, now)
    pin_idx = (
        pin.set_index(["game", "player_name"])
        if not pin.empty
        else pd.DataFrame().set_index(pd.MultiIndex.from_arrays([[], []]))
    )

    lineup_names: set[str] = set()
    lineups_posted = False
    for d in sorted(targets["game_date"].unique()):
        names, posted = lineup_fn(str(d))
        lineup_names |= names
        lineups_posted = lineups_posted or posted

    for i, row in targets.iterrows():
        k = (row["game"], _norm(row["player_name"]))
        if k in pin_idx.index:
            match = pin_idx.loc[k]
            if isinstance(match, pd.DataFrame):  # defensive: dup key
                match = match.iloc[0]
            cp = float(match["pinnacle_prob"])
            df.at[i, "closing_ts"] = now.isoformat()
            df.at[i, "closing_pinnacle_odds"] = int(match["pinnacle_odds"])
            df.at[i, "closing_pinnacle_prob"] = cp
            df.at[i, "clv_pct"] = float(row["best_retail_decimal"]) * cp - 1
        if lineups_posted:
            df.at[i, "in_lineup"] = _norm(row["player_name"]) in lineup_names
        # lineups not posted yet -> leave in_lineup blank (guard)

    df.to_csv(path, index=False)
