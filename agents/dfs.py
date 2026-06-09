"""agents/dfs.py — DFS leverage, stack, and HR-crossover analysis.

Reads a DraftKings-style CSV of player projections (columns: Player, Pos,
Tm, Opp, Lineup, Sal, Val, Own, Pts, Own z, Pts z) and merges with the
current day's HR EV DataFrame to produce:
  - Leverage plays  (high Pts z − Own z)
  - Stack targets   (by team total points + avg leverage)
  - HR crossover    (players with both DFS data and a +EV HR prop)
  - Convergence     (high DFS leverage AND +EV HR prop)

Returns a dict with keys: leverages, stacks, crossovers, convergences, meta.
Returns None if the CSV path does not exist.

Usage:
    from agents.dfs import analyze_dfs
    dfs_data = analyze_dfs("data/dfs_projections.csv", final_df)
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from rapidfuzz import fuzz as _fuzz

    _HAS_FUZZY = True
except ImportError:
    _HAS_FUZZY = False

_FUZZY_THRESHOLD = 80
_TOP_LEVERAGE = 30
_MIN_STACK_PLAYERS = 2


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def _parse_dfs_csv(source: str | Path) -> pd.DataFrame:
    """Parse a DraftKings-style CSV, handling multi-line player/hand cells.

    Cells like ``"Jacob Misiorowski\\n(R)"`` are collapsed to just the player
    name; the handedness suffix is stripped.
    """
    if isinstance(source, Path):
        text = source.read_text(encoding="utf-8")
    else:
        text = source

    # Collapse embedded newlines inside quoted cells (e.g. "Name\n(R)" → "Name")
    def _collapse(m: re.Match) -> str:
        inner = m.group(1).replace("\n", " ").strip()
        return f'"{inner}"'

    text = re.sub(r'"([^"]*)"', _collapse, text)

    df = pd.read_csv(io.StringIO(text))

    # Normalize alternate column names (e.g. DraftKings export format)
    _COL_ALIASES = {
        "Position": "Pos",
        "Team": "Tm",
        "Opponent": "Opp",
        "Salary": "Sal",
        "Value": "Val",
        "Ownership": "Own",
        "Projected Points": "Pts",
        "Projected_Points_z": "Pts z",
        "Projected Points Z": "Pts z",
        "Ownership_z": "Own z",
        "Ownership Z": "Own z",
    }
    df = df.rename(columns={k: v for k, v in _COL_ALIASES.items() if k in df.columns})

    # Strip handedness suffix "(L)", "(R)", "(B)" from Player column
    df["Player"] = df["Player"].str.replace(r"\s*\([LRB]\)\s*$", "", regex=True).str.strip()
    # Also strip DTD / injury tags that may appear after the handedness
    df["Player"] = df["Player"].str.replace(r"\s+DTD\s*$", "", regex=True).str.strip()
    # Also clean Opp column: "MIA\n(L)" style remnants
    for col in ["Opp", "Lineup"]:
        if col in df.columns:
            df[col] = df[col].str.replace(r"\s*\([LRB]\)\s*", "", regex=True).str.strip()

    # Salary: "$11,600" → 11600 (float)
    if "Sal" in df.columns:
        df["Sal"] = (
            df["Sal"]
            .astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .apply(pd.to_numeric, errors="coerce")
        )

    # Numeric coercions
    for col in ["Own", "Pts", "Own z", "Pts z", "Val"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Fuzzy player name matching
# ---------------------------------------------------------------------------

def _fuzzy_match(name: str, candidates: list[str]) -> str | None:
    """Return the best fuzzy match for *name* among *candidates*, or None."""
    if not _HAS_FUZZY or not candidates:
        return None
    best_score = 0
    best_match = None
    nl = name.lower()
    for c in candidates:
        score = _fuzz.token_sort_ratio(nl, c.lower())
        if score > best_score:
            best_score = score
            best_match = c
    return best_match if best_score >= _FUZZY_THRESHOLD else None


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def analyze_dfs(
    csv_path: str | Path,
    hr_df: pd.DataFrame,
) -> dict[str, Any] | None:
    """Return structured DFS analysis dict, or None if CSV not found.

    Parameters
    ----------
    csv_path:
        Path to a DraftKings-style CSV (Player, Pos, Tm, Opp, Lineup, Sal,
        Val, Own, Pts, Own z, Pts z columns).
    hr_df:
        ``final_df`` from the HR pipeline — must have columns: player_name,
        team, pinnacle_prob, ev_pct, kelly_units, best_retail_odds,
        best_retail_book.
    """
    path = Path(csv_path)
    if not path.exists():
        return None

    df = _parse_dfs_csv(path)

    # ---------- Leverage score = Pts z − Own z ---------------------------
    if "Pts z" in df.columns and "Own z" in df.columns:
        df["Leverage"] = df["Pts z"] - df["Own z"]
    else:
        df["Leverage"] = float("nan")

    # ---------- Build HR lookup (player_name → HR metrics) ---------------
    hr_lookup: dict[str, dict] = {}
    if hr_df is not None and not hr_df.empty:
        for _, row in hr_df.iterrows():
            hr_lookup[str(row["player_name"]).strip()] = {
                "hr_pin_pct": round(float(row["pinnacle_prob"]) * 100, 1),
                "hr_ev_pct": round(float(row["ev_pct"]) * 100, 2),
                "hr_kelly": round(float(row.get("kelly_units", 0)), 1),
                "hr_book": str(row.get("best_retail_book", "")),
                "hr_odds": int(row.get("best_retail_odds", 0)),
            }

    hr_names = list(hr_lookup.keys())

    # ---------- Merge HR data into DFS rows ------------------------------
    def _get_hr(player: str) -> dict:
        if player in hr_lookup:
            return hr_lookup[player]
        matched = _fuzzy_match(player, hr_names)
        if matched:
            return hr_lookup[matched]
        return {"hr_pin_pct": None, "hr_ev_pct": None, "hr_kelly": None,
                "hr_book": None, "hr_odds": None}

    hr_cols = df["Player"].apply(_get_hr)
    df["hr_pin_pct"] = [r["hr_pin_pct"] for r in hr_cols]
    df["hr_ev_pct"] = [r["hr_ev_pct"] for r in hr_cols]
    df["hr_kelly"] = [r["hr_kelly"] for r in hr_cols]
    df["hr_book"] = [r["hr_book"] for r in hr_cols]
    df["hr_odds"] = [r["hr_odds"] for r in hr_cols]

    # ---------- Sub-tables -----------------------------------------------
    hitters = df[df["Pos"] != "P"].copy()
    active = hitters[hitters.get("Lineup", pd.Series([""] * len(hitters))) != "BN"].copy() if "Lineup" in hitters.columns else hitters.copy()

    # All hitters sorted by leverage (no cap — user requested full grade list)
    leverage_rows = active.sort_values("Leverage", ascending=False)
    leverages = [
        {
            "player": r["Player"],
            "pos": r.get("Pos", ""),
            "team": r.get("Tm", ""),
            "opp": r.get("Opp", ""),
            "pts": round(float(r.get("Pts", 0) or 0), 1),
            "own": round(float(r.get("Own", 0) or 0), 1),
            "sal": int((r.get("Sal") or 0)),
            "leverage": round(float(r.get("Leverage", 0) or 0), 2),
            "hr_pin_pct": r["hr_pin_pct"],
            "hr_ev_pct": r["hr_ev_pct"],
            "hr_kelly": r["hr_kelly"],
            "hr_book": r["hr_book"],
            "hr_odds": r["hr_odds"],
        }
        for _, r in leverage_rows.iterrows()
    ]

    # Stack analysis by team
    def _top3_players(grp_idx: pd.Index) -> str:
        sub = hitters.loc[grp_idx]
        top_idx = sub["Pts"].nlargest(3).index
        return ", ".join(sub.loc[top_idx, "Player"].tolist())

    stack_agg = (
        hitters.groupby("Tm")
        .apply(lambda g: pd.Series({
            "Players": len(g),
            "TotalPts": g["Pts"].sum(),
            "AvgPts": g["Pts"].mean(),
            "AvgOwn": g["Own"].mean(),
            "AvgLeverage": g["Leverage"].mean(),
            "TopPlayers": _top3_players(g.index),
        }))
        .sort_values("TotalPts", ascending=False)
        .reset_index()
    )
    stacks = [
        {
            "team": r["Tm"],
            "players": int(r["Players"]),
            "total_pts": round(float(r["TotalPts"]), 1),
            "avg_pts": round(float(r["AvgPts"]), 1),
            "avg_own": round(float(r["AvgOwn"]), 1),
            "avg_leverage": round(float(r["AvgLeverage"]), 2),
            "top_players": r["TopPlayers"],
        }
        for _, r in stack_agg.iterrows()
        if int(r["Players"]) >= _MIN_STACK_PLAYERS
    ]

    # HR crossover — active hitters that have a HR price (any EV)
    crossover_rows = active[active["hr_pin_pct"].notna()].sort_values("hr_ev_pct", ascending=False)
    crossovers = [
        {
            "player": r["Player"],
            "team": r.get("Tm", ""),
            "pos": r.get("Pos", ""),
            "pts": round(float(r.get("Pts", 0) or 0), 1),
            "leverage": round(float(r.get("Leverage", 0) or 0), 2),
            "hr_pin_pct": r["hr_pin_pct"],
            "hr_ev_pct": r["hr_ev_pct"],
            "hr_kelly": r["hr_kelly"],
            "hr_book": r["hr_book"],
            "hr_odds": int(r["hr_odds"]) if pd.notna(r["hr_odds"]) else None,
        }
        for _, r in crossover_rows.iterrows()
    ]

    # Convergence — high DFS leverage (>= 0) AND +EV HR prop (ev > 0)
    conv_score_threshold = 0.0
    conv_rows = active[
        (active["Leverage"] >= conv_score_threshold)
        & (active["hr_ev_pct"].notna())
        & (active["hr_ev_pct"] > 0)
    ].copy()
    conv_rows["conv_score"] = conv_rows["Leverage"] + conv_rows["hr_ev_pct"] / 15.0
    conv_rows = conv_rows.sort_values("conv_score", ascending=False)
    convergences = [
        {
            "player": r["Player"],
            "team": r.get("Tm", ""),
            "pos": r.get("Pos", ""),
            "pts": round(float(r.get("Pts", 0) or 0), 1),
            "leverage": round(float(r.get("Leverage", 0) or 0), 2),
            "hr_ev_pct": r["hr_ev_pct"],
            "hr_pin_pct": r["hr_pin_pct"],
            "hr_kelly": r["hr_kelly"],
            "hr_book": r["hr_book"],
            "hr_odds": int(r["hr_odds"]) if pd.notna(r["hr_odds"]) else None,
            "conv_score": round(float(r["conv_score"]), 2),
        }
        for _, r in conv_rows.iterrows()
    ]

    return {
        "leverages": leverages,
        "stacks": stacks,
        "crossovers": crossovers,
        "convergences": convergences,
        "meta": {
            "total_players": len(df),
            "active_hitters": len(active),
            "hr_matches": int(active["hr_pin_pct"].notna().sum()),
        },
    }
