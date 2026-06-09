import csv
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from agents.utils import american_to_decimal

# Pinnacle Over-only lines (no Under to devig) are vig-inclusive. When a retail
# book prices the same player above this threshold, the EV calculation is
# unreliable — the anchor prob is overstated, producing false +EV. These plays
# are excluded from the output. Root cause: May 2026 Saturday slates where
# Pinnacle posted Over-only lines at +600-900 while retail had +1000-3400 via
# alternate-market inclusion, generating 250%+ fabricated "edge".
_OVER_ONLY_MAX_RETAIL_ODDS = 600

# BetOnline-anchored plays: retail odds above this threshold signal probable
# alternate-market contamination. Retail books post batter_home_runs_alternate
# (HR vs RHP, first 5 innings, etc.) which get merged into the standard market
# feed. BOL anchors the standard market only. Any BOL-anchored play where retail
# is above +600 cannot be trusted as same-market comparison.
_BOL_MAX_RETAIL_ODDS = 600
_BOL_EXCLUDED_LOG = Path("data/ev_excluded_bol.log")


def _log_bol_excluded(rows: pd.DataFrame) -> None:
    """Append excluded BOL-anchor plays to the review log. Never silently drop."""
    _BOL_EXCLUDED_LOG.parent.mkdir(parents=True, exist_ok=True)
    write_header = not _BOL_EXCLUDED_LOG.exists()
    with open(_BOL_EXCLUDED_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "player_name", "game", "bol_anchor_odds", "best_retail_odds", "best_retail_book"])
        ts = datetime.now().isoformat(timespec="seconds")
        for _, r in rows.iterrows():
            writer.writerow([
                ts,
                r.get("player_name", ""),
                r.get("game", ""),
                r.get("pinnacle_odds", ""),
                r.get("best_retail_odds", ""),
                r.get("best_retail_book", ""),
            ])


def validate_slate(df: pd.DataFrame, label: str = "") -> None:
    """Log structured warnings when a slate looks anomalously large or skewed.

    Call this after calculate_ev() with the output DataFrame. Anomalies signal
    possible data quality issues (alternate market contamination, anchor mismatch)
    that should be inspected before trusting EV/Kelly output.
    """
    tag = f"[slate-validation{' ' + label if label else ''}]"
    n = len(df)
    odds = pd.to_numeric(df.get("best_retail_odds", pd.Series(dtype=float)), errors="coerce")
    pct_high = float((odds > 600).mean()) if len(odds) else 0.0
    over_only_n = int(df.get("over_only", pd.Series(False)).fillna(False).sum())

    warnings = []
    if n > 150:
        warnings.append(
            f"large slate: {n} plays (normal 30-100) — check for alternate market contamination"
        )
    if pct_high > 0.30:
        warnings.append(
            f"{pct_high:.0%} of plays above +600 (normal <15%) — possible anchor mismatch"
        )
    if over_only_n > 10:
        warnings.append(
            f"{over_only_n} plays with over-only Pinnacle anchor (no Under to devig)"
        )

    for w in warnings:
        print(f"  {tag} WARNING: {w}")
    if not warnings:
        print(f"  {tag} OK — {n} plays, {pct_high:.0%} above +600, {over_only_n} over-only anchors")


def calculate_ev(retail_df: pd.DataFrame, pinnacle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Expected Value (EV) for each player by comparing best retail odds
    against Pinnacle's no-vig probability.

    Args:
        retail_df: Long-format DataFrame with columns:
                   player_name, game, commence_time, bookmaker, american_odds, implied_prob
        pinnacle_df: DataFrame with columns:
                     player_name, game, commence_time, pinnacle_odds, pinnacle_prob
                     Optionally: over_only (bool), sharp_anchor (str)

    Returns:
        DataFrame with one row per player, sorted descending by composite_z.
        Columns include: player_name, game, commence_time, pinnacle_odds, pinnacle_prob,
                         best_retail_odds, best_retail_decimal, ev_pct, composite_score,
                         composite_z, anchor_quality
    """
    # The model is anchored on Pinnacle's no-vig line; with no retail or no
    # Pinnacle data there is nothing to compare, so fail loudly and clearly
    # rather than with a downstream KeyError on the empty merge.
    if retail_df.empty or pinnacle_df.empty:
        missing = []
        if pinnacle_df.empty:
            missing.append("Pinnacle (sharp reference)")
        if retail_df.empty:
            missing.append("retail")
        raise ValueError(
            "Cannot compute EV: no " + " and no ".join(missing) + " HR odds available."
        )

    # Pivot retail_df from long (one row per player/book) to wide (one row per player, books as columns)
    pivot = retail_df.pivot_table(
        index=["player_name", "game", "commence_time"],
        columns="bookmaker",
        values="american_odds",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # Inner join with sharp anchor — Pinnacle players + BetOnline fallback rows.
    # Carry over_only and sharp_anchor if present for downstream quality guards.
    anchor_cols = ["player_name", "game", "pinnacle_odds", "pinnacle_prob"]
    if "sharp_anchor" in pinnacle_df.columns:
        anchor_cols.append("sharp_anchor")
    if "over_only" in pinnacle_df.columns:
        anchor_cols.append("over_only")
    merged = pivot.merge(
        pinnacle_df[anchor_cols],
        on=["player_name", "game"],
        how="inner",
    )

    meta_cols = {
        "player_name", "game", "commence_time",
        "pinnacle_odds", "pinnacle_prob", "sharp_anchor", "over_only",
    }
    book_cols = [c for c in merged.columns if c not in meta_cols]

    def _best_retail(row):
        """Find the bookmaker with the highest decimal odds (best payout) for this player."""
        best_dec, best_odds, best_book = -float("inf"), None, None
        for col in book_cols:
            val = row[col]
            if pd.isna(val):
                continue
            dec = american_to_decimal(int(val))
            if dec > best_dec:
                best_dec, best_odds, best_book = dec, int(val), col
        return pd.Series({
            "best_retail_odds": best_odds,
            "best_retail_decimal": best_dec,
            "best_retail_book": best_book,
        })

    merged[["best_retail_odds", "best_retail_decimal", "best_retail_book"]] = merged.apply(
        _best_retail, axis=1
    )

    # --- Data quality guard: exclude over-only anchor + high retail odds ----
    # Pinnacle Over-only lines carry vig-inclusive probability (no Under to
    # strip against). When retail prices the same player above the threshold,
    # the EV formula produces false edge (e.g. 250% "EV" on bench players).
    # These plays are removed entirely rather than shown with degraded Kelly.
    if "over_only" in merged.columns:
        bad = merged["over_only"].fillna(False) & (
            merged["best_retail_odds"] > _OVER_ONLY_MAX_RETAIL_ODDS
        )
        n_bad = int(bad.sum())
        if n_bad:
            print(
                f"  [EV] Excluded {n_bad} plays: Pinnacle over-only anchor "
                f"+ retail odds > +{_OVER_ONLY_MAX_RETAIL_ODDS} (false edge guard)"
            )
            merged = merged[~bad].copy()

    # Derive anchor_quality label for CLV log and dashboard display.
    def _anchor_quality(row) -> str:
        if row.get("over_only", False):
            return "pinnacle_over_only"
        anchor = row.get("sharp_anchor", "")
        if anchor == "pinnacle":
            return "pinnacle"
        if anchor:
            return str(anchor)
        return "unknown"

    merged["anchor_quality"] = merged.apply(_anchor_quality, axis=1)

    # EV formula: (pinnacle_prob × best_retail_decimal) - 1
    merged["ev_pct"] = (merged["pinnacle_prob"] * merged["best_retail_decimal"]) - 1

    # Composite score weights EV by how likely the event is (higher prob = more reliable edge)
    merged["composite_score"] = merged["ev_pct"] * merged["pinnacle_prob"]

    # Z-score using ddof=0 (population std) so mean=0 and std=1 hold exactly for any n >= 2
    mean_c = merged["composite_score"].mean()
    std_c = merged["composite_score"].std(ddof=0)
    merged["composite_z"] = (merged["composite_score"] - mean_c) / std_c

    # Quarter-Kelly stake sizing. Kelly fraction f* = (b*p - q) / b, where
    # b = net decimal odds, p = Pinnacle no-vig prob, q = 1 - p. We bet a
    # quarter of f* (standard sharp discipline — HR props are noisy), express
    # it in units under the 1u = 1% of bankroll convention (so units =
    # fraction * 100), round to 0.5u, floor at 0, and cap at 3u. 1u = $25.
    b = merged["best_retail_decimal"] - 1
    p = merged["pinnacle_prob"]
    f_star = ((b * p) - (1 - p)) / b
    quarter_units = (f_star / 4).clip(lower=0) * 100
    merged["kelly_units"] = ((quarter_units * 2).round() / 2).clip(upper=3.0)
    merged["stake_usd"] = merged["kelly_units"] * 25

    # BOL-anchor alternate market guard: BetOnline anchors the standard
    # batter_home_runs market only. When retail prices the same player above
    # +600, the retail price likely came from batter_home_runs_alternate (which
    # covers different events — HR vs RHP, first 5 innings, etc.). Comparing
    # cross-market produces false +EV. Log and exclude per CLAUDE.md rules.
    if "sharp_anchor" in merged.columns:
        bol_bad = (
            (merged["sharp_anchor"] == "betonlineag") &
            (merged["best_retail_odds"] > _BOL_MAX_RETAIL_ODDS)
        )
        n_bol_bad = int(bol_bad.sum())
        if n_bol_bad:
            _log_bol_excluded(merged[bol_bad])
            print(
                f"  [EV] Excluded {n_bol_bad} plays: BOL anchor "
                f"+ retail odds > +{_BOL_MAX_RETAIL_ODDS} (alternate market guard)"
            )
            merged = merged[~bol_bad].copy()

    # Hard stop: over-only anchor cannot de-vig (no Under exists), so EV is
    # unreliable regardless of retail odds. Zero any Kelly that survived above.
    if "over_only" in merged.columns:
        over_only_mask = merged["over_only"].fillna(False)
        n_zeroed = int(over_only_mask.sum())
        if n_zeroed:
            merged.loc[over_only_mask, "kelly_units"] = 0.0
            merged.loc[over_only_mask, "stake_usd"] = 0.0
            print(f"  [EV] Kelly/stake zeroed for {n_zeroed} over-only anchor plays")

    return merged.sort_values("composite_z", ascending=False).reset_index(drop=True)
