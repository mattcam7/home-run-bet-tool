import pandas as pd
from agents.utils import american_to_decimal


def calculate_ev(retail_df: pd.DataFrame, pinnacle_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Expected Value (EV) for each player by comparing best retail odds
    against Pinnacle's no-vig probability.

    Args:
        retail_df: Long-format DataFrame with columns:
                   player_name, game, commence_time, bookmaker, american_odds, implied_prob
        pinnacle_df: DataFrame with columns:
                     player_name, game, commence_time, pinnacle_odds, pinnacle_prob

    Returns:
        DataFrame with one row per player, sorted descending by composite_z.
        Columns include: player_name, game, commence_time, pinnacle_odds, pinnacle_prob,
                         best_retail_odds, best_retail_decimal, ev_pct, composite_score, composite_z
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
    # sharp_anchor column ("pinnacle" or "betonlineag") is carried through if
    # present so the dashboard and parlay generator can flag the confidence tier.
    anchor_cols = ["player_name", "game", "pinnacle_odds", "pinnacle_prob"]
    if "sharp_anchor" in pinnacle_df.columns:
        anchor_cols.append("sharp_anchor")
    merged = pivot.merge(
        pinnacle_df[anchor_cols],
        on=["player_name", "game"],
        how="inner",
    )

    meta_cols = {"player_name", "game", "commence_time", "pinnacle_odds", "pinnacle_prob", "sharp_anchor"}
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

    return merged.sort_values("composite_z", ascending=False).reset_index(drop=True)
