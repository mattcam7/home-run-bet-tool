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
    # Pivot retail_df from long (one row per player/book) to wide (one row per player, books as columns)
    pivot = retail_df.pivot_table(
        index=["player_name", "game", "commence_time"],
        columns="bookmaker",
        values="american_odds",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None

    # Inner join with Pinnacle data — excludes any player not listed at Pinnacle
    merged = pivot.merge(
        pinnacle_df[["player_name", "game", "pinnacle_odds", "pinnacle_prob"]],
        on=["player_name", "game"],
        how="inner",
    )

    meta_cols = {"player_name", "game", "commence_time", "pinnacle_odds", "pinnacle_prob"}
    book_cols = [c for c in merged.columns if c not in meta_cols]

    def _best_retail(row):
        """Find the bookmaker with the highest decimal odds (best payout) for this player."""
        best_dec, best_odds = -float("inf"), None
        for col in book_cols:
            val = row[col]
            if pd.isna(val):
                continue
            dec = american_to_decimal(int(val))
            if dec > best_dec:
                best_dec, best_odds = dec, int(val)
        return pd.Series({"best_retail_odds": best_odds, "best_retail_decimal": best_dec})

    merged[["best_retail_odds", "best_retail_decimal"]] = merged.apply(_best_retail, axis=1)

    # EV formula: (pinnacle_prob × best_retail_decimal) - 1
    merged["ev_pct"] = (merged["pinnacle_prob"] * merged["best_retail_decimal"]) - 1

    # Composite score weights EV by how likely the event is (higher prob = more reliable edge)
    merged["composite_score"] = merged["ev_pct"] * merged["pinnacle_prob"]

    # Z-score using ddof=0 (population std) so mean=0 and std=1 hold exactly for any n >= 2
    mean_c = merged["composite_score"].mean()
    std_c = merged["composite_score"].std(ddof=0)
    merged["composite_z"] = (merged["composite_score"] - mean_c) / std_c

    return merged.sort_values("composite_z", ascending=False).reset_index(drop=True)
