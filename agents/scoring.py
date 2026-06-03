"""Bet quality scoring: z-score composite mapped to 0-100 via sigmoid.

Formula:
  composite = ev_pct * pinnacle_prob * anchor_weight
  composite_z = (composite - mean) / std   [population std, ddof=0]
  bet_score = round(100 / (1 + exp(-k * composite_z)))

k=1.4 calibrates so z=+1 sigma -> bet_score ~80 (top ~16% of slate = "Strong").
"""
from __future__ import annotations

import math

import pandas as pd

_ANCHOR_WEIGHTS: dict[str, float] = {
    "pinnacle":           1.00,
    "betonlineag":        0.75,
    "unknown":            0.55,
    "pinnacle_over_only": 0.00,
}

_SIGMOID_K = 1.4


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-_SIGMOID_K * z))


def _grade(score: int) -> str:
    if score >= 80:
        return "Strong"
    if score >= 60:
        return "Solid"
    if score >= 40:
        return "Marginal"
    return "Skip"


def compute_bet_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add bet_score (0-100 int) and bet_grade (str) columns. Returns a copy."""
    df = df.copy()

    ev = pd.to_numeric(
        df.get("ev_pct", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    prob = pd.to_numeric(
        df.get("pinnacle_prob", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    anchor_q = df.get(
        "anchor_quality", pd.Series("unknown", index=df.index)
    ).fillna("unknown")

    anchor_w = anchor_q.map(_ANCHOR_WEIGHTS).fillna(_ANCHOR_WEIGHTS["unknown"])
    composite = ev * prob * anchor_w

    over_only_mask = anchor_q.isin(["pinnacle_over_only"]) | df.get(
        "over_only", pd.Series(False, index=df.index)
    ).fillna(False).astype(bool)

    std = composite.std(ddof=0)
    if std == 0 or pd.isna(std):
        # Uniform slate — every play scores 50
        df["bet_score"] = 50
        df["bet_grade"] = "Marginal"
        df.loc[over_only_mask, "bet_score"] = 0
        df.loc[over_only_mask, "bet_grade"] = "Skip"
        return df

    z = (composite - composite.mean()) / std
    df["bet_score"] = z.apply(lambda v: round(_sigmoid(v) * 100))
    df["bet_grade"] = df["bet_score"].apply(_grade)

    # over_only / pinnacle_over_only plays are always excluded — force score to 0
    df.loc[over_only_mask, "bet_score"] = 0
    df.loc[over_only_mask, "bet_grade"] = "Skip"
    return df
