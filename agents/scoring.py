"""Bet quality scoring: z-score composite mapped to 0-100 via sigmoid.

Formula:
  composite = ev_pct * pinnacle_prob * anchor_weight * sim_multiplier
  composite_z = (composite - mean) / std   [population std, ddof=0]
  bet_score = round(100 / (1 + exp(-k * composite_z)))

k=1.4 calibrates so z=+1 sigma -> bet_score ~80 (top ~16% of slate = "Strong").

sim_multiplier: when sim_prob is present and sim_edge < -0.05, the composite is
dampened. Penalty grows linearly from 0% at -5pp to 40% at -15pp+ divergence.
No boost for positive sim_edge — Pinnacle is the sharp anchor.
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
# Sim dampener constants
_SIM_PENALTY_START = 0.05   # sim_edge below -0.05 starts penalty
_SIM_PENALTY_MAX   = 0.40   # max 40% reduction at -0.15+ divergence
_SIM_PENALTY_RANGE = 0.10   # penalty grows over this range of divergence


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


def _sim_multiplier(sim_edge) -> float:
    """Compute composite dampener from sim_edge (sim_prob - pinnacle_prob).

    Returns a value in (0.60, 1.0]. Only penalises — never boosts.
    Returns 1.0 when sim_edge is missing or above the penalty threshold.
    """
    try:
        edge = float(sim_edge)
    except (TypeError, ValueError):
        return 1.0
    if edge >= -_SIM_PENALTY_START:
        return 1.0
    penalty = min(_SIM_PENALTY_MAX, (abs(edge) - _SIM_PENALTY_START) / _SIM_PENALTY_RANGE * _SIM_PENALTY_MAX)
    return 1.0 - penalty


def compute_bet_score(df: pd.DataFrame) -> pd.DataFrame:
    """Add bet_score (0-100 int) and bet_grade (str) columns. Returns a copy.

    Requires sim_prob and pinnacle_prob columns to apply the sim divergence
    dampener. When sim_prob is absent or NaN the dampener is a no-op.
    """
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

    # Apply sim divergence dampener when sim columns are present
    if "sim_prob" in df.columns:
        sim_prob = pd.to_numeric(df["sim_prob"], errors="coerce")
        sim_edge = sim_prob - prob  # negative = sim more bearish than Pinnacle
        composite = composite * sim_edge.apply(_sim_multiplier)

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
