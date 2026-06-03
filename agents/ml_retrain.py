"""Outcome-driven correction factors for the simulation model.

Correction formula:
    factor = actual_hr_rate / mean(pinnacle_prob_devig)
    alpha  = min(n_at_bats / 200, 0.40)
    corrected_prob = base_prob * (alpha * factor + (1 - alpha))

Artifacts:
    models/correction_factors.json   {player: {factor, n, actual_rate, predicted_rate}}
    models/retrain_log.json          [{ts, n_outcomes, n_players_updated}]
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CLV_LOG_PATH = Path("data/clv_log.csv")
CORRECTION_PATH = Path("models/correction_factors.json")
RETRAIN_LOG_PATH = Path("models/retrain_log.json")
MIN_OUTCOMES = 50
MAX_BLEND = 0.40
MIN_AT_BATS = 5


def _norm(name: str) -> str:
    return str(name).strip().title()


def compute_correction_factors(
    db_path: Path = Path("data/hr_outcomes.db"),
    clv_log_path: Path = CLV_LOG_PATH,
) -> dict[str, dict]:
    """Return {player_name: {factor, n, actual_rate, predicted_rate}}."""
    if not db_path.exists() or not clv_log_path.exists():
        return {}

    conn = sqlite3.connect(db_path)
    outcomes = pd.read_sql(
        f"""
        SELECT player_name,
               SUM(hrs_hit)  AS total_hr,
               SUM(at_bats)  AS total_ab
        FROM   outcomes
        WHERE  hit_hr IS NOT NULL AND at_bats > 0
        GROUP  BY player_name
        HAVING total_ab >= {MIN_AT_BATS}
        """,
        conn,
    )
    conn.close()

    if outcomes.empty:
        return {}

    clv = pd.read_csv(clv_log_path)
    clv["player_name"] = clv["player_name"].apply(_norm)
    clv["pinnacle_prob_devig"] = pd.to_numeric(
        clv["pinnacle_prob_devig"], errors="coerce"
    )
    pin_mean = (
        clv.groupby("player_name")["pinnacle_prob_devig"]
        .mean()
        .dropna()
    )

    factors: dict[str, dict] = {}
    for _, row in outcomes.iterrows():
        name = _norm(str(row["player_name"]))
        if name not in pin_mean.index:
            continue
        actual_rate = float(row["total_hr"]) / float(row["total_ab"])
        predicted_rate = float(pin_mean[name])
        if predicted_rate <= 0:
            continue
        factor = actual_rate / (predicted_rate + 1e-9)
        factors[name] = {
            "factor": round(factor, 4),
            "n": int(row["total_ab"]),
            "actual_rate": round(actual_rate, 4),
            "predicted_rate": round(predicted_rate, 4),
        }

    return factors


def retrain_if_ready(
    db_path: Path = Path("data/hr_outcomes.db"),
    clv_log_path: Path = CLV_LOG_PATH,
    correction_path: Path = CORRECTION_PATH,
    log_path: Path = RETRAIN_LOG_PATH,
    n_threshold: int = MIN_OUTCOMES,
) -> dict:
    """Compute and persist correction factors if enough outcomes exist."""
    try:
        if not db_path.exists():
            return {"ran": False, "reason": "no outcome DB"}

        conn = sqlite3.connect(db_path)
        n_outcomes = int(
            pd.read_sql(
                "SELECT COUNT(*) AS n FROM outcomes WHERE hit_hr IS NOT NULL", conn
            ).iloc[0]["n"]
        )
        conn.close()

        if n_outcomes < n_threshold:
            return {
                "ran": False,
                "reason": f"only {n_outcomes} outcomes, need {n_threshold}",
            }

        factors = compute_correction_factors(db_path=db_path, clv_log_path=clv_log_path)
        if not factors:
            return {"ran": False, "reason": "no correction factors computed (insufficient CLV overlap)"}

        correction_path.parent.mkdir(parents=True, exist_ok=True)
        correction_path.write_text(json.dumps(factors, indent=2))

        log_entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "n_outcomes": n_outcomes,
            "n_players_updated": len(factors),
        }
        existing = json.loads(log_path.read_text()) if log_path.exists() else []
        existing.append(log_entry)
        log_path.write_text(json.dumps(existing, indent=2))

        return {
            "ran": True,
            "n_outcomes": n_outcomes,
            "n_players_updated": len(factors),
            "reason": "success",
        }
    except Exception as exc:
        return {"ran": False, "reason": str(exc)}


def load_correction_factors(path: Path = CORRECTION_PATH) -> dict[str, dict]:
    """Load correction factors from JSON. Returns {} if not found."""
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def apply_correction(
    player_name: str,
    base_prob: float,
    correction_path: Path | None = None,
) -> float:
    """Blend correction factor into base_prob. Returns base_prob if no factor exists."""
    if correction_path is None:
        correction_path = CORRECTION_PATH
    factors = load_correction_factors(correction_path)
    name = _norm(player_name)
    if name not in factors:
        return base_prob
    entry = factors[name]
    alpha = min(entry["n"] / 200, MAX_BLEND)
    return float(base_prob * (alpha * entry["factor"] + (1.0 - alpha)))
