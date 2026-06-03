"""Per-step validation and quarantine for the WAT pipeline."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

QUARANTINE_PATH = "data/quarantine.jsonl"


@dataclass
class StepResult:
    clean: Any
    quarantined: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True


def append_quarantine(rows: list[dict], path: str = QUARANTINE_PATH) -> None:
    """Append quarantine entries to the JSONL log (one object per line)."""
    if not rows:
        return
    dirname = os.path.dirname(path)
    if dirname:
        os.makedirs(dirname, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(path, "a", encoding="utf-8") as f:
        for row in rows:
            row.setdefault("ts", ts)
            f.write(json.dumps(row) + "\n")


def validate_raw_odds(events: list[dict]) -> StepResult:
    """Validate raw OddsAPI event list before scraper processing."""
    if not events:
        return StepResult(
            clean=[],
            warnings=["No events returned from OddsAPI"],
            ok=False,
        )

    quarantined: list[dict] = []
    clean: list[dict] = []

    for event in events:
        if "id" not in event:
            quarantined.append({
                "step": "raw_validation",
                "reason": "missing_id",
                "event_preview": str(event)[:120],
            })
            continue
        if "bookmakers" not in event:
            quarantined.append({
                "step": "raw_validation",
                "reason": "missing_bookmakers",
                "event_id": event.get("id", "unknown"),
            })
            continue
        clean.append(event)

    warnings = []
    if quarantined:
        warnings.append(
            f"Quarantined {len(quarantined)} malformed events "
            f"({len(clean)} clean)"
        )

    return StepResult(clean=clean, quarantined=quarantined, warnings=warnings, ok=bool(clean))


def validate_ev_output(df: pd.DataFrame) -> StepResult:
    """Validate calculate_ev() output: drop impossible rows, warn on slate anomalies."""
    if df.empty:
        return StepResult(clean=df, warnings=["EV output is empty"], ok=False)

    df = df.copy()
    quarantined: list[dict] = []
    warnings: list[str] = []

    ev = pd.to_numeric(df.get("ev_pct", pd.Series(dtype=float)), errors="coerce")

    # Drop rows with impossible EV (|ev| > 200%)
    impossible_mask = ev.abs() > 2.0
    for idx in df[impossible_mask].index:
        quarantined.append({
            "step": "ev_calculation",
            "reason": "impossible_ev_over_200pct",
            "player": str(df.at[idx, "player_name"]),
            "ev_pct": float(ev.at[idx]) if not pd.isna(ev.at[idx]) else None,
        })
    clean = df[~impossible_mask].copy()

    # Invariant re-check: over_only must have kelly=0 (defence-in-depth)
    if "over_only" in clean.columns:
        mask = clean["over_only"].fillna(False)
        kelly = pd.to_numeric(
            clean.get("kelly_units", pd.Series(0, index=clean.index)), errors="coerce"
        ).fillna(0)
        violated = mask & (kelly > 0)
        if violated.any():
            warnings.append(
                f"INVARIANT VIOLATED: {violated.sum()} over_only plays had kelly>0 — zeroing"
            )
            clean.loc[violated, "kelly_units"] = 0.0
            clean.loc[violated, "stake_usd"] = 0.0

    # Non-fatal slate-size warnings
    n = len(clean)
    if n > 150:
        warnings.append(
            f"Large slate: {n} plays (normal 30–100) — possible alternate market contamination"
        )

    odds = pd.to_numeric(
        clean.get("best_retail_odds", pd.Series(dtype=float)), errors="coerce"
    )
    if len(odds):
        pct_high = float((odds > 600).mean())
        if pct_high > 0.30:
            warnings.append(
                f"{pct_high:.0%} of plays above +600 (threshold: 30%) — check for anchor mismatch"
            )

    return StepResult(clean=clean, quarantined=quarantined, warnings=warnings)


def validate_clv_log(df: pd.DataFrame) -> StepResult:
    """Scan captured CLV entries for extreme values (|CLV| > 50%)."""
    if df.empty:
        return StepResult(clean=df, warnings=["CLV log is empty"])

    captured = df[df["closing_pinnacle_prob"].notna()].copy()
    if captured.empty:
        return StepResult(clean=df, warnings=["No closing lines captured yet"])

    clv = pd.to_numeric(captured["clv_pct"], errors="coerce")
    anomalies = captured[clv.abs() > 0.50]

    quarantined: list[dict] = []
    for idx, row in anomalies.iterrows():
        quarantined.append({
            "step": "clv_validation",
            "reason": "clv_exceeds_50pct",
            "player": str(row.get("player_name", "")),
            "game_date": str(row.get("game_date", "")),
            "clv_pct": float(clv.at[idx]) if not pd.isna(clv.at[idx]) else None,
        })

    return StepResult(clean=df, quarantined=quarantined, warnings=[])


def validate_outcomes(outcomes: dict, date_str: str) -> StepResult:
    """Validate MLB batter outcome dict returned by get_all_hr_hitters()."""
    if not outcomes:
        return StepResult(
            clean={},
            warnings=[f"No outcomes found for {date_str}"],
            ok=False,
        )

    n_total = len(outcomes)
    n_hr = sum(1 for v in outcomes.values() if v.get("hrs_hit", 0) > 0)
    hit_rate = n_hr / n_total if n_total else 0.0

    warnings: list[str] = []
    if hit_rate > 0.50:
        warnings.append(
            f"Anomalous HR rate {hit_rate:.1%} on {date_str} "
            f"({n_hr}/{n_total} players hit HRs) — inspect for data error"
        )

    return StepResult(clean=outcomes, quarantined=[], warnings=warnings, ok=True)
