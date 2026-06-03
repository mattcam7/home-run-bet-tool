"""WAT Pipeline orchestrator — Phase 1 (Steps 1-5 + dashboard).

Steps:
  1  Data Acquisition       fetch_odds()
  2  Raw Validation         validate_raw_odds()
  3  Dataframe Build        extract_retail_odds() + extract_sharp_anchor()
  4  EV Calculation         calculate_ev() + validate_ev_output() + compute_bet_score()
  5  CLV Log                log_open_plays()
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd
from dotenv import load_dotenv

from agents.clv_log import log_open_plays
from agents.dfs import analyze_dfs
from agents.ev_calculator import calculate_ev, validate_slate
from agents.odds_scraper import extract_retail_odds
from agents.parlay import format_parlays, generate_parlays
from agents.pinnacle_scraper import extract_sharp_anchor
from agents.scoring import compute_bet_score
from agents.simulation import add_simulation
from agents.validation import StepResult, append_quarantine, validate_ev_output, validate_raw_odds
from dashboard.generator import generate_dashboard
from run import fetch_odds, fetch_player_teams


@dataclass
class PipelineContext:
    api_key: str = ""
    now: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    raw_events: list = field(default_factory=list)
    retail_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    anchor_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    final_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    step_results: dict = field(default_factory=dict)
    total_quarantined: int = 0
    halted_at: str = ""


def _log_result(ctx: PipelineContext, name: str, result: StepResult) -> None:
    ctx.step_results[name] = {
        "quarantined": len(result.quarantined),
        "warnings": result.warnings,
        "ok": result.ok,
    }
    ctx.total_quarantined += len(result.quarantined)
    if result.quarantined:
        append_quarantine(result.quarantined)
    for w in result.warnings:
        print(f"  [{name}] WARNING: {w}")


def run_phase1() -> PipelineContext:
    """Run Steps 1-5 and generate the dashboard. Returns PipelineContext."""
    load_dotenv()
    ctx = PipelineContext(
        api_key=os.environ["ODDS_API_KEY"],
        now=datetime.now(timezone.utc),
    )

    # Step 1: Data Acquisition
    print("[Step 1/5] Fetching odds from OddsAPI...")
    ctx.raw_events = fetch_odds(ctx.api_key, ctx.now)

    # Step 2: Raw Validation
    print("[Step 2/5] Validating raw data...")
    raw_result = validate_raw_odds(ctx.raw_events)
    _log_result(ctx, "raw_validation", raw_result)
    ctx.raw_events = raw_result.clean
    if not raw_result.ok:
        ctx.halted_at = "raw_validation"
        print("  HALT: No valid events after raw validation.")
        return ctx

    # Step 3: Build Dataframes
    print("[Step 3/5] Building dataframes...")
    ctx.retail_df = extract_retail_odds(ctx.raw_events, ctx.now)
    ctx.anchor_df = extract_sharp_anchor(ctx.raw_events, ctx.now)

    if ctx.retail_df.empty or ctx.anchor_df.empty:
        missing = "anchor" if ctx.anchor_df.empty else "retail"
        ctx.halted_at = "dataframe_build"
        print(f"  HALT: No {missing} data available.")
        return ctx

    # Step 4: EV Calculation + Validation + Scoring
    print("[Step 4/5] Calculating EV, validating, and scoring...")
    ctx.final_df = calculate_ev(ctx.retail_df, ctx.anchor_df)
    validate_slate(ctx.final_df)

    ev_result = validate_ev_output(ctx.final_df)
    _log_result(ctx, "ev_validation", ev_result)
    ctx.final_df = ev_result.clean
    ctx.final_df = compute_bet_score(ctx.final_df)

    player_teams = fetch_player_teams()
    ctx.final_df["team"] = ctx.final_df["player_name"].map(player_teams).fillna("")
    ctx.final_df = add_simulation(ctx.final_df)

    # Step 5: CLV Log
    print("[Step 5/5] Logging open plays to CLV log...")
    log_open_plays(ctx.final_df, now=ctx.now)

    # Dashboard
    parlays = generate_parlays(ctx.final_df)
    if parlays:
        print(format_parlays(parlays))

    dfs_data = analyze_dfs("data/dfs_projections.csv", ctx.final_df)
    generate_dashboard(ctx.final_df, parlays=parlays, dfs_data=dfs_data)

    if ctx.total_quarantined:
        print(f"\n  [{ctx.total_quarantined} rows quarantined -> data/quarantine.jsonl]")

    return ctx


if __name__ == "__main__":
    run_phase1()
