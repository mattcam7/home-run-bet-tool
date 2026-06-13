"""Discord webhook delivery for HR picks, results, and health status."""
import logging
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

ET = ZoneInfo("America/New_York")
_LOG = Path("data/discord.log")
_LOG.parent.mkdir(exist_ok=True)
logging.basicConfig(filename=str(_LOG), level=logging.ERROR,
                    format="%(asctime)s %(levelname)s %(message)s")

_GRADE_EMOJI = {"Strong": "🔴", "Solid": "🟡", "Marginal": "🟠"}


def _wh(env_var: str) -> str:
    val = os.environ.get(env_var, "")
    if not val:
        raise EnvironmentError(f"{env_var} must be set")
    return val


def _american_to_decimal(odds: int) -> float:
    return (odds / 100) + 1 if odds > 0 else (100 / abs(odds)) + 1


def _odds_str(odds: int) -> str:
    return f"+{odds}" if odds > 0 else str(odds)


def _bet_by(commence_time, buffer_min: int = 30) -> str:
    if pd.isna(commence_time):
        return "before game"
    bt = commence_time - timedelta(minutes=buffer_min)
    s = bt.astimezone(ET).strftime("%I:%M %p ET")
    return s.lstrip("0")


def _running_metrics() -> tuple:
    """Return (roi, mean_clv_pct) for featured bets. Returns (None, None) on error."""
    try:
        from agents.outcome_tracker import compute_roi_metrics
        m = compute_roi_metrics(featured_only=True)
        roi = m.get("roi")
        clv = None
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        if supabase_key:
            try:
                from agents.supabase_client import fetch_clv_log
                df = fetch_clv_log(featured_only=True)
            except Exception:
                df = pd.DataFrame()
        else:
            clv_path = Path("data/clv_log.csv")
            df = pd.read_csv(clv_path) if clv_path.exists() else pd.DataFrame()
            if not df.empty and "featured_bet" in df.columns:
                df = df[df["featured_bet"].astype(str).str.lower() == "true"]
        if not df.empty and "clv_pct" in df.columns:
            vals = pd.to_numeric(df["clv_pct"], errors="coerce").dropna()
            clv = float(vals.mean()) if len(vals) > 0 else None
        return roi, clv
    except Exception:
        return None, None


def post_status(message: str) -> None:
    """Post to #system-status. Never raises."""
    try:
        wh = _wh("DISCORD_STATUS_WEBHOOK")
        requests.post(wh, json={"content": message}, timeout=10)
    except Exception:
        pass


def post_alert(
    player_name: str,
    old_odds: int,
    new_odds: int,
    old_ev: float,
    new_ev: float,
    alert_type: str,
) -> None:
    """Post a line movement alert or withdrawal to #picks."""
    try:
        wh = _wh("DISCORD_PICKS_WEBHOOK")
        old_str = _odds_str(old_odds)
        new_str = _odds_str(new_odds)
        if alert_type == "withdrawal":
            msg = (f"❌ Withdrawal — {player_name}: {new_str} · "
                   f"EV now {new_ev*100:+.1f}% · skip this play")
        else:
            msg = (f"⚠️ Line alert — {player_name}: {old_str} → {new_str} · "
                   f"EV {old_ev*100:+.1f}% → {new_ev*100:+.1f}% · edge reduced")
        requests.post(wh, json={"content": msg}, timeout=10).raise_for_status()
    except Exception as e:
        logging.error(f"post_alert failed: {e}")


def post_picks(final_df: pd.DataFrame, now: datetime = None) -> None:
    """Post daily picks to #picks. Called from run.py --no-browser."""
    now = now or datetime.now(timezone.utc)
    try:
        picks_wh = _wh("DISCORD_PICKS_WEBHOOK")
        status_wh = _wh("DISCORD_STATUS_WEBHOOK")

        today_str = now.astimezone(ET).strftime("%Y-%m-%d")
        supabase_key = os.environ.get("SUPABASE_KEY", "")

        # Filter to featured bets, exclude games starting within 90 min
        if final_df.empty or "featured_bet" not in final_df.columns:
            featured = pd.DataFrame()
        else:
            featured = final_df[final_df["featured_bet"] == True].copy()
            if "commence_time" in featured.columns:
                featured = featured[featured["commence_time"].apply(
                    lambda ct: pd.isna(ct) or (
                        (ct.replace(tzinfo=timezone.utc) if ct.tzinfo is None else ct) - now
                    ).total_seconds() / 60 > 90
                )]

            # Exclude picks already posted to Discord earlier today (dedup across multi-run days)
            already_posted: set = set()
            if supabase_key and not featured.empty:
                try:
                    from agents.supabase_client import fetch_discord_posted_players
                    already_posted = fetch_discord_posted_players(today_str)
                except Exception:
                    pass
            if already_posted and "player_name" in featured.columns:
                featured = featured[~featured["player_name"].isin(already_posted)]

            featured = featured.sort_values(
                ["kelly_units", "ev_pct"], ascending=False
            )

        # Platform-safe day format (Windows doesn't support %-d)
        if os.name == "nt":
            date_label = now.astimezone(ET).strftime("%a %b %#d")
        else:
            date_label = now.astimezone(ET).strftime("%a %b %-d")

        if featured.empty:
            msg = f"⚾ HR Picks — {date_label}\n\nNo featured plays today."
            n_plays = 0
        else:
            lines = [f"⚾ HR Picks — {date_label}\n"]
            for _, r in featured.iterrows():
                grade = str(r.get("bet_grade", ""))
                emoji = _GRADE_EMOJI.get(grade, "⚪")
                kelly = float(r.get("kelly_units", 0))
                ev = float(r.get("ev_pct", 0)) * 100
                anchor = "PIN" if str(r.get("anchor_quality", "")) == "pinnacle" else "BOL"
                bet_by = _bet_by(r.get("commence_time"))
                odds = int(r["best_retail_odds"])
                line = (f"{emoji} {grade:<8} {r['player_name']} · "
                        f"{r['best_retail_book']} {_odds_str(odds)} · "
                        f"EV {ev:+.1f}% · {kelly:.1f}u · {anchor} · Bet by {bet_by}")
                lines.append(line)

            roi, clv = _running_metrics()
            footer_parts = [f"{len(featured)} plays", "1u = $25"]
            if roi is not None:
                footer_parts.append(f"Running ROI: {roi*100:+.1f}%")
            if clv is not None:
                footer_parts.append(f"CLV: {clv*100:+.1f}%")
            lines.append("\n" + " · ".join(footer_parts))
            msg = "\n".join(lines)
            n_plays = len(featured)

        requests.post(picks_wh, json={"content": msg}, timeout=10).raise_for_status()

        # Mark posted picks so subsequent same-day runs skip them
        if not featured.empty and supabase_key:
            try:
                from agents.supabase_client import mark_picks_discord_posted
                mark_picks_discord_posted(today_str, featured["player_name"].tolist())
            except Exception:
                pass

        if os.name == "nt":
            time_label = now.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
            date_d = now.astimezone(ET).strftime("%b %#d")
        else:
            time_label = now.astimezone(ET).strftime("%I:%M %p ET").lstrip("0")
            date_d = now.astimezone(ET).strftime("%b %-d")

        if n_plays == 0:
            post_status(f"⚠️ Picks ran — 0 featured plays found · {date_d} {time_label}")
        else:
            post_status(f"✅ Picks posted — {n_plays} plays · {date_d} {time_label}")

    except Exception as e:
        logging.error(f"post_picks failed: {e}")
        try:
            if os.name == "nt":
                ts = now.astimezone(ET).strftime("%b %d %I:%M %p ET")
            else:
                ts = now.astimezone(ET).strftime("%b %-d %I:%M %p ET")
            post_status(f"❌ Picks FAILED — {ts}: {e}")
        except Exception:
            pass


def post_results(date_str: str, now: datetime = None) -> None:
    """Post settled results for date_str to #results."""
    now = now or datetime.now(timezone.utc)
    try:
        results_wh = _wh("DISCORD_RESULTS_WEBHOOK")

        supabase_key = os.environ.get("SUPABASE_KEY", "")
        if supabase_key:
            try:
                from agents.supabase_client import fetch_clv_log, fetch_outcomes as _fetch_outcomes
                picks = fetch_clv_log(game_date=date_str, featured_only=True)
                outcomes_df = _fetch_outcomes(game_date=date_str)
            except Exception as e:
                post_status(f"⚠️ Results skipped — Supabase read failed: {e}")
                return
        else:
            clv_path = Path("data/clv_log.csv")
            if not clv_path.exists():
                post_status(f"⚠️ Results skipped — clv_log.csv not found")
                return
            clv = pd.read_csv(clv_path)
            if "featured_bet" not in clv.columns:
                post_status(f"⚠️ Results skipped — featured_bet column missing")
                return
            picks = clv[
                (clv["game_date"] == date_str) &
                (clv["featured_bet"].astype(str).str.lower() == "true")
            ].copy()
            from agents.outcome_tracker import load_outcomes
            outcomes_df = load_outcomes()

        if picks.empty:
            post_status(f"✅ Results — no featured bets on {date_str}")
            return

        if outcomes_df.empty:
            post_status(f"⚠️ Results — outcomes DB empty for {date_str}")
            return

        merged = picks.merge(
            outcomes_df[["game_date", "player_name", "hit_hr"]],
            on=["game_date", "player_name"], how="left"
        )

        lines = [f"📋 Results — {date_str}\n"]
        day_pnl = 0.0
        for _, r in merged.iterrows():
            name = str(r["player_name"])
            odds = int(r["best_retail_odds"])
            book = str(r["best_retail_book"])
            stake = float(r.get("stake_usd", 0))
            decimal = float(r.get("best_retail_decimal", _american_to_decimal(odds)))

            if pd.isna(r.get("hit_hr")):
                logging.warning(f"post_results: no outcome for {name} on {date_str} — possible name mismatch")
                lines.append(f"➖ {name} · scratched — no result")
            elif int(r["hit_hr"]) == 1:
                pnl = stake * (decimal - 1)
                day_pnl += pnl
                lines.append(f"✅ {name} · {book} {_odds_str(odds)} · HIT · +${pnl:.2f}")
            else:
                day_pnl -= stake
                lines.append(f"❌ {name} · {book} {_odds_str(odds)} · miss · -${stake:.2f}")

        roi, _ = _running_metrics()
        footer_parts = [f"Day: {day_pnl:+.2f}"]
        if roi is not None:
            footer_parts.append(f"Running ROI: {roi*100:+.1f}%")
        lines.append("\n" + " · ".join(footer_parts))

        requests.post(results_wh, json={"content": "\n".join(lines)}, timeout=10).raise_for_status()
        post_status(f"✅ Results posted — {date_str} · {len(merged)} settled")

    except Exception as e:
        logging.error(f"post_results failed: {e}")
        try:
            post_status(f"❌ Results FAILED — {date_str}: {e}")
        except Exception:
            pass


def post_weekly_recap(now: datetime = None) -> None:
    """Post Sunday weekly recap to #weekly-recap."""
    now = now or datetime.now(timezone.utc)
    try:
        recap_wh = _wh("DISCORD_RECAP_WEBHOOK")

        from agents.outcome_tracker import compute_roi_metrics
        metrics = compute_roi_metrics(featured_only=True)

        from datetime import date
        today = now.astimezone(ET).date()
        last_monday = today - timedelta(days=today.weekday())

        week_pnl = None
        week_plays = 0
        week_hits = 0
        supabase_key = os.environ.get("SUPABASE_KEY", "")
        try:
            if supabase_key:
                from agents.supabase_client import fetch_clv_log, fetch_outcomes as _fetch_out
                clv = fetch_clv_log(featured_only=True)
                outcomes = _fetch_out()
            else:
                clv_path = Path("data/clv_log.csv")
                clv = pd.read_csv(clv_path) if clv_path.exists() else pd.DataFrame()
                if not clv.empty and "featured_bet" in clv.columns:
                    clv = clv[clv["featured_bet"].astype(str).str.lower() == "true"]
                from agents.outcome_tracker import load_outcomes
                outcomes = load_outcomes()

            if not clv.empty and not outcomes.empty:
                feat = clv[pd.to_datetime(clv["game_date"]) >= pd.Timestamp(last_monday)]
                merged = feat.merge(outcomes[["game_date", "player_name", "hit_hr"]],
                                    on=["game_date", "player_name"], how="left")
                settled = merged[merged["hit_hr"].notna()].copy()
                settled["hit_hr"] = settled["hit_hr"].astype(int)
                settled["stake_usd"] = pd.to_numeric(settled["stake_usd"], errors="coerce").fillna(0)
                settled["best_retail_decimal"] = pd.to_numeric(settled["best_retail_decimal"], errors="coerce")
                pnl = settled.apply(
                    lambda r: r["stake_usd"] * (r["best_retail_decimal"] - 1) if r["hit_hr"] == 1
                    else -r["stake_usd"], axis=1
                )
                week_pnl = float(pnl.sum())
                week_plays = len(settled)
                week_hits = int(settled["hit_hr"].sum())
        except Exception:
            pass

        n_total = metrics.get("n_with_outcome", 0)
        roi = metrics.get("roi")
        hit_rate = metrics.get("hit_rate")

        if os.name == "nt":
            mon_str = last_monday.strftime("%#m/%#d")
            sun_str = today.strftime("%#m/%#d")
        else:
            mon_str = last_monday.strftime("%-m/%-d")
            sun_str = today.strftime("%-m/%-d")

        lines = [f"📊 Weekly Recap · {mon_str} – {sun_str}"]
        if week_pnl is not None:
            hit_pct = (week_hits / week_plays * 100) if week_plays else 0
            lines.append(f"This week: {week_plays} plays · {week_hits} hits · {hit_pct:.1f}% · P&L: {week_pnl:+.2f}")
        if n_total and roi is not None and hit_rate is not None:
            lines.append(
                f"All-time:  {n_total} plays · {roi*100:+.1f}% ROI · "
                f"{hit_rate*100:.1f}% hit rate"
            )
        lines.append("Anchor: Pinnacle/BOL devig · 1u = $25 · Kelly ¼ sizing")

        requests.post(recap_wh, json={"content": "\n".join(lines)}, timeout=10).raise_for_status()
        post_status(f"✅ Weekly recap posted · {sun_str}")

    except Exception as e:
        logging.error(f"post_weekly_recap failed: {e}")
        try:
            post_status(f"❌ Weekly recap FAILED: {e}")
        except Exception:
            pass
