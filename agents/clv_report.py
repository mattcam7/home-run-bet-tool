# agents/clv_report.py
"""CLV calibration report — the evidence gate for the de-vigged model.

Closing Line Value is the accepted proxy for sharpness without waiting for a
full settled-bet sample: if our flagged plays consistently beat Pinnacle's
*closing* no-vig line, the model is picking real edges. This reads the CLV log
(populated by capture_closing.py) and summarises whether that's happening.

Designed to be called by AGENT 4 (Audit Agent). compute_metrics() is pure and
testable; format_report() renders the terminal summary; main() wires it to the
on-disk log.

Caveat (per CLAUDE.md): small samples lie. Anything under MIN_SAMPLE captured
plays is reported with a prominent insufficient-sample banner — do not act on
or claim sharpness from it.
"""
from agents.clv_log import DEFAULT_PATH

import pandas as pd

MIN_SAMPLE = 20

KELLY_TIERS = [
    ("0u", lambda u: u <= 0),
    ("0.5-1u", lambda u: 0 < u <= 1),
    ("1.5-2u", lambda u: 1 < u <= 2),
    ("2.5-3u", lambda u: u > 2),
]


def _seg(frame: pd.DataFrame) -> dict:
    return {
        "n": int(len(frame)),
        "mean_clv_pct": float(frame["clv_pct"].mean()) if len(frame) else None,
        "beat_close_rate": float((frame["clv_pct"] > 0).mean()) if len(frame) else None,
    }


def compute_metrics(df: pd.DataFrame) -> dict:
    """Summarise CLV performance over rows that have a captured closing line."""
    captured = df[df["closing_pinnacle_prob"].notna()].copy()
    captured["clv_pct"] = pd.to_numeric(captured["clv_pct"], errors="coerce")
    captured = captured.dropna(subset=["clv_pct"])

    metrics = {
        "n_logged": int(len(df)),
        "n_captured": int(len(captured)),
        "n_pending": int(len(df) - len(captured)),
        "small_sample": len(captured) < MIN_SAMPLE,
    }
    if captured.empty:
        return metrics

    ev = pd.to_numeric(captured["ev_pct"], errors="coerce")
    metrics["mean_clv_pct"] = float(captured["clv_pct"].mean())
    metrics["median_clv_pct"] = float(captured["clv_pct"].median())
    metrics["beat_close_rate"] = float((captured["clv_pct"] > 0).mean())
    metrics["positive_ev"] = _seg(captured[ev > 0])
    metrics["non_positive_ev"] = _seg(captured[ev <= 0])

    units = pd.to_numeric(captured["kelly_units"], errors="coerce").fillna(0)
    metrics["by_kelly_tier"] = [
        {"tier": label, **_seg(captured[units.apply(pred)])}
        for label, pred in KELLY_TIERS
    ]
    metrics["by_book"] = [
        {"book": book, **_seg(grp)}
        for book, grp in captured.groupby("best_retail_book")
    ]

    lineup = captured["in_lineup"].dropna()
    metrics["lineup_checked"] = int(len(lineup))
    metrics["in_lineup_rate"] = (
        float(lineup.astype(str).str.lower().eq("true").mean())
        if len(lineup) else None
    )
    return metrics


def _pct(v) -> str:
    return "--" if v is None else f"{v * 100:+.2f}%"


def _rate(v) -> str:
    return "--" if v is None else f"{v * 100:.1f}%"


def format_report(df: pd.DataFrame) -> str:
    m = compute_metrics(df)
    L = ["=" * 56, "CLV CALIBRATION REPORT", "=" * 56,
         f"Logged plays: {m['n_logged']}  |  Closing captured: "
         f"{m['n_captured']}  |  Pending: {m['n_pending']}"]
    if m["n_captured"] == 0:
        L.append("")
        L.append("No closing lines captured yet - run after Phase 2 "
                 "(capture_closing.py) has executed near game time.")
        L.append("=" * 56)
        return "\n".join(L)

    if m["small_sample"]:
        L += ["", "!! INSUFFICIENT SAMPLE (<%d) - directional only, do "
              "NOT claim sharpness !!" % MIN_SAMPLE]

    L += [
        "",
        f"Overall   beat-close: {_rate(m['beat_close_rate'])}   "
        f"mean CLV: {_pct(m['mean_clv_pct'])}   "
        f"median: {_pct(m['median_clv_pct'])}",
        "",
        "By selection (the sharpness test - +EV picks should win the close):",
        f"  +EV picks      n={m['positive_ev']['n']:<4} "
        f"beat={_rate(m['positive_ev']['beat_close_rate'])} "
        f"mean CLV={_pct(m['positive_ev']['mean_clv_pct'])}",
        f"  non-+EV        n={m['non_positive_ev']['n']:<4} "
        f"beat={_rate(m['non_positive_ev']['beat_close_rate'])} "
        f"mean CLV={_pct(m['non_positive_ev']['mean_clv_pct'])}",
        "",
        "By Kelly tier:",
    ]
    for t in m["by_kelly_tier"]:
        L.append(f"  {t['tier']:<8} n={t['n']:<4} "
                 f"mean CLV={_pct(t['mean_clv_pct'])}")
    L.append("")
    L.append("By book:")
    for b in sorted(m["by_book"], key=lambda x: -(x["mean_clv_pct"] or -9)):
        L.append(f"  {b['book']:<14} n={b['n']:<4} "
                 f"mean CLV={_pct(b['mean_clv_pct'])}")
    if m.get("in_lineup_rate") is not None:
        L += ["", f"Lineup-confirmed: {_rate(m['in_lineup_rate'])} "
              f"of {m['lineup_checked']} checked"]
    L.append("=" * 56)
    return "\n".join(L)


def format_roi_report() -> str:
    """Append ROI summary from outcome_tracker if DB exists."""
    from agents.outcome_tracker import compute_roi_metrics
    from pathlib import Path

    m = compute_roi_metrics()
    if not m["has_outcomes"]:
        return "\n[Outcome DB not found — run: python -m agents.outcome_tracker --backfill]"

    lines = [
        "",
        "=" * 56,
        "ACTUAL OUTCOME RESULTS (MLB box scores)",
        "=" * 56,
        f"Settled picks : {m['n_with_outcome']}  (scratched/no AB: {m['n_scratched']})",
        f"HR hit rate   : {m['hit_rate']*100:.1f}%  ({m['n_hr_hits']} HRs)",
        f"Total staked  : ${m['total_staked']:,.0f}",
        f"Total P&L     : ${m['total_pnl']:+,.2f}",
    ]
    if m["roi"] is not None:
        lines.append(f"ROI           : {m['roi']*100:+.2f}%")

    pev = m["positive_ev"]
    nev = m["negative_ev"]

    def _outcome_line(seg, label):
        if not seg["n"]:
            return f"  {label}  none"
        roi_str = f"ROI={seg['roi']*100:+.2f}%" if seg["roi"] is not None else "ROI=N/A (no stake)"
        hr_str = f"hit%={seg['hit_rate']*100:.1f}%" if seg["hit_rate"] is not None else "hit%=N/A"
        return f"  {label}  n={seg['n']:<4}  {hr_str}  {roi_str}  P&L=${seg['total_pnl']:+,.0f}"

    lines += [
        "",
        "By EV selection:",
        _outcome_line(pev, "+EV picks   "),
        _outcome_line(nev, "non-+EV     "),
    ]
    if m.get("clv_outcome_correlation") is not None:
        lines.append(f"\nCLV<->outcome correlation: {m['clv_outcome_correlation']:+.3f}")
    lines.append("=" * 56)
    return "\n".join(lines)


def main(path: str = DEFAULT_PATH) -> None:
    import os

    if not os.path.exists(path):
        print(f"No CLV log at {path} — run Phase 1 (run.py) first.")
        return
    print(format_report(pd.read_csv(path)))
    print(format_roi_report())


if __name__ == "__main__":
    main()
