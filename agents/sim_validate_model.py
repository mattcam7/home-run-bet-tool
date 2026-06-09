"""
agents/sim_validate_model.py — Compare v1 (10-feature) vs v2 (14-feature) HR models.

Temporal split: 2022-2024 train, 2025 test.
Reports AUC-ROC, Brier score, log-loss, calibration.
Exit 0 if v2 AUC >= v1 AUC, else exit 1.

Usage:
    python -m agents.sim_validate_model
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TRAINING_CACHE_PATH = Path("data/sim_training_cache.parquet")

FEATURES_V1 = [
    "brl_percent", "avg_hit_speed", "ev95percent", "iso",
    "bat_speed", "park_factor", "same_hand", "pitcher_hr9",
    "fb_pct", "hr_fb",
]

FEATURES_V2 = [
    "brl_pct_vs_hand", "iso_vs_hand", "fb_pct_vs_hand", "hr_fb_vs_hand",
    "avg_hit_speed", "ev95percent", "bat_speed",
    "park_factor", "same_hand",
    "rolling_brl_pct", "rolling_avg_ev",
    "rolling_pitcher_hr9", "pitcher_gb_pct",
    "lineup_slot",
]


def _train_eval(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]
) -> dict:
    """Train logistic regression on train_df, evaluate on test_df.

    Returns dict with keys: n_train, n_test, auc, brier, logloss, y_test, y_prob.
    auc/brier/logloss are NaN when test set is empty.
    """
    available = [f for f in features if f in train_df.columns and f in test_df.columns]
    train = train_df.dropna(subset=available + ["hit_hr"])
    test = test_df.dropna(subset=available + ["hit_hr"])

    if train.empty or test.empty:
        return {
            "n_train": len(train), "n_test": len(test),
            "auc": float("nan"), "brier": float("nan"), "logloss": float("nan"),
            "y_test": np.array([]), "y_prob": np.array([]),
        }

    X_train = train[available].fillna(0.0)
    y_train = train["hit_hr"].astype(int)
    X_test = test[available].fillna(0.0)
    y_test = test["hit_hr"].astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    y_prob = pipe.predict_proba(X_test)[:, 1]

    return {
        "n_train": len(train),
        "n_test": len(test),
        "auc": float(roc_auc_score(y_test, y_prob)),
        "brier": float(brier_score_loss(y_test, y_prob)),
        "logloss": float(log_loss(y_test, y_prob)),
        "y_test": y_test.values,
        "y_prob": y_prob,
    }


def _calibration_table(
    y_test: np.ndarray, y_prob: np.ndarray, n_bins: int = 5
) -> list[tuple[str, float, int]]:
    """Return calibration rows: [(label, actual_rate, count), ...]"""
    edges = np.percentile(y_prob, np.linspace(0, 100, n_bins + 1))
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob <= hi)
        n = int(mask.sum())
        actual = float(y_test[mask].mean()) if n > 0 else float("nan")
        rows.append((f"{lo*100:.0f}-{hi*100:.0f}%", actual, n))
    return rows


def main() -> None:
    if not TRAINING_CACHE_PATH.exists():
        print(f"ERROR: {TRAINING_CACHE_PATH} not found. Run sim_build_training_data first.")
        sys.exit(1)

    df = pd.read_parquet(TRAINING_CACHE_PATH)

    train_df = df[df["season"] < 2025]
    test_df = df[df["season"] == 2025]

    print(
        f"Temporal split: train={len(train_df):,} rows (seasons < 2025), "
        f"test={len(test_df):,} rows (2025)\n"
    )

    v1 = _train_eval(train_df, test_df, FEATURES_V1)
    v2 = _train_eval(train_df, test_df, FEATURES_V2)

    print(f"{'Model Comparison -- 2025 holdout':}")
    print(f"{'':25s}  {'Baseline (v1)':>14s}  {'v2 (14f)':>14s}  {'Delta':>8s}")
    print(f"{'AUC-ROC':25s}  {v1['auc']:>14.4f}  {v2['auc']:>14.4f}  {v2['auc']-v1['auc']:>+8.4f}")
    print(f"{'Brier Score':25s}  {v1['brier']:>14.4f}  {v2['brier']:>14.4f}  {v2['brier']-v1['brier']:>+8.4f}")
    print(f"{'Log-Loss':25s}  {v1['logloss']:>14.4f}  {v2['logloss']:>14.4f}  {v2['logloss']-v1['logloss']:>+8.4f}")
    print(f"\n(n_train v1={v1['n_train']:,}, v2={v2['n_train']:,})")
    print(f"(n_test  v1={v1['n_test']:,}, v2={v2['n_test']:,})")

    missing_v2 = [f for f in FEATURES_V2 if f not in df.columns]
    if missing_v2:
        print(f"\nWARNING: v2 features missing from training cache: {missing_v2}")
        print("Re-run sim_build_training_data to rebuild the cache with v2 features.")

    if v2["n_test"] > 0 and not np.isnan(v2["auc"]):
        print("\nCalibration (v2):")
        for label, actual, n in _calibration_table(v2["y_test"], v2["y_prob"]):
            actual_str = f"{actual*100:.1f}%" if not np.isnan(actual) else "n/a"
            print(f"  Predicted {label}: actual {actual_str} (N={n:,})")

    if np.isnan(v2["auc"]):
        print("\nWARNING: v2 could not be evaluated (no valid test rows).")
        sys.exit(1)

    if v2["auc"] >= v1["auc"]:
        print(f"\nPASS: v2 AUC {v2['auc']:.4f} >= v1 AUC {v1['auc']:.4f}")
        sys.exit(0)
    else:
        print(f"\nFAIL: v2 AUC {v2['auc']:.4f} < v1 AUC {v1['auc']:.4f}")
        sys.exit(1)


if __name__ == "__main__":
    main()
