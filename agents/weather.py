"""
agents/weather.py — Ballpark weather features for HR simulation model.

Public interface:
    get_weather(team, date, hour=19) -> dict[str, float]
        Returns {temp_f, wind_toward_cf} for a park at the given date+hour.
        Uses open-meteo.com (free, no API key).

    enrich_weather_features(df) -> pd.DataFrame
        Batch-adds temp_f and wind_toward_cf columns to a training DataFrame.
        Expects columns: home_team (MLB API abbreviation), game_date (YYYY-MM-DD).

    WEATHER_FEATURES: list[str]  — ["temp_f", "wind_toward_cf"]
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEATHER_FEATURES = ["temp_f", "wind_toward_cf"]

_STADIUM_PATH = Path("data/stadium_coords.json")
_CACHE_DIR = Path("data/weather_cache")
_HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

_INDOOR_TEMP_F = 72.0
_INDOOR_WIND = 0.0
_DEFAULT_GAME_HOUR = 19   # 7 pm local — covers most night games

# FanGraphs-style → MLB API style (same as _MLB_API_TO_FG_ABBREV reversed).
# Handles callers that pass FG-style abbreviations from TEAM_NAME_TO_ABBREV.
_FG_TO_API: dict[str, str] = {
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "ARI": "AZ",
    "OAK": "ATH",
}

_stadiums_cache: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_stadiums() -> dict[str, Any]:
    global _stadiums_cache
    if _stadiums_cache is None:
        _stadiums_cache = json.loads(_STADIUM_PATH.read_text(encoding="utf-8"))
    return _stadiums_cache


def _normalize_abbrev(team: str) -> str:
    """Convert FanGraphs-style abbreviation to MLB API style used in stadium_coords.json."""
    return _FG_TO_API.get(team.strip(), team.strip())


def _wind_toward_cf(speed_kmh: float, from_deg: float, cf_bearing: float) -> float:
    """
    Wind component (mph) blowing from home plate toward center field.
    Positive = tailwind (favors HRs). Negative = headwind (suppresses HRs).

    from_deg: meteorological convention — direction the wind is FROM (0=N, 90=E).
    cf_bearing: azimuth from home plate to CF (0=N, 90=E).
    """
    toward_deg = (from_deg + 180.0) % 360.0          # direction wind blows TOWARD
    angle = math.radians(toward_deg - cf_bearing)
    speed_mph = speed_kmh * 0.621371
    return speed_mph * math.cos(angle)


def _c_to_f(celsius: float) -> float:
    return celsius * 9.0 / 5.0 + 32.0


def _fetch_hourly_raw(lat: float, lon: float, tz: str,
                      start_date: str, end_date: str,
                      forecast: bool = False) -> dict[str, list]:
    """
    Call open-meteo API and return the 'hourly' block.
    Raises requests.HTTPError on non-200.
    """
    url = _FORECAST_URL if forecast else _HISTORICAL_URL
    params: dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,wind_speed_10m,wind_direction_10m",
        "timezone": tz,
        "wind_speed_unit": "kmh",
    }
    if forecast:
        params["forecast_days"] = 3
    else:
        params["start_date"] = start_date
        params["end_date"] = end_date

    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["hourly"]


def _hourly_to_lookup(hourly: dict[str, list]) -> dict[str, tuple[float, float, float]]:
    """Index hourly API response as {datetime_str: (temp_c, wind_kmh, wind_dir)}."""
    return {
        ts: (t, w, d)
        for ts, t, w, d in zip(
            hourly["time"],
            hourly["temperature_2m"],
            hourly["wind_speed_10m"],
            hourly["wind_direction_10m"],
        )
    }


def _cache_path_for(team_api: str, date: str) -> Path:
    return _CACHE_DIR / f"{team_api}_{date}.json"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_weather(team: str, date: str, hour: int = _DEFAULT_GAME_HOUR) -> dict[str, float]:
    """
    Return {temp_f, wind_toward_cf} for *team*'s park at *date* and *hour* (local time).

    team: MLB Stats API abbreviation (KC, SD, SF, TB, AZ, ATH) or FanGraphs style
          (KCR, SDP, SFG, TBR, ARI, OAK) — both accepted.
    date: "YYYY-MM-DD"
    hour: local hour 0-23 (default 19 = 7 pm, typical night game start)

    Falls back to indoor defaults if the park is indoor, if the API fails,
    or if the team is unknown.
    """
    team_api = _normalize_abbrev(team)
    stadiums = _load_stadiums()
    park = stadiums.get(team_api)

    if park is None or park.get("indoor", False):
        return {"temp_f": _INDOOR_TEMP_F, "wind_toward_cf": _INDOOR_WIND}

    # Check per-date cache
    cache_file = _cache_path_for(team_api, date)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    hourly_lookup: dict[str, tuple[float, float, float]] | None = None

    if cache_file.exists():
        try:
            hourly_lookup = {
                ts: tuple(v)  # type: ignore[assignment]
                for ts, v in json.loads(cache_file.read_text(encoding="utf-8")).items()
            }
        except Exception:
            hourly_lookup = None

    if hourly_lookup is None:
        try:
            from datetime import date as date_cls, timedelta
            d = date_cls.fromisoformat(date)
            today = date_cls.today()
            forecast = d >= today
            raw = _fetch_hourly_raw(
                lat=park["lat"], lon=park["lon"], tz=park["tz"],
                start_date=date, end_date=date, forecast=forecast,
            )
            hourly_lookup = _hourly_to_lookup(raw)
            cache_file.write_text(
                json.dumps({k: list(v) for k, v in hourly_lookup.items()}, indent=None),
                encoding="utf-8",
            )
        except Exception:
            return {"temp_f": _INDOOR_TEMP_F, "wind_toward_cf": _INDOOR_WIND}

    ts_key = f"{date}T{hour:02d}:00"
    entry = hourly_lookup.get(ts_key)
    if entry is None:
        return {"temp_f": _INDOOR_TEMP_F, "wind_toward_cf": _INDOOR_WIND}

    temp_c, wind_kmh, wind_dir = entry
    tf = _c_to_f(float(temp_c)) if temp_c is not None else _INDOOR_TEMP_F
    wtc = _wind_toward_cf(float(wind_kmh or 0), float(wind_dir or 0), park["cf_bearing"])

    return {"temp_f": round(tf, 1), "wind_toward_cf": round(wtc, 2)}


def enrich_weather_features(df: pd.DataFrame, game_hour: int = _DEFAULT_GAME_HOUR) -> pd.DataFrame:
    """
    Batch-add temp_f and wind_toward_cf columns to a training DataFrame.

    Expects columns: home_team (MLB API abbreviation), game_date (str YYYY-MM-DD).
    Fetches per team in bulk date-range requests — roughly 30 API calls for a
    full 4-season training set regardless of row count.

    Returns df with two new columns; rows where weather couldn't be fetched
    receive indoor defaults (72°F / 0 wind).
    """
    stadiums = _load_stadiums()
    df = df.copy()
    df["temp_f"] = _INDOOR_TEMP_F
    df["wind_toward_cf"] = _INDOOR_WIND

    # Build (team, date) → row index mapping
    needs_weather = df[["home_team", "game_date"]].copy()
    needs_weather["home_team_api"] = needs_weather["home_team"].apply(_normalize_abbrev)

    unique_teams = needs_weather["home_team_api"].unique()

    for team_api in unique_teams:
        park = stadiums.get(team_api)
        if park is None or park.get("indoor", False):
            continue

        team_rows = needs_weather[needs_weather["home_team_api"] == team_api]
        dates = pd.to_datetime(team_rows["game_date"]).dt.date
        start_date = str(dates.min())
        end_date = str(dates.max())

        # Fetch full season range in one API call
        team_cache = _CACHE_DIR / f"{team_api}_{start_date}_{end_date}.parquet"
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

        hourly_lookup: dict[str, tuple] = {}

        if team_cache.exists():
            try:
                cached_df = pd.read_parquet(team_cache)
                for _, row in cached_df.iterrows():
                    hourly_lookup[str(row["ts"])] = (
                        row["temp_c"], row["wind_kmh"], row["wind_dir"]
                    )
            except Exception:
                hourly_lookup = {}

        if not hourly_lookup:
            try:
                print(f"  [weather] Fetching {team_api} {start_date} to {end_date}...", flush=True)
                raw = _fetch_hourly_raw(
                    lat=park["lat"], lon=park["lon"], tz=park["tz"],
                    start_date=start_date, end_date=end_date, forecast=False,
                )
                hourly_lookup = _hourly_to_lookup(raw)
                # Cache as parquet
                cache_rows = [
                    {"ts": ts, "temp_c": t, "wind_kmh": w, "wind_dir": d}
                    for ts, (t, w, d) in hourly_lookup.items()
                ]
                pd.DataFrame(cache_rows).to_parquet(team_cache, index=False)
                time.sleep(0.3)   # polite rate-limiting
            except Exception as exc:
                print(f"  [weather] WARNING: fetch failed for {team_api}: {exc}")
                continue

        # Map weather to each game row
        cf_bearing = park["cf_bearing"]
        for idx in team_rows.index:
            date_str = str(needs_weather.loc[idx, "game_date"])
            ts_key = f"{date_str}T{game_hour:02d}:00"
            entry = hourly_lookup.get(ts_key)
            if entry is None:
                continue
            temp_c, wind_kmh, wind_dir = entry
            tf = _c_to_f(float(temp_c)) if temp_c is not None else _INDOOR_TEMP_F
            wtc = _wind_toward_cf(
                float(wind_kmh or 0), float(wind_dir or 0), cf_bearing
            )
            df.at[idx, "temp_f"] = round(tf, 1)
            df.at[idx, "wind_toward_cf"] = round(wtc, 2)

    print(
        f"[weather] Enriched {df[['temp_f']].notna().sum().item():,} rows. "
        f"Outdoor parks: {(df['temp_f'] != _INDOOR_TEMP_F).sum():,} rows with real weather."
    )
    return df
