import os
import pandas as pd


def _client():
    url = os.environ.get("SUPABASE_URL", "")
    key = os.environ.get("SUPABASE_KEY", "")
    if not url or not key:
        raise EnvironmentError("SUPABASE_URL and SUPABASE_KEY must be set")
    from supabase import create_client
    return create_client(url, key)


def insert_clv_rows(rows: list[dict]) -> None:
    if not rows:
        return
    _client().table("clv_log").upsert(
        rows, on_conflict="game_date,game,player_name"
    ).execute()


def fetch_clv_log(game_date: str | None = None, featured_only: bool = False) -> pd.DataFrame:
    q = _client().table("clv_log").select("*")
    if game_date:
        q = q.eq("game_date", game_date)
    if featured_only:
        q = q.eq("featured_bet", True)
    resp = q.execute()
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)


def upsert_outcome(
    game_date: str,
    player_name: str,
    hit_hr,
    hrs_hit: int,
    at_bats: int,
    game_pk=None,
    team: str = "",
    game_str: str = "",
    captured_ts: str = "",
) -> None:
    _client().table("hr_outcomes").upsert(
        {
            "game_date": game_date,
            "player_name": player_name,
            "hit_hr": hit_hr,
            "hrs_hit": hrs_hit,
            "at_bats": at_bats,
            "game_pk": game_pk,
            "team": team,
            "game": game_str,
            "captured_ts": captured_ts,
        },
        on_conflict="game_date,player_name",
    ).execute()


def fetch_outcomes(game_date: str | None = None) -> pd.DataFrame:
    q = _client().table("hr_outcomes").select("*")
    if game_date:
        q = q.eq("game_date", game_date)
    resp = q.execute()
    if not resp.data:
        return pd.DataFrame()
    return pd.DataFrame(resp.data)


def mark_withdrawn(game_date: str, player_name: str) -> None:
    _client().table("clv_log").update({"withdrawn": True}).eq(
        "game_date", game_date
    ).eq("player_name", player_name).execute()
