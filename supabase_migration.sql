-- Drop and recreate clv_log with correct schema matching agents/clv_log.py COLUMNS
-- Run this in Supabase SQL Editor

DROP TABLE IF EXISTS clv_log;

CREATE TABLE clv_log (
    id              bigserial PRIMARY KEY,
    run_ts          text,
    game_date       date        NOT NULL,
    commence_iso    text,
    game            text        NOT NULL,
    player_name     text        NOT NULL,
    team            text,
    best_retail_book        text,
    best_retail_odds        integer,
    best_retail_decimal     numeric,
    pinnacle_over_odds      integer,
    pinnacle_prob_devig     numeric,
    ev_pct                  numeric,
    kelly_units             numeric,
    stake_usd               numeric,
    anchor_quality          text,
    sim_prob                numeric,
    featured_bet            boolean,
    closing_ts              text,
    closing_pinnacle_odds   integer,
    closing_pinnacle_prob   numeric,
    clv_pct                 numeric,
    in_lineup               boolean,
    withdrawn               boolean DEFAULT false,
    posted_to_discord       boolean DEFAULT false,
    UNIQUE (game_date, game, player_name)
);

-- Grant access to the anon/service role
GRANT ALL ON clv_log TO anon, authenticated, service_role;
GRANT USAGE, SELECT ON SEQUENCE clv_log_id_seq TO anon, authenticated, service_role;

-- ---------------------------------------------------------------------------
-- hr_outcomes: actual game results joined against CLV log picks
-- ---------------------------------------------------------------------------
DROP TABLE IF EXISTS hr_outcomes;

CREATE TABLE hr_outcomes (
    id              bigserial PRIMARY KEY,
    game_date       date        NOT NULL,
    player_name     text        NOT NULL,
    team            text,
    game            text,
    game_pk         integer,
    hit_hr          integer,
    hrs_hit         integer     DEFAULT 0,
    at_bats         integer     DEFAULT 0,
    captured_ts     text,
    UNIQUE (game_date, player_name)
);

GRANT ALL ON hr_outcomes TO anon, authenticated, service_role;
GRANT USAGE, SELECT ON SEQUENCE hr_outcomes_id_seq TO anon, authenticated, service_role;
