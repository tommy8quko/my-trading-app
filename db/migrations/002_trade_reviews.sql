-- Trade reviews: per-trade tags, notes, MAE/MFE
-- Run this once in your Supabase SQL editor

CREATE TABLE IF NOT EXISTS trade_reviews (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol          TEXT NOT NULL,
    entry_date      DATE NOT NULL,
    setup_tag       TEXT,           -- Breakout / Pullback / Reversal / Buyable Gapup / Other
    outcome_reason  TEXT,           -- Stop Hit / Target Hit / Manual Exit / Trail Stop
    market_condition TEXT,          -- Trending / Choppy / News-Driven / Ranging
    emotional_notes TEXT,
    mae             NUMERIC(18,6),  -- max adverse excursion (price units, auto-fetched)
    mfe             NUMERIC(18,6),  -- max favorable excursion (price units, auto-fetched)
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    updated_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, entry_date)
);
