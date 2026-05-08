-- ============================================================
-- Trading Journal — Base Schema
-- Run this once in the Supabase SQL Editor.
-- All tables use UUIDs.  No RLS for now (personal use).
-- ============================================================

-- Brokers
CREATE TABLE IF NOT EXISTS brokers (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    short_code   TEXT UNIQUE NOT NULL,          -- 'IBKR', 'CHIEF'
    display_name TEXT NOT NULL,
    -- array of email sender patterns (ILIKE matching)
    email_sender_patterns TEXT[] DEFAULT '{}',
    created_at   TIMESTAMPTZ DEFAULT NOW()
);

-- Accounts
CREATE TABLE IF NOT EXISTS accounts (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    broker_id       UUID REFERENCES brokers(id) ON DELETE RESTRICT,
    account_number  TEXT,
    display_name    TEXT NOT NULL,
    base_currency   TEXT NOT NULL DEFAULT 'HKD',
    initial_capital NUMERIC(18,4) DEFAULT 0,
    is_active       BOOLEAN DEFAULT TRUE,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Instruments
CREATE TABLE IF NOT EXISTS instruments (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol      TEXT NOT NULL,
    exchange    TEXT,           -- 'NYSE','NASDAQ','HKEX'
    asset_class TEXT DEFAULT 'equity',
    currency    TEXT NOT NULL,
    description TEXT,
    created_at  TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (symbol, exchange)
);

-- Imported Documents  (provenance / audit trail)
CREATE TABLE IF NOT EXISTS imported_documents (
    id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id        UUID REFERENCES accounts(id),
    source_type       TEXT NOT NULL,    -- 'email'|'csv'|'xlsx'|'pdf'
    original_reference TEXT,            -- filename or email message-id
    sha256_hash       TEXT,             -- idempotency key for files
    status            TEXT DEFAULT 'processed',
    imported_at       TIMESTAMPTZ DEFAULT NOW(),
    metadata_json     JSONB
);

-- Email Records
CREATE TABLE IF NOT EXISTS email_records (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    gmail_message_id TEXT UNIQUE NOT NULL,   -- Gmail's immutable message-id
    from_address     TEXT,
    subject          TEXT,
    received_at      TIMESTAMPTZ,
    broker_id        UUID REFERENCES brokers(id),
    document_id      UUID REFERENCES imported_documents(id),
    raw_body         TEXT,
    processed        BOOLEAN DEFAULT FALSE,
    created_at       TIMESTAMPTZ DEFAULT NOW()
);

-- Orders  (one row per execution / fill)
CREATE TABLE IF NOT EXISTS orders (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id          UUID REFERENCES accounts(id) ON DELETE RESTRICT,
    instrument_id       UUID REFERENCES instruments(id) ON DELETE RESTRICT,
    document_id         UUID REFERENCES imported_documents(id),
    external_order_id   TEXT,               -- broker's own order/ref ID
    order_date          DATE NOT NULL,
    order_time          TIMETZ,
    action              TEXT NOT NULL,      -- 'BUY'|'SELL'|'SHORT'|'COVER'
    price               NUMERIC(18,6) NOT NULL,
    quantity            NUMERIC(18,4) NOT NULL,
    fees                NUMERIC(18,4) DEFAULT 0,
    currency            TEXT NOT NULL,
    stop_loss_at_entry  NUMERIC(18,6),
    raw_data_json       JSONB,              -- preserve original parsed row
    created_at          TIMESTAMPTZ DEFAULT NOW(),
    -- prevent duplicate imports from same account+broker ref
    UNIQUE (account_id, external_order_id)
);

-- Positions  (one per complete trade lifecycle: open → close)
CREATE TABLE IF NOT EXISTS positions (
    id                          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id                  UUID REFERENCES accounts(id) ON DELETE RESTRICT,
    instrument_id               UUID REFERENCES instruments(id) ON DELETE RESTRICT,
    direction                   TEXT NOT NULL,      -- 'long'|'short'
    open_date                   DATE NOT NULL,
    close_date                  DATE,
    status                      TEXT DEFAULT 'open', -- 'open'|'closed'|'partial'
    avg_entry_price             NUMERIC(18,6),
    avg_exit_price              NUMERIC(18,6),
    total_quantity              NUMERIC(18,4),
    realized_pnl_raw            NUMERIC(18,4),  -- in instrument currency
    realized_pnl_hkd            NUMERIC(18,4),  -- converted to HKD
    initial_risk                NUMERIC(18,4),  -- |entry - initial_stop| × qty
    trade_r                     NUMERIC(10,4),
    strategy                    TEXT,
    emotion                     TEXT,
    market_condition            TEXT,
    mistake_tags                TEXT[],
    notes                       TEXT,
    chart_url                   TEXT,
    created_at                  TIMESTAMPTZ DEFAULT NOW(),
    updated_at                  TIMESTAMPTZ DEFAULT NOW()
);

-- Position ↔ Order link
CREATE TABLE IF NOT EXISTS position_orders (
    position_id UUID REFERENCES positions(id) ON DELETE CASCADE,
    order_id    UUID REFERENCES orders(id) ON DELETE CASCADE UNIQUE,
    role        TEXT NOT NULL,  -- 'entry'|'add'|'trim'|'exit'|'stop_loss'
    PRIMARY KEY (position_id, order_id)
);

-- Review Queue  (staging area before writing to orders)
CREATE TABLE IF NOT EXISTS review_queue (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id         UUID REFERENCES imported_documents(id),
    email_record_id     UUID REFERENCES email_records(id),
    raw_parsed_json     JSONB NOT NULL,
    normalized_json     JSONB,
    confidence_score    NUMERIC(4,3) DEFAULT 1.0,
    status              TEXT DEFAULT 'pending',  -- 'pending'|'approved'|'rejected'|'auto_approved'
    review_notes        TEXT,
    reviewed_at         TIMESTAMPTZ,
    created_at          TIMESTAMPTZ DEFAULT NOW()
);

-- Cash Movements
CREATE TABLE IF NOT EXISTS cash_movements (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    account_id    UUID REFERENCES accounts(id),
    document_id   UUID REFERENCES imported_documents(id),
    movement_date DATE NOT NULL,
    type          TEXT NOT NULL,  -- 'fee'|'dividend'|'deposit'|'withdrawal'
    amount        NUMERIC(18,4) NOT NULL,
    currency      TEXT NOT NULL,
    description   TEXT,
    created_at    TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================
-- Seed: brokers
-- ============================================================
INSERT INTO brokers (short_code, display_name, email_sender_patterns)
VALUES
    ('IBKR',  'Interactive Brokers',   ARRAY['%ibkr%', '%interactivebrokers%']),
    ('CHIEF', 'Chief Securities',      ARRAY['%chief%'])
ON CONFLICT (short_code) DO NOTHING;
