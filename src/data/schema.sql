-- Cryprobotik database schema.
--
-- Applied idempotently by Storage.apply_schema() on every bot startup.
-- TimescaleDB hypertables are optional: if the extension is not available,
-- the tables are still created as regular Postgres tables and the bot works
-- without the time-partitioning optimisation.

-- ──────────────────────────── TimescaleDB extension (optional) ────────────────

CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ──────────────────────────── market data ─────────────────────────────────────

CREATE TABLE IF NOT EXISTS ohlcv (
    ts          TIMESTAMPTZ      NOT NULL,
    exchange    TEXT             NOT NULL,
    symbol      TEXT             NOT NULL,
    timeframe   TEXT             NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      DOUBLE PRECISION NOT NULL,
    PRIMARY KEY (exchange, symbol, timeframe, ts)
);

CREATE INDEX IF NOT EXISTS ohlcv_ts_idx ON ohlcv (ts DESC);

DO $$
BEGIN
    PERFORM create_hypertable('ohlcv', 'ts', if_not_exists => TRUE);
EXCEPTION
    WHEN undefined_function THEN NULL;
    WHEN others THEN NULL;
END$$;

CREATE TABLE IF NOT EXISTS funding_rates (
    ts              TIMESTAMPTZ      NOT NULL,
    exchange        TEXT             NOT NULL,
    symbol          TEXT             NOT NULL,
    rate            DOUBLE PRECISION NOT NULL,
    next_funding_ts TIMESTAMPTZ,
    PRIMARY KEY (exchange, symbol, ts)
);

CREATE INDEX IF NOT EXISTS funding_rates_ts_idx ON funding_rates (ts DESC);

DO $$
BEGIN
    PERFORM create_hypertable('funding_rates', 'ts', if_not_exists => TRUE);
EXCEPTION
    WHEN undefined_function THEN NULL;
    WHEN others THEN NULL;
END$$;

-- ──────────────────────────── signal / order / fill pipeline ──────────────────

CREATE TABLE IF NOT EXISTS signals (
    id              BIGSERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ      NOT NULL,
    strategy        TEXT             NOT NULL,
    exchange        TEXT             NOT NULL,
    symbol          TEXT             NOT NULL,
    timeframe       TEXT,
    side            TEXT             NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    regime          TEXT,
    suggested_sl    DOUBLE PRECISION,
    suggested_tp    DOUBLE PRECISION,
    meta            JSONB
);

CREATE INDEX IF NOT EXISTS signals_ts_idx ON signals (ts DESC);
CREATE INDEX IF NOT EXISTS signals_symbol_idx ON signals (symbol, ts DESC);

CREATE TABLE IF NOT EXISTS orders (
    id                  BIGSERIAL PRIMARY KEY,
    ts                  TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    mode                TEXT             NOT NULL,
    exchange            TEXT             NOT NULL,
    symbol              TEXT             NOT NULL,
    side                TEXT             NOT NULL,
    order_type          TEXT             NOT NULL,
    quantity            DOUBLE PRECISION NOT NULL,
    price               DOUBLE PRECISION,
    status              TEXT             NOT NULL,
    client_order_id     TEXT             NOT NULL UNIQUE,
    exchange_order_id   TEXT,
    strategy            TEXT,
    signal_id           BIGINT           REFERENCES signals(id) ON DELETE SET NULL,
    stop_loss           DOUBLE PRECISION,
    take_profit         DOUBLE PRECISION,
    raw_response        JSONB,
    updated_at          TIMESTAMPTZ      NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS orders_ts_idx ON orders (ts DESC);
CREATE INDEX IF NOT EXISTS orders_symbol_idx ON orders (symbol, ts DESC);
CREATE INDEX IF NOT EXISTS orders_status_idx ON orders (status);

CREATE TABLE IF NOT EXISTS fills (
    id                  BIGSERIAL PRIMARY KEY,
    ts                  TIMESTAMPTZ      NOT NULL,
    order_id            BIGINT           REFERENCES orders(id) ON DELETE SET NULL,
    client_order_id     TEXT,
    exchange            TEXT             NOT NULL,
    symbol              TEXT             NOT NULL,
    side                TEXT             NOT NULL,
    quantity            DOUBLE PRECISION NOT NULL,
    price               DOUBLE PRECISION NOT NULL,
    fee                 DOUBLE PRECISION NOT NULL DEFAULT 0,
    fee_currency        TEXT,
    realized_pnl        DOUBLE PRECISION,
    raw                 JSONB
);

CREATE INDEX IF NOT EXISTS fills_ts_idx ON fills (ts DESC);
CREATE INDEX IF NOT EXISTS fills_client_order_idx ON fills (client_order_id);

-- ──────────────────────────── equity + positions ──────────────────────────────

CREATE TABLE IF NOT EXISTS equity (
    ts                  TIMESTAMPTZ      NOT NULL,
    mode                TEXT             NOT NULL,
    equity              DOUBLE PRECISION NOT NULL,
    balance             DOUBLE PRECISION NOT NULL,
    unrealized_pnl      DOUBLE PRECISION NOT NULL DEFAULT 0,
    open_positions      INTEGER          NOT NULL DEFAULT 0,
    drawdown_pct        DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (mode, ts)
);

CREATE INDEX IF NOT EXISTS equity_ts_idx ON equity (ts DESC);

DO $$
BEGIN
    PERFORM create_hypertable('equity', 'ts', if_not_exists => TRUE);
EXCEPTION
    WHEN undefined_function THEN NULL;
    WHEN others THEN NULL;
END$$;

CREATE OR REPLACE VIEW equity_daily AS
SELECT date_trunc('day', ts)                                        AS day,
       mode,
       MIN(equity)                                                  AS low_equity,
       MAX(equity)                                                  AS high_equity,
       (ARRAY_AGG(equity ORDER BY ts ASC))[1]                       AS open_equity,
       (ARRAY_AGG(equity ORDER BY ts DESC))[1]                      AS close_equity
FROM equity
GROUP BY date_trunc('day', ts), mode;

CREATE TABLE IF NOT EXISTS positions_snapshot (
    ts                  TIMESTAMPTZ      NOT NULL,
    mode                TEXT             NOT NULL,
    exchange            TEXT             NOT NULL,
    symbol              TEXT             NOT NULL,
    side                TEXT             NOT NULL,
    quantity            DOUBLE PRECISION NOT NULL,
    entry_price         DOUBLE PRECISION,
    mark_price          DOUBLE PRECISION,
    liquidation_price   DOUBLE PRECISION,
    unrealized_pnl      DOUBLE PRECISION,
    leverage            DOUBLE PRECISION,
    stop_loss           DOUBLE PRECISION,
    take_profit         DOUBLE PRECISION,
    strategy            TEXT,
    PRIMARY KEY (ts, exchange, symbol)
);

CREATE INDEX IF NOT EXISTS positions_snapshot_symbol_idx
    ON positions_snapshot (exchange, symbol, ts DESC);

-- ──────────────────────────── ML decisions ────────────────────────────────────

CREATE TABLE IF NOT EXISTS ml_decisions (
    ts              TIMESTAMPTZ      NOT NULL DEFAULT NOW(),
    symbol          TEXT             NOT NULL,
    exchange        TEXT             NOT NULL,
    side            TEXT             NOT NULL,
    confidence      DOUBLE PRECISION NOT NULL,
    ml_score        DOUBLE PRECISION NOT NULL,
    accepted        BOOLEAN          NOT NULL,
    cold_start      BOOLEAN          NOT NULL DEFAULT FALSE,
    model_version   INTEGER          NOT NULL DEFAULT 0,
    regime          TEXT,
    net_vote        DOUBLE PRECISION,
    features        JSONB
);

CREATE INDEX IF NOT EXISTS ml_decisions_ts_idx ON ml_decisions (ts DESC);
CREATE INDEX IF NOT EXISTS ml_decisions_symbol_idx ON ml_decisions (symbol, ts DESC);

-- ──────────────────────────── key-value state + audit ────────────────────────

CREATE TABLE IF NOT EXISTS bot_state (
    key         TEXT         PRIMARY KEY,
    value       JSONB        NOT NULL,
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    level       TEXT         NOT NULL,
    event       TEXT         NOT NULL,
    payload     JSONB
);

CREATE INDEX IF NOT EXISTS events_ts_idx ON events (ts DESC);
CREATE INDEX IF NOT EXISTS events_event_idx ON events (event, ts DESC);
