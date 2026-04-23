"""
Thin asyncpg wrapper.

Owns the connection pool for all Postgres / TimescaleDB writes. Consumers:
    - KillSwitch, MLSignalFilter → get_state / set_state (bot_state table)
    - Orchestrator → upsert_ohlcv, record_funding_rate, record_signal,
                     update_order_status
    - OrderExecutor → record_order, update_order_status
    - PortfolioTracker → record_fill, record_equity, snapshot_positions
    - Analytics, monitoring.health → read-only SQL via `storage.pool`

apply_schema() reads `schema.sql` from this package and runs it inside a
transaction. The DDL uses CREATE … IF NOT EXISTS everywhere, so re-applying
the schema on every boot is safe and idempotent.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import asyncpg

from src.utils.logging import get_logger

log = get_logger(__name__)


_SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Storage:
    """asyncpg-backed persistence layer."""

    def __init__(
        self,
        dsn: str,
        pool_min: int = 2,
        pool_max: int = 10,
        statement_cache_size: int = 1024,
    ) -> None:
        self._dsn = dsn
        self._pool_min = pool_min
        self._pool_max = pool_max
        self._statement_cache_size = statement_cache_size
        self._pool: asyncpg.Pool | None = None

    # ─────────────────────── lifecycle ───────────────────────

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(
            dsn=self._dsn,
            min_size=self._pool_min,
            max_size=self._pool_max,
            statement_cache_size=self._statement_cache_size,
            init=self._init_connection,
        )
        log.info("storage.connected", pool_min=self._pool_min, pool_max=self._pool_max)

    @staticmethod
    async def _init_connection(conn: asyncpg.Connection) -> None:
        """Register the JSONB codec so Python dicts round-trip through jsonb columns."""
        await conn.set_type_codec(
            "jsonb",
            encoder=json.dumps,
            decoder=json.loads,
            schema="pg_catalog",
        )

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            log.info("storage.closed")

    @property
    def pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError(
                "Storage pool not connected; call await storage.connect() first"
            )
        return self._pool

    async def apply_schema(self) -> None:
        sql_text = _SCHEMA_PATH.read_text(encoding="utf-8")
        async with self.pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(sql_text)
        log.info("storage.schema_applied")

    # ─────────────────────── bot_state key-value ───────────────────────

    async def get_state(self, key: str) -> dict[str, Any] | None:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT value FROM bot_state WHERE key = $1",
                key,
            )
        if row is None:
            return None
        raw = row["value"]
        if isinstance(raw, str):
            return json.loads(raw)
        return dict(raw) if raw is not None else None

    async def set_state(self, key: str, value: dict[str, Any]) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO bot_state (key, value, updated_at)
                VALUES ($1, $2, NOW())
                ON CONFLICT (key) DO UPDATE
                    SET value = EXCLUDED.value, updated_at = NOW()
                """,
                key,
                value,
            )

    # ─────────────────────── market data ───────────────────────

    async def upsert_ohlcv(
        self,
        *,
        exchange: str,
        symbol: str,
        timeframe: str,
        bars: list[tuple[datetime, float, float, float, float, float]],
    ) -> None:
        if not bars:
            return
        rows = [
            (ts, exchange, symbol, timeframe, o, h, low, c, v)
            for (ts, o, h, low, c, v) in bars
        ]
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO ohlcv
                    (ts, exchange, symbol, timeframe, open, high, low, close, volume)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (exchange, symbol, timeframe, ts) DO UPDATE
                    SET open = EXCLUDED.open,
                        high = EXCLUDED.high,
                        low  = EXCLUDED.low,
                        close = EXCLUDED.close,
                        volume = EXCLUDED.volume
                """,
                rows,
            )

    async def record_funding_rate(
        self,
        *,
        ts: datetime,
        exchange: str,
        symbol: str,
        rate: float,
        next_funding_ts: datetime | None,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO funding_rates (ts, exchange, symbol, rate, next_funding_ts)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (exchange, symbol, ts) DO UPDATE
                    SET rate = EXCLUDED.rate,
                        next_funding_ts = EXCLUDED.next_funding_ts
                """,
                ts, exchange, symbol, rate, next_funding_ts,
            )

    # ─────────────────────── signal / order / fill ───────────────────────

    async def record_signal(
        self,
        *,
        ts: datetime,
        strategy: str,
        exchange: str,
        symbol: str,
        timeframe: str | None,
        side: str,
        confidence: float,
        regime: str | None,
        suggested_sl: float | None,
        suggested_tp: float | None,
        meta: dict[str, Any] | None,
    ) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO signals
                    (ts, strategy, exchange, symbol, timeframe, side, confidence,
                     regime, suggested_sl, suggested_tp, meta)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                RETURNING id
                """,
                ts, strategy, exchange, symbol, timeframe, side, confidence,
                regime, suggested_sl, suggested_tp, meta or {},
            )
        return int(row["id"])

    async def record_order(
        self,
        *,
        mode: str,
        exchange: str,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float | None,
        status: str,
        client_order_id: str,
        strategy: str | None,
        signal_id: int | None,
        stop_loss: float | None,
        take_profit: float | None,
    ) -> int:
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO orders
                    (mode, exchange, symbol, side, order_type, quantity, price,
                     status, client_order_id, strategy, signal_id,
                     stop_loss, take_profit)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                RETURNING id
                """,
                mode, exchange, symbol, side, order_type, quantity, price,
                status, client_order_id, strategy, signal_id,
                stop_loss, take_profit,
            )
        return int(row["id"])

    async def update_order_status(
        self,
        client_order_id: str,
        status: str,
        *,
        exchange_order_id: str | None = None,
        raw_response: dict[str, Any] | None = None,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                UPDATE orders
                SET status = $2,
                    exchange_order_id = COALESCE($3, exchange_order_id),
                    raw_response = COALESCE($4, raw_response),
                    updated_at = NOW()
                WHERE client_order_id = $1
                """,
                client_order_id, status, exchange_order_id, raw_response,
            )

    async def record_fill(
        self,
        *,
        ts: datetime,
        client_order_id: str | None,
        exchange: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        fee: float,
        fee_currency: str | None,
        realized_pnl: float | None,
        raw: dict[str, Any] | None,
    ) -> None:
        async with self.pool.acquire() as conn:
            order_id: int | None = None
            if client_order_id:
                row = await conn.fetchrow(
                    "SELECT id FROM orders WHERE client_order_id = $1",
                    client_order_id,
                )
                if row is not None:
                    order_id = int(row["id"])
            await conn.execute(
                """
                INSERT INTO fills
                    (ts, order_id, client_order_id, exchange, symbol, side,
                     quantity, price, fee, fee_currency, realized_pnl, raw)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """,
                ts, order_id, client_order_id, exchange, symbol, side,
                quantity, price, fee, fee_currency, realized_pnl, raw or {},
            )

    # ─────────────────────── equity + positions ───────────────────────

    async def record_equity(
        self,
        *,
        ts: datetime,
        mode: str,
        equity: float,
        balance: float,
        unrealized_pnl: float,
        open_positions: int,
        drawdown_pct: float,
    ) -> None:
        async with self.pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO equity
                    (ts, mode, equity, balance, unrealized_pnl,
                     open_positions, drawdown_pct)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (mode, ts) DO UPDATE
                    SET equity = EXCLUDED.equity,
                        balance = EXCLUDED.balance,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        open_positions = EXCLUDED.open_positions,
                        drawdown_pct = EXCLUDED.drawdown_pct
                """,
                ts, mode, equity, balance, unrealized_pnl,
                open_positions, drawdown_pct,
            )

    async def snapshot_positions(
        self,
        *,
        mode: str,
        positions: list[dict[str, Any]],
    ) -> None:
        if not positions:
            return
        async with self.pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO positions_snapshot
                    (ts, mode, exchange, symbol, side, quantity, entry_price,
                     mark_price, liquidation_price, unrealized_pnl, leverage,
                     stop_loss, take_profit, strategy)
                VALUES (NOW(), $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13)
                ON CONFLICT (ts, exchange, symbol) DO UPDATE
                    SET side = EXCLUDED.side,
                        quantity = EXCLUDED.quantity,
                        entry_price = EXCLUDED.entry_price,
                        mark_price = EXCLUDED.mark_price,
                        liquidation_price = EXCLUDED.liquidation_price,
                        unrealized_pnl = EXCLUDED.unrealized_pnl,
                        leverage = EXCLUDED.leverage,
                        stop_loss = EXCLUDED.stop_loss,
                        take_profit = EXCLUDED.take_profit,
                        strategy = EXCLUDED.strategy
                """,
                [
                    (
                        mode,
                        p["exchange"], p["symbol"], p["side"], p["quantity"],
                        p.get("entry_price"), p.get("mark_price"),
                        p.get("liquidation_price"), p.get("unrealized_pnl"),
                        p.get("leverage"), p.get("stop_loss"), p.get("take_profit"),
                        p.get("strategy"),
                    )
                    for p in positions
                ],
            )
