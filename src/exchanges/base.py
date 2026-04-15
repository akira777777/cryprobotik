"""
Exchange connector abstract base class and shared types.

Every exchange implementation (OKX, Bybit, paper) must implement
`ExchangeConnector`. Strategies and the executor depend ONLY on this ABC — they
never import from `okx.py` or `bybit.py` directly. This keeps the code paths
swappable and makes paper mode a genuine drop-in.

All events flow through asyncio.Queue objects exposed as read-only properties:
  - kline_events       : KlineEvent (bar updates, closed or forming)
  - funding_events     : FundingRateEvent
  - order_events       : OrderUpdateEvent (private)
  - position_events    : PositionUpdateEvent (private)

Timestamps are tz-aware UTC. Symbols use the ccxt unified format
("BTC/USDT:USDT" for USDT-margined perps) so one symbol string identifies the
same instrument across venues.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class OrderSide(StrEnum):
    BUY = "buy"
    SELL = "sell"


class OrderType(StrEnum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_MARKET = "stop_market"
    STOP_LIMIT = "stop_limit"


class OrderStatus(StrEnum):
    NEW = "new"
    OPEN = "open"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionSide(StrEnum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


# ──────────────────────────────────────────────────────────────────────────────
# Event dataclasses — normalized shape across exchanges
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class KlineEvent:
    exchange: str
    symbol: str
    timeframe: str
    ts: datetime       # bar OPEN time, aligned to timeframe boundary
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool       # True when the bar has closed; False for in-progress updates


@dataclass(slots=True)
class FundingRateEvent:
    exchange: str
    symbol: str
    ts: datetime
    rate: float                       # current/next funding rate, decimal (0.0001 = 0.01%)
    next_funding_ts: datetime | None  # when the next settlement happens


@dataclass(slots=True)
class TradeEvent:
    """Individual aggressor trade — used to compute CVD (cumulative volume delta)."""
    exchange: str
    symbol: str
    ts: datetime
    side: "OrderSide"   # BUY = taker bought (aggressor was buyer)
    qty: float          # base-currency quantity
    price: float


@dataclass(slots=True)
class OIEvent:
    """Open-interest snapshot — used to compute OI rate-of-change."""
    exchange: str
    symbol: str
    ts: datetime
    oi_contracts: float   # open interest in base-currency contracts


@dataclass(slots=True)
class OrderRequest:
    exchange: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float                    # in base currency
    price: float | None = None         # required for limit orders
    client_order_id: str | None = None
    reduce_only: bool = False
    stop_loss: float | None = None
    take_profit: float | None = None
    leverage: int | None = None        # set leverage per-symbol before order
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrderResult:
    """Returned by place_order once the exchange accepts (or rejects) the request."""
    client_order_id: str
    exchange_order_id: str | None
    status: OrderStatus
    filled_quantity: float
    average_price: float | None
    raw: dict[str, Any]


@dataclass(slots=True)
class OrderUpdateEvent:
    """Private WS event: order transitioned to a new state."""
    exchange: str
    symbol: str
    client_order_id: str | None
    exchange_order_id: str
    status: OrderStatus
    side: OrderSide
    quantity: float
    filled_quantity: float
    average_price: float | None
    ts: datetime
    raw: dict[str, Any]


@dataclass(slots=True)
class FillEvent:
    exchange: str
    symbol: str
    client_order_id: str | None
    exchange_order_id: str
    side: OrderSide
    quantity: float
    price: float
    fee: float
    fee_currency: str | None
    realized_pnl: float | None
    ts: datetime
    raw: dict[str, Any]


@dataclass(slots=True)
class PositionUpdateEvent:
    exchange: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float | None
    mark_price: float | None
    liquidation_price: float | None
    unrealized_pnl: float | None
    leverage: float | None
    ts: datetime


@dataclass(slots=True)
class Balance:
    total: float           # total equity including unrealized pnl
    free: float            # available for new positions
    used: float            # locked in open orders / positions
    currency: str = "USDT"


@dataclass(slots=True)
class TickerInfo:
    """Minimal shape used by UniverseSelector."""
    symbol: str
    volume_usd_24h: float
    last_price: float
    raw: dict[str, Any]


# ──────────────────────────────────────────────────────────────────────────────
# ExchangeConnector ABC
# ──────────────────────────────────────────────────────────────────────────────

class ExchangeConnector(ABC):
    """
    Contract that every exchange backend must fulfil.

    Responsibilities:
        - Manage one REST client and one (or more) WS client(s).
        - Keep WS subscriptions alive (reconnect + resubscribe).
        - Push normalized events onto the shared asyncio queues.
        - Execute orders with idempotency (client_order_id).

    Implementations do NOT take trading decisions. They ONLY translate requests
    and events between the exchange's wire format and the normalized dataclasses
    defined above.
    """

    name: str = "abstract"

    def __init__(self, queue_maxsize: int = 10000) -> None:
        self._kline_q: asyncio.Queue[KlineEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._funding_q: asyncio.Queue[FundingRateEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._order_q: asyncio.Queue[OrderUpdateEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._fill_q: asyncio.Queue[FillEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._position_q: asyncio.Queue[PositionUpdateEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._trade_q: asyncio.Queue[TradeEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._oi_q: asyncio.Queue[OIEvent] = asyncio.Queue(maxsize=queue_maxsize)

    # ─────────────────── event queues (read-only) ───────────────────

    @property
    def kline_events(self) -> asyncio.Queue[KlineEvent]:
        return self._kline_q

    @property
    def funding_events(self) -> asyncio.Queue[FundingRateEvent]:
        return self._funding_q

    @property
    def order_events(self) -> asyncio.Queue[OrderUpdateEvent]:
        return self._order_q

    @property
    def fill_events(self) -> asyncio.Queue[FillEvent]:
        return self._fill_q

    @property
    def position_events(self) -> asyncio.Queue[PositionUpdateEvent]:
        return self._position_q

    @property
    def trade_events(self) -> asyncio.Queue[TradeEvent]:
        """Individual aggressor trades for CVD calculation."""
        return self._trade_q

    @property
    def oi_events(self) -> asyncio.Queue[OIEvent]:
        """Open-interest snapshots for OI ROC calculation."""
        return self._oi_q

    # ─────────────────── lifecycle ───────────────────

    @abstractmethod
    async def connect(self) -> None:
        """Open REST + WS connections. Must be idempotent."""

    @abstractmethod
    async def close(self) -> None:
        """Close all connections gracefully."""

    # ─────────────────── subscriptions (public WS) ───────────────────

    @abstractmethod
    async def subscribe_klines(self, symbol: str, timeframe: str) -> None:
        """Subscribe to kline updates for (symbol, timeframe)."""

    @abstractmethod
    async def unsubscribe_klines(self, symbol: str, timeframe: str) -> None:
        ...

    @abstractmethod
    async def subscribe_funding(self, symbol: str) -> None:
        ...

    @abstractmethod
    async def unsubscribe_funding(self, symbol: str) -> None:
        ...

    @abstractmethod
    async def subscribe_trades(self, symbol: str) -> None:
        """Subscribe to individual trade events for CVD computation."""

    @abstractmethod
    async def unsubscribe_trades(self, symbol: str) -> None:
        ...

    # ─────────────────── REST: market data ───────────────────

    @abstractmethod
    async def fetch_ohlcv_backfill(
        self, symbol: str, timeframe: str, limit: int = 500
    ) -> list[KlineEvent]:
        """Fetch historical bars via REST for feature_store seeding."""

    @abstractmethod
    async def fetch_24h_tickers(
        self, quote: str = "USDT", instrument_type: str = "swap"
    ) -> list[TickerInfo]:
        """Fetch 24h volume rankings for universe selection."""

    @abstractmethod
    async def fetch_funding_rate(self, symbol: str) -> FundingRateEvent:
        """Return current funding rate for a symbol."""

    @abstractmethod
    async def fetch_open_interest(self, symbol: str) -> OIEvent:
        """Return current open interest snapshot for OI ROC tracking."""

    # ─────────────────── REST: account ───────────────────

    @abstractmethod
    async def fetch_balance(self) -> Balance:
        ...

    @abstractmethod
    async def fetch_positions(self) -> list[PositionUpdateEvent]:
        """Return all currently open positions."""

    @abstractmethod
    async def set_leverage(self, symbol: str, leverage: int) -> None:
        """Set per-symbol leverage. No-op if already at that leverage."""

    # ─────────────────── trading ───────────────────

    @abstractmethod
    async def place_order(self, req: OrderRequest) -> OrderResult:
        """Place an order. Idempotent when client_order_id is set."""

    @abstractmethod
    async def cancel_order(self, symbol: str, *, client_order_id: str | None = None,
                           exchange_order_id: str | None = None) -> None:
        ...

    @abstractmethod
    async def cancel_all(self, symbol: str | None = None) -> None:
        """Cancel all open orders, optionally scoped to one symbol."""

    # ─────────────────── misc ───────────────────

    @abstractmethod
    async def close_position(self, symbol: str) -> OrderResult:
        """Flatten a position at market using a reduce-only order."""
