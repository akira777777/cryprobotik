"""
Paper-trading connector.

Wraps a REAL read-only connector (OKX or Bybit in live mode) so the bot sees
true mainnet prices and funding rates, but intercepts every `place_order` /
`cancel_order` / `close_position` call into an in-memory matching engine.

Simulated fills:
- Market orders fill instantly at the last-seen close price of the symbol,
  plus configurable slippage (in bps) applied against the taker.
- Limit orders are tracked and filled the next time the live price crosses
  the limit.
- Fees are charged from a simulated balance (starting_balance_usd).

Positions and PnL are persisted to the same TimescaleDB tables as live mode,
with `mode='paper'` so analytics are identical.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.exchanges.base import (
    Balance,
    ExchangeConnector,
    FillEvent,
    FundingRateEvent,
    KlineEvent,
    OIEvent,
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    OrderUpdateEvent,
    PositionSide,
    PositionUpdateEvent,
    TickerInfo,
    TradeEvent,
)
from src.settings import PaperConfig
from src.utils.logging import get_logger
from src.utils.time import now_utc

log = get_logger(__name__)


@dataclass(slots=True)
class _PaperPosition:
    symbol: str
    side: PositionSide = PositionSide.FLAT
    quantity: float = 0.0
    entry_price: float = 0.0
    stop_loss: float | None = None
    take_profit: float | None = None
    strategy: str | None = None
    initial_stop_distance: float = 0.0   # |entry - original_sl|, set on open
    breakeven_triggered: bool = False     # True once SL moved to entry price

    def notional(self, mark_price: float) -> float:
        return self.quantity * mark_price

    def unrealized_pnl(self, mark_price: float) -> float:
        if self.side == PositionSide.FLAT or self.quantity == 0:
            return 0.0
        if self.side == PositionSide.LONG:
            return (mark_price - self.entry_price) * self.quantity
        return (self.entry_price - mark_price) * self.quantity


@dataclass(slots=True)
class _PaperOrder:
    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    limit_price: float | None
    reduce_only: bool
    stop_loss: float | None
    take_profit: float | None
    status: OrderStatus = OrderStatus.OPEN
    strategy: str | None = None


class PaperConnector(ExchangeConnector):
    """
    Paper-trading wrapper. The `name` is the underlying exchange's name with a
    'paper-' prefix — that way the DB rows still identify which venue the
    simulated trades corresponded to.
    """

    def __init__(
        self,
        underlying: ExchangeConnector,
        paper_config: PaperConfig,
    ) -> None:
        super().__init__()
        self._u = underlying
        self._cfg = paper_config
        # Keep the full 'paper-{name}' as the public connector name for DB
        # mode tagging and logging.  But expose the underlying exchange name
        # separately so event routing in the orchestrator works correctly —
        # the orchestrator stores connectors under the underlying name (e.g.
        # "okx"), and PortfolioTracker must find positions under the same key.
        self.name = f"paper-{underlying.name}"  # type: ignore[assignment]
        self._routing_name: str = underlying.name

        self._balance = paper_config.starting_balance_usd
        self._positions: dict[str, _PaperPosition] = {}
        self._open_orders: dict[str, _PaperOrder] = {}
        self._last_prices: dict[str, float] = {}
        self._lock = asyncio.Lock()
        self._pump_task: asyncio.Task[None] | None = None

    # ─────────────────────── lifecycle ───────────────────────

    async def connect(self) -> None:
        await self._u.connect()
        # Spawn a task that forwards underlying kline/funding events to our own
        # queues and maintains last_prices, and fills resting limit orders.
        self._pump_task = asyncio.create_task(self._pump_events(), name=f"{self.name}.pump")

    async def close(self) -> None:
        if self._pump_task and not self._pump_task.done():
            self._pump_task.cancel()
            try:
                await self._pump_task
            except (asyncio.CancelledError, Exception):
                pass
        await self._u.close()

    async def _pump_events(self) -> None:
        """Forward events from underlying and run the matching engine on each kline."""
        kline_task = asyncio.create_task(self._pump_klines())
        funding_task = asyncio.create_task(self._pump_funding())
        trades_task = asyncio.create_task(self._pump_trades())
        try:
            await asyncio.gather(kline_task, funding_task, trades_task)
        except asyncio.CancelledError:
            kline_task.cancel()
            funding_task.cancel()
            trades_task.cancel()
            raise

    async def _pump_klines(self) -> None:
        while True:
            evt = await self._u.kline_events.get()
            # Mark this instance as the event source so downstream storage rows
            # get the 'paper-*' name.
            forwarded = KlineEvent(
                exchange=self.name,
                symbol=evt.symbol,
                timeframe=evt.timeframe,
                ts=evt.ts,
                open=evt.open,
                high=evt.high,
                low=evt.low,
                close=evt.close,
                volume=evt.volume,
                closed=evt.closed,
            )
            self._last_prices[evt.symbol] = evt.close
            try:
                self._kline_q.put_nowait(forwarded)
            except asyncio.QueueFull:
                pass

            # Run the matching engine on every kline update.
            await self._match_resting_orders(evt.symbol, evt.high, evt.low, evt.close)
            await self._maybe_trigger_sl_tp(evt.symbol, evt.high, evt.low)

    async def _pump_funding(self) -> None:
        while True:
            evt = await self._u.funding_events.get()
            forwarded = FundingRateEvent(
                exchange=self.name,
                symbol=evt.symbol,
                ts=evt.ts,
                rate=evt.rate,
                next_funding_ts=evt.next_funding_ts,
            )
            try:
                self._funding_q.put_nowait(forwarded)
            except asyncio.QueueFull:
                pass

    async def _pump_trades(self) -> None:
        while True:
            evt = await self._u.trade_events.get()
            forwarded = TradeEvent(
                exchange=self.name,
                symbol=evt.symbol,
                ts=evt.ts,
                side=evt.side,
                qty=evt.qty,
                price=evt.price,
            )
            try:
                self._trade_q.put_nowait(forwarded)
            except asyncio.QueueFull:
                pass

    # ─────────────────────── subscriptions pass-through ───────────────────────

    async def subscribe_klines(self, symbol: str, timeframe: str) -> None:
        await self._u.subscribe_klines(symbol, timeframe)

    async def unsubscribe_klines(self, symbol: str, timeframe: str) -> None:
        await self._u.unsubscribe_klines(symbol, timeframe)

    async def subscribe_funding(self, symbol: str) -> None:
        await self._u.subscribe_funding(symbol)

    async def unsubscribe_funding(self, symbol: str) -> None:
        await self._u.unsubscribe_funding(symbol)

    async def subscribe_trades(self, symbol: str) -> None:
        await self._u.subscribe_trades(symbol)

    async def unsubscribe_trades(self, symbol: str) -> None:
        await self._u.unsubscribe_trades(symbol)

    # ─────────────────────── REST pass-through (market data) ───────────────────────

    async def fetch_ohlcv_backfill(self, symbol: str, timeframe: str, limit: int = 500) -> list[KlineEvent]:
        return await self._u.fetch_ohlcv_backfill(symbol, timeframe, limit)

    async def fetch_24h_tickers(self, quote: str = "USDT", instrument_type: str = "swap") -> list[TickerInfo]:
        return await self._u.fetch_24h_tickers(quote=quote, instrument_type=instrument_type)

    async def fetch_funding_rate(self, symbol: str) -> FundingRateEvent:
        return await self._u.fetch_funding_rate(symbol)

    async def fetch_open_interest(self, symbol: str) -> OIEvent:
        return await self._u.fetch_open_interest(symbol)

    # ─────────────────────── REST: simulated account ───────────────────────

    async def fetch_balance(self) -> Balance:
        async with self._lock:
            leverage = getattr(self._cfg, "leverage", 1) or 1
            margin_used = sum(
                abs(p.quantity * p.entry_price) / max(1, leverage)
                for p in self._positions.values()
                if p.side != PositionSide.FLAT
            )
            unrealized = sum(
                p.unrealized_pnl(self._last_prices.get(p.symbol, p.entry_price))
                for p in self._positions.values()
            )
            total = self._balance + unrealized
            free = max(0.0, total - margin_used)
            return Balance(
                total=total,
                free=free,
                used=margin_used,
                currency="USDT",
            )

    async def fetch_positions(self) -> list[PositionUpdateEvent]:
        async with self._lock:
            out: list[PositionUpdateEvent] = []
            for p in self._positions.values():
                if p.side == PositionSide.FLAT or p.quantity == 0:
                    continue
                mark = self._last_prices.get(p.symbol, p.entry_price)
                out.append(
                    PositionUpdateEvent(
                        exchange=self._routing_name,
                        symbol=p.symbol,
                        side=p.side,
                        quantity=p.quantity,
                        entry_price=p.entry_price,
                        mark_price=mark,
                        liquidation_price=None,
                        unrealized_pnl=p.unrealized_pnl(mark),
                        leverage=None,
                        ts=now_utc(),
                    )
                )
            return out

    async def set_leverage(self, symbol: str, leverage: int) -> None:
        # Paper mode doesn't enforce leverage independently — the risk module
        # already sizes positions accordingly.
        return None

    # ─────────────────────── trading (simulated) ───────────────────────

    async def place_order(self, req: OrderRequest) -> OrderResult:
        async with self._lock:
            client_id = req.client_order_id or f"paper-{uuid.uuid4().hex[:12]}"
            exch_id = f"PAPER-{uuid.uuid4().hex[:14]}"
            mark = self._last_prices.get(req.symbol)

            if req.order_type == OrderType.MARKET:
                if mark is None:
                    return OrderResult(
                        client_order_id=client_id,
                        exchange_order_id=None,
                        status=OrderStatus.REJECTED,
                        filled_quantity=0.0,
                        average_price=None,
                        raw={"reason": "no live price available for symbol"},
                    )
                fill_price = self._apply_slippage(mark, req.side)
                await self._execute_fill(
                    client_id,
                    exch_id,
                    req.symbol,
                    req.side,
                    req.quantity,
                    fill_price,
                    is_maker=False,
                    reduce_only=req.reduce_only,
                    stop_loss=req.stop_loss,
                    take_profit=req.take_profit,
                    strategy=req.meta.get("strategy"),
                )
                return OrderResult(
                    client_order_id=client_id,
                    exchange_order_id=exch_id,
                    status=OrderStatus.FILLED,
                    filled_quantity=req.quantity,
                    average_price=fill_price,
                    raw={"mode": "paper"},
                )

            # Limit order — track as resting
            order = _PaperOrder(
                client_order_id=client_id,
                exchange_order_id=exch_id,
                symbol=req.symbol,
                side=req.side,
                order_type=req.order_type,
                quantity=req.quantity,
                limit_price=req.price,
                reduce_only=req.reduce_only,
                stop_loss=req.stop_loss,
                take_profit=req.take_profit,
                strategy=req.meta.get("strategy"),
            )
            self._open_orders[client_id] = order
            # Emit open event (use routing name so tracker lookup works)
            self._order_q.put_nowait(
                OrderUpdateEvent(
                    exchange=self._routing_name,
                    symbol=req.symbol,
                    client_order_id=client_id,
                    exchange_order_id=exch_id,
                    status=OrderStatus.OPEN,
                    side=req.side,
                    quantity=req.quantity,
                    filled_quantity=0.0,
                    average_price=None,
                    ts=now_utc(),
                    raw={"mode": "paper"},
                )
            )
            return OrderResult(
                client_order_id=client_id,
                exchange_order_id=exch_id,
                status=OrderStatus.OPEN,
                filled_quantity=0.0,
                average_price=None,
                raw={"mode": "paper"},
            )

    async def cancel_order(
        self,
        symbol: str,
        *,
        client_order_id: str | None = None,
        exchange_order_id: str | None = None,
    ) -> None:
        async with self._lock:
            target_id: str | None = client_order_id
            if target_id is None and exchange_order_id is not None:
                target_id = next(
                    (cid for cid, o in self._open_orders.items() if o.exchange_order_id == exchange_order_id),
                    None,
                )
            if target_id and target_id in self._open_orders:
                order = self._open_orders.pop(target_id)
                self._order_q.put_nowait(
                    OrderUpdateEvent(
                        exchange=self._routing_name,
                        symbol=order.symbol,
                        client_order_id=target_id,
                        exchange_order_id=order.exchange_order_id,
                        status=OrderStatus.CANCELLED,
                        side=order.side,
                        quantity=order.quantity,
                        filled_quantity=0.0,
                        average_price=None,
                        ts=now_utc(),
                        raw={"mode": "paper"},
                    )
                )

    async def cancel_all(self, symbol: str | None = None) -> None:
        async with self._lock:
            to_cancel = [cid for cid, o in self._open_orders.items() if symbol is None or o.symbol == symbol]
        for cid in to_cancel:
            await self.cancel_order(symbol or self._open_orders[cid].symbol, client_order_id=cid)

    async def close_position(self, symbol: str) -> OrderResult:
        async with self._lock:
            pos = self._positions.get(symbol)
            if pos is None or pos.side == PositionSide.FLAT or pos.quantity == 0:
                return OrderResult(
                    client_order_id=f"paper-{uuid.uuid4().hex[:12]}",
                    exchange_order_id=None,
                    status=OrderStatus.REJECTED,
                    filled_quantity=0.0,
                    average_price=None,
                    raw={"reason": "no open position"},
                )
            close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            qty = pos.quantity
        req = OrderRequest(
            exchange=self.name,
            symbol=symbol,
            side=close_side,
            order_type=OrderType.MARKET,
            quantity=qty,
            reduce_only=True,
        )
        return await self.place_order(req)

    # ─────────────────────── matching engine ───────────────────────

    async def _match_resting_orders(self, symbol: str, high: float, low: float, close: float) -> None:
        async with self._lock:
            to_fill: list[_PaperOrder] = []
            for order in list(self._open_orders.values()):
                if order.symbol != symbol or order.limit_price is None:
                    continue
                # Limit buy fills when low <= limit; limit sell when high >= limit.
                if order.side == OrderSide.BUY and low <= order.limit_price:
                    to_fill.append(order)
                elif order.side == OrderSide.SELL and high >= order.limit_price:
                    to_fill.append(order)

            for order in to_fill:
                del self._open_orders[order.client_order_id]
                fill_px = order.limit_price  # maker fill at the limit
                assert fill_px is not None
                await self._execute_fill(
                    order.client_order_id,
                    order.exchange_order_id,
                    order.symbol,
                    order.side,
                    order.quantity,
                    fill_px,
                    is_maker=True,
                    reduce_only=order.reduce_only,
                    stop_loss=order.stop_loss,
                    take_profit=order.take_profit,
                    strategy=order.strategy,
                )

    async def _maybe_trigger_sl_tp(self, symbol: str, high: float, low: float) -> None:
        """Check if SL/TP on any open position has been touched this bar."""
        async with self._lock:
            pos = self._positions.get(symbol)
            if pos is None or pos.side == PositionSide.FLAT:
                return

            hit: str | None = None
            if pos.side == PositionSide.LONG:
                if pos.stop_loss is not None and low <= pos.stop_loss:
                    hit = "sl"
                elif pos.take_profit is not None and high >= pos.take_profit:
                    hit = "tp"
            else:  # SHORT
                if pos.stop_loss is not None and high >= pos.stop_loss:
                    hit = "sl"
                elif pos.take_profit is not None and low <= pos.take_profit:
                    hit = "tp"

            if hit is None:
                # ── Trailing stop management ──────────────────────────────────
                if pos.initial_stop_distance > 0 and pos.stop_loss is not None:
                    if pos.side == PositionSide.LONG:
                        gain = high - pos.entry_price
                        if not pos.breakeven_triggered and gain >= pos.initial_stop_distance:
                            new_sl = pos.entry_price
                            if new_sl > pos.stop_loss:
                                pos.stop_loss = new_sl
                                pos.breakeven_triggered = True
                        elif pos.breakeven_triggered and gain >= 2 * pos.initial_stop_distance:
                            new_sl = high - 1.5 * pos.initial_stop_distance
                            if new_sl > pos.stop_loss:
                                pos.stop_loss = new_sl
                    else:  # SHORT
                        gain = pos.entry_price - low
                        if not pos.breakeven_triggered and gain >= pos.initial_stop_distance:
                            new_sl = pos.entry_price
                            if new_sl < pos.stop_loss:
                                pos.stop_loss = new_sl
                                pos.breakeven_triggered = True
                        elif pos.breakeven_triggered and gain >= 2 * pos.initial_stop_distance:
                            new_sl = low + 1.5 * pos.initial_stop_distance
                            if new_sl < pos.stop_loss:
                                pos.stop_loss = new_sl
                return
            trigger_price = pos.stop_loss if hit == "sl" else pos.take_profit
            assert trigger_price is not None
            close_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
            qty = pos.quantity
            await self._execute_fill(
                f"paper-{hit}-{uuid.uuid4().hex[:10]}",
                f"PAPER-{hit.upper()}-{uuid.uuid4().hex[:10]}",
                symbol,
                close_side,
                qty,
                trigger_price,
                is_maker=False,
                reduce_only=True,
                stop_loss=None,
                take_profit=None,
                strategy=pos.strategy,
            )

    async def _execute_fill(
        self,
        client_id: str,
        exch_id: str,
        symbol: str,
        side: OrderSide,
        quantity: float,
        price: float,
        is_maker: bool,
        reduce_only: bool,
        stop_loss: float | None,
        take_profit: float | None,
        strategy: str | None,
    ) -> None:
        """Must be called with self._lock held."""
        fee_bps = self._cfg.maker_fee_bps if is_maker else self._cfg.taker_fee_bps
        notional = quantity * price
        fee = notional * fee_bps / 10000.0

        pos = self._positions.get(symbol)
        if pos is None:
            pos = _PaperPosition(symbol=symbol)
            self._positions[symbol] = pos

        realized_pnl = 0.0

        # Apply to position
        if pos.side == PositionSide.FLAT or pos.quantity == 0:
            pos.side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
            pos.quantity = quantity
            pos.entry_price = price
            pos.stop_loss = stop_loss
            pos.take_profit = take_profit
            pos.strategy = strategy
            pos.initial_stop_distance = abs(price - stop_loss) if stop_loss is not None else 0.0
            pos.breakeven_triggered = False
        else:
            same_side = (pos.side == PositionSide.LONG and side == OrderSide.BUY) or (
                pos.side == PositionSide.SHORT and side == OrderSide.SELL
            )
            if same_side:
                # Scale into existing position — weighted-average entry
                total_qty = pos.quantity + quantity
                pos.entry_price = (pos.entry_price * pos.quantity + price * quantity) / total_qty
                pos.quantity = total_qty
                if stop_loss is not None:
                    pos.stop_loss = stop_loss
                if take_profit is not None:
                    pos.take_profit = take_profit
            else:
                # Reduce or flip
                close_qty = min(pos.quantity, quantity)
                if pos.side == PositionSide.LONG:
                    realized_pnl += (price - pos.entry_price) * close_qty
                else:
                    realized_pnl += (pos.entry_price - price) * close_qty
                pos.quantity -= close_qty
                remaining = quantity - close_qty
                if pos.quantity <= 1e-12:
                    pos.side = PositionSide.FLAT
                    pos.quantity = 0.0
                    pos.entry_price = 0.0
                    pos.stop_loss = None
                    pos.take_profit = None
                    pos.strategy = None
                    pos.initial_stop_distance = 0.0
                    pos.breakeven_triggered = False
                    if remaining > 0 and not reduce_only:
                        # Flipped — open opposite side with the leftover
                        pos.side = PositionSide.LONG if side == OrderSide.BUY else PositionSide.SHORT
                        pos.quantity = remaining
                        pos.entry_price = price
                        pos.stop_loss = stop_loss
                        pos.take_profit = take_profit
                        pos.strategy = strategy

        self._balance += realized_pnl - fee

        # Use _routing_name (e.g. "okx") for all event exchange fields so that
        # PortfolioTracker lookups match the connector dict key in the orchestrator.
        # self.name ("paper-okx") is kept for the raw dict / DB mode tag only.
        fill_evt = FillEvent(
            exchange=self._routing_name,
            symbol=symbol,
            client_order_id=client_id,
            exchange_order_id=exch_id,
            side=side,
            quantity=quantity,
            price=price,
            fee=fee,
            fee_currency="USDT",
            realized_pnl=realized_pnl or None,
            ts=now_utc(),
            raw={"mode": "paper", "maker": is_maker},
        )
        self._fill_q.put_nowait(fill_evt)

        order_evt = OrderUpdateEvent(
            exchange=self._routing_name,
            symbol=symbol,
            client_order_id=client_id,
            exchange_order_id=exch_id,
            status=OrderStatus.FILLED,
            side=side,
            quantity=quantity,
            filled_quantity=quantity,
            average_price=price,
            ts=now_utc(),
            raw={"mode": "paper"},
        )
        self._order_q.put_nowait(order_evt)

        pos_evt = PositionUpdateEvent(
            exchange=self._routing_name,
            symbol=symbol,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price or None,
            mark_price=price,
            liquidation_price=None,
            unrealized_pnl=0.0,
            leverage=None,
            ts=now_utc(),
        )
        self._position_q.put_nowait(pos_evt)

        log.info(
            "paper.fill",
            symbol=symbol,
            side=side.value,
            qty=quantity,
            price=price,
            fee=fee,
            realized_pnl=realized_pnl,
            balance=self._balance,
        )

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        """Apply the configured slippage to a market-order fill price."""
        slip = self._cfg.slippage_bps / 10000.0
        return price * (1 + slip) if side == OrderSide.BUY else price * (1 - slip)
