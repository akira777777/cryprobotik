"""
Portfolio tracker — the in-memory source of truth for equity, positions, and PnL.

Event flow:
    fill_events  ──┐
    position_events ──┼─→ tracker.update_* ─→ in-memory state ─→ storage.record_* / kill_switch.update
    equity periodic ─┘

The tracker aggregates across all connected exchanges. Positions are keyed by
(exchange, symbol). Equity is the sum of (balance + unrealized_pnl) across
all exchanges, queried via REST on a reconcile timer to guard against missed
WS events.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

from src.exchanges.base import PositionSide
from src.utils.logging import get_logger
from src.utils.time import now_utc

if TYPE_CHECKING:
    from src.data.storage import Storage
    from src.exchanges.base import (
        ExchangeConnector,
        FillEvent,
        PositionUpdateEvent,
    )
    from src.risk.kill_switch import KillSwitch
    from src.settings import RuntimeMode

log = get_logger(__name__)


@dataclass(slots=True)
class TrackedPosition:
    exchange: str
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    mark_price: float
    unrealized_pnl: float
    strategy: str | None = None
    updated_at: datetime = field(default_factory=now_utc)

    @property
    def notional(self) -> float:
        return self.quantity * (self.mark_price or self.entry_price)


class PortfolioTracker:
    def __init__(
        self,
        connectors: dict[str, "ExchangeConnector"],
        storage: "Storage",
        mode: "RuntimeMode",
        kill_switch: "KillSwitch",
        reconcile_interval_sec: float = 30.0,
    ) -> None:
        self._connectors = connectors
        self._storage = storage
        self._mode = mode
        self._kill_switch = kill_switch
        self._reconcile_interval = reconcile_interval_sec

        # (exchange, symbol) → TrackedPosition
        self._positions: dict[tuple[str, str], TrackedPosition] = {}
        # exchange → latest Balance snapshot (from REST reconciliation)
        self._balances: dict[str, float] = {}  # total equity per exchange
        self._free_margin: dict[str, float] = {}
        self._peak_equity: float = 0.0
        self._lock = asyncio.Lock()

    # ─────────────────────── inspection ───────────────────────

    def open_positions(self) -> list[TrackedPosition]:
        # Take a snapshot first to avoid RuntimeError if another coroutine
        # mutates _positions while we're iterating (single event-loop, but
        # dict can be mutated between yields in other coroutines).
        positions = list(self._positions.values())
        return [p for p in positions if p.side != PositionSide.FLAT and p.quantity > 0]

    def position(self, exchange: str, symbol: str) -> TrackedPosition | None:
        return self._positions.get((exchange, symbol))

    def total_equity(self) -> float:
        return sum(self._balances.values())

    def total_unrealized_pnl(self) -> float:
        return sum(p.unrealized_pnl for p in self.open_positions())

    def free_margin(self, exchange: str) -> float:
        return self._free_margin.get(exchange, 0.0)

    def exposure_by_exchange(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for p in self.open_positions():
            out[p.exchange] = out.get(p.exchange, 0.0) + abs(p.notional)
        return out

    # ─────────────────────── write path ───────────────────────

    async def on_position_update(self, evt: "PositionUpdateEvent") -> None:
        async with self._lock:
            key = (evt.exchange, evt.symbol)
            if evt.side == PositionSide.FLAT or evt.quantity == 0:
                if key in self._positions:
                    del self._positions[key]
                    log.info("tracker.position_closed", exchange=evt.exchange, symbol=evt.symbol)
                return
            pos = self._positions.get(key)
            if pos is None:
                pos = TrackedPosition(
                    exchange=evt.exchange,
                    symbol=evt.symbol,
                    side=evt.side,
                    quantity=evt.quantity,
                    entry_price=evt.entry_price or 0.0,
                    mark_price=evt.mark_price or evt.entry_price or 0.0,
                    unrealized_pnl=evt.unrealized_pnl or 0.0,
                )
                self._positions[key] = pos
                log.info(
                    "tracker.position_opened",
                    exchange=evt.exchange,
                    symbol=evt.symbol,
                    side=evt.side.value,
                    qty=evt.quantity,
                    entry=evt.entry_price,
                )
            else:
                pos.side = evt.side
                pos.quantity = evt.quantity
                if evt.entry_price:
                    pos.entry_price = evt.entry_price
                if evt.mark_price:
                    pos.mark_price = evt.mark_price
                if evt.unrealized_pnl is not None:
                    pos.unrealized_pnl = evt.unrealized_pnl
                pos.updated_at = now_utc()

    async def on_fill(self, evt: "FillEvent") -> None:
        """
        Record a fill to DB. The corresponding position update comes through
        a separate PositionUpdateEvent — we don't mutate position state here
        to avoid double-counting.
        """
        await self._storage.record_fill(
            ts=evt.ts,
            client_order_id=evt.client_order_id,
            exchange=evt.exchange,
            symbol=evt.symbol,
            side=evt.side.value,
            quantity=evt.quantity,
            price=evt.price,
            fee=evt.fee,
            fee_currency=evt.fee_currency,
            realized_pnl=evt.realized_pnl,
            raw=evt.raw,
        )
        log.info(
            "tracker.fill",
            exchange=evt.exchange,
            symbol=evt.symbol,
            side=evt.side.value,
            qty=evt.quantity,
            price=evt.price,
            fee=evt.fee,
            realized_pnl=evt.realized_pnl,
        )

    # ─────────────────────── reconciliation loop ───────────────────────

    async def run_reconcile_loop(self) -> None:
        """
        Long-running task: every `reconcile_interval_sec`, pull REST balances
        + positions from each exchange, update in-memory state, persist to DB,
        and feed the kill switch.
        """
        while True:
            try:
                await self._reconcile_once()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("tracker.reconcile_failed", error=str(e), exc_info=True)
            await asyncio.sleep(self._reconcile_interval)

    async def _reconcile_once(self) -> None:
        total_balance = 0.0
        total_equity = 0.0
        for name, conn in self._connectors.items():
            try:
                bal = await conn.fetch_balance()
                positions = await conn.fetch_positions()
            except Exception as e:
                log.warning("tracker.reconcile_fetch_failed", exchange=name, error=str(e))
                continue
            self._balances[name] = bal.total
            self._free_margin[name] = bal.free
            total_balance += bal.free + bal.used
            total_equity += bal.total

            # Update open positions from REST (source of truth)
            async with self._lock:
                # Remove any positions on this exchange not in the REST snapshot.
                rest_syms = {p.symbol for p in positions}
                for key in list(self._positions.keys()):
                    ex, sym = key
                    if ex != name:
                        continue
                    if sym not in rest_syms:
                        del self._positions[key]
                for p in positions:
                    self._positions[(p.exchange, p.symbol)] = TrackedPosition(
                        exchange=p.exchange,
                        symbol=p.symbol,
                        side=p.side,
                        quantity=p.quantity,
                        entry_price=p.entry_price or 0.0,
                        mark_price=p.mark_price or p.entry_price or 0.0,
                        unrealized_pnl=p.unrealized_pnl or 0.0,
                    )

        unrealized = self.total_unrealized_pnl()
        equity_now = total_equity  # fetch_balance already returns total = free + used + uPnL
        if equity_now > self._peak_equity:
            self._peak_equity = equity_now
        dd_from_peak = (
            0.0 if self._peak_equity == 0 else max(0.0, (self._peak_equity - equity_now) / self._peak_equity)
        )

        await self._storage.record_equity(
            ts=now_utc(),
            mode=self._mode.value,
            equity=equity_now,
            balance=total_balance,
            unrealized_pnl=unrealized,
            open_positions=len(self.open_positions()),
            drawdown_pct=dd_from_peak,
        )
        await self._storage.snapshot_positions(
            mode=self._mode.value,
            positions=[
                {
                    "exchange": p.exchange,
                    "symbol": p.symbol,
                    "side": p.side.value,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "mark_price": p.mark_price,
                    "liquidation_price": None,
                    "unrealized_pnl": p.unrealized_pnl,
                    "leverage": None,
                    "stop_loss": None,
                    "take_profit": None,
                    "strategy": p.strategy,
                }
                for p in self.open_positions()
            ],
        )
        # Feed the kill switch
        await self._kill_switch.on_equity_update(equity_now)

        log.debug(
            "tracker.reconciled",
            equity=equity_now,
            balance=total_balance,
            upnl=unrealized,
            positions=len(self.open_positions()),
            dd_from_peak=round(dd_from_peak, 4),
        )
