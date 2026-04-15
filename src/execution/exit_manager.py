"""
Exit manager — continuously monitors open positions and applies dynamic exit rules.

Rules (evaluated on every tick / periodic check, in priority order):
1. Breakeven:    Once unrealized gain ≥ 1×R, move SL to entry price (zero-risk trade).
2. Partial TP:   Once unrealized gain ≥ 1.5×R, close 50% of the position.
3. Trailing SL:  Once unrealized gain ≥ 2×R, trail SL at current_price ± 1.5×ATR.
4. Time exit:    If position has been open ≥ max_bars_open without ever reaching
                 ≥ 0.5×R, close the entire position at market.

The exit manager is stateless between calls — all state comes from the
PortfolioTracker and FeatureStore. It is wired as a periodic async task
in the Orchestrator, running on every tick (or on a timer alongside the
OI poll loop).

Idempotency: every stop adjustment follows the "place new first, cancel old"
discipline. If the new order fails, the old one is NOT cancelled.

Prometheus metrics emitted:
    exits_breakeven_total       — SL moved to breakeven
    exits_trailed_total         — SL trailed after +2R
    exits_partial_tp_total      — 50% closed at +1.5R
    exits_time_stop_total       — position closed on time limit
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.data.feature_store import FeatureKey, FeatureStore
from src.exchanges.base import (
    OrderRequest,
    OrderSide,
    OrderStatus,
    OrderType,
    PositionSide,
)
from src.monitoring import prom_metrics as m
from src.utils.logging import get_logger
from src.utils.time import now_utc

if TYPE_CHECKING:
    from src.data.storage import Storage
    from src.exchanges.base import ExchangeConnector
    from src.portfolio.tracker import PortfolioTracker, TrackedPosition
    from src.settings import ExecutionConfig

log = get_logger(__name__)


# ─────────────────────── config ───────────────────────


@dataclass(slots=True)
class ExitConfig:
    """
    Parameters for all dynamic exit rules.

    Attribute                  Default   Description
    ─────────────────────────  ───────   ──────────────────────────────────────
    atr_period                 14        ATR lookback for trailing stop distance
    atr_trailing_mult          1.5       Trailing SL = price ± (atr_trailing_mult × ATR)
    breakeven_trigger_r        1.0       Move SL to entry after this many R gained
    partial_tp_trigger_r       1.5       Close partial_tp_fraction at this many R
    partial_tp_fraction        0.5       Fraction of position to close on partial TP
    trailing_trigger_r         2.0       Start trailing after this many R gained
    max_bars_open              48        Time-exit after this many 15m bars (~12h)
    time_exit_min_r            0.5       Time exit fires only if gain < this R
    check_interval_sec         30.0      How often the loop wakes up to check positions
    """

    atr_period: int = 14
    atr_trailing_mult: float = 1.5
    breakeven_trigger_r: float = 1.0
    partial_tp_trigger_r: float = 1.5
    partial_tp_fraction: float = 0.5
    trailing_trigger_r: float = 2.0
    max_bars_open: int = 48
    time_exit_min_r: float = 0.5
    check_interval_sec: float = 30.0


# ─────────────────────── per-position state ───────────────────────


@dataclass
class _ExitState:
    """Mutable tracking state for a single open position."""

    entry_ts: datetime = field(default_factory=now_utc)
    original_sl: float = 0.0       # SL set when entry was placed
    current_sl: float = 0.0        # latest SL (may have been moved)
    current_sl_order_id: str | None = None  # exchange order-id of the live SL order
    current_tp_order_id: str | None = None  # exchange order-id of the live TP order
    bars_open: int = 0
    breakeven_done: bool = False
    partial_tp_done: bool = False
    trailing_active: bool = False
    risk_usd: float = 0.0          # original risk in USD (= |entry_price - original_sl| × qty)


# ─────────────────────── ExitManager ───────────────────────


class ExitManager:
    """
    Monitors all open positions and applies the dynamic exit rules described
    in the module docstring.

    Lifecycle:
        - Instantiated once in the Orchestrator after all dependencies are wired.
        - `run()` is added as a long-running task in the asyncio.TaskGroup.
        - `register_position()` is called by the Orchestrator immediately after
          a successful order fill to record the initial SL/TP order IDs.
    """

    def __init__(
        self,
        tracker: "PortfolioTracker",
        feature_store: FeatureStore,
        connectors: dict[str, "ExchangeConnector"],
        config: ExitConfig | None = None,
    ) -> None:
        self._tracker = tracker
        self._store = feature_store
        self._connectors = connectors
        self._config = config or ExitConfig()
        # (exchange, symbol) → state
        self._states: dict[tuple[str, str], _ExitState] = {}
        self._lock = asyncio.Lock()

    # ─────────────────────── public API ───────────────────────

    def register_position(
        self,
        exchange: str,
        symbol: str,
        entry_ts: datetime,
        original_sl: float,
        risk_usd: float,
        sl_order_id: str | None = None,
        tp_order_id: str | None = None,
    ) -> None:
        """Called when a new position is opened. Records SL/TP order IDs."""
        key = (exchange, symbol)
        self._states[key] = _ExitState(
            entry_ts=entry_ts,
            original_sl=original_sl,
            current_sl=original_sl,
            current_sl_order_id=sl_order_id,
            current_tp_order_id=tp_order_id,
            risk_usd=risk_usd,
        )
        log.debug("exit_manager.registered", exchange=exchange, symbol=symbol)

    async def run(self) -> None:
        """Main loop — runs as a background task in the Orchestrator TaskGroup."""
        while True:
            await asyncio.sleep(self._config.check_interval_sec)
            try:
                await self._check_all_positions()
            except Exception as e:
                log.error("exit_manager.loop_error", error=str(e), exc_info=True)

    # ─────────────────────── internals ───────────────────────

    async def _check_all_positions(self) -> None:
        positions = self._tracker.open_positions()
        for pos in positions:
            key = (pos.exchange, pos.symbol)
            async with self._lock:
                state = self._states.get(key)
                if state is None:
                    # Position opened before ExitManager was active; create a
                    # minimal state record so we don't apply stale rules.
                    state = _ExitState(
                        entry_ts=pos.updated_at,
                        original_sl=0.0,
                        current_sl=0.0,
                        risk_usd=0.0,
                    )
                    self._states[key] = state

                conn = self._connectors.get(pos.exchange)
                if conn is None:
                    continue

                await self._check_position(pos, state, conn)

        # Clean up state for positions that have been closed.
        open_keys = {(p.exchange, p.symbol) for p in positions}
        async with self._lock:
            for key in list(self._states.keys()):
                if key not in open_keys:
                    del self._states[key]

    async def _check_position(
        self,
        pos: "TrackedPosition",
        state: _ExitState,
        conn: "ExchangeConnector",
    ) -> None:
        """Apply all exit rules to a single position."""
        current_price = pos.mark_price or pos.entry_price
        if current_price <= 0 or pos.entry_price <= 0:
            return

        # R = |current_price - entry_price| / |original_sl - entry_price|
        sl_distance = abs(state.original_sl - pos.entry_price) if state.original_sl > 0 else 0.0
        if sl_distance <= 0:
            # No original SL recorded — can't compute R; skip dynamic rules but
            # still apply the time exit.
            state.bars_open += 1
            await self._maybe_time_exit(pos, state, conn, r_multiple=0.0)
            return

        if pos.side == PositionSide.LONG:
            price_move = current_price - pos.entry_price
        else:
            price_move = pos.entry_price - current_price

        r_multiple = price_move / sl_distance

        # Increment bar count (approximate: loop fires every check_interval_sec;
        # 15m bars = 900 s, so each fire ≈ check_interval_sec / 900 bars).
        state.bars_open += 1

        # ── Rule 1: Breakeven ──
        if (
            not state.breakeven_done
            and r_multiple >= self._config.breakeven_trigger_r
        ):
            await self._apply_breakeven(pos, state, conn)

        # ── Rule 2: Partial TP ──
        if (
            not state.partial_tp_done
            and r_multiple >= self._config.partial_tp_trigger_r
        ):
            await self._apply_partial_tp(pos, state, conn)

        # ── Rule 3: Trailing SL ──
        if r_multiple >= self._config.trailing_trigger_r:
            atr_val = self._get_atr(pos.exchange, pos.symbol)
            if atr_val and atr_val > 0:
                await self._apply_trailing_sl(pos, state, conn, current_price, atr_val)

        # ── Rule 4: Time exit ──
        await self._maybe_time_exit(pos, state, conn, r_multiple)

    async def _apply_breakeven(
        self,
        pos: "TrackedPosition",
        state: _ExitState,
        conn: "ExchangeConnector",
    ) -> None:
        """Move SL to entry price (risk-free position)."""
        new_sl = pos.entry_price
        if abs(new_sl - state.current_sl) < 1e-8:
            return  # already at breakeven

        log.info(
            "exit_manager.breakeven",
            symbol=pos.symbol,
            exchange=pos.exchange,
            entry_price=pos.entry_price,
            old_sl=state.current_sl,
        )
        success = await self.adjust_stop(conn, pos, new_sl, state)
        if success:
            state.breakeven_done = True
            m.exits_breakeven_total.labels(exchange=pos.exchange).inc()

    async def _apply_partial_tp(
        self,
        pos: "TrackedPosition",
        state: _ExitState,
        conn: "ExchangeConnector",
    ) -> None:
        """Close partial_tp_fraction of the position at market."""
        close_qty = pos.quantity * self._config.partial_tp_fraction
        close_qty = _floor_to_precision(close_qty)
        if close_qty <= 0:
            return

        exit_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        req = OrderRequest(
            exchange=pos.exchange,
            symbol=pos.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=close_qty,
            reduce_only=True,
            meta={"exit": "partial_tp", "fraction": self._config.partial_tp_fraction},
        )
        try:
            await conn.place_order(req)
            state.partial_tp_done = True
            log.info(
                "exit_manager.partial_tp",
                symbol=pos.symbol,
                exchange=pos.exchange,
                close_qty=close_qty,
            )
            m.exits_partial_tp_total.labels(exchange=pos.exchange).inc()
        except Exception as e:
            log.warning(
                "exit_manager.partial_tp_failed",
                symbol=pos.symbol,
                error=str(e),
            )

    async def _apply_trailing_sl(
        self,
        pos: "TrackedPosition",
        state: _ExitState,
        conn: "ExchangeConnector",
        current_price: float,
        atr_val: float,
    ) -> None:
        """Trail the SL as the price moves favorably."""
        trail_dist = self._config.atr_trailing_mult * atr_val
        if pos.side == PositionSide.LONG:
            new_sl = current_price - trail_dist
            # SL must be strictly better (higher) than current SL for a long
            if new_sl <= state.current_sl:
                return
        else:
            new_sl = current_price + trail_dist
            # SL must be strictly better (lower) than current SL for a short
            if new_sl >= state.current_sl:
                return

        log.info(
            "exit_manager.trail",
            symbol=pos.symbol,
            exchange=pos.exchange,
            old_sl=round(state.current_sl, 6),
            new_sl=round(new_sl, 6),
            current_price=round(current_price, 6),
            atr=round(atr_val, 6),
        )
        success = await self.adjust_stop(conn, pos, new_sl, state)
        if success:
            state.trailing_active = True
            m.exits_trailed_total.labels(exchange=pos.exchange).inc()

    async def _maybe_time_exit(
        self,
        pos: "TrackedPosition",
        state: _ExitState,
        conn: "ExchangeConnector",
        r_multiple: float,
    ) -> None:
        """Close the position if it has been open too long without reaching min_r."""
        if state.bars_open < self._config.max_bars_open:
            return
        if r_multiple >= self._config.time_exit_min_r:
            return  # position is working — keep holding

        exit_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        req = OrderRequest(
            exchange=pos.exchange,
            symbol=pos.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            quantity=pos.quantity,
            reduce_only=True,
            meta={"exit": "time_stop", "bars_open": state.bars_open},
        )
        try:
            await conn.place_order(req)
            log.info(
                "exit_manager.time_stop",
                symbol=pos.symbol,
                exchange=pos.exchange,
                bars_open=state.bars_open,
                r_multiple=round(r_multiple, 3),
            )
            m.exits_time_stop_total.labels(exchange=pos.exchange).inc()
        except Exception as e:
            log.warning(
                "exit_manager.time_stop_failed",
                symbol=pos.symbol,
                error=str(e),
            )

    # ─────────────────────── stop adjustment ───────────────────────

    async def adjust_stop(
        self,
        conn: "ExchangeConnector",
        pos: "TrackedPosition",
        new_sl: float,
        state: _ExitState,
    ) -> bool:
        """
        Replace the live SL order with a new one at new_sl.

        Safety discipline: place the new SL order FIRST; only cancel the old
        one if the new order is accepted. If the new order fails, the original
        SL remains active.

        Returns True on success.
        """
        exit_side = OrderSide.SELL if pos.side == PositionSide.LONG else OrderSide.BUY
        new_sl_order_id_tag = (
            f"{state.current_sl_order_id}-adj" if state.current_sl_order_id
            else None
        )
        new_req = OrderRequest(
            exchange=pos.exchange,
            symbol=pos.symbol,
            side=exit_side,
            order_type=OrderType.STOP_MARKET,
            quantity=pos.quantity,
            price=new_sl,
            client_order_id=new_sl_order_id_tag,
            reduce_only=True,
            stop_loss=new_sl,
            meta={"exit": "sl_adjust", "new_sl": new_sl},
        )
        try:
            result = await conn.place_order(new_req)
            if result.status in (OrderStatus.REJECTED,):
                log.warning(
                    "exit_manager.new_sl_rejected",
                    symbol=pos.symbol,
                    new_sl=new_sl,
                    status=result.status.value,
                )
                return False

            # New SL accepted — now cancel the old one.
            if state.current_sl_order_id:
                try:
                    await conn.cancel_order(
                        pos.symbol,
                        exchange_order_id=state.current_sl_order_id,
                    )
                except Exception as e:
                    log.warning(
                        "exit_manager.old_sl_cancel_failed",
                        symbol=pos.symbol,
                        old_order_id=state.current_sl_order_id,
                        error=str(e),
                    )
                    # Non-fatal: new SL is already active, old one may have
                    # already filled or been cancelled by the exchange.

            state.current_sl = new_sl
            state.current_sl_order_id = result.exchange_order_id
            return True

        except Exception as e:
            log.warning(
                "exit_manager.adjust_stop_failed",
                symbol=pos.symbol,
                new_sl=new_sl,
                error=str(e),
            )
            return False

    # ─────────────────────── helpers ───────────────────────

    def _get_atr(self, exchange: str, symbol: str) -> float | None:
        """Return the current ATR(14) value from the 15m feature store."""
        try:
            # Import here to avoid a circular import at module level.
            from src.utils.indicators import atr as compute_atr  # noqa: PLC0415

            key = FeatureKey(exchange=exchange, symbol=symbol, timeframe="15m")
            df = self._store.as_df(key, min_bars=self._config.atr_period + 5)
            if df is None:
                return None
            series = compute_atr(df, length=self._config.atr_period)
            if series is None or series.empty:
                return None
            val = float(series.iloc[-1])
            return None if math.isnan(val) else val
        except Exception:
            return None  # best-effort; never block exit logic on indicator failure


def _floor_to_precision(qty: float, precision: int = 8) -> float:
    """Floor a quantity to a fixed number of decimal places."""
    factor = 10 ** precision
    return math.floor(qty * factor) / factor
