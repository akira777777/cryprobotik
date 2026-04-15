"""
Order executor.

Responsibilities:
1. Receive a SizedTrade from the risk manager.
2. Build an OrderRequest and place it via the routed connector.
3. Retry on transient errors (network, 5xx, 429) with exponential backoff.
4. Persist the order row to DB.
5. On successful entry, arm a server-side SL/TP pair via a follow-up reduce-only
   order pair. If the exchange supports native attached orders the connector
   could do this in one shot — we keep it explicit here so behaviour is
   identical on both venues.

Idempotency: client_order_id is generated once per SizedTrade and carried
through every retry, so a retried request never double-fills.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import TYPE_CHECKING, Any  # noqa: UP035

from tenacity import (
    AsyncRetrying,
    RetryError,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from src.exchanges.base import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.utils.logging import bind_trade_context, clear_trade_context, get_logger
from src.utils.time import now_utc

if TYPE_CHECKING:
    from src.data.storage import Storage
    from src.execution.order_router import OrderRouter
    from src.risk.manager import SizedTrade
    from src.settings import ExecutionConfig, RuntimeMode

log = get_logger(__name__)


def _round_quantity_to_step(qty: float, connector: Any, symbol: str) -> float:
    """Round qty DOWN to the exchange's lot step from ccxt market precision."""
    import math

    try:
        markets = getattr(connector, "markets", None) or getattr(connector, "_ccxt", None)
        if hasattr(markets, "markets"):
            markets = markets.markets  # type: ignore[assignment]
        if not markets:
            return qty
        step_val = ((markets.get(symbol) or {}).get("precision") or {}).get("amount")
        if step_val is None:
            return qty
        step = float(step_val)
        if step <= 0:
            return qty
        return math.floor(qty / step) * step
    except Exception:
        return qty  # best-effort — never block the trade on a precision lookup failure


# Errors that are safe to retry. asyncio.TimeoutError covers REST timeouts; we
# conservatively treat any generic Exception as non-retryable EXCEPT the ones
# listed here.
class TransientOrderError(Exception):
    """Wrap exchange errors we consider safe to retry."""


class PermanentOrderError(Exception):
    """Wrap exchange errors that must NOT be retried (e.g. insufficient margin)."""


class OrderExecutor:
    def __init__(
        self,
        router: "OrderRouter",
        storage: "Storage",
        config: "ExecutionConfig",
        mode: "RuntimeMode",
    ) -> None:
        self._router = router
        self._storage = storage
        self._config = config
        self._mode = mode

    # ─────────────────────── public API ───────────────────────

    async def execute(
        self,
        trade: "SizedTrade",
        signal_id: int | None = None,
    ) -> OrderResult | None:
        """
        Place the entry for a sized trade. Returns the OrderResult on success,
        None on failure (already logged + persisted).
        """
        client_order_id = self._make_client_id(trade.strategy, trade.symbol)
        bind_trade_context(client_order_id, trade.symbol, trade.strategy)

        try:
            connector = self._router._connectors.get(trade.exchange)  # type: ignore[attr-defined]
            if connector is None:
                log.error("executor.no_connector_for_exchange", exchange=trade.exchange)
                return None

            # Apply lot-size rounding from ccxt market precision before sending.
            quantity = _round_quantity_to_step(
                trade.quantity,
                connector,
                trade.symbol,  # type: ignore[arg-type]
            )
            if quantity <= 0:
                log.warning(
                    "executor.quantity_rounds_to_zero",
                    symbol=trade.symbol,
                    raw_qty=trade.quantity,
                )
                return None

            req = OrderRequest(
                exchange=trade.exchange,
                symbol=trade.symbol,
                side=trade.side,
                order_type=OrderType(self._config.default_order_type),
                quantity=quantity,
                price=None,  # market order
                client_order_id=client_order_id,
                reduce_only=False,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
                leverage=trade.leverage,
                meta={
                    "strategy": trade.strategy,
                    "confidence": trade.confidence,
                    "risk_usd": trade.risk_usd,
                },
            )

            # Persist 'new' row BEFORE sending, so we have a record even if the
            # network fails mid-flight.
            order_id = await self._storage.record_order(
                mode=self._mode.value,
                exchange=trade.exchange,
                symbol=trade.symbol,
                side=trade.side.value,
                order_type=req.order_type.value,
                quantity=trade.quantity,
                price=None,
                status="new",
                client_order_id=client_order_id,
                strategy=trade.strategy,
                signal_id=signal_id,
                stop_loss=trade.stop_loss,
                take_profit=trade.take_profit,
            )

            # Retry loop — only TransientOrderError is retried.
            try:
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self._config.max_retries),
                    wait=wait_exponential_jitter(
                        initial=self._config.retry_base_seconds,
                        jitter=self._config.retry_jitter_seconds,
                        max=10.0,
                    ),
                    retry=retry_if_exception_type(TransientOrderError),
                    reraise=True,
                ):
                    with attempt:
                        result = await self._place_order_with_timeout(connector, req)
            except RetryError as e:
                log.error("executor.retry_exhausted", error=str(e))
                await self._storage.update_order_status(
                    client_order_id,
                    "rejected",
                    raw_response={"error": "retry_exhausted"},
                )
                return None
            except PermanentOrderError as e:
                log.error("executor.permanent_error", error=str(e))
                await self._storage.update_order_status(
                    client_order_id,
                    "rejected",
                    raw_response={"error": str(e)},
                )
                return None
            except Exception as e:
                log.error("executor.unexpected_error", error=str(e), exc_info=True)
                await self._storage.update_order_status(
                    client_order_id,
                    "rejected",
                    raw_response={"error": f"unexpected: {e}"},
                )
                return None

            # Persist result
            await self._storage.update_order_status(
                client_order_id,
                result.status.value,
                exchange_order_id=result.exchange_order_id,
                raw_response=result.raw,
            )
            log.info(
                "executor.order_placed",
                status=result.status.value,
                exchange_order_id=result.exchange_order_id,
                quantity=trade.quantity,
                notional_usd=trade.notional_usd,
            )

            # Arm protective SL/TP via reduce-only stop orders. Many exchanges
            # accept attached TP/SL on the entry but behaviour differs — doing
            # it explicitly keeps semantics identical.
            if result.status in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED, OrderStatus.OPEN):
                await self._arm_exits(connector, trade, result)

            return result
        finally:
            clear_trade_context()

    async def cancel_all_for_symbol(self, connector: object, symbol: str) -> None:
        try:
            await connector.cancel_all(symbol)  # type: ignore[attr-defined]
        except Exception as e:
            log.warning("executor.cancel_all_failed", symbol=symbol, error=str(e))

    async def flatten_position(self, connector: object, symbol: str) -> None:
        try:
            await connector.close_position(symbol)  # type: ignore[attr-defined]
        except Exception as e:
            log.warning("executor.flatten_failed", symbol=symbol, error=str(e))

    # ─────────────────────── internals ───────────────────────

    async def _place_order_with_timeout(self, connector: object, req: OrderRequest) -> OrderResult:
        """
        Single attempt. Wraps the connector call in a timeout and translates
        exchange errors into our Transient/Permanent split.
        """
        from src.monitoring import prom_metrics as m

        t0 = time.monotonic()
        try:
            result = await asyncio.wait_for(
                connector.place_order(req),  # type: ignore[attr-defined]
                timeout=self._config.order_timeout_sec,
            )
            m.order_latency_seconds.labels(exchange=req.exchange).observe(time.monotonic() - t0)
            return result
        except asyncio.TimeoutError as e:
            m.order_latency_seconds.labels(exchange=req.exchange).observe(time.monotonic() - t0)
            raise TransientOrderError(f"order timeout after {self._config.order_timeout_sec}s") from e
        except Exception as e:
            m.order_latency_seconds.labels(exchange=req.exchange).observe(time.monotonic() - t0)
            msg = str(e).lower()
            if any(
                k in msg
                for k in (
                    "insufficient",
                    "margin",
                    "min_notional",
                    "too small",
                    "invalid symbol",
                    "permission",
                    "bad request",
                    "parameter",
                )
            ):
                raise PermanentOrderError(str(e)) from e
            if any(
                k in msg
                for k in (
                    "timeout",
                    "connection",
                    "rate limit",
                    "too many",
                    "503",
                    "502",
                    "504",
                    "429",
                    "network",
                )
            ):
                raise TransientOrderError(str(e)) from e
            # Default: unknown errors are treated as PERMANENT to avoid double-fill.
            log.error("executor.unknown_order_error", error=str(e), exc_info=True)
            raise PermanentOrderError(f"unknown: {e}") from e

    async def _arm_exits(
        self,
        connector: object,
        trade: "SizedTrade",
        entry: OrderResult,
    ) -> None:
        """
        Place reduce-only stop-market and take-profit orders after a filled entry.
        If either fails, log and continue — the in-memory position tracker will
        still monitor SL/TP triggers from ticker updates and flatten via market.
        """
        exit_side = OrderSide.SELL if trade.side == OrderSide.BUY else OrderSide.BUY
        # Use the ACTUAL filled quantity — partial fills would cause reduce-only
        # orders at the originally requested size to be rejected.
        exit_qty = entry.filled_quantity or trade.quantity
        # Stop-loss
        sl_req = OrderRequest(
            exchange=trade.exchange,
            symbol=trade.symbol,
            side=exit_side,
            order_type=OrderType.STOP_MARKET,
            quantity=exit_qty,
            price=trade.stop_loss,
            client_order_id=f"{entry.client_order_id}-sl",
            reduce_only=True,
            stop_loss=trade.stop_loss,
            meta={"parent": entry.client_order_id, "exit": "sl"},
        )
        tp_req = OrderRequest(
            exchange=trade.exchange,
            symbol=trade.symbol,
            side=exit_side,
            order_type=OrderType.LIMIT,
            quantity=exit_qty,
            price=trade.take_profit,
            client_order_id=f"{entry.client_order_id}-tp",
            reduce_only=True,
            take_profit=trade.take_profit,
            meta={"parent": entry.client_order_id, "exit": "tp"},
        )
        for req in (sl_req, tp_req):
            try:
                await connector.place_order(req)  # type: ignore[attr-defined]
                log.info("executor.exit_armed", exit=req.meta.get("exit"))
            except Exception as e:
                log.warning(
                    "executor.exit_arm_failed",
                    exit=req.meta.get("exit"),
                    error=str(e),
                )

    @staticmethod
    def _make_client_id(strategy: str, symbol: str) -> str:
        """Strategy-symbol-uuid. Keeps chars safe for both OKX (<=32) and Bybit (<=36)."""
        tag = f"{strategy[:4]}{symbol.split('/')[0][:4]}"
        return f"cpb{tag}{uuid.uuid4().hex[:20]}"
