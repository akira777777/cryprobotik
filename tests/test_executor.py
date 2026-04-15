"""
OrderExecutor tests.

Focus: the executor sits between risk sizing and the exchange, and owns the
properties that protect us from double-fills:

  - `client_order_id` generated ONCE per SizedTrade and carried through every
    retry (if a retry creates a NEW id, the exchange may accept it twice).
  - Error classification decides retry vs reject. Classifying a permanent
    error as transient = infinite loop. Classifying transient as permanent =
    legit orders lost.
  - On partial fill, reduce-only SL/TP must use filled_quantity, otherwise
    the exchange rejects them for over-reduce.
  - Exit-arm failures must not sink a successful entry.

These tests cover those invariants without touching any real exchange.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from src.exchanges.base import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from src.execution.executor import (
    OrderExecutor,
    PermanentOrderError,
    TransientOrderError,
    _round_quantity_to_step,
)
from src.risk.manager import SizedTrade
from src.settings import ExecutionConfig, RuntimeMode

from tests.conftest import FakeStorage


# ──────────────────────────────────────────────────────────────────────────────
# Test doubles
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class _FakeRouter:
    """Minimal stand-in — executor only reads `._connectors` off it."""

    _connectors: dict[str, Any]


class _TrackingStorage(FakeStorage):
    """FakeStorage that also records status updates for assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.status_updates: list[dict[str, Any]] = []

    async def update_order_status(self, *args: Any, **kwargs: Any) -> None:
        # Signature in real Storage: (client_order_id, status, *, exchange_order_id=?, raw_response=?)
        client_order_id = args[0] if args else kwargs.get("client_order_id")
        status = args[1] if len(args) > 1 else kwargs.get("status")
        self.status_updates.append({
            "client_order_id": client_order_id,
            "status": status,
            "exchange_order_id": kwargs.get("exchange_order_id"),
            "raw_response": kwargs.get("raw_response"),
        })


class _FakeConnector:
    """Records every place_order call. Behaviour configurable per test."""

    def __init__(
        self,
        *,
        markets: dict[str, Any] | None = None,
        responder: Any = None,
    ) -> None:
        # ccxt-style markets dict; executor's _round_quantity_to_step reads
        # connector.markets[symbol].precision.amount.
        self.markets = markets or {}
        # `responder` can be a list (consumed in order), a callable, or None.
        self._responder = responder
        self.calls: list[OrderRequest] = []

    async def place_order(self, req: OrderRequest) -> OrderResult:
        self.calls.append(req)
        r = self._responder
        if r is None:
            return OrderResult(
                client_order_id=req.client_order_id or "?",
                exchange_order_id=f"ex_{len(self.calls)}",
                status=OrderStatus.FILLED,
                filled_quantity=req.quantity,
                average_price=req.price or 50_000.0,
                raw={},
            )
        if callable(r):
            return await r(req, len(self.calls))
        # list[callable|exception|OrderResult]
        if isinstance(r, list):
            nth = r[min(len(self.calls) - 1, len(r) - 1)]
            if isinstance(nth, BaseException):
                raise nth
            if callable(nth):
                return await nth(req, len(self.calls))
            return nth  # type: ignore[return-value]
        raise AssertionError(f"unsupported responder type: {type(r)}")


def _make_trade(
    *,
    quantity: float = 0.01,
    exchange: str = "okx",
    stop_loss: float = 49_000.0,
    take_profit: float = 52_000.0,
) -> SizedTrade:
    return SizedTrade(
        symbol="BTC/USDT:USDT",
        exchange=exchange,
        side=OrderSide.BUY,
        quantity=quantity,
        entry_price=50_000.0,
        stop_loss=stop_loss,
        take_profit=take_profit,
        leverage=3,
        risk_usd=20.0,
        notional_usd=500.0,
        strategy="momentum",
        confidence=0.7,
    )


def _make_executor(
    connectors: dict[str, _FakeConnector],
    storage: _TrackingStorage,
    execution_config: ExecutionConfig,
) -> OrderExecutor:
    router = _FakeRouter(_connectors=connectors)  # type: ignore[arg-type]
    return OrderExecutor(
        router=router,  # type: ignore[arg-type]
        storage=storage,  # type: ignore[arg-type]
        config=execution_config,
        mode=RuntimeMode.PAPER,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pure helpers — synchronous
# ──────────────────────────────────────────────────────────────────────────────


class TestMakeClientId:
    """Format invariants for the client_order_id generator."""

    def test_prefix_is_cpb(self) -> None:
        cid = OrderExecutor._make_client_id("momentum", "BTC/USDT:USDT")
        assert cid.startswith("cpb")

    def test_length_within_okx_limit(self) -> None:
        """OKX caps clOrdId at 32 chars; ours must fit."""
        for strategy in ("momentum", "x", "mean_reversion", "funding_contrarian"):
            for symbol in ("BTC/USDT:USDT", "ETH/USDT:USDT", "A/USDT:USDT"):
                cid = OrderExecutor._make_client_id(strategy, symbol)
                assert len(cid) <= 32, (strategy, symbol, len(cid), cid)

    def test_unique_between_calls(self) -> None:
        ids = {OrderExecutor._make_client_id("momentum", "BTC/USDT:USDT") for _ in range(200)}
        # UUID-backed randomness → effectively no collisions in 200 draws
        assert len(ids) == 200

    def test_contains_strategy_and_symbol_tag(self) -> None:
        cid = OrderExecutor._make_client_id("momentum", "BTC/USDT:USDT")
        # strategy[:4]=mome, symbol base[:4]=BTC (only 3 chars) → "momeBTC" in id
        assert "mome" in cid
        assert "BTC" in cid


class TestRoundQtyToStep:
    """Lot-size rounding must never raise and must round DOWN."""

    def test_no_markets_returns_qty_unchanged(self) -> None:
        class NoMarkets:
            pass

        assert _round_quantity_to_step(1.2345, NoMarkets(), "BTC/USDT:USDT") == 1.2345

    def test_zero_step_returns_qty_unchanged(self) -> None:
        conn = _FakeConnector(markets={"BTC/USDT:USDT": {"precision": {"amount": 0.0}}})
        # step=0 would otherwise div-by-zero; executor must bail
        assert _round_quantity_to_step(1.2345, conn, "BTC/USDT:USDT") == 1.2345

    def test_floors_to_step(self) -> None:
        conn = _FakeConnector(markets={"BTC/USDT:USDT": {"precision": {"amount": 0.01}}})
        # 1.2345 rounded DOWN to 0.01 → 1.23 (not 1.24)
        assert _round_quantity_to_step(1.2345, conn, "BTC/USDT:USDT") == pytest.approx(1.23)

    def test_symbol_missing_returns_qty_unchanged(self) -> None:
        conn = _FakeConnector(markets={"ETH/USDT:USDT": {"precision": {"amount": 0.01}}})
        assert _round_quantity_to_step(1.2345, conn, "BTC/USDT:USDT") == 1.2345

    def test_raises_are_swallowed(self) -> None:
        class BoomMarkets:
            @property
            def markets(self) -> dict[str, Any]:  # pragma: no cover - getter raises
                raise RuntimeError("boom")

        # Must never bubble up — precision lookup is best-effort only.
        assert _round_quantity_to_step(1.2345, BoomMarkets(), "BTC/USDT:USDT") == 1.2345


# ──────────────────────────────────────────────────────────────────────────────
# execute() — full async path
# ──────────────────────────────────────────────────────────────────────────────


class TestExecuteBasicPaths:
    async def test_unknown_exchange_returns_none(
        self, execution_config: ExecutionConfig,
    ) -> None:
        storage = _TrackingStorage()
        connector = _FakeConnector()
        executor = _make_executor({"okx": connector}, storage, execution_config)
        trade = _make_trade(exchange="bybit")  # not in router

        result = await executor.execute(trade)

        assert result is None
        assert connector.calls == []
        # no order row persisted, no status update
        assert storage.orders_recorded == []
        assert storage.status_updates == []

    async def test_quantity_rounds_to_zero_skips_order(
        self, execution_config: ExecutionConfig,
    ) -> None:
        # lot step is 1.0, trade qty is 0.01 → floor to 0 → executor bails
        connector = _FakeConnector(
            markets={"BTC/USDT:USDT": {"precision": {"amount": 1.0}}},
        )
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)
        trade = _make_trade(quantity=0.01)

        result = await executor.execute(trade)

        assert result is None
        assert connector.calls == []
        # No DB row either — we exit BEFORE record_order
        assert storage.orders_recorded == []

    async def test_happy_path_fills_and_arms_exits(
        self, execution_config: ExecutionConfig,
    ) -> None:
        connector = _FakeConnector()  # defaults: FILLED with full quantity
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        result = await executor.execute(_make_trade())

        assert result is not None
        assert result.status == OrderStatus.FILLED
        # 1 entry + 2 exit arms (SL, TP)
        assert len(connector.calls) == 3
        # entry order recorded BEFORE send
        assert len(storage.orders_recorded) == 1
        # status transitioned to FILLED
        filled_updates = [u for u in storage.status_updates if u["status"] == "filled"]
        assert len(filled_updates) == 1


class TestClientOrderIdIdempotency:
    """
    THE critical property: a retried order must carry the same client_order_id
    so the exchange deduplicates.
    """

    async def test_client_order_id_stable_across_retries(
        self, execution_config: ExecutionConfig,
    ) -> None:
        # Responder: fail transient once, then succeed.
        seq = [
            TransientOrderError("503 temporary"),
            OrderResult(
                client_order_id="filled-by-exchange",
                exchange_order_id="ex_2",
                status=OrderStatus.FILLED,
                filled_quantity=0.01,
                average_price=50_000.0,
                raw={},
            ),
        ]
        connector = _FakeConnector(responder=seq)
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        result = await executor.execute(_make_trade())

        assert result is not None
        # Entry was retried exactly once
        entry_calls = [c for c in connector.calls if not c.reduce_only]
        assert len(entry_calls) == 2
        # Same client_order_id on both attempts — THE property
        assert entry_calls[0].client_order_id == entry_calls[1].client_order_id
        assert entry_calls[0].client_order_id is not None
        assert entry_calls[0].client_order_id.startswith("cpb")


class TestErrorClassification:
    async def test_permanent_error_does_not_retry(
        self, execution_config: ExecutionConfig,
    ) -> None:
        # "insufficient margin" → PermanentOrderError → no retry
        connector = _FakeConnector(responder=[RuntimeError("insufficient margin")])
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        result = await executor.execute(_make_trade())

        assert result is None
        # Exactly one attempt, no retries
        assert len(connector.calls) == 1
        # Order row recorded then marked rejected
        rejected = [u for u in storage.status_updates if u["status"] == "rejected"]
        assert len(rejected) == 1

    async def test_transient_error_exhausts_retries_then_rejects(
        self, execution_config: ExecutionConfig,
    ) -> None:
        # Always a transient-classified error → retry until max_retries is hit
        connector = _FakeConnector(
            responder=[RuntimeError("connection reset")] * 20
        )
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        result = await executor.execute(_make_trade())

        assert result is None
        # Exactly max_retries attempts
        assert len(connector.calls) == execution_config.max_retries
        rejected = [u for u in storage.status_updates if u["status"] == "rejected"]
        assert len(rejected) == 1

    async def test_unknown_error_treated_as_permanent(
        self, execution_config: ExecutionConfig,
    ) -> None:
        """
        An unclassifiable error must be treated as PERMANENT, not retried.
        This is a deliberate bias toward 'avoid double-fills' over 'maximize
        executions' — see executor.py:296.
        """
        connector = _FakeConnector(responder=[RuntimeError("wat some weird thing")])
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        result = await executor.execute(_make_trade())

        assert result is None
        assert len(connector.calls) == 1  # no retry

    async def test_timeout_is_transient(
        self, execution_config: ExecutionConfig,
    ) -> None:
        """asyncio.TimeoutError → TransientOrderError (retried)."""

        entry_attempts = {"n": 0}

        async def responder(req: OrderRequest, n: int) -> OrderResult:
            if not req.reduce_only:
                entry_attempts["n"] += 1
                if entry_attempts["n"] == 1:
                    raise asyncio.TimeoutError("mock timeout")
            return OrderResult(
                client_order_id=req.client_order_id or "?",
                exchange_order_id="ex_ok",
                status=OrderStatus.FILLED,
                filled_quantity=req.quantity,
                average_price=50_000.0,
                raw={},
            )

        connector = _FakeConnector(responder=responder)
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        result = await executor.execute(_make_trade())

        assert result is not None and result.status == OrderStatus.FILLED
        # One timeout, then success → 2 entry attempts
        assert entry_attempts["n"] == 2
        entry_calls = [c for c in connector.calls if not c.reduce_only]
        assert len(entry_calls) == 2


class TestExitArming:
    async def test_exit_qty_uses_filled_quantity_on_partial_fill(
        self, execution_config: ExecutionConfig,
    ) -> None:
        """
        Reduce-only orders placed at the originally-requested size are rejected
        when the entry only partially filled. Executor must use filled_quantity.
        """
        partial = OrderResult(
            client_order_id="cpb-fake-id",
            exchange_order_id="ex_1",
            status=OrderStatus.PARTIALLY_FILLED,
            filled_quantity=0.004,  # of requested 0.01
            average_price=50_000.0,
            raw={},
        )
        connector = _FakeConnector(responder=[partial])
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        await executor.execute(_make_trade(quantity=0.01))

        reduce_only_calls = [c for c in connector.calls if c.reduce_only]
        assert len(reduce_only_calls) == 2  # SL + TP
        for c in reduce_only_calls:
            assert c.quantity == pytest.approx(0.004)

    async def test_exit_arm_failure_does_not_fail_execute(
        self, execution_config: ExecutionConfig,
    ) -> None:
        """SL/TP placement errors must log-and-continue, not sink the entry."""

        async def responder(req: OrderRequest, n: int) -> OrderResult:
            if req.reduce_only:
                raise RuntimeError("exit arm rejected by exchange")
            return OrderResult(
                client_order_id=req.client_order_id or "?",
                exchange_order_id="ex_entry",
                status=OrderStatus.FILLED,
                filled_quantity=req.quantity,
                average_price=50_000.0,
                raw={},
            )

        connector = _FakeConnector(responder=responder)
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        result = await executor.execute(_make_trade())

        # Entry succeeded even though both exit arms threw
        assert result is not None
        assert result.status == OrderStatus.FILLED
        # 1 entry + 2 attempted exit arms
        assert len(connector.calls) == 3

    async def test_exit_client_order_ids_are_derivative(
        self, execution_config: ExecutionConfig,
    ) -> None:
        """SL/TP ids = entry id + '-sl'/'-tp' — deterministic, traceable."""
        connector = _FakeConnector()
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        await executor.execute(_make_trade())

        entry = next(c for c in connector.calls if not c.reduce_only)
        reduce_only = [c for c in connector.calls if c.reduce_only]
        sl = next(c for c in reduce_only if c.order_type == OrderType.STOP_MARKET)
        tp = next(c for c in reduce_only if c.order_type == OrderType.LIMIT)
        assert sl.client_order_id == f"{entry.client_order_id}-sl"
        assert tp.client_order_id == f"{entry.client_order_id}-tp"

    async def test_exits_not_armed_when_entry_is_rejected(
        self, execution_config: ExecutionConfig,
    ) -> None:
        connector = _FakeConnector(responder=[RuntimeError("invalid symbol")])
        storage = _TrackingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        await executor.execute(_make_trade())

        # only 1 call (the entry attempt); no reduce-only arms
        assert len(connector.calls) == 1
        assert all(not c.reduce_only for c in connector.calls)


class TestOrdering:
    """Storage must be called BEFORE the exchange — so we don't lose records."""

    async def test_record_order_called_before_connector_place(
        self, execution_config: ExecutionConfig,
    ) -> None:
        events: list[str] = []

        class OrderingStorage(_TrackingStorage):
            async def record_order(self, **kwargs: Any) -> int:  # type: ignore[override]
                events.append("storage.record_order")
                return await super().record_order(**kwargs)

        class OrderingConnector(_FakeConnector):
            async def place_order(self, req: OrderRequest) -> OrderResult:
                events.append("connector.place_order")
                return await super().place_order(req)

        connector = OrderingConnector()
        storage = OrderingStorage()
        executor = _make_executor({"okx": connector}, storage, execution_config)

        await executor.execute(_make_trade())

        # First two events must be storage BEFORE connector
        assert events[:2] == ["storage.record_order", "connector.place_order"]
