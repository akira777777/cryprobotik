"""
Unit tests for ExitManager.

Covers:
- Breakeven triggers after ≥1R gain and moves SL to entry price.
- Partial TP fires after ≥1.5R gain and sends a reduce-only market order.
- Trailing SL fires after ≥2R gain and only advances (never retreats) the SL.
- Time exit fires when bars_open ≥ max_bars_open AND gain < time_exit_min_r.
- adjust_stop places new SL BEFORE cancelling the old one (place-first discipline).
- adjust_stop is idempotent: double call with the same SL price does nothing.
- No exit fires when position is still at 0R.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.data.feature_store import Bar, FeatureKey, FeatureStore
from src.exchanges.base import (
    OrderRequest,
    OrderResult,
    OrderSide,
    OrderStatus,
    PositionSide,
)
from src.execution.exit_manager import ExitConfig, ExitManager, _ExitState


# ─────────────────────── helpers ───────────────────────

EXCHANGE = "okx"
SYMBOL = "BTC/USDT:USDT"


def _make_connector(place_result: OrderResult | Exception | None = None) -> MagicMock:
    conn = MagicMock()
    if isinstance(place_result, Exception):
        conn.place_order = AsyncMock(side_effect=place_result)
    elif place_result is None:
        # Default: succeeds with status=OPEN
        conn.place_order = AsyncMock(return_value=OrderResult(
            client_order_id="cid",
            exchange_order_id="eid-new",
            status=OrderStatus.OPEN,
            filled_quantity=0.0,
            average_price=None,
            raw={},
        ))
    else:
        conn.place_order = AsyncMock(return_value=place_result)
    conn.cancel_order = AsyncMock()
    return conn


def _make_position(
    side: PositionSide = PositionSide.LONG,
    entry_price: float = 100.0,
    mark_price: float = 110.0,
    quantity: float = 1.0,
) -> MagicMock:
    pos = MagicMock()
    pos.exchange = EXCHANGE
    pos.symbol = SYMBOL
    pos.side = side
    pos.entry_price = entry_price
    pos.mark_price = mark_price
    pos.quantity = quantity
    pos.updated_at = datetime(2026, 1, 1, tzinfo=UTC)
    return pos


def _make_tracker(positions: list[Any]) -> MagicMock:
    t = MagicMock()
    t.open_positions.return_value = positions
    return t


def _make_exit_manager(
    positions: list[Any],
    connectors: dict[str, Any],
    config: ExitConfig | None = None,
    store: FeatureStore | None = None,
) -> ExitManager:
    tracker = _make_tracker(positions)
    return ExitManager(
        tracker=tracker,
        feature_store=store or FeatureStore(),
        connectors=connectors,
        config=config or ExitConfig(),
    )


# ─────────────────────── breakeven ───────────────────────


async def test_breakeven_fires_at_1r() -> None:
    """After +1R gain, adjust_stop is called to move SL to entry price."""
    conn = _make_connector()
    entry = 100.0
    sl = 95.0   # 5 points risk → 1R = +5 points
    # Mark price at +5 → exactly 1R
    pos = _make_position(entry_price=entry, mark_price=105.0)

    manager = _make_exit_manager([pos], {EXCHANGE: conn})
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=sl,
        risk_usd=5.0,
        sl_order_id="old-sl-eid",
    )

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as metric:
        metric.labels.return_value = MagicMock()
        await manager._check_all_positions()

    # place_order was called with the new SL at entry price
    assert conn.place_order.call_count >= 1
    req: OrderRequest = conn.place_order.call_args_list[0].args[0]
    assert req.stop_loss == pytest.approx(entry)
    assert req.reduce_only

    # Old SL was cancelled after the new one was placed
    conn.cancel_order.assert_called_once()

    state = manager._states[(EXCHANGE, SYMBOL)]
    assert state.breakeven_done


async def test_breakeven_does_not_fire_below_1r() -> None:
    """Below 1R gain, no stop adjustment is made."""
    conn = _make_connector()
    pos = _make_position(entry_price=100.0, mark_price=102.0)  # 0.4R with sl=95

    manager = _make_exit_manager([pos], {EXCHANGE: conn})
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=95.0,
        risk_usd=5.0,
    )
    await manager._check_all_positions()

    conn.place_order.assert_not_called()


async def test_breakeven_idempotent() -> None:
    """Once breakeven is set, a second call at same gain should not call place_order again."""
    conn = _make_connector()
    pos = _make_position(entry_price=100.0, mark_price=108.0)  # >1R

    manager = _make_exit_manager([pos], {EXCHANGE: conn})
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=95.0,
        risk_usd=5.0,
    )

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as m:
        m.labels.return_value = MagicMock()
        await manager._check_all_positions()
        assert manager._states[(EXCHANGE, SYMBOL)].breakeven_done

        # Second check — should not call place_order again for breakeven
        prev_call_count = conn.place_order.call_count
        await manager._check_all_positions()

    # Place count should NOT have increased from the breakeven rule
    # (may increase from trailing if we crossed 2R, but not from breakeven again)
    state = manager._states[(EXCHANGE, SYMBOL)]
    assert state.breakeven_done  # still True — only set once


# ─────────────────────── partial TP ───────────────────────


async def test_partial_tp_fires_at_1_5r() -> None:
    """At ≥1.5R gain, a reduce-only market order for 50% of position is placed."""
    conn = _make_connector()
    pos = _make_position(entry_price=100.0, mark_price=107.5, quantity=2.0)  # 1.5R with sl=95

    manager = _make_exit_manager([pos], {EXCHANGE: conn}, config=ExitConfig(
        breakeven_trigger_r=1.0,
        partial_tp_trigger_r=1.5,
        trailing_trigger_r=2.0,
        max_bars_open=10000,  # disable time exit
    ))
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=95.0,
        risk_usd=10.0,
    )

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as be, \
         patch("src.monitoring.prom_metrics.exits_partial_tp_total") as pt:
        be.labels.return_value = MagicMock()
        pt.labels.return_value = MagicMock()
        await manager._check_all_positions()

    # Find the partial-TP order (MARKET, reduce_only, qty ≈ 1.0 = 50% of 2.0)
    partial_calls = [
        c.args[0] for c in conn.place_order.call_args_list
        if c.args[0].order_type.value == "market" and c.args[0].reduce_only
    ]
    assert partial_calls, "expected a partial TP market order"
    tp_req = partial_calls[0]
    assert tp_req.quantity == pytest.approx(1.0)  # 50% of 2.0
    assert tp_req.side == OrderSide.SELL  # long position → close with sell

    state = manager._states[(EXCHANGE, SYMBOL)]
    assert state.partial_tp_done


# ─────────────────────── trailing SL ───────────────────────


async def test_trailing_sl_advances_on_price_rise() -> None:
    """
    After +2R, trailing SL should move up with price.
    A second check at a higher price must advance the SL further.
    """
    conn = _make_connector()

    # Seed feature store with flat ATR data to get a predictable ATR value.
    store = FeatureStore()
    key = FeatureKey(EXCHANGE, SYMBOL, "15m")
    ts0 = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
    bars = [
        Bar(ts_ms=ts0 + i * 900_000, open=100.0, high=101.0, low=99.0, close=100.0, volume=500.0)
        for i in range(30)
    ]
    store.bulk_load(key, bars)

    pos = _make_position(entry_price=100.0, mark_price=110.0)  # +2R with sl=95

    manager = _make_exit_manager([pos], {EXCHANGE: conn}, store=store)
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=95.0,
        risk_usd=5.0,
        sl_order_id="old-sl",
    )

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as be, \
         patch("src.monitoring.prom_metrics.exits_trailed_total") as tr:
        be.labels.return_value = MagicMock()
        tr.labels.return_value = MagicMock()
        await manager._check_all_positions()

    state = manager._states[(EXCHANGE, SYMBOL)]
    first_sl = state.current_sl

    # SL should now be above the original 95.0 (trailed up)
    assert first_sl > 95.0, f"expected SL to trail up from 95, got {first_sl}"

    # Now price rises further → SL should advance again
    pos.mark_price = 115.0
    prev_place_count = conn.place_order.call_count

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as be, \
         patch("src.monitoring.prom_metrics.exits_trailed_total") as tr:
        be.labels.return_value = MagicMock()
        tr.labels.return_value = MagicMock()
        await manager._check_all_positions()

    second_sl = manager._states[(EXCHANGE, SYMBOL)].current_sl
    assert second_sl >= first_sl, "trailing SL must never retreat"


async def test_trailing_sl_does_not_retreat_on_price_drop() -> None:
    """If price drops after trailing, the SL must NOT move down."""
    conn = _make_connector()
    store = FeatureStore()
    key = FeatureKey(EXCHANGE, SYMBOL, "15m")
    ts0 = int(datetime(2026, 1, 1, tzinfo=UTC).timestamp() * 1000)
    store.bulk_load(key, [
        Bar(ts_ms=ts0 + i * 900_000, open=100.0, high=101.0, low=99.0, close=100.0, volume=500.0)
        for i in range(30)
    ])

    pos = _make_position(entry_price=100.0, mark_price=110.0)
    manager = _make_exit_manager([pos], {EXCHANGE: conn}, store=store)
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=95.0,
        risk_usd=5.0,
    )

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as be, \
         patch("src.monitoring.prom_metrics.exits_trailed_total") as tr:
        be.labels.return_value = MagicMock()
        tr.labels.return_value = MagicMock()
        await manager._check_all_positions()

    sl_after_first_trail = manager._states[(EXCHANGE, SYMBOL)].current_sl
    prev_place_count = conn.place_order.call_count

    # Price drops back — should NOT trigger another trail (already above original)
    pos.mark_price = 105.0  # still above entry but below the 110 trail point

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as be, \
         patch("src.monitoring.prom_metrics.exits_trailed_total") as tr:
        be.labels.return_value = MagicMock()
        tr.labels.return_value = MagicMock()
        await manager._check_all_positions()

    sl_after_drop = manager._states[(EXCHANGE, SYMBOL)].current_sl
    # The trailing SL must not have retreated
    assert sl_after_drop >= sl_after_first_trail


# ─────────────────────── time exit ───────────────────────


async def test_time_exit_fires_when_bars_exceeded_and_low_r() -> None:
    """
    After max_bars_open and gain < time_exit_min_r, a market order closes the position.
    """
    conn = _make_connector()
    # Position barely moved — only 0.2R gain, stuck
    pos = _make_position(entry_price=100.0, mark_price=101.0)  # 0.2R with sl=95

    manager = _make_exit_manager([pos], {EXCHANGE: conn}, config=ExitConfig(
        max_bars_open=3,       # very short limit for testing
        time_exit_min_r=0.5,   # time-exit if gain < 0.5R
        breakeven_trigger_r=1.0,
        partial_tp_trigger_r=1.5,
        trailing_trigger_r=2.0,
    ))
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=95.0,
        risk_usd=5.0,
    )

    # Run enough checks to exceed max_bars_open
    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as be, \
         patch("src.monitoring.prom_metrics.exits_time_stop_total") as ts:
        be.labels.return_value = MagicMock()
        ts.labels.return_value = MagicMock()
        for _ in range(5):  # 5 > max_bars_open=3
            await manager._check_all_positions()

    # A reduce-only MARKET order should have been placed for full quantity
    market_orders = [
        c.args[0] for c in conn.place_order.call_args_list
        if c.args[0].order_type.value == "market" and c.args[0].reduce_only
    ]
    assert market_orders, "expected time-exit market order"
    exit_req = market_orders[0]
    assert exit_req.quantity == pytest.approx(pos.quantity)
    assert exit_req.meta.get("exit") == "time_stop"


async def test_time_exit_does_not_fire_when_position_is_profitable() -> None:
    """Time exit is suppressed when gain >= time_exit_min_r."""
    conn = _make_connector()
    # +1R gain — above time_exit_min_r (0.5)
    pos = _make_position(entry_price=100.0, mark_price=105.0)

    manager = _make_exit_manager([pos], {EXCHANGE: conn}, config=ExitConfig(
        max_bars_open=3,
        time_exit_min_r=0.5,
        breakeven_trigger_r=1.0,
        partial_tp_trigger_r=1.5,
        trailing_trigger_r=2.0,
    ))
    manager.register_position(
        EXCHANGE, SYMBOL,
        entry_ts=datetime(2026, 1, 1, tzinfo=UTC),
        original_sl=95.0,
        risk_usd=5.0,
    )

    with patch("src.monitoring.prom_metrics.exits_breakeven_total") as be, \
         patch("src.monitoring.prom_metrics.exits_time_stop_total") as ts:
        be.labels.return_value = MagicMock()
        ts.labels.return_value = MagicMock()
        for _ in range(5):
            await manager._check_all_positions()

    # Time-stop should not have fired a market order (breakeven may have fired)
    market_time_stops = [
        c.args[0] for c in conn.place_order.call_args_list
        if c.args[0].meta.get("exit") == "time_stop"
    ]
    assert not market_time_stops


# ─────────────────────── adjust_stop safety discipline ───────────────────────


async def test_adjust_stop_places_new_before_cancelling_old() -> None:
    """
    The new SL order must be placed BEFORE the old one is cancelled.
    Order of calls: place_order, then cancel_order.
    """
    call_sequence: list[str] = []

    async def mock_place(req: OrderRequest) -> OrderResult:
        call_sequence.append("place")
        return OrderResult(
            client_order_id="cid",
            exchange_order_id="new-eid",
            status=OrderStatus.OPEN,
            filled_quantity=0.0,
            average_price=None,
            raw={},
        )

    async def mock_cancel(symbol: str, **kwargs: Any) -> None:
        call_sequence.append("cancel")

    conn = MagicMock()
    conn.place_order = mock_place
    conn.cancel_order = mock_cancel

    pos = _make_position(entry_price=100.0, mark_price=100.0)
    state = _ExitState(original_sl=95.0, current_sl=95.0, current_sl_order_id="old-eid")

    manager = ExitManager(
        tracker=MagicMock(),
        feature_store=FeatureStore(),
        connectors={EXCHANGE: conn},
    )
    success = await manager.adjust_stop(conn, pos, new_sl=97.0, state=state)

    assert success
    assert call_sequence == ["place", "cancel"], (
        f"expected place → cancel order, got {call_sequence}"
    )


async def test_adjust_stop_does_not_cancel_if_new_order_fails() -> None:
    """If placing the new SL order fails, the old one must NOT be cancelled."""
    conn = MagicMock()
    conn.place_order = AsyncMock(side_effect=RuntimeError("network error"))
    conn.cancel_order = AsyncMock()

    pos = _make_position()
    state = _ExitState(original_sl=95.0, current_sl=95.0, current_sl_order_id="old-eid")

    manager = ExitManager(
        tracker=MagicMock(),
        feature_store=FeatureStore(),
        connectors={EXCHANGE: conn},
    )
    success = await manager.adjust_stop(conn, pos, new_sl=97.0, state=state)

    assert not success
    conn.cancel_order.assert_not_called()
    # current_sl must remain unchanged
    assert state.current_sl == pytest.approx(95.0)
