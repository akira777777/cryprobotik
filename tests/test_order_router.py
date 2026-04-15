"""
Tests for OrderRouter.

Covers:
- Explicit preferred_exchange is honoured (highest priority).
- Funding-rate edge: long → prefer venue with lower (or negative) funding rate.
- Funding-rate edge: short → prefer venue with higher (positive) funding rate.
- Exposure tiebreaker: when funding rates are equal, route to the less-utilized
  venue.
- Single-exchange configuration always returns that exchange.
- Missing preferred_exchange falls through to automatic selection.
- No connectors raises RuntimeError.
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import MagicMock

import pytest

from src.exchanges.base import OrderSide
from src.execution.order_router import OrderRouter, RoutedLeg
from src.strategies.base import Signal, SignalAction


# ─────────────────────── helpers ───────────────────────


def _make_connector(name: str) -> MagicMock:
    """Return a mock ExchangeConnector identified by *name*."""
    c = MagicMock()
    c.name = name
    return c


def _make_tracker(exposure: dict[str, float]) -> MagicMock:
    """Return a mock PortfolioTracker that reports the given per-exchange exposure."""
    t = MagicMock()
    t.exposure_by_exchange.return_value = exposure
    return t


def _make_signal(
    side: OrderSide,
    symbol: str = "BTC/USDT:USDT",
    preferred_exchange: str | None = None,
) -> Signal:
    return Signal(
        strategy="test",
        symbol=symbol,
        action=SignalAction.OPEN,
        side=side,
        confidence=0.7,
        ts=datetime(2026, 1, 1, tzinfo=UTC),
        preferred_exchange=preferred_exchange,
    )


def _build_router(
    connectors: dict[str, object],
    exposure: dict[str, float] | None = None,
) -> OrderRouter:
    tracker = _make_tracker(exposure or {})
    return OrderRouter(connectors=connectors, tracker=tracker)  # type: ignore[arg-type]


# ─────────────────────── preferred_exchange ───────────────────────


def test_preferred_exchange_is_honoured() -> None:
    """If the signal names an exchange, we use it regardless of funding."""
    okx = _make_connector("okx")
    bybit = _make_connector("bybit")
    router = _build_router({"okx": okx, "bybit": bybit})

    # Funding strongly favours bybit for a long, but preferred_exchange wins.
    router.update_funding("okx", "BTC/USDT:USDT", rate=0.01)   # okx expensive for long
    router.update_funding("bybit", "BTC/USDT:USDT", rate=-0.01) # bybit cheap for long

    signal = _make_signal(OrderSide.BUY, preferred_exchange="okx")
    leg = router.route(signal)

    assert isinstance(leg, RoutedLeg)
    assert leg.exchange_name == "okx"
    assert leg.connector is okx


def test_preferred_exchange_unknown_falls_through() -> None:
    """preferred_exchange pointing to a non-configured exchange is ignored."""
    okx = _make_connector("okx")
    router = _build_router({"okx": okx})

    signal = _make_signal(OrderSide.BUY, preferred_exchange="binance")  # not in connectors
    leg = router.route(signal)

    assert leg.exchange_name == "okx"
    assert leg.connector is okx


# ─────────────────────── funding-rate edge ───────────────────────


def test_long_prefers_lower_funding_rate() -> None:
    """
    For a LONG position we pay funding when rate > 0 and receive when rate < 0.
    Router should pick the venue with the lower (or negative) rate.
    """
    okx = _make_connector("okx")
    bybit = _make_connector("bybit")
    router = _build_router({"okx": okx, "bybit": bybit})

    router.update_funding("okx", "ETH/USDT:USDT", rate=0.005)   # expensive for long
    router.update_funding("bybit", "ETH/USDT:USDT", rate=-0.002) # longs receive here

    signal = _make_signal(OrderSide.BUY, symbol="ETH/USDT:USDT")
    leg = router.route(signal)

    # bybit has the more favourable (lower) rate for a long
    assert leg.exchange_name == "bybit"


def test_short_prefers_higher_funding_rate() -> None:
    """
    For a SHORT position we RECEIVE funding when rate > 0 and PAY when rate < 0.
    Router should pick the venue with the higher (positive) rate.
    """
    okx = _make_connector("okx")
    bybit = _make_connector("bybit")
    router = _build_router({"okx": okx, "bybit": bybit})

    router.update_funding("okx", "ETH/USDT:USDT", rate=0.010)   # shorts receive more here
    router.update_funding("bybit", "ETH/USDT:USDT", rate=0.001) # shorts receive less here

    signal = _make_signal(OrderSide.SELL, symbol="ETH/USDT:USDT")
    leg = router.route(signal)

    assert leg.exchange_name == "okx"


def test_funding_rate_tie_broken_by_exposure() -> None:
    """
    When funding rates are identical across venues, the less-utilized (lower
    exposure) venue should be preferred.
    """
    okx = _make_connector("okx")
    bybit = _make_connector("bybit")

    # Equal funding rates → exposure tiebreaker activates.
    router = _build_router(
        {"okx": okx, "bybit": bybit},
        exposure={"okx": 10_000.0, "bybit": 3_000.0},  # bybit less loaded
    )
    router.update_funding("okx", "BTC/USDT:USDT", rate=0.001)
    router.update_funding("bybit", "BTC/USDT:USDT", rate=0.001)

    signal = _make_signal(OrderSide.BUY)
    leg = router.route(signal)

    # bybit has lower exposure and identical funding → should win
    assert leg.exchange_name == "bybit"


def test_no_funding_data_falls_back_to_exposure() -> None:
    """Without any funding data, routing is governed purely by exposure balance."""
    okx = _make_connector("okx")
    bybit = _make_connector("bybit")

    router = _build_router(
        {"okx": okx, "bybit": bybit},
        exposure={"okx": 50_000.0, "bybit": 5_000.0},
    )
    # No update_funding calls — both default to 0.0.

    signal = _make_signal(OrderSide.BUY)
    leg = router.route(signal)

    # bybit has less exposure → should be selected
    assert leg.exchange_name == "bybit"


# ─────────────────────── single-exchange ───────────────────────


def test_single_exchange_always_wins() -> None:
    """With only one venue there's nothing to route — just return it."""
    okx = _make_connector("okx")
    router = _build_router({"okx": okx})

    for side in (OrderSide.BUY, OrderSide.SELL):
        leg = router.route(_make_signal(side))
        assert leg.exchange_name == "okx"
        assert leg.connector is okx


# ─────────────────────── no connectors ───────────────────────


def test_no_connectors_raises() -> None:
    """Router with no connectors must raise RuntimeError, not silently return None."""
    router = _build_router({})

    with pytest.raises(RuntimeError, match="no exchanges configured"):
        router.route(_make_signal(OrderSide.BUY))


# ─────────────────────── funding-arb context ───────────────────────


def test_update_funding_overwrites_previous_rate() -> None:
    """Calling update_funding twice keeps only the latest rate."""
    okx = _make_connector("okx")
    router = _build_router({"okx": okx})

    router.update_funding("okx", "BTC/USDT:USDT", rate=0.002)
    router.update_funding("okx", "BTC/USDT:USDT", rate=-0.005)  # updated

    # The internal table must reflect the latest value.
    assert router._funding[("okx", "BTC/USDT:USDT")] == pytest.approx(-0.005)


def test_funding_arb_legs_routed_to_different_venues() -> None:
    """
    funding_arb emits a PairSignal (handled by the orchestrator), not a Signal.
    Verify that manually routing two opposing legs ends up on different connectors.

    This mirrors the orchestrator's logic: long_leg to long_exchange,
    short_leg to short_exchange — each via preferred_exchange.
    """
    okx = _make_connector("okx")
    bybit = _make_connector("bybit")
    router = _build_router({"okx": okx, "bybit": bybit})

    # Funding arb: long on bybit, short on okx
    long_signal = _make_signal(
        OrderSide.BUY, symbol="BTC/USDT:USDT", preferred_exchange="bybit"
    )
    short_signal = _make_signal(
        OrderSide.SELL, symbol="BTC/USDT:USDT", preferred_exchange="okx"
    )

    long_leg = router.route(long_signal)
    short_leg = router.route(short_signal)

    assert long_leg.exchange_name == "bybit"
    assert short_leg.exchange_name == "okx"
    # The two connectors are distinct objects.
    assert long_leg.connector is not short_leg.connector


# ─────────────────────── returned RoutedLeg shape ───────────────────────


def test_routed_leg_carries_connector_reference() -> None:
    """RoutedLeg.connector must be the exact object registered in the router."""
    okx = _make_connector("okx")
    bybit = _make_connector("bybit")
    router = _build_router({"okx": okx, "bybit": bybit})

    signal = _make_signal(OrderSide.BUY, preferred_exchange="okx")
    leg = router.route(signal)

    assert leg.connector is okx   # identity check, not equality
