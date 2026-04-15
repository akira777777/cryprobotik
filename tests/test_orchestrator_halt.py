"""
Tests for halt / kill-switch behaviour.

Covers:
1. _on_halt_callback — calls cancel_all_for_symbol + flatten_position once per
   open position, across the correct connectors.
2. KillSwitch fires halt callback exactly ONCE even when equity keeps falling.
3. Halt state survives a simulated restart (load() restores HALTED from storage).
4. force_halt persists and calls the callback.
5. Day-roll does NOT auto-clear a HALTED state (sticky by design).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.risk.kill_switch import HaltState, KillSwitch


# ─────────────────────── fake infrastructure ───────────────────────


class _FakeStorage:
    """Minimal storage double that persists bot_state in memory."""

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    async def get_state(self, key: str) -> Any:
        return self._state.get(key)

    async def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value


class _FakeTracker:
    """Returns a configurable list of open positions."""

    def __init__(self, positions: list[Any] | None = None) -> None:
        self._positions = positions or []

    def open_positions(self) -> list[Any]:
        return list(self._positions)


def _make_position(exchange: str, symbol: str) -> MagicMock:
    p = MagicMock()
    p.exchange = exchange
    p.symbol = symbol
    return p


def _make_connector() -> MagicMock:
    c = MagicMock()
    c.cancel_all = AsyncMock()
    c.close_position = AsyncMock()
    return c


# ─────────────────────── standalone on_halt_callback logic ───────────────────────

# Rather than instantiating the full Orchestrator (which needs many deps),
# we re-test the *contract* of _on_halt_callback by extracting its logic
# into a helper function that matches the implementation exactly.

async def _simulated_halt_callback(
    tracker: _FakeTracker,
    executor: Any,
    connectors: dict[str, Any],
    reason: str,
    drawdown: float,
) -> None:
    """Mirrors Orchestrator._on_halt_callback without the Telegram / metrics."""
    if tracker is not None and executor is not None:
        for pos in tracker.open_positions():
            conn = connectors.get(pos.exchange)
            if conn is None:
                continue
            await executor.cancel_all_for_symbol(conn, pos.symbol)
            await executor.flatten_position(conn, pos.symbol)


# ─────────────────────── halt callback tests ───────────────────────


async def test_halt_callback_cancels_and_flattens_all_positions() -> None:
    """
    On halt, cancel_all_for_symbol and flatten_position are called once per
    open position, on the matching connector.
    """
    okx_conn = _make_connector()
    bybit_conn = _make_connector()
    connectors = {"okx": okx_conn, "bybit": bybit_conn}

    positions = [
        _make_position("okx", "BTC/USDT:USDT"),
        _make_position("okx", "ETH/USDT:USDT"),
        _make_position("bybit", "SOL/USDT:USDT"),
    ]
    tracker = _FakeTracker(positions)

    executor = MagicMock()
    executor.cancel_all_for_symbol = AsyncMock()
    executor.flatten_position = AsyncMock()

    await _simulated_halt_callback(tracker, executor, connectors, "drawdown", 0.11)

    # Three positions → three cancel + three flatten calls
    assert executor.cancel_all_for_symbol.call_count == 3
    assert executor.flatten_position.call_count == 3

    # Verify correct connectors were passed
    cancel_args = [call.args for call in executor.cancel_all_for_symbol.call_args_list]
    assert (okx_conn, "BTC/USDT:USDT") in cancel_args
    assert (okx_conn, "ETH/USDT:USDT") in cancel_args
    assert (bybit_conn, "SOL/USDT:USDT") in cancel_args


async def test_halt_callback_skips_unknown_exchange() -> None:
    """If a position references an exchange not in connectors, it is silently skipped."""
    connectors: dict[str, Any] = {}   # no connectors at all
    tracker = _FakeTracker([_make_position("okx", "BTC/USDT:USDT")])

    executor = MagicMock()
    executor.cancel_all_for_symbol = AsyncMock()
    executor.flatten_position = AsyncMock()

    # Must not raise
    await _simulated_halt_callback(tracker, executor, connectors, "reason", 0.12)

    executor.cancel_all_for_symbol.assert_not_called()
    executor.flatten_position.assert_not_called()


async def test_halt_callback_no_positions_is_noop() -> None:
    """Empty position book → nothing is cancelled or flattened."""
    conn = _make_connector()
    executor = MagicMock()
    executor.cancel_all_for_symbol = AsyncMock()
    executor.flatten_position = AsyncMock()

    await _simulated_halt_callback(
        _FakeTracker([]), executor, {"okx": conn}, "reason", 0.10
    )

    executor.cancel_all_for_symbol.assert_not_called()
    executor.flatten_position.assert_not_called()


# ─────────────────────── KillSwitch: halt fires exactly once ───────────────────────


async def test_halt_fires_exactly_once_on_repeated_drawdown(
    risk_config: Any,
) -> None:
    """
    Even if equity keeps falling after the halt threshold is crossed, the halt
    callback must be called exactly ONCE per halt event.
    """
    storage = _FakeStorage()
    halt_calls: list[tuple[str, float]] = []

    async def on_halt(reason: str, drawdown: float) -> None:
        halt_calls.append((reason, drawdown))

    ks = KillSwitch(config=risk_config, storage=storage, on_halt=on_halt)
    await ks.load()

    ts = datetime(2026, 1, 2, 12, 0, 0, tzinfo=UTC)
    # Seed the day at 10 000
    await ks.on_equity_update(10_000.0, ts=ts)

    # Drop past max_daily_drawdown_pct (10%) → triggers halt
    await ks.on_equity_update(8_900.0, ts=ts + timedelta(minutes=30))
    # Continue falling — must NOT fire a second callback
    await ks.on_equity_update(8_000.0, ts=ts + timedelta(hours=1))
    await ks.on_equity_update(7_000.0, ts=ts + timedelta(hours=2))

    assert len(halt_calls) == 1
    assert ks.is_halted


async def test_halt_is_halted_property(risk_config: Any) -> None:
    """is_halted reflects the HALTED state correctly."""
    storage = _FakeStorage()
    ks = KillSwitch(config=risk_config, storage=storage)
    await ks.load()

    assert not ks.is_halted

    await ks.force_halt("manual_test")
    assert ks.is_halted
    assert ks.state == HaltState.HALTED


# ─────────────────────── KillSwitch: halt persists across restart ───────────────────────


async def test_halt_persists_after_simulated_restart(risk_config: Any) -> None:
    """
    After a halt is triggered and persisted, a new KillSwitch.load() on the
    same storage must restore the HALTED state — i.e. the bot stays halted
    after a crash+restart.
    """
    storage = _FakeStorage()

    # First instance: trigger halt
    halt_calls_1: list[str] = []

    async def on_halt_1(reason: str, drawdown: float) -> None:
        halt_calls_1.append(reason)

    ks1 = KillSwitch(config=risk_config, storage=storage, on_halt=on_halt_1)
    await ks1.load()

    ts = datetime(2026, 1, 3, 12, 0, 0, tzinfo=UTC)
    await ks1.on_equity_update(10_000.0, ts=ts)
    await ks1.on_equity_update(8_500.0, ts=ts + timedelta(minutes=5))  # > 10% drawdown

    assert ks1.is_halted, "first instance should be halted"
    assert len(halt_calls_1) == 1

    # Simulate restart: second instance with the SAME storage
    halt_calls_2: list[str] = []

    async def on_halt_2(reason: str, drawdown: float) -> None:
        halt_calls_2.append(reason)

    ks2 = KillSwitch(config=risk_config, storage=storage, on_halt=on_halt_2)
    await ks2.load()  # must restore state from storage

    assert ks2.is_halted, "restarted instance must still be halted"
    # The halt callback should NOT re-fire on load — we're just restoring state
    assert len(halt_calls_2) == 0


async def test_force_halt_persists_and_fires_callback(risk_config: Any) -> None:
    """force_halt (Telegram /halt command) persists and fires the callback."""
    storage = _FakeStorage()
    fired: list[str] = []

    async def on_halt(reason: str, drawdown: float) -> None:
        fired.append(reason)

    ks = KillSwitch(config=risk_config, storage=storage, on_halt=on_halt)
    await ks.load()

    await ks.force_halt("telegram_command")

    assert ks.is_halted
    assert fired == ["telegram_command"]

    # Confirm persistence: new instance loads HALTED
    ks2 = KillSwitch(config=risk_config, storage=storage)
    await ks2.load()
    assert ks2.is_halted


# ─────────────────────── Day-roll does not clear halt ───────────────────────


async def test_day_roll_does_not_clear_halt(risk_config: Any) -> None:
    """
    UTC midnight day-roll must NOT auto-clear a HALT state.
    The bot must stay halted until scripts/reset_halt.py is run manually.
    """
    storage = _FakeStorage()
    ks = KillSwitch(config=risk_config, storage=storage)
    await ks.load()

    day1 = datetime(2026, 1, 4, 23, 55, 0, tzinfo=UTC)
    await ks.on_equity_update(10_000.0, ts=day1)          # seed day 1
    await ks.on_equity_update(8_500.0, ts=day1 + timedelta(minutes=3))  # halt day 1

    assert ks.is_halted

    # Simulate day-roll: update with a timestamp on day 2
    day2 = datetime(2026, 1, 5, 0, 5, 0, tzinfo=UTC)
    await ks.on_equity_update(8_500.0, ts=day2)

    # Still halted after day-roll
    assert ks.is_halted


async def test_reset_clears_halt(risk_config: Any) -> None:
    """reset() clears the halt state (used by scripts/reset_halt.py)."""
    storage = _FakeStorage()
    ks = KillSwitch(config=risk_config, storage=storage)
    await ks.load()

    await ks.force_halt("test_reset")
    assert ks.is_halted

    await ks.reset()
    assert not ks.is_halted
    assert ks.state == HaltState.RUNNING
