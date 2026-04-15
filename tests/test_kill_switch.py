"""Kill switch state-machine tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from src.risk.kill_switch import HaltState, KillSwitch
from src.settings import RiskConfig


class _Recorder:
    def __init__(self) -> None:
        self.halts: list[tuple[str, float]] = []
        self.warnings: list[float] = []

    async def on_halt(self, reason: str, dd: float) -> None:
        self.halts.append((reason, dd))

    async def on_warning(self, dd: float) -> None:
        self.warnings.append(dd)


async def _build(risk_config: RiskConfig, storage: Any) -> tuple[KillSwitch, _Recorder]:
    rec = _Recorder()
    ks = KillSwitch(config=risk_config, storage=storage,
                    on_halt=rec.on_halt, on_warning=rec.on_warning)
    await ks.load()
    return ks, rec


async def test_fresh_start_is_running(risk_config: RiskConfig, fake_storage: Any) -> None:
    ks, _ = await _build(risk_config, fake_storage)
    assert ks.state == HaltState.RUNNING
    assert not ks.is_halted


async def test_day_roll_sets_anchor(risk_config: RiskConfig, fake_storage: Any) -> None:
    ks, _ = await _build(risk_config, fake_storage)
    ts = datetime(2026, 4, 9, 0, 5, tzinfo=UTC)
    await ks.on_equity_update(10000.0, ts=ts)
    assert ks.state == HaltState.RUNNING
    # anchor is start of day
    # drawdown at same equity = 0
    assert ks.current_drawdown(10000.0) == 0.0


async def test_warning_then_halt_sequence(
    risk_config: RiskConfig, fake_storage: Any
) -> None:
    # warning at 5%, halt at 10%
    ks, rec = await _build(risk_config, fake_storage)
    ts0 = datetime(2026, 4, 9, 0, 0, tzinfo=UTC)
    await ks.on_equity_update(10000.0, ts=ts0)

    # 4% drawdown — nothing fires
    await ks.on_equity_update(9600.0, ts=ts0 + timedelta(minutes=1))
    assert not rec.warnings
    assert not rec.halts
    assert ks.state == HaltState.RUNNING

    # 6% drawdown — warning only
    await ks.on_equity_update(9400.0, ts=ts0 + timedelta(minutes=2))
    assert rec.warnings
    assert not rec.halts
    assert ks.state == HaltState.WARNING

    # 11% drawdown — halt
    await ks.on_equity_update(8900.0, ts=ts0 + timedelta(minutes=3))
    assert rec.halts
    assert ks.state == HaltState.HALTED
    assert ks.is_halted


async def test_halt_persists_across_reload(
    risk_config: RiskConfig, fake_storage: Any
) -> None:
    ks, _ = await _build(risk_config, fake_storage)
    ts0 = datetime(2026, 4, 9, 0, 0, tzinfo=UTC)
    await ks.on_equity_update(10000.0, ts=ts0)
    await ks.on_equity_update(8500.0, ts=ts0 + timedelta(minutes=1))
    assert ks.is_halted

    # Reload a fresh KillSwitch from the same storage — must still be halted.
    ks2 = KillSwitch(config=risk_config, storage=fake_storage)
    await ks2.load()
    assert ks2.is_halted


async def test_manual_reset_clears_halt(
    risk_config: RiskConfig, fake_storage: Any
) -> None:
    ks, _ = await _build(risk_config, fake_storage)
    await ks.force_halt("test")
    assert ks.is_halted
    await ks.reset()
    assert not ks.is_halted
    assert ks.state == HaltState.RUNNING


async def test_halt_survives_day_boundary(
    risk_config: RiskConfig, fake_storage: Any
) -> None:
    """
    A halt on day N must NOT automatically clear at midnight. Manual reset
    is required.
    """
    ks, _ = await _build(risk_config, fake_storage)
    ts0 = datetime(2026, 4, 9, 23, 0, tzinfo=UTC)
    await ks.on_equity_update(10000.0, ts=ts0)
    await ks.on_equity_update(8500.0, ts=ts0 + timedelta(minutes=1))
    assert ks.is_halted

    # Day rolls over
    ts1 = datetime(2026, 4, 10, 0, 1, tzinfo=UTC)
    await ks.on_equity_update(8500.0, ts=ts1)
    assert ks.is_halted, "halt state must survive day boundary"
