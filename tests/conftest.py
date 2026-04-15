"""
Shared pytest fixtures.

Provides:
- `sample_ohlcv_df`: a deterministic 200-bar synthetic OHLCV frame usable
  by all indicator/strategy tests.
- `risk_config`, `execution_config`: populated dataclasses with defaults.
- `fake_storage`: a stub Storage that records calls without hitting a DB.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.settings import (
    AppConfig,
    ExecutionConfig,
    RegimeConfig,
    RiskConfig,
)


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Deterministic synthetic OHLCV: upward drift with sinusoidal overlay."""
    rng = np.random.default_rng(42)
    n = 300
    t = np.arange(n)
    # Base drift + sine
    close = 100 + t * 0.05 + np.sin(t / 15) * 3 + rng.normal(0, 0.6, n).cumsum() * 0.3
    high = close + rng.uniform(0.3, 1.5, n)
    low = close - rng.uniform(0.3, 1.5, n)
    open_ = close + rng.uniform(-0.8, 0.8, n)
    volume = rng.uniform(100, 1000, n)

    ts = [datetime(2026, 1, 1, tzinfo=UTC) + timedelta(minutes=15 * i) for i in range(n)]
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low, "close": close, "volume": volume,
    }, index=pd.DatetimeIndex(ts, name="ts"))
    return df


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig(
        max_daily_drawdown_pct=0.10,
        warning_drawdown_pct=0.05,
        risk_per_trade_pct=0.02,
        max_open_positions=3,
        max_positions_per_symbol=1,
        leverage=3,
        sl_atr_multiplier=1.5,
        min_reward_to_risk=1.5,
        max_correlation=0.80,
        correlation_lookback_bars=100,
        max_margin_utilization=0.70,
        flatten_on_shutdown=False,
    )


@pytest.fixture
def execution_config() -> ExecutionConfig:
    return ExecutionConfig(
        max_retries=3, retry_base_seconds=0.1, retry_jitter_seconds=0.05,
        order_timeout_sec=5.0, default_order_type="market",
        slippage_tolerance_bps=10.0,
    )


@pytest.fixture
def regime_config() -> RegimeConfig:
    return RegimeConfig(
        adx_period=14, adx_trend_threshold=25, adx_range_threshold=20,
        vol_window_bars=96, vol_high_threshold=0.015,
        weights={
            "trend_high_vol": {"momentum": 1.0, "volatility_breakout": 0.8,
                               "mean_reversion": 0.0, "funding_arb": 1.0},
            "trend_low_vol": {"momentum": 0.9, "volatility_breakout": 0.5,
                              "mean_reversion": 0.0, "funding_arb": 1.0},
            "range_high_vol": {"momentum": 0.2, "volatility_breakout": 0.3,
                               "mean_reversion": 0.8, "funding_arb": 1.0},
            "range_low_vol": {"momentum": 0.0, "volatility_breakout": 0.1,
                              "mean_reversion": 1.0, "funding_arb": 1.0},
            "chop": {"momentum": 0.0, "volatility_breakout": 0.0,
                     "mean_reversion": 0.3, "funding_arb": 1.0},
        },
    )


class FakeStorage:
    """In-memory stand-in for Storage. Records every call for assertions."""
    def __init__(self) -> None:
        self.state: dict[str, Any] = {}
        self.events: list[dict[str, Any]] = []
        self.signals_recorded: list[dict[str, Any]] = []
        self.orders_recorded: list[dict[str, Any]] = []
        self.equity_recorded: list[dict[str, Any]] = []

    async def get_state(self, key: str) -> Any:
        return self.state.get(key)

    async def set_state(self, key: str, value: Any) -> None:
        self.state[key] = value

    async def record_signal(self, **kwargs: Any) -> int:
        self.signals_recorded.append(kwargs)
        return len(self.signals_recorded)

    async def record_order(self, **kwargs: Any) -> int:
        self.orders_recorded.append(kwargs)
        return len(self.orders_recorded)

    async def update_order_status(self, *args: Any, **kwargs: Any) -> None:
        return None

    async def record_fill(self, **kwargs: Any) -> int:
        return 1

    async def record_equity(self, **kwargs: Any) -> None:
        self.equity_recorded.append(kwargs)

    async def snapshot_positions(self, mode: str, positions: Any) -> None:
        return None

    async def record_event(self, *args: Any, **kwargs: Any) -> None:
        self.events.append({"args": args, "kwargs": kwargs})

    async def record_funding_rate(self, **kwargs: Any) -> None:
        return None


@pytest.fixture
def fake_storage() -> FakeStorage:
    return FakeStorage()
