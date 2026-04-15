"""Risk manager position-sizing tests."""

from __future__ import annotations

import pytest

from src.exchanges.base import OrderSide
from src.risk.manager import RejectedTrade, RiskManager, SizedTrade
from src.settings import RiskConfig


def _sizer(risk_config: RiskConfig) -> RiskManager:
    return RiskManager(config=risk_config)


def test_basic_long_sizing(risk_config: RiskConfig) -> None:
    sizer = _sizer(risk_config)
    result = sizer.size_trade(
        symbol="BTC/USDT:USDT", exchange="okx", side=OrderSide.BUY,
        entry_price=50000.0, atr=500.0,
        equity=10000.0, free_margin=10000.0,
        strategy="momentum", confidence=0.8,
    )
    assert isinstance(result, SizedTrade)
    # risk_usd = equity * 2% * confidence_scalar
    #   confidence_scalar = 0.5 + 0.5 * 0.8 = 0.9
    #   → 10000 * 0.02 * 0.9 = 180
    assert result.risk_usd == pytest.approx(180.0)
    # stop distance = atr * 1.5 = 750
    # qty = 180 / 750 = 0.24
    assert result.quantity == pytest.approx(180.0 / 750.0, rel=1e-6)
    # SL = entry - 750
    assert result.stop_loss == pytest.approx(49250.0)
    # TP = entry + 750 * 1.5 = entry + 1125
    assert result.take_profit == pytest.approx(51125.0)
    # RR check: (tp - entry) / (entry - sl) == 1.5
    rr = (result.take_profit - result.entry_price) / (result.entry_price - result.stop_loss)
    assert rr == pytest.approx(1.5)


def test_basic_short_sizing(risk_config: RiskConfig) -> None:
    sizer = _sizer(risk_config)
    result = sizer.size_trade(
        symbol="ETH/USDT:USDT", exchange="bybit", side=OrderSide.SELL,
        entry_price=3000.0, atr=30.0,
        equity=10000.0, free_margin=10000.0,
        strategy="momentum", confidence=0.7,
    )
    assert isinstance(result, SizedTrade)
    # stop = 30 * 1.5 = 45 → SL above entry for short
    assert result.stop_loss == pytest.approx(3045.0)
    # TP = entry - 45 * 1.5 = 2932.5
    assert result.take_profit == pytest.approx(2932.5)
    # Same RR
    rr = (result.entry_price - result.take_profit) / (result.stop_loss - result.entry_price)
    assert rr == pytest.approx(1.5)


def test_rejects_zero_atr(risk_config: RiskConfig) -> None:
    sizer = _sizer(risk_config)
    result = sizer.size_trade(
        symbol="BTC/USDT:USDT", exchange="okx", side=OrderSide.BUY,
        entry_price=50000.0, atr=0.0,
        equity=10000.0, free_margin=10000.0,
        strategy="momentum", confidence=0.8,
    )
    assert isinstance(result, RejectedTrade)
    assert result.reason == "invalid_atr"


def test_rejects_zero_equity(risk_config: RiskConfig) -> None:
    sizer = _sizer(risk_config)
    result = sizer.size_trade(
        symbol="BTC/USDT:USDT", exchange="okx", side=OrderSide.BUY,
        entry_price=50000.0, atr=500.0,
        equity=0.0, free_margin=0.0,
        strategy="momentum", confidence=0.8,
    )
    assert isinstance(result, RejectedTrade)
    assert result.reason == "zero_equity"


def test_rejects_when_notional_below_minimum(risk_config: RiskConfig) -> None:
    sizer = _sizer(risk_config)
    # Tiny equity → minuscule notional
    result = sizer.size_trade(
        symbol="BTC/USDT:USDT", exchange="okx", side=OrderSide.BUY,
        entry_price=50000.0, atr=500.0,
        equity=0.1,    # too little equity
        free_margin=0.1,
        strategy="momentum", confidence=0.8,
        min_notional_usd=5.0,
    )
    assert isinstance(result, RejectedTrade)
    assert result.reason == "below_min_notional"


def test_rejects_insufficient_margin(risk_config: RiskConfig) -> None:
    sizer = _sizer(risk_config)
    # Force sizing that requires more margin than the free_margin budget.
    result = sizer.size_trade(
        symbol="BTC/USDT:USDT", exchange="okx", side=OrderSide.BUY,
        entry_price=50000.0, atr=500.0,
        equity=1_000_000.0,   # equity allows huge notional
        free_margin=10.0,     # but only $10 free margin available
        strategy="momentum", confidence=0.8,
    )
    assert isinstance(result, RejectedTrade)
    assert result.reason == "insufficient_margin"


def test_hard_ceilings_enforced_in_config() -> None:
    with pytest.raises(ValueError, match="max_daily_drawdown_pct"):
        RiskConfig(max_daily_drawdown_pct=0.50)
    with pytest.raises(ValueError, match="risk_per_trade_pct"):
        RiskConfig(risk_per_trade_pct=0.20)
    with pytest.raises(ValueError, match="leverage"):
        RiskConfig(leverage=50)
