"""
Risk manager: position sizing, SL/TP computation, signal validation.

Responsibilities:
- Given a validated Signal and current equity + ATR, compute position size
  such that the stop distance corresponds to exactly `risk_per_trade_pct` of
  equity. This is volatility-normalized sizing and is the single most important
  feature of the risk module.
- Compute stop-loss and take-profit prices using the ATR and the minimum
  reward-to-risk multiplier.
- Reject signals that violate leverage / margin / min-notional constraints.

This module is stateless w.r.t. PnL — it ONLY sizes individual trades. The
kill switch and portfolio-level limits live in `risk/limits.py` and
`risk/kill_switch.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.exchanges.base import OrderSide
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.settings import RiskConfig

log = get_logger(__name__)


@dataclass(slots=True)
class SizedTrade:
    """Output of the risk manager — ready for the executor."""

    symbol: str
    exchange: str
    side: OrderSide
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    leverage: int
    risk_usd: float  # dollar amount at risk if SL hits
    notional_usd: float  # position notional
    strategy: str
    confidence: float


@dataclass(slots=True)
class RejectedTrade:
    reason: str
    details: dict[str, float | str] | None = None


class RiskManager:
    """
    Volatility-normalized position sizing with ATR-based stops.

    The core sizing formula:

        risk_dollars = equity * risk_per_trade_pct * confidence_scalar
            (confidence_scalar = 0.5 + 0.5 * clamp(confidence, 0, 1))
        stop_distance = atr * sl_atr_multiplier
        quantity = risk_dollars / stop_distance

    Then:
        stop_loss  = entry ± stop_distance
        take_profit = entry ± stop_distance * min_reward_to_risk

    A trade is rejected if:
        - quantity rounds to zero
        - notional * leverage > max_margin_utilization * free_margin
        - ATR is zero/NaN (insufficient data)
    """

    def __init__(self, config: "RiskConfig") -> None:
        self._config = config

    def size_trade(
        self,
        *,
        symbol: str,
        exchange: str,
        side: OrderSide,
        entry_price: float,
        atr: float,
        equity: float,
        free_margin: float,
        strategy: str,
        confidence: float,
        min_notional_usd: float = 5.0,
        contract_size: float = 1.0,
    ) -> SizedTrade | RejectedTrade:
        """
        Size a single trade. See class docstring for the formula.

        Args:
            entry_price:   reference price (last close or mid).
            atr:           Average True Range at the entry timeframe, in price units.
            equity:        total account equity (balance + unrealized PnL).
            free_margin:   cash currently free to open new positions.
            contract_size: for exchanges where qty is in contracts not base
                           currency (defaults to 1, i.e. qty == base units).
        """
        if entry_price <= 0:
            return RejectedTrade(reason="invalid_entry_price", details={"entry": entry_price})
        if atr is None or atr <= 0 or not math.isfinite(float(atr)):
            return RejectedTrade(reason="invalid_atr", details={"atr": float(atr or 0)})
        if math.isnan(equity) or equity <= 0:
            return RejectedTrade(reason="zero_equity")

        confidence_scalar = 0.5 + 0.5 * max(0.0, min(1.0, confidence))
        risk_dollars = equity * self._config.risk_per_trade_pct * confidence_scalar
        stop_distance = atr * self._config.sl_atr_multiplier
        if stop_distance <= 0:
            return RejectedTrade(reason="zero_stop_distance")

        # Quantity in base currency units. The executor converts to contracts
        # if the exchange requires it (contract_size param kept for caller
        # compatibility but NOT applied here — applying it would cancel out
        # and break the ATR-based dollar-risk formula).
        raw_quantity = risk_dollars / stop_distance
        quantity = raw_quantity  # base units (e.g. BTC, ETH)

        if quantity <= 0:
            return RejectedTrade(reason="quantity_zero")

        notional = quantity * entry_price  # base units × price = USD notional
        if notional < min_notional_usd:
            return RejectedTrade(
                reason="below_min_notional",
                details={"notional": notional, "min": min_notional_usd},
            )

        # Margin check: the margin the exchange will lock is notional / leverage.
        required_margin = notional / max(1, self._config.leverage)
        margin_budget = free_margin * self._config.max_margin_utilization
        if required_margin > margin_budget:
            return RejectedTrade(
                reason="insufficient_margin",
                details={
                    "required_margin": required_margin,
                    "margin_budget": margin_budget,
                },
            )

        # Compute SL/TP
        if side == OrderSide.BUY:
            stop_loss = entry_price - stop_distance
            take_profit = entry_price + stop_distance * self._config.min_reward_to_risk
        else:
            stop_loss = entry_price + stop_distance
            take_profit = entry_price - stop_distance * self._config.min_reward_to_risk

        if stop_loss <= 0 or take_profit <= 0:
            return RejectedTrade(
                reason="invalid_sl_tp",
                details={"sl": stop_loss, "tp": take_profit},
            )

        return SizedTrade(
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            leverage=self._config.leverage,
            risk_usd=risk_dollars,
            notional_usd=notional,
            strategy=strategy,
            confidence=confidence,
        )
