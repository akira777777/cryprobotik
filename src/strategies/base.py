"""
Strategy ABC and shared Signal type.

Every strategy is a pure function over a feature store snapshot: given the
current OHLCV buffers, emit zero or more signals. Strategies do NOT:
- place orders directly
- talk to the database
- care about risk sizing (the risk manager owns that)

The ensemble aggregates per-bar signals from all active strategies, applies
regime weights, and produces a single consolidated Signal which then flows
through risk → router → executor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from src.exchanges.base import OrderSide

if TYPE_CHECKING:
    from src.data.feature_store import FeatureKey, FeatureStore


class SignalAction(StrEnum):
    OPEN = "open"      # open a new position
    CLOSE = "close"    # flatten an existing position
    NONE = "none"      # no action


@dataclass(slots=True)
class Signal:
    """A single-leg trading signal produced by a strategy."""
    strategy: str
    symbol: str
    action: SignalAction
    side: OrderSide           # for OPEN: direction of the new position
    confidence: float         # [0, 1]
    ts: datetime
    timeframe: str | None = None
    preferred_exchange: str | None = None  # route hint (None = let router choose)
    suggested_sl: float | None = None
    suggested_tp: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PairSignal:
    """
    A delta-neutral pair signal (used by funding arbitrage).

    Both legs are executed together; if either fails, the other must be
    unwound. Orchestrator handles this coordination.
    """
    strategy: str
    symbol: str
    ts: datetime
    long_exchange: str        # place a long on this venue
    short_exchange: str       # place a short on this venue
    confidence: float
    meta: dict[str, Any] = field(default_factory=dict)


class Strategy(ABC):
    """
    Strategy base class.

    Contract:
        - `name` identifies the strategy in logs and config.
        - `evaluate(symbol, store, timeframe)` returns zero or more Signals.
        - The base confidence level is read from config; strategies may scale
          it by indicator strength but should not exceed 1.0.

    Strategies are pure — calling `evaluate` multiple times on the same data
    must return equivalent signals. State belongs in the feature store, not in
    the strategy instance.
    """

    name: str = "abstract"

    def __init__(self, base_confidence: float = 0.5) -> None:
        self._base_confidence = base_confidence

    @abstractmethod
    def evaluate(
        self,
        symbol: str,
        store: "FeatureStore",
        exchange: str,
        ts: datetime,
    ) -> list[Signal]:
        """Return a (possibly empty) list of signals for one symbol at time `ts`."""

    @property
    def base_confidence(self) -> float:
        return self._base_confidence

    @staticmethod
    def _clip(conf: float) -> float:
        return max(0.0, min(1.0, conf))
