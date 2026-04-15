"""
Order router — chooses WHICH exchange to place an order on.

The ensemble produces a symbol-level Signal. Most strategies target a single
venue; the funding arb strategy produces a *pair* of legs (one per venue).

Routing preferences (in priority order):
    1. If the signal already names an exchange, honour it.
    2. For non-arb signals, prefer the venue currently RECEIVING funding for a
       long signal (or PAYING funding for a short), to get a small edge.
    3. Break ties by exposure balance — place on the less-utilized venue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.exchanges.base import OrderSide
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.exchanges.base import ExchangeConnector
    from src.portfolio.tracker import PortfolioTracker
    from src.strategies.base import Signal

log = get_logger(__name__)


@dataclass(slots=True)
class RoutedLeg:
    exchange_name: str
    connector: "ExchangeConnector"


class OrderRouter:
    def __init__(
        self,
        connectors: dict[str, "ExchangeConnector"],
        tracker: "PortfolioTracker",
    ) -> None:
        self._connectors = connectors
        self._tracker = tracker
        # Latest known funding rate per (exchange, symbol) — populated by the
        # orchestrator when funding events arrive.
        self._funding: dict[tuple[str, str], float] = {}

    def update_funding(self, exchange: str, symbol: str, rate: float) -> None:
        self._funding[(exchange, symbol)] = rate

    def route(self, signal: "Signal") -> RoutedLeg:
        """
        Decide which exchange to route a single-leg signal to.

        This is NOT used for funding_arb — that strategy emits a PairSignal
        that the orchestrator splits into two pre-routed legs directly.
        """
        # 1) Honour explicit venue
        if signal.preferred_exchange and signal.preferred_exchange in self._connectors:
            ex = signal.preferred_exchange
            return RoutedLeg(exchange_name=ex, connector=self._connectors[ex])

        # 2) Funding-rate edge
        candidates = list(self._connectors.items())
        if not candidates:
            raise RuntimeError("no exchanges configured for routing")

        def funding_score(ex_name: str) -> float:
            rate = self._funding.get((ex_name, signal.symbol), 0.0)
            # For a LONG, we PAY funding if rate > 0, RECEIVE if rate < 0.
            # Prefer the venue where we'd receive (or pay least).
            if signal.side == OrderSide.BUY:
                return -rate   # lower rate = better for long
            return rate        # higher rate = better for short (we receive)

        # 3) Exposure balance — prefer the less-utilized venue
        def exposure_score(ex_name: str) -> float:
            return self._tracker.exposure_by_exchange().get(ex_name, 0.0)

        # Composite rank: funding edge first (higher = better), exposure as
        # tiebreaker (lower = better, so negate so it sorts correctly with
        # reverse=True — i.e. the venue with less exposure scores "higher").
        ranked = sorted(
            candidates,
            key=lambda kv: (funding_score(kv[0]), -exposure_score(kv[0])),
            reverse=True,
        )
        best_name, best_conn = ranked[0]
        log.debug(
            "router.selected",
            signal_symbol=signal.symbol,
            signal_side=signal.side.value,
            selected=best_name,
            candidates=[n for n, _ in candidates],
        )
        return RoutedLeg(exchange_name=best_name, connector=best_conn)
