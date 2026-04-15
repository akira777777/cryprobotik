"""
Strategy ensemble.

Owns all enabled strategies and combines their signals using regime-dependent
weights. For each (symbol, bar) the ensemble:

1. Determines the current regime.
2. Runs every enabled strategy's `evaluate()`.
3. Multiplies each signal's confidence by its regime weight.
4. Aggregates opposing signals into a net direction: long votes − short votes.
5. If the net absolute value exceeds a minimum threshold, emits a single
   consolidated Signal with side = sign of the net vote and confidence = the
   absolute net weighted sum (capped at 1.0).

Funding arb is NOT routed through the ensemble — it produces PairSignals
which the orchestrator handles directly.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.exchanges.base import OrderSide
from src.strategies.base import Signal, SignalAction, Strategy
from src.strategies.regime import Regime, RegimeClassifier
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data.feature_store import FeatureStore

log = get_logger(__name__)


class Ensemble:
    def __init__(
        self,
        strategies: list[Strategy],
        regime_classifier: RegimeClassifier,
        min_net_vote: float = 0.20,
    ) -> None:
        self._strategies = strategies
        self._regime = regime_classifier
        self._min_net_vote = min_net_vote

    @property
    def strategies(self) -> list[Strategy]:
        return list(self._strategies)

    def evaluate_symbol(
        self,
        symbol: str,
        store: "FeatureStore",
        exchange: str,
        ts: datetime,
    ) -> tuple[Signal | None, Regime, list[Signal]]:
        """
        Run all directional strategies for one symbol and return:
            - consolidated Signal or None
            - current regime
            - full list of raw per-strategy signals (for logging / analytics)
        """
        regime = self._regime.classify(symbol, store, exchange)
        raw_signals: list[Signal] = []
        weighted_long = 0.0
        weighted_short = 0.0
        contributors: list[str] = []

        for strat in self._strategies:
            # Skip strategies that emit via non-bar paths (funding_arb).
            if strat.name == "funding_arb":
                continue
            try:
                sigs = strat.evaluate(symbol, store, exchange, ts)
            except Exception as e:
                log.error("ensemble.strategy_error", strategy=strat.name,
                          symbol=symbol, error=str(e), exc_info=True)
                continue
            for s in sigs:
                if s.action != SignalAction.OPEN:
                    continue
                weight = self._regime.weight_for(regime, strat.name)
                contribution = s.confidence * weight
                if contribution <= 0:
                    continue
                if s.side == OrderSide.BUY:
                    weighted_long += contribution
                else:
                    weighted_short += contribution
                contributors.append(f"{strat.name}({s.side.value},{contribution:.2f})")
                raw_signals.append(s)

        net = weighted_long - weighted_short
        if abs(net) < self._min_net_vote:
            return (None, regime, raw_signals)

        side = OrderSide.BUY if net > 0 else OrderSide.SELL
        consolidated = Signal(
            strategy="ensemble",
            symbol=symbol,
            action=SignalAction.OPEN,
            side=side,
            confidence=min(1.0, abs(net)),
            ts=ts,
            timeframe=None,
            meta={
                "regime": regime.value,
                "net_vote": round(net, 3),
                "weighted_long": round(weighted_long, 3),
                "weighted_short": round(weighted_short, 3),
                "contributors": contributors,
            },
        )
        log.info(
            "ensemble.consolidated",
            symbol=symbol, side=side.value, confidence=consolidated.confidence,
            regime=regime.value, contributors=contributors,
        )
        return (consolidated, regime, raw_signals)
