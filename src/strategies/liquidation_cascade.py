"""
Liquidation Cascade Strategy.

Thesis: when open interest drops sharply (>3% in one poll interval) while
price moves significantly (|bar move| > 1.5×ATR), forced liquidations are
occurring. The cascade typically overshoots fair value, creating a contrarian
mean-reversion opportunity.

Long setup:  OI drops >3% AND price fell >1.5×ATR  → BUY (long liq cascade)
Short setup: OI drops >3% AND price rose >1.5×ATR  → SELL (short liq cascade)

Confidence scales with OI drop magnitude.

OI data comes from the OIStore (REST-polled every 5 min by the orchestrator).
The signal fires once per closed bar; the OI poll lag is accepted — a cascade
large enough to matter will be visible even with a few minutes of lag.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from src.data.feature_store import FeatureKey
from src.exchanges.base import OrderSide
from src.strategies.base import Signal, SignalAction, Strategy
from src.utils.indicators import atr
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.data.feature_store import FeatureStore, OIStore

log = get_logger(__name__)


class LiquidationCascadeStrategy(Strategy):
    """
    Contrarian strategy that fades forced-liquidation moves.

    Parameters
    ----------
    oi_store:
        OIStore instance shared with the orchestrator.
    timeframe:
        Bar timeframe to read from FeatureStore (default "15m").
    oi_roc_threshold:
        Minimum OI rate-of-change to trigger (negative, e.g. -0.03 = −3%).
    atr_period:
        ATR period for price-move filter.
    atr_multiplier:
        |price_move| must exceed atr_multiplier × ATR to qualify.
    base_confidence:
        Minimum signal confidence (scaled up with OI drop magnitude).
    """

    name = "liquidation_cascade"

    def __init__(
        self,
        *,
        oi_store: "OIStore",
        timeframe: str = "15m",
        oi_roc_threshold: float = -0.03,
        atr_period: int = 14,
        atr_multiplier: float = 1.5,
        base_confidence: float = 0.60,
    ) -> None:
        super().__init__(base_confidence=base_confidence)
        self._oi_store = oi_store
        self._timeframe = timeframe
        self._oi_roc_threshold = oi_roc_threshold  # must be ≤ 0
        self._atr_period = atr_period
        self._atr_multiplier = atr_multiplier

    def evaluate(
        self,
        symbol: str,
        store: "FeatureStore",
        exchange: str,
        ts: datetime,
    ) -> list[Signal]:
        df = store.as_df(
            FeatureKey(exchange, symbol, self._timeframe),
            min_bars=self._atr_period + 2,
        )
        if df is None or len(df) < self._atr_period + 2:
            return []

        # ── OI condition ─────────────────────────────────────────────────────
        oi_roc = self._oi_store.oi_roc(exchange, symbol, periods=1)
        if oi_roc > self._oi_roc_threshold:
            # OI didn't drop enough — no cascade
            return []

        # ── Price-move condition ──────────────────────────────────────────────
        atr_series = atr(df, self._atr_period)
        atr_val = float(atr_series.iloc[-1])
        if atr_val <= 0:
            return []

        close = df["close"]
        price_move = abs(float(close.iloc[-1]) - float(close.iloc[-2]))
        if price_move < self._atr_multiplier * atr_val:
            # Move is within normal noise — not a cascade bar
            return []

        # ── Direction: contrarian to the move ────────────────────────────────
        price_fell = float(close.iloc[-1]) < float(close.iloc[-2])
        side = OrderSide.BUY if price_fell else OrderSide.SELL

        # ── Confidence: scale with |OI drop| beyond the threshold ────────────
        # At threshold (|oi_roc| = |threshold|) → base_confidence.
        # At 3× threshold → 1.0.  Linear interpolation.
        oi_drop_mag = abs(oi_roc)
        threshold_mag = abs(self._oi_roc_threshold)
        boost_frac = min(1.0, (oi_drop_mag - threshold_mag) / (2.0 * threshold_mag))
        confidence = self._clip(
            self._base_confidence + boost_frac * (1.0 - self._base_confidence)
        )

        log.debug(
            "liquidation_cascade.signal",
            symbol=symbol,
            side=side.value,
            oi_roc=round(oi_roc, 5),
            price_move=round(price_move, 6),
            atr=round(atr_val, 6),
            confidence=round(confidence, 3),
        )

        return [
            Signal(
                strategy=self.name,
                symbol=symbol,
                action=SignalAction.OPEN,
                side=side,
                confidence=confidence,
                ts=ts,
                timeframe=self._timeframe,
                meta={
                    "oi_roc": round(oi_roc, 5),
                    "price_move": round(price_move, 6),
                    "atr": round(atr_val, 6),
                    "atr_ratio": round(price_move / atr_val, 2),
                },
            )
        ]
