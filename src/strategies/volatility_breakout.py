"""
Volatility breakout strategy.

Detects consolidation (range compression) by measuring the ratio of the
Donchian channel width to ATR. When that ratio stays below
`squeeze_atr_ratio_max` for `squeeze_bars` consecutive bars, the market is in
a squeeze. The next bar that breaks out of the channel with volume above
`volume_multiple * average_volume_20` triggers a signal in the breakout
direction.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.data.feature_store import FeatureKey, FeatureStore
from src.exchanges.base import OrderSide
from src.strategies.base import Signal, SignalAction, Strategy
from src.utils.indicators import atr, donchian, rolling_volume_ratio
from src.utils.logging import get_logger

log = get_logger(__name__)


class VolatilityBreakoutStrategy(Strategy):
    name = "volatility_breakout"

    def __init__(
        self,
        *,
        timeframe: str,
        donchian_period: int,
        squeeze_atr_ratio_max: float,
        squeeze_bars: int,
        volume_multiple: float,
        base_confidence: float,
    ) -> None:
        super().__init__(base_confidence=base_confidence)
        if volume_multiple < 1.0:
            raise ValueError(f"volume_multiple must be >= 1.0, got {volume_multiple}")
        self._tf = timeframe
        self._dc_period = donchian_period
        self._squeeze_ratio_max = squeeze_atr_ratio_max
        self._squeeze_bars = squeeze_bars
        self._volume_multiple = volume_multiple

    def evaluate(self, symbol: str, store: FeatureStore, exchange: str, ts: datetime) -> list[Signal]:
        df = store.as_df(
            FeatureKey(exchange, symbol, self._tf), min_bars=self._dc_period + self._squeeze_bars + 10
        )
        if df is None:
            return []

        dc = donchian(df, length=self._dc_period)
        if dc is None or dc.empty:
            return []
        upper_col = f"DCU_{self._dc_period}_{self._dc_period}"
        lower_col = f"DCL_{self._dc_period}_{self._dc_period}"
        if upper_col not in dc.columns or lower_col not in dc.columns:
            return []

        a_series = atr(df, length=14)
        if a_series is None or a_series.empty:
            return []

        # range / atr ratio at each bar
        dc_range = (dc[upper_col] - dc[lower_col]).astype(float)
        ratio = dc_range / a_series
        tail_ratio = ratio.tail(self._squeeze_bars + 1)

        # Prior N bars (excluding current) must all be in squeeze; current bar
        # is the potential breakout bar. NaN check on prior only — the current
        # bar's ratio is expected to be large (breakout widens the channel).
        prior = tail_ratio.iloc[:-1]
        if prior.isna().any():
            return []
        if not (prior < self._squeeze_ratio_max).all():
            return []

        # Volume confirmation on the current bar.
        vol_ratio = rolling_volume_ratio(df, length=20).iloc[-1]
        if vol_ratio is None or np.isnan(vol_ratio) or vol_ratio < self._volume_multiple:
            return []

        # Use the LAST CLOSED bar (iloc[-2]) for the breakout check.  This
        # strategy is evaluated on every 15m close but operates on the 1h TF,
        # so iloc[-1] is the FORMING 1h bar.  Checking it would re-trigger the
        # same signal 4 times per 1h bar — always use the most-recently CLOSED bar.
        if len(df) < 3:
            return []
        last_close = float(df["close"].iloc[-2])
        # Channel at the bar before last_close — that's the level it breaks.
        upper_prev = float(dc[upper_col].iloc[-3])
        lower_prev = float(dc[lower_col].iloc[-3])
        if np.isnan(upper_prev) or np.isnan(lower_prev):
            return []

        strength = max(0.0, min(1.0, (float(vol_ratio) / self._volume_multiple) - 1.0 + 0.5))

        if last_close > upper_prev:
            conf = self._clip(self._base_confidence * (0.6 + 0.4 * strength))
            return [
                Signal(
                    strategy=self.name,
                    symbol=symbol,
                    action=SignalAction.OPEN,
                    side=OrderSide.BUY,
                    confidence=conf,
                    ts=ts,
                    timeframe=self._tf,
                    meta={
                        "donchian_upper": round(upper_prev, 4),
                        "last_close": round(last_close, 4),
                        "vol_ratio": round(float(vol_ratio), 2),
                        "squeeze_bars": self._squeeze_bars,
                    },
                )
            ]

        if last_close < lower_prev:
            conf = self._clip(self._base_confidence * (0.6 + 0.4 * strength))
            return [
                Signal(
                    strategy=self.name,
                    symbol=symbol,
                    action=SignalAction.OPEN,
                    side=OrderSide.SELL,
                    confidence=conf,
                    ts=ts,
                    timeframe=self._tf,
                    meta={
                        "donchian_lower": round(lower_prev, 4),
                        "last_close": round(last_close, 4),
                        "vol_ratio": round(float(vol_ratio), 2),
                        "squeeze_bars": self._squeeze_bars,
                    },
                )
            ]

        return []
