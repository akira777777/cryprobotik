"""
VWAP Mean-Reversion Strategy.

VWAP (session-reset at UTC midnight) acts as the institutional benchmark.
Long setup:
    - 15m: EMA(50) slope positive (trend is up)
    - 15m: price has pulled back to within vwap_band_pct of VWAP from above
    - 15m: previous bar was above VWAP, current bar close is at or below VWAP + band

Short setup is the mirror.

Signal confidence is boosted when the pre-pullback distance from VWAP is large
(price overshot, more mean-reversion energy).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.data.feature_store import FeatureKey, FeatureStore
from src.exchanges.base import OrderSide
from src.strategies.base import Signal, SignalAction, Strategy
from src.utils.indicators import ema
from src.utils.logging import get_logger

log = get_logger(__name__)


class VWAPStrategy(Strategy):
    name = "vwap"

    def __init__(
        self,
        *,
        timeframe: str = "15m",
        ema_period: int = 50,
        vwap_band_pct: float = 0.001,   # 0.1% band around VWAP for touch detection
        base_confidence: float = 0.60,
    ) -> None:
        super().__init__(base_confidence=base_confidence)
        self._timeframe = timeframe
        self._ema_period = ema_period
        self._vwap_band_pct = vwap_band_pct

    def evaluate(self, symbol: str, store: FeatureStore, exchange: str, ts: datetime) -> list[Signal]:
        df = store.as_df(
            FeatureKey(exchange, symbol, self._timeframe),
            min_bars=self._ema_period + 10,
        )
        if df is None or len(df) < self._ema_period + 5:
            return []

        # Calculate session VWAP (reset at UTC midnight)
        df = df.copy()
        typical = (df["high"] + df["low"] + df["close"]) / 3.0
        df["tp_vol"] = typical * df["volume"]

        # Cumulative VWAP within the same UTC day
        vwap_vals = []
        cum_tp_vol = 0.0
        cum_vol = 0.0
        prev_day = None
        for i in range(len(df)):
            row_ts = df.index[i]
            try:
                day = row_ts.date()
            except Exception:
                day = i  # fallback
            if day != prev_day:
                cum_tp_vol = 0.0
                cum_vol = 0.0
                prev_day = day
            cum_tp_vol += float(df["tp_vol"].iloc[i])
            cum_vol += float(df["volume"].iloc[i])
            vwap_vals.append(cum_tp_vol / cum_vol if cum_vol > 0 else float(df["close"].iloc[i]))

        df["vwap"] = vwap_vals

        # EMA for trend direction
        ema_series = ema(df, self._ema_period)
        if ema_series is None or len(ema_series) < 3:
            return []

        vwap_now = float(df["vwap"].iloc[-1])
        vwap_prev = float(df["vwap"].iloc[-2])
        close_now = float(df["close"].iloc[-1])
        close_prev = float(df["close"].iloc[-2])
        ema_now = float(ema_series.iloc[-1])
        ema_prev = float(ema_series.iloc[-3])  # 3-bar slope

        if any(np.isnan(v) for v in (vwap_now, close_now, ema_now, ema_prev)):
            return []

        band = vwap_now * self._vwap_band_pct
        trend_up = ema_now > ema_prev
        trend_down = ema_now < ema_prev

        # Long: price was above VWAP last bar, pulled back into VWAP band this bar
        long_touch = (
            close_prev > vwap_prev                           # was above VWAP
            and close_now <= vwap_now + band                  # touched or crossed VWAP
            and close_now >= vwap_now - band                  # not crashed through it
        )
        # Short: price was below VWAP last bar, bounced up into VWAP band
        short_touch = (
            close_prev < vwap_prev
            and close_now >= vwap_now - band
            and close_now <= vwap_now + band
        )

        if long_touch and trend_up:
            # Distance boosts confidence: how far above VWAP was price before pullback
            prior_high = float(df["high"].iloc[-5:-1].max()) if len(df) >= 5 else close_prev
            distance_pct = max(0.0, (prior_high - vwap_now) / vwap_now)
            conf = self._clip(self._base_confidence * (1.0 + min(1.0, distance_pct * 20)))
            return [Signal(
                strategy=self.name, symbol=symbol,
                action=SignalAction.OPEN, side=OrderSide.BUY,
                confidence=conf, ts=ts, timeframe=self._timeframe,
                meta={
                    "vwap": round(vwap_now, 6),
                    "close": round(close_now, 6),
                    "distance_pct": round(distance_pct, 4),
                },
            )]

        if short_touch and trend_down:
            prior_low = float(df["low"].iloc[-5:-1].min()) if len(df) >= 5 else close_prev
            distance_pct = max(0.0, (vwap_now - prior_low) / vwap_now)
            conf = self._clip(self._base_confidence * (1.0 + min(1.0, distance_pct * 20)))
            return [Signal(
                strategy=self.name, symbol=symbol,
                action=SignalAction.OPEN, side=OrderSide.SELL,
                confidence=conf, ts=ts, timeframe=self._timeframe,
                meta={
                    "vwap": round(vwap_now, 6),
                    "close": round(close_now, 6),
                    "distance_pct": round(distance_pct, 4),
                },
            )]

        return []
