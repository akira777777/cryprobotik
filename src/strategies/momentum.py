"""
Multi-timeframe momentum strategy.

Long setup (all must be true on the latest closed bar):
    - 15m: EMA(fast) > EMA(mid) > EMA(slow)   (stacked uptrend)
    - 1h : RSI(14) > rsi_long_threshold       (strength confirmation)
    - 4h : MACD histogram > 0                 (higher-TF alignment)

Short setup is the mirror image.

The signal strength is scaled by how strong the MACD histogram is relative to
a rolling reference, so the ensemble can weight strong vs borderline signals.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.data.feature_store import FeatureKey, FeatureStore
from src.exchanges.base import OrderSide
from src.strategies.base import Signal, SignalAction, Strategy
from src.utils.indicators import ema, macd, rsi
from src.utils.logging import get_logger

log = get_logger(__name__)


class MomentumStrategy(Strategy):
    name = "momentum"

    def __init__(
        self,
        *,
        timeframes: list[str],
        ema_fast: int,
        ema_mid: int,
        ema_slow: int,
        rsi_period: int,
        rsi_long_threshold: float,
        rsi_short_threshold: float,
        macd_fast: int,
        macd_slow: int,
        macd_signal: int,
        base_confidence: float,
        volume_multiplier: float = 0.0,
    ) -> None:
        super().__init__(base_confidence=base_confidence)
        if len(timeframes) != 3:
            raise ValueError("momentum strategy requires exactly 3 timeframes (low, mid, high)")
        self._tf_low, self._tf_mid, self._tf_high = timeframes
        self._ema_fast = ema_fast
        self._ema_mid = ema_mid
        self._ema_slow = ema_slow
        self._rsi_period = rsi_period
        self._rsi_long_thr = rsi_long_threshold
        self._rsi_short_thr = rsi_short_threshold
        self._macd_fast = macd_fast
        self._macd_slow = macd_slow
        self._macd_signal = macd_signal
        self._volume_multiplier = volume_multiplier

    def evaluate(self, symbol: str, store: FeatureStore, exchange: str, ts: datetime) -> list[Signal]:
        low_df = store.as_df(FeatureKey(exchange, symbol, self._tf_low), min_bars=self._ema_slow + 5)
        mid_df = store.as_df(FeatureKey(exchange, symbol, self._tf_mid), min_bars=self._rsi_period + 5)
        high_df = store.as_df(
            FeatureKey(exchange, symbol, self._tf_high),
            min_bars=self._macd_slow + self._macd_signal + 5,
        )
        if low_df is None or mid_df is None or high_df is None:
            return []
        # Need at least 2 bars on higher TFs so we can read the last CLOSED bar.
        if len(mid_df) < 2 or len(high_df) < 2:
            return []

        # ── low TF: EMA stack (last closed 15m bar = iloc[-1]) ──
        ema_f = ema(low_df, self._ema_fast).iloc[-1]
        ema_m = ema(low_df, self._ema_mid).iloc[-1]
        ema_s = ema(low_df, self._ema_slow).iloc[-1]
        last_close = float(low_df["close"].iloc[-1])

        if any(v is None or np.isnan(v) for v in (ema_f, ema_m, ema_s)):
            return []
        long_stack = ema_f > ema_m > ema_s and last_close > ema_f
        short_stack = ema_f < ema_m < ema_s and last_close < ema_f

        # ── volume confirmation (optional) ──
        # Use iloc[-21:-1] to exclude the signal bar from the baseline mean.
        if self._volume_multiplier > 0 and "volume" in low_df.columns:
            if len(low_df) < 22:
                return []
            vol_mean = float(low_df["volume"].iloc[-21:-1].mean())
            last_vol = float(low_df["volume"].iloc[-1])
            if vol_mean > 0 and last_vol < self._volume_multiplier * vol_mean:
                return []

        # ── mid TF: RSI (use iloc[-2] = last CLOSED 1h bar, not the forming one) ──
        rsi_series = rsi(mid_df, length=self._rsi_period)
        r = rsi_series.iloc[-2]  # last CLOSED higher-TF bar
        if r is None or np.isnan(r):
            return []
        # Slope: RSI must be moving in the signal direction over the last 3 bars.
        r_prev = float(rsi_series.iloc[-5]) if len(rsi_series) >= 5 else float(r)

        # ── high TF: MACD histogram (iloc[-2] = last CLOSED 4h bar) ──
        macd_df = macd(high_df, fast=self._macd_fast, slow=self._macd_slow, signal=self._macd_signal)
        if macd_df is None or macd_df.empty:
            return []
        hist_col = f"MACDh_{self._macd_fast}_{self._macd_slow}_{self._macd_signal}"
        if hist_col not in macd_df.columns:
            return []
        hist = float(macd_df[hist_col].iloc[-2])  # last CLOSED 4h bar
        if np.isnan(hist):
            return []

        # MACD histogram "strength" relative to its recent stdev — used to
        # scale confidence. Rolling 50 bars of histogram stdev.
        hist_std = float(macd_df[hist_col].tail(50).std())
        if hist_std and hist_std > 0:
            strength = min(1.0, abs(hist) / (2 * hist_std))
        else:
            strength = 0.5

        # ── combine ──
        # Long: above threshold, rising, not overbought (< 75 avoids exhaustion entries).
        if long_stack and self._rsi_long_thr < r < 75.0 and r > r_prev and hist > 0:
            conf = self._clip(self._base_confidence * (0.7 + 0.3 * strength))
            return [
                Signal(
                    strategy=self.name,
                    symbol=symbol,
                    action=SignalAction.OPEN,
                    side=OrderSide.BUY,
                    confidence=conf,
                    ts=ts,
                    timeframe=self._tf_low,
                    meta={
                        "rsi": round(float(r), 2),
                        "macd_hist": round(hist, 6),
                        "strength": round(strength, 3),
                    },
                )
            ]

        # Short: below threshold, falling, not oversold (> 25 avoids exhaustion entries).
        if short_stack and 25.0 < r < self._rsi_short_thr and r < r_prev and hist < 0:
            conf = self._clip(self._base_confidence * (0.7 + 0.3 * strength))
            return [
                Signal(
                    strategy=self.name,
                    symbol=symbol,
                    action=SignalAction.OPEN,
                    side=OrderSide.SELL,
                    confidence=conf,
                    ts=ts,
                    timeframe=self._tf_low,
                    meta={
                        "rsi": round(float(r), 2),
                        "macd_hist": round(hist, 6),
                        "strength": round(strength, 3),
                    },
                )
            ]

        return []
