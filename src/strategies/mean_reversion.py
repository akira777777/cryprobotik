"""
Bollinger Band mean reversion (RSI(2) + ADX filter).

Entry rules:
    - ADX(14) < adx_max    → only trade in ranging markets
    - Price touches or closes beyond a BB band
    - RSI(2) confirms the extreme (< rsi_long_threshold for longs)

Exit philosophy: the risk manager always attaches SL/TP, so we don't need a
separate exit signal from the strategy. The TP at 1.5x ATR will typically be
the BB midline.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.data.feature_store import FeatureKey, FeatureStore
from src.exchanges.base import OrderSide
from src.strategies.base import Signal, SignalAction, Strategy
from src.utils.indicators import adx, bollinger, rsi
from src.utils.logging import get_logger

log = get_logger(__name__)


class MeanReversionStrategy(Strategy):
    name = "mean_reversion"

    def __init__(
        self,
        *,
        timeframe: str,
        bb_period: int,
        bb_std: float,
        rsi_period: int,
        rsi_long_threshold: float,
        rsi_short_threshold: float,
        adx_max: float,
        base_confidence: float,
    ) -> None:
        super().__init__(base_confidence=base_confidence)
        self._tf = timeframe
        self._bb_period = bb_period
        self._bb_std = bb_std
        self._rsi_period = rsi_period
        self._rsi_long_thr = rsi_long_threshold
        self._rsi_short_thr = rsi_short_threshold
        self._adx_max = adx_max

    def evaluate(
        self, symbol: str, store: FeatureStore, exchange: str, ts: datetime
    ) -> list[Signal]:
        df = store.as_df(FeatureKey(exchange, symbol, self._tf),
                         min_bars=self._bb_period + 10)
        if df is None:
            return []

        bb = bollinger(df, length=self._bb_period, std=self._bb_std)
        if bb is None or bb.empty:
            return []
        lower_col = f"BBL_{self._bb_period}_{self._bb_std}"
        upper_col = f"BBU_{self._bb_period}_{self._bb_std}"
        if lower_col not in bb.columns or upper_col not in bb.columns:
            return []

        last_close = float(df["close"].iloc[-1])
        last_low = float(df["low"].iloc[-1])
        last_high = float(df["high"].iloc[-1])
        lower = float(bb[lower_col].iloc[-1])
        upper = float(bb[upper_col].iloc[-1])
        if any(np.isnan(x) for x in (lower, upper)):
            return []

        # ADX regime filter
        adx_df = adx(df, length=14)
        if adx_df is None or adx_df.empty:
            return []
        adx_col = "ADX_14"
        if adx_col not in adx_df.columns:
            return []
        a = float(adx_df[adx_col].iloc[-1])
        if np.isnan(a) or a >= self._adx_max:
            return []

        # RSI extreme confirmation
        r = float(rsi(df, length=self._rsi_period).iloc[-1])
        if np.isnan(r):
            return []

        band_width = upper - lower

        # Long on lower-band touch + oversold RSI
        if last_low <= lower and r < self._rsi_long_thr:
            # Falling knife veto: suppress LONG if 4h EMA21 is trending down.
            # ADX slope filter: only enter if ADX is flat or declining on 4h.
            df_4h = store.as_df(FeatureKey(exchange=exchange, symbol=symbol, timeframe="4h"), min_bars=10)
            if df_4h is not None and len(df_4h) >= 5:
                import pandas_ta as ta  # type: ignore
                ema21 = ta.ema(df_4h["close"], length=21)
                if ema21 is not None and len(ema21) >= 5 and not np.isnan(ema21.iloc[-1]) and not np.isnan(ema21.iloc[-5]):
                    if float(ema21.iloc[-1]) < float(ema21.iloc[-5]):
                        log.debug(
                            "mean_reversion.falling_knife_veto",
                            symbol=symbol,
                            ema21_now=round(float(ema21.iloc[-1]), 4),
                            ema21_prev=round(float(ema21.iloc[-5]), 4),
                        )
                        return []
                adx_4h_df = adx(df_4h, length=14)
                if adx_4h_df is not None and "ADX_14" in adx_4h_df.columns and len(adx_4h_df) >= 3:
                    adx_now = float(adx_4h_df["ADX_14"].iloc[-1])
                    adx_prev = float(adx_4h_df["ADX_14"].iloc[-3])
                    if not np.isnan(adx_now) and not np.isnan(adx_prev):
                        if adx_now > adx_prev + 1.0:
                            log.debug(
                                "mean_reversion.adx_slope_veto_long",
                                symbol=symbol,
                                adx_now=round(adx_now, 2),
                                adx_prev=round(adx_prev, 2),
                            )
                            return []

            # Confidence scales with how deep into the band we are, normalized
            # by band width (ATR-proportional) so it's consistent across assets.
            if band_width > 0:
                depth = min(1.0, (lower - last_low) / band_width + 0.2)
            else:
                depth = 0.2
            conf = self._clip(self._base_confidence * (0.6 + 0.4 * depth))
            return [Signal(
                strategy=self.name,
                symbol=symbol,
                action=SignalAction.OPEN,
                side=OrderSide.BUY,
                confidence=conf,
                ts=ts,
                timeframe=self._tf,
                meta={
                    "rsi": round(r, 2),
                    "adx": round(a, 2),
                    "bb_lower": round(lower, 4),
                    "last_low": round(last_low, 4),
                },
            )]

        # Short on upper-band touch + overbought RSI
        if last_high >= upper and r > self._rsi_short_thr:
            # ADX slope filter for shorts: only enter if ADX is flat or declining.
            df_4h_short = store.as_df(FeatureKey(exchange=exchange, symbol=symbol, timeframe="4h"), min_bars=10)
            if df_4h_short is not None and len(df_4h_short) >= 3:
                adx_4h_short = adx(df_4h_short, length=14)
                if adx_4h_short is not None and "ADX_14" in adx_4h_short.columns and len(adx_4h_short) >= 3:
                    adx_now_s = float(adx_4h_short["ADX_14"].iloc[-1])
                    adx_prev_s = float(adx_4h_short["ADX_14"].iloc[-3])
                    if not np.isnan(adx_now_s) and not np.isnan(adx_prev_s):
                        if adx_now_s > adx_prev_s + 1.0:
                            log.debug(
                                "mean_reversion.adx_slope_veto_short",
                                symbol=symbol,
                                adx_now=round(adx_now_s, 2),
                                adx_prev=round(adx_prev_s, 2),
                            )
                            return []

            if band_width > 0:
                depth = min(1.0, (last_high - upper) / band_width + 0.2)
            else:
                depth = 0.2
            conf = self._clip(self._base_confidence * (0.6 + 0.4 * depth))
            return [Signal(
                strategy=self.name,
                symbol=symbol,
                action=SignalAction.OPEN,
                side=OrderSide.SELL,
                confidence=conf,
                ts=ts,
                timeframe=self._tf,
                meta={
                    "rsi": round(r, 2),
                    "adx": round(a, 2),
                    "bb_upper": round(upper, 4),
                    "last_high": round(last_high, 4),
                },
            )]

        return []
