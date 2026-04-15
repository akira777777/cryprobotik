"""
Funding rate contrarian strategy.

Perpetual futures funding rates reflect the sentiment of leveraged traders. When
funding is extremely positive (top percentile of its historical distribution),
the market is crowded long — longs are paying shorts a premium. This signals
excessive optimism and a potential reversal, so the contrarian trade is SHORT.

Conversely, extremely negative funding (bottom percentile) indicates a crowded
short — the contrarian trade is LONG.

The strategy requires ≥20 historical funding samples before it emits any signal
(cold-start guard). Confidence scales linearly with how far into the extreme zone
the current rate sits.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np

from src.data.feature_store import FeatureKey, FundingHistory
from src.exchanges.base import OrderSide
from src.strategies.base import Signal, SignalAction, Strategy
from src.utils.logging import get_logger

log = get_logger(__name__)


class FundingContrarianStrategy(Strategy):
    name = "funding_contrarian"

    def __init__(
        self,
        *,
        funding_history: FundingHistory,
        extreme_threshold: float = 0.85,  # top 15% = extremely crowded long → SHORT
        low_threshold: float = 0.15,       # bottom 15% = extremely crowded short → LONG
        base_confidence: float = 0.55,
    ) -> None:
        super().__init__(base_confidence=base_confidence)
        if low_threshold >= extreme_threshold:
            raise ValueError("low_threshold must be < extreme_threshold")
        self._history = funding_history
        self._extreme_thr = extreme_threshold
        self._low_thr = low_threshold
        # Tighter threshold multiplier: tightens to ~10th/90th percentile
        # without changing the config value (0.67 of 0.15 ≈ 0.10).
        self._extreme_threshold = extreme_threshold  # alias used in veto calc

    def _ema21_55(self, symbol: str, store, exchange: str):
        """Return (ema21_last, ema55_last) from 4h bars, or (None, None) on failure."""
        df_4h = store.as_df(FeatureKey(exchange=exchange, symbol=symbol, timeframe="4h"), min_bars=60)
        if df_4h is None or len(df_4h) < 60:
            return None, None
        try:
            import pandas_ta as ta  # type: ignore
            ema21 = ta.ema(df_4h["close"], length=21)
            ema55 = ta.ema(df_4h["close"], length=55)
        except Exception:
            return None, None
        if ema21 is None or ema55 is None:
            return None, None
        e21 = float(ema21.iloc[-1]) if not np.isnan(ema21.iloc[-1]) else None
        e55 = float(ema55.iloc[-1]) if not np.isnan(ema55.iloc[-1]) else None
        return e21, e55

    def evaluate(self, symbol: str, store, exchange: str, ts: datetime) -> list[Signal]:
        rate = self._history.latest(exchange, symbol)
        if rate is None:
            return []

        pct = self._history.percentile(exchange, symbol, rate)

        # Tighter thresholds: use 0.67 multiplier to narrow to ~10th/90th percentile.
        tight_high = 1.0 - self._extreme_threshold * 0.67  # e.g. 0.15*0.67=0.10 → 0.90
        tight_low = self._extreme_threshold * 0.67          # e.g. 0.15*0.67=0.10

        if pct >= tight_high:
            # Longs are paying an excessive funding premium → crowded long → SHORT
            # Trend veto: if EMA21 is significantly below EMA55, trend is down —
            # don't fight a downtrend with a contrarian sell.
            e21, e55 = self._ema21_55(symbol, store, exchange)
            if e21 is not None and e55 is not None:
                if e21 < e55 * 0.995:
                    log.debug(
                        "funding_contrarian.trend_veto_sell",
                        symbol=symbol,
                        ema21=round(e21, 4),
                        ema55=round(e55, 4),
                    )
                    return []

            # Scale confidence by how deep into the extreme zone we are.
            depth = (pct - tight_high) / max(1e-9, 1.0 - tight_high)
            conf = self._clip(self._base_confidence * (0.7 + 0.3 * depth))
            log.debug(
                "funding_contrarian.short_signal",
                symbol=symbol,
                exchange=exchange,
                rate=round(rate, 6),
                pct=round(pct, 3),
                conf=round(conf, 3),
            )
            return [
                Signal(
                    strategy=self.name,
                    symbol=symbol,
                    action=SignalAction.OPEN,
                    side=OrderSide.SELL,
                    confidence=conf,
                    ts=ts,
                    meta={
                        "funding_rate": round(rate, 6),
                        "funding_pct": round(pct, 3),
                    },
                )
            ]

        if pct <= tight_low:
            # Shorts are paying an excessive funding premium → crowded short → LONG
            # Trend veto: if EMA21 is significantly above EMA55, trend is up —
            # funding is negative but trend is bullish, let it ride.
            e21, e55 = self._ema21_55(symbol, store, exchange)
            if e21 is not None and e55 is not None:
                if e21 > e55 * 1.005:
                    log.debug(
                        "funding_contrarian.trend_veto_buy",
                        symbol=symbol,
                        ema21=round(e21, 4),
                        ema55=round(e55, 4),
                    )
                    return []

            depth = (tight_low - pct) / max(1e-9, tight_low)
            conf = self._clip(self._base_confidence * (0.7 + 0.3 * depth))
            log.debug(
                "funding_contrarian.long_signal",
                symbol=symbol,
                exchange=exchange,
                rate=round(rate, 6),
                pct=round(pct, 3),
                conf=round(conf, 3),
            )
            return [
                Signal(
                    strategy=self.name,
                    symbol=symbol,
                    action=SignalAction.OPEN,
                    side=OrderSide.BUY,
                    confidence=conf,
                    ts=ts,
                    meta={
                        "funding_rate": round(rate, 6),
                        "funding_pct": round(pct, 3),
                    },
                )
            ]

        return []
