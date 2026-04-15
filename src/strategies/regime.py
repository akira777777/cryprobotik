"""
Market regime classifier.

Classifies the current market state for a single symbol into one of:
    trend_high_vol | trend_low_vol | range_high_vol | range_low_vol | chop

Inputs:
    - ADX(14) on the base timeframe (15m) for trend strength
    - Realized volatility over N bars for vol level

The ensemble queries this on every bar close to weight strategies:
trend regimes favor momentum + breakout, range regimes favor mean reversion,
chop disables all directional strategies (funding arb is unaffected since it's
market-neutral).
"""

from __future__ import annotations

from collections import deque
from enum import StrEnum
from typing import TYPE_CHECKING

import numpy as np

from src.data.feature_store import FeatureKey, FeatureStore
from src.utils.indicators import adx, realized_vol
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.settings import RegimeConfig

log = get_logger(__name__)


def _hurst(prices, max_lag: int = 20) -> float:
    """Compute the Hurst exponent via the R/S (rescaled range) method.

    Parameters
    ----------
    prices:
        A pandas Series of close prices.
    max_lag:
        Maximum lag window for R/S calculation.

    Returns a value in [0, 1]:
        H > 0.55  → persistent / trending
        H < 0.45  → anti-persistent / mean-reverting
        otherwise → random walk / chop
    """

    if len(prices) < max_lag * 2:
        return 0.5  # not enough data
    lags = range(2, max_lag)
    rs_list = []
    for lag in lags:
        subseries = [prices.iloc[i:i + lag] for i in range(0, len(prices) - lag, lag)]
        if not subseries:
            continue
        rs_vals = []
        for sub in subseries:
            mean = sub.mean()
            devs = (sub - mean).cumsum()
            r = devs.max() - devs.min()
            s = sub.std()
            if s > 0:
                rs_vals.append(r / s)
        if rs_vals:
            rs_list.append((lag, sum(rs_vals) / len(rs_vals)))
    if len(rs_list) < 3:
        return 0.5
    lags_arr = np.log([x[0] for x in rs_list])
    rs_arr = np.log([x[1] for x in rs_list])
    hurst = float(np.polyfit(lags_arr, rs_arr, 1)[0])
    return max(0.0, min(1.0, hurst))


class Regime(StrEnum):
    TREND_HIGH_VOL = "trend_high_vol"
    TREND_LOW_VOL = "trend_low_vol"
    RANGE_HIGH_VOL = "range_high_vol"
    RANGE_LOW_VOL = "range_low_vol"
    CHOP = "chop"


class RegimeClassifier:
    def __init__(self, config: "RegimeConfig", base_timeframe: str = "15m") -> None:
        self._config = config
        self._tf = base_timeframe
        # Raw history per (exchange, symbol) — plain list so we can slice it.
        self._history: dict[tuple[str, str], list[Regime]] = {}
        # Last confirmed (stable) regime per (exchange, symbol).
        # Only changes when n consecutive bars agree on a new value.
        self._current_regime: dict[tuple[str, str], Regime] = {}

    def classify(self, symbol: str, store: FeatureStore, exchange: str) -> Regime:
        """Return the regime for a symbol. Defaults to CHOP if insufficient data.

        Applies true hysteresis: the regime only changes when the last
        ``regime_hysteresis_bars`` raw classifications all agree on the SAME
        NEW value. Between transitions the previously-accepted regime is held.
        """
        raw = self._raw_classify(symbol, store, exchange)

        key = (exchange, symbol)
        history = self._history.setdefault(key, [])
        history.append(raw)

        n = self._config.regime_hysteresis_bars
        # Trim history to 2×n so the list never grows unbounded.
        if len(history) > 2 * n:
            history[:] = history[-(2 * n) :]

        # Only commit a regime change when the last n bars unanimously agree.
        if len(history) >= n:
            recent = history[-n:]
            if len(set(recent)) == 1:
                # All n bars agree — accept this as the new current regime.
                self._current_regime[key] = recent[0]

        # Return the last accepted regime; default to CHOP until first consensus.
        return self._current_regime.get(key, Regime.CHOP)

    def _raw_classify(self, symbol: str, store: FeatureStore, exchange: str) -> Regime:
        """Compute the raw (non-hysteresis) regime from indicators."""
        df = store.as_df(
            FeatureKey(exchange, symbol, self._tf),
            min_bars=max(self._config.adx_period * 3, self._config.vol_window_bars + 5),
        )
        if df is None:
            return Regime.CHOP

        # ADX
        adx_df = adx(df, length=self._config.adx_period)
        if adx_df is None or adx_df.empty:
            return Regime.CHOP
        adx_col = f"ADX_{self._config.adx_period}"
        if adx_col not in adx_df.columns:
            return Regime.CHOP
        a = float(adx_df[adx_col].iloc[-1])
        if np.isnan(a):
            return Regime.CHOP

        # Realized vol
        rv_series = realized_vol(df, window=self._config.vol_window_bars)
        rv = float(rv_series.iloc[-1]) if rv_series is not None and len(rv_series) else float("nan")
        high_vol = not np.isnan(rv) and rv > self._config.vol_high_threshold

        if a >= self._config.adx_trend_threshold:
            regime = Regime.TREND_HIGH_VOL if high_vol else Regime.TREND_LOW_VOL
        elif a <= self._config.adx_range_threshold:
            regime = Regime.RANGE_HIGH_VOL if high_vol else Regime.RANGE_LOW_VOL
        else:
            regime = Regime.CHOP

        # Hurst exponent tiebreaker for chop/range vs trend ambiguity.
        hurst = _hurst(df["close"])
        if regime == Regime.CHOP:
            if hurst > 0.55:
                regime = Regime.TREND_HIGH_VOL if high_vol else Regime.TREND_LOW_VOL
            elif hurst < 0.45:
                regime = Regime.RANGE_HIGH_VOL if high_vol else Regime.RANGE_LOW_VOL
        elif regime in (Regime.RANGE_HIGH_VOL, Regime.RANGE_LOW_VOL):
            if hurst > 0.55:
                regime = Regime.TREND_HIGH_VOL if high_vol else Regime.TREND_LOW_VOL
        elif regime in (Regime.TREND_HIGH_VOL, Regime.TREND_LOW_VOL):
            if hurst < 0.42:
                regime = Regime.RANGE_HIGH_VOL if high_vol else Regime.RANGE_LOW_VOL

        return regime

    def weight_for(self, regime: Regime, strategy: str) -> float:
        """Look up the strategy weight for a regime from config."""
        weights = self._config.weights.get(regime.value, {})
        return float(weights.get(strategy, 0.0))
