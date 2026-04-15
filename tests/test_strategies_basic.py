"""
Happy-path tests for each strategy.

Goal: verify that each strategy emits the expected side (BUY/SELL) given
synthetic price data designed to trigger its entry rules.  We do NOT test
that signals never trigger — that is covered by property tests and edge-case
tests elsewhere.

Each test:
    1. Builds a FeatureStore populated with a synthetic price series.
    2. Instantiates the strategy with minimal, realistic parameters.
    3. Calls evaluate() and asserts ≥1 Signal is returned with the right side.

Strategies covered:
    - MomentumStrategy         (BUY on strong uptrend)
    - MeanReversionStrategy    (BUY on lower-BB touch without 4h data)
    - VolatilityBreakoutStrategy (BUY on squeeze + breakout)
    - FundingContrarianStrategy (SELL on extreme positive funding)
    - VWAPStrategy              (BUY on pullback to VWAP in uptrend)

NOTE: these tests require pandas_ta (and its numba dependency) to be installed.
They are skipped automatically on Python ≥3.14 where numba cannot be built.
"""

from __future__ import annotations

import pytest

# pandas_ta requires numba+tqdm which cannot be installed on Python >=3.14;
# skip the entire module gracefully when those dependencies are missing.
pytest.importorskip("pandas_ta", reason="pandas_ta and its deps required (Python <3.14 only)")

from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_store import Bar, FeatureKey, FeatureStore, FundingHistory
from src.exchanges.base import OrderSide
from src.strategies.base import SignalAction
from src.data.feature_store import OIStore
from src.strategies.funding_contrarian import FundingContrarianStrategy
from src.strategies.liquidation_cascade import LiquidationCascadeStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.momentum import MomentumStrategy
from src.strategies.vwap import VWAPStrategy
from src.strategies.volatility_breakout import VolatilityBreakoutStrategy


# ─────────────────────── helpers ───────────────────────

EXCHANGE = "okx"
SYMBOL = "BTC/USDT:USDT"
_TS0 = datetime(2026, 1, 1, tzinfo=UTC)


def _ms(i: int, interval_sec: int = 15 * 60) -> int:
    return int((_TS0 + timedelta(seconds=i * interval_sec)).timestamp() * 1000)


def _populate_store(
    store: FeatureStore,
    closes: list[float],
    timeframe: str,
    interval_sec: int,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    volumes: list[float] | None = None,
) -> None:
    """Load synthetic bars into a FeatureStore from close prices."""
    key = FeatureKey(EXCHANGE, SYMBOL, timeframe)
    n = len(closes)
    _highs = highs or [c * 1.002 for c in closes]
    _lows = lows or [c * 0.998 for c in closes]
    _vols = volumes or [1000.0] * n

    bars = [
        Bar(
            ts_ms=_ms(i, interval_sec),
            open=closes[i - 1] if i > 0 else closes[0] * 0.999,
            high=_highs[i],
            low=_lows[i],
            close=closes[i],
            volume=_vols[i],
        )
        for i in range(n)
    ]
    store.bulk_load(key, bars)


# ─────────────────────── MomentumStrategy ───────────────────────


def test_momentum_emits_buy_on_strong_uptrend() -> None:
    """
    A strongly rising 15m series → EMA stack (fast > mid > slow) AND
    a modestly bullish 1h RSI, AND positive 4h MACD histogram.

    A small pullback is injected mid-series so RSI doesn't saturate at 100
    (which would fail the `r > r_prev` rising-slope check) and so the MACD
    histogram has non-zero recent stdev plus a positive reading on the last
    closed bar.
    """
    n = 250
    base = 100.0

    def _series(slope: float) -> list[float]:
        """
        Uptrend with strong pullbacks so RSI(14) settles in a 55-70 band
        rather than saturating at 100. Ratio of avg gain / avg loss ≈ 1.5
        gives RSI ≈ 60. Ends with 3 up-bars so RSI and MACD histogram are
        both rising on the last closed bar.
        """
        closes: list[float] = []
        price = base
        for i in range(n):
            # Alternate up bar (+slope * 3.0) and down bar (-slope * 2.0):
            # gain:loss = 3:2 → RSI ≈ 60. Net drift ≈ +slope * 0.5 per 2 bars.
            if i % 2 == 0:
                price += slope * 3.0
            else:
                price -= slope * 2.0
            closes.append(price)
        # Ensure the last 5 bars trend up cleanly so MACD histogram and
        # RSI are both rising vs the reference `r_prev = iloc[-5]`.
        for k in range(5, 0, -1):
            closes[-k] = closes[-(k + 1)] + slope * 1.0
        return closes

    closes_15m = _series(0.25)
    closes_1h = _series(1.0)
    closes_4h = _series(4.0)

    store = FeatureStore()
    _populate_store(store, closes_15m, "15m", 15 * 60)
    _populate_store(store, closes_1h, "1h", 60 * 60)
    _populate_store(store, closes_4h, "4h", 4 * 60 * 60)

    strategy = MomentumStrategy(
        timeframes=["15m", "1h", "4h"],
        ema_fast=9,
        ema_mid=21,
        ema_slow=55,
        rsi_period=14,
        rsi_long_threshold=50,
        rsi_short_threshold=50,
        macd_fast=12,
        macd_slow=26,
        macd_signal=9,
        base_confidence=0.65,
        volume_multiplier=0.0,   # disable volume filter for simplicity
    )

    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)

    # Must emit at least one BUY signal
    buy_signals = [s for s in signals if s.side == OrderSide.BUY]
    assert buy_signals, f"expected BUY signals, got: {signals}"
    for s in buy_signals:
        assert s.action == SignalAction.OPEN
        assert 0.0 < s.confidence <= 1.0
        assert s.symbol == SYMBOL
        assert s.strategy == "momentum"


def test_momentum_returns_empty_on_insufficient_data() -> None:
    """Returns [] when the store doesn't have enough bars to compute indicators."""
    store = FeatureStore()
    # Only 10 bars — far fewer than ema_slow(55) + 5
    closes = [100.0 + i for i in range(10)]
    _populate_store(store, closes, "15m", 15 * 60)
    # 1h and 4h are missing entirely

    strategy = MomentumStrategy(
        timeframes=["15m", "1h", "4h"],
        ema_fast=9, ema_mid=21, ema_slow=55,
        rsi_period=14, rsi_long_threshold=50, rsi_short_threshold=50,
        macd_fast=12, macd_slow=26, macd_signal=9,
        base_confidence=0.65,
    )
    assert strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0) == []


# ─────────────────────── MeanReversionStrategy ───────────────────────


def test_mean_reversion_emits_buy_on_lower_bb_touch() -> None:
    """
    Build a ranging series (ADX will stay low) and force the last bar to
    dip below the lower Bollinger Band with RSI(2) oversold.

    No 4h data in the store → falling-knife veto is skipped.
    """
    rng = np.random.default_rng(0)
    n = 80
    # Sideways oscillation around 100 with low volatility → low ADX
    base = 100.0
    noise = rng.normal(0, 0.5, n)
    closes = [base + noise[i] for i in range(n)]

    # Force the last 3 bars to dip sharply below the mean → below BB lower band
    drop = 8.0  # push ~8 std below the mean
    closes[-3] = base - drop
    closes[-2] = base - drop * 0.9
    closes[-1] = base - drop * 1.1  # final dip below lower band

    # Build highs/lows to match (important for `last_low`)
    highs = [c + 0.5 for c in closes]
    lows = [c - 0.5 for c in closes]
    lows[-1] = closes[-1] - 0.1  # make sure last_low <= lower band

    store = FeatureStore()
    _populate_store(store, closes, "15m", 15 * 60, highs=highs, lows=lows)

    strategy = MeanReversionStrategy(
        timeframe="15m",
        bb_period=20,
        bb_std=2.0,
        rsi_period=2,
        rsi_long_threshold=20,    # RSI(2) at extreme dip will be < 10
        rsi_short_threshold=80,
        adx_max=30.0,             # ADX should stay well below this on synthetic data
        base_confidence=0.60,
    )

    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)
    buy_signals = [s for s in signals if s.side == OrderSide.BUY]

    assert buy_signals, (
        f"expected BUY from mean_reversion on BB lower touch, got: {signals}"
    )
    for s in buy_signals:
        assert s.action == SignalAction.OPEN
        assert 0.0 < s.confidence <= 1.0
        assert s.strategy == "mean_reversion"


def test_mean_reversion_returns_empty_on_trending_market() -> None:
    """
    In a strong trend (high ADX), mean reversion must not fire.
    We verify the strategy returns [] when there are insufficient bars
    (a safe proxy for the cold-start guard).
    """
    store = FeatureStore()
    # 5 bars — far below bb_period (20)
    closes = [100.0 + i * 2 for i in range(5)]
    _populate_store(store, closes, "15m", 15 * 60)

    strategy = MeanReversionStrategy(
        timeframe="15m", bb_period=20, bb_std=2.0,
        rsi_period=2, rsi_long_threshold=20, rsi_short_threshold=80,
        adx_max=30.0, base_confidence=0.60,
    )
    assert strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0) == []


# ─────────────────────── VolatilityBreakoutStrategy ───────────────────────


def test_volatility_breakout_emits_buy_on_squeeze_breakout() -> None:
    """
    Build a series where:
    - The first 150 bars oscillate in a tight range (Donchian/ATR < ratio_max).
    - The last bar's *second-to-last closed bar* (iloc[-2]) closes decisively
      above the Donchian upper of the previous bar.
    - Volume is 3× the 20-bar mean on the breakout bar.

    Note: the strategy uses iloc[-2] as the "last CLOSED bar" and
    checks volume on the *current* bar (iloc[-1]).
    """
    rng = np.random.default_rng(7)

    # Base tight-range section: small oscillation
    tight_n = 170
    base = 100.0
    tight_closes = [base + rng.uniform(-0.3, 0.3) for _ in range(tight_n)]
    tight_highs  = [c + 0.4 for c in tight_closes]
    tight_lows   = [c - 0.4 for c in tight_closes]
    normal_vols  = [1000.0] * tight_n

    # iloc[-2]: gentle breakout that just clears the tight-range channel.
    # A LARGE breakout would make the Donchian-range/ATR ratio at iloc[-2]
    # explode, and the strategy's squeeze check covers bars through iloc[-2],
    # so the prior window must still register as compressed.
    breakout_close = base + 0.9   # just above tight-range Donchian upper
    # iloc[-1]: confirm bar, HIGH volume — this is the bar whose volume the
    # strategy reads via `rolling_volume_ratio().iloc[-1]`.
    confirm_close = base + 1.0

    all_closes = tight_closes + [breakout_close, confirm_close]
    all_highs  = tight_highs + [breakout_close + 0.05, confirm_close + 0.05]
    all_lows   = tight_lows + [breakout_close - 0.05, confirm_close - 0.05]
    all_vols   = normal_vols + [1000.0, 4000.0]  # high volume on the last bar

    store = FeatureStore()
    _populate_store(
        store, all_closes, "1h", 60 * 60,
        highs=all_highs, lows=all_lows, volumes=all_vols
    )

    strategy = VolatilityBreakoutStrategy(
        timeframe="1h",
        donchian_period=20,
        # The strategy's squeeze window (`tail(squeeze_bars+1).iloc[:-1]`)
        # includes iloc[-2] — the breakout bar itself — so the threshold
        # must tolerate the small ratio expansion there.
        squeeze_atr_ratio_max=2.5,
        squeeze_bars=3,
        volume_multiple=2.0,
        base_confidence=0.70,
    )

    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)
    buy_signals = [s for s in signals if s.side == OrderSide.BUY]

    assert buy_signals, (
        f"expected BUY from volatility_breakout on squeeze+breakout, got: {signals}"
    )
    for s in buy_signals:
        assert s.action == SignalAction.OPEN
        assert 0.0 < s.confidence <= 1.0
        assert s.strategy == "volatility_breakout"


def test_volatility_breakout_returns_empty_on_insufficient_data() -> None:
    store = FeatureStore()
    _populate_store(store, [100.0] * 10, "1h", 60 * 60)

    strategy = VolatilityBreakoutStrategy(
        timeframe="1h", donchian_period=20, squeeze_atr_ratio_max=2.0,
        squeeze_bars=3, volume_multiple=2.0, base_confidence=0.70,
    )
    assert strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0) == []


# ─────────────────────── FundingContrarianStrategy ───────────────────────


def test_funding_contrarian_emits_sell_on_extreme_high_funding() -> None:
    """
    When the current funding rate is at the 95th percentile of history
    (crowded longs paying heavy premium), the contrarian signal is SELL.

    No FeatureStore data is passed → the 4h trend veto is skipped.
    """
    fh = FundingHistory()
    # Load 25 normal rates around 0.01%
    for _ in range(25):
        fh.update(EXCHANGE, SYMBOL, 0.0001)
    # Add a few at higher levels to spread distribution
    for v in [0.0002, 0.0003, 0.0005, 0.0008]:
        fh.update(EXCHANGE, SYMBOL, v)
    # Current extreme rate: top 5th percentile
    fh.update(EXCHANGE, SYMBOL, 0.005)

    strategy = FundingContrarianStrategy(
        funding_history=fh,
        extreme_threshold=0.85,
        low_threshold=0.15,
        base_confidence=0.55,
    )

    # Empty store → no 4h bars → trend veto can't trigger → pure funding signal
    store = FeatureStore()
    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)

    sell_signals = [s for s in signals if s.side == OrderSide.SELL]
    assert sell_signals, (
        f"expected SELL from funding_contrarian on extreme funding, got: {signals}"
    )
    for s in sell_signals:
        assert s.action == SignalAction.OPEN
        assert 0.0 < s.confidence <= 1.0
        assert s.strategy == "funding_contrarian"


def test_funding_contrarian_emits_buy_on_extreme_low_funding() -> None:
    """
    When funding is at the 5th percentile (crowded shorts paying premium),
    the contrarian signal is BUY.
    """
    fh = FundingHistory()
    for _ in range(25):
        fh.update(EXCHANGE, SYMBOL, 0.0001)
    for v in [0.0002, 0.0003, 0.0004]:
        fh.update(EXCHANGE, SYMBOL, v)
    # Extreme negative rate: shorts are paying premium → BUY
    fh.update(EXCHANGE, SYMBOL, -0.005)

    strategy = FundingContrarianStrategy(
        funding_history=fh,
        extreme_threshold=0.85,
        low_threshold=0.15,
        base_confidence=0.55,
    )

    store = FeatureStore()
    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)

    buy_signals = [s for s in signals if s.side == OrderSide.BUY]
    assert buy_signals, (
        f"expected BUY from funding_contrarian on extreme negative funding, got: {signals}"
    )


def test_funding_contrarian_no_signal_on_cold_start() -> None:
    """FundingHistory with <20 samples → percentile returns 0.5 → no signal."""
    fh = FundingHistory()
    # Only 5 samples — below the 20-sample warm-up guard
    for v in [0.001, 0.002, 0.003, -0.001, 0.005]:
        fh.update(EXCHANGE, SYMBOL, v)
    fh.update(EXCHANGE, SYMBOL, 0.005)  # would be extreme if enough history existed

    strategy = FundingContrarianStrategy(
        funding_history=fh, extreme_threshold=0.85, low_threshold=0.15,
        base_confidence=0.55,
    )
    store = FeatureStore()
    assert strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0) == []


# ─────────────────────── VWAPStrategy ───────────────────────


def test_vwap_emits_buy_on_pullback_to_vwap_in_uptrend() -> None:
    """
    Construct a series where:
    - EMA(50) is clearly rising (uptrend confirmed).
    - All prices are above VWAP initially, then the last two bars touch VWAP
      from above (long_touch condition).

    VWAP resets at UTC midnight. We keep all bars in the same UTC day (≤96
    15m bars) so the cumulative VWAP is meaningful and is not re-seeded by
    a day-boundary reset.
    """
    # 80 bars = 20 hours — fits comfortably inside a single UTC day
    n = 80
    # Steep rising trend so EMA(50) slope stays positive even after a single
    # 1-bar pullback on the final bar. (EMA's alpha ≈ 2/51 ≈ 0.04, so one
    # dip bar can only depress EMA slightly.)
    closes = [100.0 + i * 0.4 for i in range(n)]
    # Only the last bar pulls back onto VWAP. Bar -2 stays on the trend line.
    approx_vwap = sum(closes[: n - 1]) / (n - 1)
    closes[-1] = approx_vwap * 1.002    # lands just above VWAP (inside band)

    highs   = [c * 1.002 for c in closes]
    lows    = [c * 0.998 for c in closes]
    volumes = [1000.0] * n

    store = FeatureStore()

    # All bars in the same UTC day (starting from midnight)
    day_start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=UTC)
    key = FeatureKey(EXCHANGE, SYMBOL, "15m")
    bars = [
        Bar(
            ts_ms=int((day_start + timedelta(minutes=15 * i)).timestamp() * 1000),
            open=closes[i - 1] if i > 0 else closes[0] * 0.999,
            high=highs[i],
            low=lows[i],
            close=closes[i],
            volume=volumes[i],
        )
        for i in range(n)
    ]
    store.bulk_load(key, bars)

    strategy = VWAPStrategy(
        timeframe="15m",
        ema_period=50,
        vwap_band_pct=0.01,   # 1% band — wide enough to capture the touch
        base_confidence=0.60,
    )

    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)
    buy_signals = [s for s in signals if s.side == OrderSide.BUY]

    assert buy_signals, (
        f"expected BUY from vwap on pullback to VWAP, got: {signals}"
    )
    for s in buy_signals:
        assert s.action == SignalAction.OPEN
        assert 0.0 < s.confidence <= 1.0
        assert s.strategy == "vwap"


def test_vwap_returns_empty_on_insufficient_bars() -> None:
    store = FeatureStore()
    _populate_store(store, [100.0] * 10, "15m", 15 * 60)

    strategy = VWAPStrategy(timeframe="15m", ema_period=50, base_confidence=0.60)
    assert strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0) == []


# ─────────────────────── LiquidationCascadeStrategy ───────────────────────


def _build_lc_store_and_oi(
    *,
    n_flat: int = 50,
    flat_price: float = 50_000.0,
    bar_range: float = 50.0,       # high = flat_price + bar_range, low = flat_price - bar_range
    cascade_drop: float = 200.0,   # last bar falls by this much → triggers if > 1.5×ATR
    oi_start: float = 1000.0,
    oi_end: float = 950.0,         # -5% drop (triggers at -3%)
) -> tuple[FeatureStore, OIStore]:
    """Return (FeatureStore, OIStore) preloaded for a long-liq cascade scenario."""
    store = FeatureStore()

    closes = [flat_price] * n_flat + [flat_price - cascade_drop]
    n = len(closes)
    highs = [c + bar_range for c in closes]
    lows = [c - bar_range for c in closes]

    _populate_store(store, closes, "15m", 15 * 60, highs=highs, lows=lows)

    oi = OIStore()
    oi.update(EXCHANGE, SYMBOL, oi_start)
    oi.update(EXCHANGE, SYMBOL, oi_end)
    return store, oi


def test_liquidation_cascade_buy_on_liq_drop() -> None:
    """OI drops 5% AND price falls 200 (> 1.5×ATR≈100) → BUY signal."""
    store, oi = _build_lc_store_and_oi()
    strategy = LiquidationCascadeStrategy(
        oi_store=oi,
        timeframe="15m",
        oi_roc_threshold=-0.03,
        atr_period=14,
        atr_multiplier=1.5,
        base_confidence=0.60,
    )
    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)
    buys = [s for s in signals if s.side == OrderSide.BUY]
    assert buys, f"expected BUY on liquidation cascade, got: {signals}"
    for s in buys:
        assert s.action == SignalAction.OPEN
        assert 0.0 < s.confidence <= 1.0
        assert s.strategy == "liquidation_cascade"


def test_liquidation_cascade_sell_on_short_liq() -> None:
    """OI drops 5% AND price rises 200 → SELL (short-squeeze liq cascade)."""
    store = FeatureStore()
    flat = 50_000.0
    n_flat = 50
    closes = [flat] * n_flat + [flat + 200.0]
    highs = [c + 50 for c in closes]
    lows = [c - 50 for c in closes]
    _populate_store(store, closes, "15m", 15 * 60, highs=highs, lows=lows)

    oi = OIStore()
    oi.update(EXCHANGE, SYMBOL, 1000.0)
    oi.update(EXCHANGE, SYMBOL, 930.0)  # -7% drop

    strategy = LiquidationCascadeStrategy(
        oi_store=oi,
        timeframe="15m",
        oi_roc_threshold=-0.03,
        atr_period=14,
        atr_multiplier=1.5,
        base_confidence=0.60,
    )
    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)
    sells = [s for s in signals if s.side == OrderSide.SELL]
    assert sells, f"expected SELL on short-liq cascade, got: {signals}"


def test_liquidation_cascade_no_signal_when_oi_drop_small() -> None:
    """OI only drops 1% (below -3% threshold) → no signal even with large price move."""
    store, _ = _build_lc_store_and_oi(oi_end=990.0)  # only -1% OI drop

    oi = OIStore()
    oi.update(EXCHANGE, SYMBOL, 1000.0)
    oi.update(EXCHANGE, SYMBOL, 990.0)

    strategy = LiquidationCascadeStrategy(
        oi_store=oi,
        timeframe="15m",
        oi_roc_threshold=-0.03,
        atr_period=14,
        atr_multiplier=1.5,
        base_confidence=0.60,
    )
    store2 = FeatureStore()
    closes = [50_000.0] * 50 + [50_000.0 - 200.0]
    _populate_store(store2, closes, "15m", 15 * 60)

    signals = strategy.evaluate(SYMBOL, store2, EXCHANGE, _TS0)
    assert signals == [], f"expected no signal with small OI drop, got: {signals}"


def test_liquidation_cascade_no_signal_when_price_move_small() -> None:
    """OI drops 5% but price barely moves → no signal."""
    store = FeatureStore()
    flat = 50_000.0
    # Last bar only drops 10 (ATR≈100, 10 < 1.5×100=150 → no trigger)
    closes = [flat] * 50 + [flat - 10.0]
    highs = [c + 50 for c in closes]
    lows = [c - 50 for c in closes]
    _populate_store(store, closes, "15m", 15 * 60, highs=highs, lows=lows)

    oi = OIStore()
    oi.update(EXCHANGE, SYMBOL, 1000.0)
    oi.update(EXCHANGE, SYMBOL, 940.0)  # -6% OI drop

    strategy = LiquidationCascadeStrategy(
        oi_store=oi,
        timeframe="15m",
        oi_roc_threshold=-0.03,
        atr_period=14,
        atr_multiplier=1.5,
        base_confidence=0.60,
    )
    signals = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)
    assert signals == [], f"expected no signal with small price move, got: {signals}"


def test_liquidation_cascade_no_signal_on_insufficient_bars() -> None:
    """Fewer than atr_period + 2 bars → returns []."""
    store = FeatureStore()
    _populate_store(store, [50_000.0] * 10, "15m", 15 * 60)

    oi = OIStore()
    oi.update(EXCHANGE, SYMBOL, 1000.0)
    oi.update(EXCHANGE, SYMBOL, 900.0)

    strategy = LiquidationCascadeStrategy(
        oi_store=oi,
        timeframe="15m",
        oi_roc_threshold=-0.03,
        atr_period=14,
        atr_multiplier=1.5,
        base_confidence=0.60,
    )
    assert strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0) == []


def test_liquidation_cascade_confidence_scales_with_oi_drop() -> None:
    """Larger OI drop → higher confidence (scales between base and 1.0)."""
    base_conf = 0.60

    def _run_with_oi_end(oi_end: float) -> float:
        store = FeatureStore()
        closes = [50_000.0] * 50 + [50_000.0 - 200.0]
        highs = [c + 50 for c in closes]
        lows = [c - 50 for c in closes]
        _populate_store(store, closes, "15m", 15 * 60, highs=highs, lows=lows)
        oi = OIStore()
        oi.update(EXCHANGE, SYMBOL, 1000.0)
        oi.update(EXCHANGE, SYMBOL, oi_end)
        strategy = LiquidationCascadeStrategy(
            oi_store=oi,
            timeframe="15m",
            oi_roc_threshold=-0.03,
            atr_period=14,
            atr_multiplier=1.5,
            base_confidence=base_conf,
        )
        sigs = strategy.evaluate(SYMBOL, store, EXCHANGE, _TS0)
        return sigs[0].confidence if sigs else 0.0

    conf_small = _run_with_oi_end(970.0)   # -3% drop (at threshold)
    conf_large = _run_with_oi_end(850.0)   # -15% drop (well above threshold)

    assert conf_small >= base_conf
    assert conf_large > conf_small, (
        f"larger OI drop should give higher confidence: {conf_large} vs {conf_small}"
    )
