"""
In-memory feature / signal stores.

All four stores below are populated by the Orchestrator's event pumps and read
by strategies, the ensemble, and the ML filter. They are pure Python state —
no disk I/O, no network. Persistence (OHLCV, fills, etc.) happens separately
through `src.data.storage.Storage` against Postgres.

Stores:
    FeatureStore     — rolling OHLCV bars per (exchange, symbol, timeframe).
    CVDStore         — per-bar taker-buy / taker-sell flow for CVD ratio.
    OIStore          — rolling open-interest snapshots for ROC.
    FundingHistory   — rolling funding-rate history for percentile ranking.

All stores are single-event-loop safe (no locks): every call site is inside
the asyncio event loop and mutations are synchronous.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import pandas as pd

if TYPE_CHECKING:
    from src.exchanges.base import OrderSide


# ─────────────────────── value types ───────────────────────


@dataclass(slots=True)
class Bar:
    """One OHLCV bar. `ts_ms` is the bar OPEN time in Unix milliseconds, UTC."""

    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float


class FeatureKey(NamedTuple):
    """Composite key for a rolling OHLCV buffer."""

    exchange: str
    symbol: str
    timeframe: str


# ─────────────────────── FeatureStore ───────────────────────


class _BarBuffer:
    """Single (exchange, symbol, timeframe) buffer with cached DataFrame."""

    __slots__ = ("bars", "_df_cache", "_df_cache_last_ts")

    def __init__(self, max_bars: int) -> None:
        self.bars: deque[Bar] = deque(maxlen=max_bars)
        self._df_cache: pd.DataFrame | None = None
        self._df_cache_last_ts: int = -1

    def append(self, bar: Bar) -> None:
        """Append a bar, replacing the tail if the timestamp matches (forming
        bar), and dropping strictly-older timestamps (stale WS snapshot)."""
        if self.bars:
            tail = self.bars[-1]
            if bar.ts_ms < tail.ts_ms:
                return
            if bar.ts_ms == tail.ts_ms:
                self.bars[-1] = bar
                self._df_cache = None
                return
        self.bars.append(bar)
        self._df_cache = None

    def bulk_load(self, bars_iter: list[Bar]) -> None:
        self.bars.clear()
        for b in bars_iter:
            self.bars.append(b)
        self._df_cache = None

    def latest(self) -> Bar | None:
        return self.bars[-1] if self.bars else None

    def size(self) -> int:
        return len(self.bars)

    def as_list(self) -> list[Bar]:
        return list(self.bars)

    def as_df(self, min_bars: int) -> pd.DataFrame | None:
        n = len(self.bars)
        if n == 0 or n < max(min_bars, 1):
            return None
        latest_ts = self.bars[-1].ts_ms
        if (
            self._df_cache is not None
            and self._df_cache_last_ts == latest_ts
            and len(self._df_cache) == n
        ):
            return self._df_cache
        rows = [(b.ts_ms, b.open, b.high, b.low, b.close, b.volume) for b in self.bars]
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        df = df.set_index("ts")
        self._df_cache = df
        self._df_cache_last_ts = latest_ts
        return df


class FeatureStore:
    """Rolling OHLCV per (exchange, symbol, timeframe)."""

    def __init__(self, max_bars: int = 1500) -> None:
        self._max_bars = max_bars
        self._buffers: dict[FeatureKey, _BarBuffer] = {}

    def _buf(self, key: FeatureKey) -> _BarBuffer:
        buf = self._buffers.get(key)
        if buf is None:
            buf = _BarBuffer(max_bars=self._max_bars)
            self._buffers[key] = buf
        return buf

    def append_bar(self, key: FeatureKey, bar: Bar) -> None:
        self._buf(key).append(bar)

    def bulk_load(self, key: FeatureKey, bars: list[Bar]) -> None:
        self._buf(key).bulk_load(bars)

    def latest(self, key: FeatureKey) -> Bar | None:
        buf = self._buffers.get(key)
        return buf.latest() if buf else None

    def size(self, key: FeatureKey) -> int:
        buf = self._buffers.get(key)
        return buf.size() if buf else 0

    def bars(self, key: FeatureKey) -> list[Bar]:
        buf = self._buffers.get(key)
        return buf.as_list() if buf else []

    def drop(self, key: FeatureKey) -> None:
        self._buffers.pop(key, None)

    def as_df(self, key: FeatureKey, min_bars: int = 0) -> pd.DataFrame | None:
        buf = self._buffers.get(key)
        if buf is None:
            return None
        return buf.as_df(min_bars)

    def keys(self) -> list[FeatureKey]:
        return list(self._buffers.keys())


# ─────────────────────── CVDStore ───────────────────────


class _CVDBar:
    """Accumulator for one bar's taker flow."""

    __slots__ = ("buy_qty", "sell_qty")

    def __init__(self) -> None:
        self.buy_qty: float = 0.0
        self.sell_qty: float = 0.0

    def total(self) -> float:
        return self.buy_qty + self.sell_qty


class CVDStore:
    """
    Per-bar taker buy/sell volume.

    `on_trade` accumulates into the in-progress bar; `on_bar_close` finalises
    it and starts a new one. `cvd_ratio` reports the taker-buy fraction over
    the last *lookback* finalised bars (range [0, 1]; 0.5 = neutral).
    """

    def __init__(self, max_bars: int = 200) -> None:
        self._max_bars = max_bars
        self._history: dict[tuple[str, str], deque[_CVDBar]] = {}
        self._current: dict[tuple[str, str], _CVDBar] = {}

    def _current_bar(self, exchange: str, symbol: str) -> _CVDBar:
        key = (exchange, symbol)
        bar = self._current.get(key)
        if bar is None:
            bar = _CVDBar()
            self._current[key] = bar
        return bar

    def on_trade(self, exchange: str, symbol: str, side: "OrderSide", qty: float) -> None:
        from src.exchanges.base import OrderSide as _OrderSide

        bar = self._current_bar(exchange, symbol)
        if side == _OrderSide.BUY:
            bar.buy_qty += qty
        else:
            bar.sell_qty += qty

    def on_bar_close(self, exchange: str, symbol: str) -> None:
        key = (exchange, symbol)
        bar = self._current.pop(key, None)
        if bar is None:
            return
        hist = self._history.setdefault(key, deque(maxlen=self._max_bars))
        hist.append(bar)

    def has_data(self, exchange: str, symbol: str, min_bars: int = 3) -> bool:
        hist = self._history.get((exchange, symbol))
        return hist is not None and len(hist) >= min_bars

    def cvd_ratio(self, exchange: str, symbol: str, lookback: int = 20) -> float:
        """Taker-buy fraction over the last *lookback* finalised bars."""
        hist = self._history.get((exchange, symbol))
        if not hist:
            return 0.5
        recent = list(hist)[-lookback:]
        buy_sum = sum(b.buy_qty for b in recent)
        total = sum(b.total() for b in recent)
        if total <= 0.0:
            return 0.5
        return max(0.0, min(1.0, buy_sum / total))

    def trend_aligned(
        self, exchange: str, symbol: str, side: "OrderSide", lookback: int = 20
    ) -> bool:
        """True when CVD agrees with the intended side (>0.55 BUY, <0.45 SELL)."""
        from src.exchanges.base import OrderSide as _OrderSide

        ratio = self.cvd_ratio(exchange, symbol, lookback=lookback)
        if side == _OrderSide.BUY:
            return ratio > 0.55
        return ratio < 0.45


# ─────────────────────── OIStore ───────────────────────


class OIStore:
    """Rolling open-interest history with rate-of-change computation."""

    def __init__(self, max_samples: int = 200) -> None:
        self._max_samples = max_samples
        self._history: dict[tuple[str, str], deque[float]] = {}

    def update(self, exchange: str, symbol: str, oi_contracts: float) -> None:
        key = (exchange, symbol)
        hist = self._history.setdefault(key, deque(maxlen=self._max_samples))
        hist.append(float(oi_contracts))

    def has_data(self, exchange: str, symbol: str, min_samples: int = 2) -> bool:
        hist = self._history.get((exchange, symbol))
        return hist is not None and len(hist) >= min_samples

    def oi_roc(self, exchange: str, symbol: str, periods: int = 5) -> float:
        hist = self._history.get((exchange, symbol))
        if not hist or len(hist) < 2:
            return 0.0
        arr = list(hist)
        n = min(periods, len(arr) - 1)
        if n <= 0:
            return 0.0
        start = arr[-(n + 1)]
        end = arr[-1]
        if start <= 0.0:
            return 0.0
        return (end - start) / start


# ─────────────────────── FundingHistory ───────────────────────


_FUNDING_MIN_SAMPLES: int = 20


class FundingHistory:
    """
    Rolling funding-rate history per (exchange, symbol).

    `percentile(rate)` returns the rank of *rate* within the stored history,
    normalised to [0, 1]. Returns the neutral 0.5 until at least
    `_FUNDING_MIN_SAMPLES` samples have been observed (cold-start guard).
    """

    def __init__(self, max_samples: int = 500) -> None:
        self._max_samples = max_samples
        self._history: dict[tuple[str, str], deque[float]] = {}

    def update(self, exchange: str, symbol: str, rate: float) -> None:
        key = (exchange, symbol)
        hist = self._history.setdefault(key, deque(maxlen=self._max_samples))
        hist.append(float(rate))

    def latest(self, exchange: str, symbol: str) -> float | None:
        hist = self._history.get((exchange, symbol))
        if not hist:
            return None
        return hist[-1]

    def percentile(self, exchange: str, symbol: str, rate: float) -> float:
        hist = self._history.get((exchange, symbol))
        if not hist or len(hist) < _FUNDING_MIN_SAMPLES:
            return 0.5
        arr = sorted(hist)
        n = len(arr)
        less = 0
        equal = 0
        for v in arr:
            if v < rate:
                less += 1
            elif v == rate:
                equal += 1
            else:
                break
        rank = (less + 0.5 * equal) / n
        return max(0.0, min(1.0, rank))
