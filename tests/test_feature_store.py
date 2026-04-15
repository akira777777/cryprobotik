"""Feature store buffer + DataFrame materialization tests."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.feature_store import Bar, FeatureKey, FeatureStore


def _bar(ts_ms: int, close: float = 100.0) -> Bar:
    return Bar(
        ts_ms=ts_ms,
        open=close - 0.5,
        high=close + 1.0,
        low=close - 1.0,
        close=close,
        volume=500.0,
    )


def test_append_and_latest() -> None:
    store = FeatureStore(max_bars=100)
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    assert store.latest(key) is None

    store.append_bar(key, _bar(1_000, 100.0))
    store.append_bar(key, _bar(2_000, 101.0))
    latest = store.latest(key)
    assert latest is not None
    assert latest.ts_ms == 2_000
    assert latest.close == 101.0
    assert store.size(key) == 2


def test_same_ts_replaces_last_bar() -> None:
    """
    A forming-bar update (same ts) must replace the last bar, not append.
    Otherwise indicators would see phantom duplicate prints.
    """
    store = FeatureStore()
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    store.append_bar(key, _bar(1_000, 100.0))
    store.append_bar(key, _bar(1_000, 103.0))  # in-progress bar updated
    assert store.size(key) == 1
    assert store.latest(key).close == 103.0  # type: ignore[union-attr]


def test_out_of_order_bars_dropped() -> None:
    store = FeatureStore()
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    store.append_bar(key, _bar(2_000, 100.0))
    store.append_bar(key, _bar(1_000, 99.0))  # stale, must be dropped
    assert store.size(key) == 1
    assert store.latest(key).ts_ms == 2_000  # type: ignore[union-attr]


def test_max_bars_enforced() -> None:
    store = FeatureStore(max_bars=5)
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    for i in range(10):
        store.append_bar(key, _bar((i + 1) * 1_000, 100.0 + i))
    assert store.size(key) == 5
    # The oldest 5 were dropped
    bars = store.bars(key)
    assert bars[0].ts_ms == 6_000
    assert bars[-1].ts_ms == 10_000


def test_bulk_load() -> None:
    store = FeatureStore(max_bars=100)
    key = FeatureKey("okx", "ETH/USDT:USDT", "1h")
    seed = [_bar(i * 1_000, 50.0 + i) for i in range(20)]
    store.bulk_load(key, seed)
    assert store.size(key) == 20
    assert store.latest(key).close == 50.0 + 19  # type: ignore[union-attr]


def test_drop_removes_buffer() -> None:
    store = FeatureStore()
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    store.append_bar(key, _bar(1_000))
    assert store.size(key) == 1
    store.drop(key)
    assert store.latest(key) is None
    assert store.size(key) == 0


def test_as_df_returns_none_when_insufficient() -> None:
    store = FeatureStore()
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    store.append_bar(key, _bar(1_000))
    assert store.as_df(key, min_bars=10) is None


def test_as_df_returns_ohlcv_dataframe() -> None:
    store = FeatureStore()
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    for i in range(30):
        store.append_bar(key, _bar((i + 1) * 60_000, 100.0 + i))
    df = store.as_df(key, min_bars=10)
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]
    assert df.index.name == "ts"
    assert len(df) == 30
    # Timestamps are UTC DatetimeIndex
    assert df.index.tz is not None
    # Last close matches
    assert df["close"].iloc[-1] == pytest.approx(129.0)


def test_as_df_cache_invalidated_on_new_bar() -> None:
    store = FeatureStore()
    key = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    for i in range(5):
        store.append_bar(key, _bar((i + 1) * 60_000, 100.0 + i))
    df1 = store.as_df(key)
    assert df1 is not None and len(df1) == 5

    # Same data → cached frame returned
    df2 = store.as_df(key)
    assert df2 is df1

    # Append a new bar → cache must be invalidated and a fresh DF built
    store.append_bar(key, _bar(6 * 60_000, 200.0))
    df3 = store.as_df(key)
    assert df3 is not None
    assert len(df3) == 6
    assert df3["close"].iloc[-1] == pytest.approx(200.0)
    assert df3 is not df1  # new object


def test_keys_lists_all_buffers() -> None:
    store = FeatureStore()
    k1 = FeatureKey("okx", "BTC/USDT:USDT", "15m")
    k2 = FeatureKey("bybit", "ETH/USDT:USDT", "1h")
    store.append_bar(k1, _bar(1_000))
    store.append_bar(k2, _bar(1_000))
    keys = store.keys()
    assert set(keys) == {k1, k2}
