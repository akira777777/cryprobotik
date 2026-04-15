"""Smoke tests for indicator wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.utils.indicators import (
    adx, atr, bollinger, donchian, ema, macd, realized_vol, rolling_volume_ratio, rsi,
)


def test_ema_returns_series_of_same_length(sample_ohlcv_df: pd.DataFrame) -> None:
    out = ema(sample_ohlcv_df, length=20)
    assert isinstance(out, pd.Series)
    assert len(out) == len(sample_ohlcv_df)
    # First 19 values are NaN; after that values are finite
    assert out.tail(50).notna().all()


def test_rsi_in_bounds(sample_ohlcv_df: pd.DataFrame) -> None:
    out = rsi(sample_ohlcv_df, length=14)
    tail = out.dropna()
    assert (tail >= 0).all() and (tail <= 100).all()


def test_atr_positive(sample_ohlcv_df: pd.DataFrame) -> None:
    out = atr(sample_ohlcv_df, length=14).dropna()
    assert (out > 0).all()


def test_macd_returns_expected_columns(sample_ohlcv_df: pd.DataFrame) -> None:
    out = macd(sample_ohlcv_df, fast=12, slow=26, signal=9)
    assert out is not None
    assert any(c.startswith("MACDh_") for c in out.columns)


def test_bollinger_bands_ordering(sample_ohlcv_df: pd.DataFrame) -> None:
    bb = bollinger(sample_ohlcv_df, length=20, std=2.0)
    assert bb is not None
    lower = bb.filter(like="BBL_").iloc[:, 0].dropna()
    mid = bb.filter(like="BBM_").iloc[:, 0].dropna()
    upper = bb.filter(like="BBU_").iloc[:, 0].dropna()
    assert (upper.loc[lower.index] >= mid.loc[lower.index]).all()
    assert (mid.loc[lower.index] >= lower).all()


def test_donchian_contains_close(sample_ohlcv_df: pd.DataFrame) -> None:
    dc = donchian(sample_ohlcv_df, length=20)
    assert dc is not None
    upper = dc.filter(like="DCU_").iloc[:, 0].dropna()
    lower = dc.filter(like="DCL_").iloc[:, 0].dropna()
    highs = sample_ohlcv_df["high"].loc[upper.index]
    lows = sample_ohlcv_df["low"].loc[lower.index]
    # Donchian upper is rolling max of high → should equal or exceed last N highs
    assert (upper >= highs).all()
    assert (lower <= lows).all()


def test_adx_nonneg(sample_ohlcv_df: pd.DataFrame) -> None:
    out = adx(sample_ohlcv_df, length=14)
    assert out is not None
    col = "ADX_14"
    assert col in out.columns
    tail = out[col].dropna()
    assert (tail >= 0).all()


def test_realized_vol_nonneg(sample_ohlcv_df: pd.DataFrame) -> None:
    rv = realized_vol(sample_ohlcv_df, window=50).dropna()
    assert (rv >= 0).all()


def test_rolling_volume_ratio(sample_ohlcv_df: pd.DataFrame) -> None:
    rr = rolling_volume_ratio(sample_ohlcv_df, length=20).dropna()
    assert (rr > 0).all()


def test_indicator_with_missing_columns_raises() -> None:
    df = pd.DataFrame({"x": [1, 2, 3]})
    with pytest.raises(ValueError):
        ema(df, length=3)
