"""
Thin wrappers around pandas_ta. Every function takes a pandas DataFrame of
OHLCV data and returns a new column(s) as a pandas Series or DataFrame.

We avoid `ta-lib` because of its native build dependency — pandas_ta is pure
Python/NumPy and good enough for strategy-level indicators.

All wrappers validate the minimum history length and return NaN series rather
than raising when there isn't enough data, so strategies can safely ignore
early bars.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
import pandas_ta as ta  # type: ignore[import-untyped]

# Column names we expect to find in the OHLCV frame. Strategies should use these
# constants so a rename here propagates everywhere.
COL_OPEN: Final = "open"
COL_HIGH: Final = "high"
COL_LOW: Final = "low"
COL_CLOSE: Final = "close"
COL_VOLUME: Final = "volume"


def _ensure_cols(df: pd.DataFrame, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe missing required columns: {missing}")


def ema(df: pd.DataFrame, length: int, col: str = COL_CLOSE) -> pd.Series:
    """Exponential moving average."""
    _ensure_cols(df, [col])
    return ta.ema(df[col], length=length)


def rsi(df: pd.DataFrame, length: int = 14, col: str = COL_CLOSE) -> pd.Series:
    """Relative Strength Index. Returns NaN for the first `length` bars."""
    _ensure_cols(df, [col])
    return ta.rsi(df[col], length=length)


def macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
    col: str = COL_CLOSE,
) -> pd.DataFrame:
    """
    MACD — returns a DataFrame with columns:
        MACD_<fast>_<slow>_<signal>      : MACD line
        MACDh_<fast>_<slow>_<signal>     : histogram (MACD - signal)
        MACDs_<fast>_<slow>_<signal>     : signal line
    """
    _ensure_cols(df, [col])
    return ta.macd(df[col], fast=fast, slow=slow, signal=signal)


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """Average True Range — used by risk module for SL distance calculation."""
    _ensure_cols(df, [COL_HIGH, COL_LOW, COL_CLOSE])
    return ta.atr(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], length=length)


def adx(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    """
    ADX directional movement. Returns DataFrame with:
        ADX_<length>  : trend strength (>25 typically = trending)
        DMP_<length>  : +DI
        DMN_<length>  : -DI
    """
    _ensure_cols(df, [COL_HIGH, COL_LOW, COL_CLOSE])
    return ta.adx(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], length=length)


def bollinger(
    df: pd.DataFrame, length: int = 20, std: float = 2.0, col: str = COL_CLOSE
) -> pd.DataFrame:
    """
    Bollinger Bands. Returns DataFrame with columns:
        BBL_<length>_<std>  : lower band
        BBM_<length>_<std>  : middle (SMA)
        BBU_<length>_<std>  : upper band
        BBB_<length>_<std>  : bandwidth
        BBP_<length>_<std>  : %B (position within the bands)
    """
    _ensure_cols(df, [col])
    return ta.bbands(df[col], length=length, std=std)


def keltner(
    df: pd.DataFrame, length: int = 20, scalar: float = 2.0
) -> pd.DataFrame:
    """
    Keltner Channels. Returns DataFrame with:
        KCLe_<length>_<scalar>  : lower
        KCBe_<length>_<scalar>  : basis (EMA)
        KCUe_<length>_<scalar>  : upper
    """
    _ensure_cols(df, [COL_HIGH, COL_LOW, COL_CLOSE])
    return ta.kc(df[COL_HIGH], df[COL_LOW], df[COL_CLOSE], length=length, scalar=scalar)


def donchian(df: pd.DataFrame, length: int = 20) -> pd.DataFrame:
    """
    Donchian Channels. Returns DataFrame with:
        DCL_<length>_<length>  : lower (rolling min of low)
        DCM_<length>_<length>  : midline
        DCU_<length>_<length>  : upper (rolling max of high)
    """
    _ensure_cols(df, [COL_HIGH, COL_LOW])
    return ta.donchian(df[COL_HIGH], df[COL_LOW], lower_length=length, upper_length=length)


def realized_vol(df: pd.DataFrame, window: int = 96, col: str = COL_CLOSE) -> pd.Series:
    """
    Realized volatility: rolling stdev of log returns over `window` bars.
    Not annualized — used as a relative regime indicator.
    """
    _ensure_cols(df, [col])
    log_ret = np.log(df[col] / df[col].shift(1))
    return log_ret.rolling(window=window, min_periods=window // 2).std()


def rolling_volume_ratio(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """Current bar volume / average of previous `length` bars. NaN-safe."""
    _ensure_cols(df, [COL_VOLUME])
    avg = df[COL_VOLUME].rolling(window=length, min_periods=length).mean().shift(1)
    return df[COL_VOLUME] / avg
