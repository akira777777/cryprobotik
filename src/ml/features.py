"""
Feature extraction for ML signal scoring.

Computes a fixed 23-element float vector from a FeatureStore snapshot + Signal.
Uses pure pandas/numpy — no pandas-ta — to stay fast on the hot path.

Feature layout:
  [0-6]   15m technical indicators
  [7-10]  1h technical indicators
  [11-12] 4h technical indicators
  [13-17] regime one-hot (5 binary flags: trend_high_vol … chop)
  [18-21] time features (hour sin/cos, weekday sin/cos)
  [22]    4h ATR ratio

Note: ensemble meta-outputs (net_vote, weighted_long, weighted_short,
confidence) were removed in FEATURE_VERSION 2 to prevent the ML filter
from degenerating into a replica of the ensemble's confidence ranking.
Volume context and strategy_encoded features were consolidated in v2 as
well to keep the vector compact (23 features).
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from src.data.feature_store import CVDStore, FeatureKey, FeatureStore, OIStore
    from src.strategies.base import Signal

# ─────────────────────── last-bar result cache ───────────────────────
# Keyed by (symbol, exchange, bar_ts_ms_15m) — the 15m bar timestamp
# uniquely identifies the market snapshot for all three timeframes.
# Max 50 entries; oldest key is evicted when the limit is reached.
# Safe because extract_features() is called from the asyncio event loop
# (single-threaded) — no locking needed.
_FEATURE_CACHE_MAX: int = 50
_feature_cache: dict[tuple, list[float]] = {}

REGIME_MAP: dict[str, int] = {
    "trend_high_vol": 0,
    "trend_low_vol": 1,
    "range_high_vol": 2,
    "range_low_vol": 3,
    "chop": 4,
}

STRATEGY_MAP: dict[str, int] = {
    "momentum": 0,
    "mean_reversion": 1,
    "volatility_breakout": 2,
    "funding_arb": 3,
    "ensemble": 4,
}

FEATURE_NAMES: list[str] = [
    # 15m (7 features)
    "rsi_15m",
    "macd_hist_15m",
    "ema9_slope_15m",
    "bb_width_15m",
    "adx_15m",
    "donch_pos_15m",
    "atr_ratio_15m",
    # 1h (4 features)
    "rsi_1h",
    "ema9_slope_1h",
    "ema21_slope_1h",
    "atr_ratio_1h",
    # 4h (2 features)
    "rsi_4h",
    "ema9_slope_4h",
    # regime one-hot (5 binary features — replaces single ordinal encoding)
    "regime_trend_high_vol",
    "regime_trend_low_vol",
    "regime_range_high_vol",
    "regime_range_low_vol",
    "regime_chop",
    # time (4 features)
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    # 4h ATR (1 feature)
    "atr_ratio_4h",
    # perp-native signals (2 features) — added in FEATURE_VERSION 3
    "cvd_ratio",   # taker buy fraction [0, 1]; 0.5 = neutral
    "oi_roc",      # open-interest rate-of-change (normalised)
]

N_FEATURES: int = len(FEATURE_NAMES)  # 25
# Bump this whenever the feature vector layout changes so persisted models
# are automatically discarded and a fresh cold start begins.
FEATURE_VERSION: int = 3


# ─────────────────────── indicator helpers ───────────────────────


def _rsi(close: pd.Series, period: int = 14) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs = gain / loss.replace(0.0, float("nan"))
    rsi_s = 100.0 - 100.0 / (1.0 + rs)
    v = rsi_s.iloc[-1]
    return float(v) if not (math.isnan(v) or math.isinf(v)) else 50.0


def _macd_hist(close: pd.Series) -> float:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    sig = macd.ewm(span=9, adjust=False).mean()
    hist = macd - sig
    v = hist.iloc[-1]
    # Normalize by price so the value is scale-invariant
    price = float(close.iloc[-1]) or 1.0
    return float(v) / price if not (math.isnan(v) or math.isinf(v)) else 0.0


def _ema_slope(close: pd.Series, span: int, lookback: int = 5) -> float:
    ema = close.ewm(span=span, adjust=False).mean()
    if len(ema) < lookback + 1:
        return 0.0
    old = float(ema.iloc[-(lookback + 1)])
    new = float(ema.iloc[-1])
    if old == 0.0:
        return 0.0
    v = (new - old) / old
    return v if not (math.isnan(v) or math.isinf(v)) else 0.0


def _bb_width(close: pd.Series, period: int = 20) -> float:
    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + 2.0 * std
    lower = mid - 2.0 * std
    m = float(mid.iloc[-1])
    if m == 0.0:
        return 0.0
    v = float((upper.iloc[-1] - lower.iloc[-1]) / m)
    return v if not (math.isnan(v) or math.isinf(v)) else 0.0


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_s = tr.ewm(com=period - 1, adjust=False).mean()
    diff_high = high.diff()
    diff_low = -low.diff()
    dm_plus = diff_high.where((diff_high > diff_low) & (diff_high > 0), 0.0)
    dm_minus = diff_low.where((diff_low > diff_high) & (diff_low > 0), 0.0)
    di_plus = 100.0 * dm_plus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0.0, float("nan"))
    di_minus = 100.0 * dm_minus.ewm(com=period - 1, adjust=False).mean() / atr_s.replace(0.0, float("nan"))
    dx = 100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0.0, float("nan"))
    adx_s = dx.ewm(com=period - 1, adjust=False).mean()
    v = adx_s.iloc[-1]
    return float(v) if not (math.isnan(v) or math.isinf(v)) else 0.0


def _donch_pos(df: pd.DataFrame, period: int = 20) -> float:
    """Position of close within the Donchian channel: 0.0 = at low, 1.0 = at high."""
    high_max = float(df["high"].rolling(period).max().iloc[-1])
    low_min = float(df["low"].rolling(period).min().iloc[-1])
    rng = high_max - low_min
    if rng == 0.0:
        return 0.5
    v = (float(df["close"].iloc[-1]) - low_min) / rng
    return max(0.0, min(1.0, v))


def _atr_ratio(df: pd.DataFrame, period: int = 14) -> float:
    """ATR / close — normalized volatility."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    tr = pd.concat(
        [
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_s = tr.ewm(com=period - 1, adjust=False).mean()
    price = float(close.iloc[-1])
    if price == 0.0:
        return 0.0
    v = float(atr_s.iloc[-1]) / price
    return v if not (math.isnan(v) or math.isinf(v)) else 0.0


def _safe_df_feature(fn, df: pd.DataFrame | None, default: float = 0.0) -> float:
    if df is None:
        return default
    try:
        return fn(df)
    except Exception:
        return default


def _safe_close_feature(fn, df: pd.DataFrame | None, default: float = 0.0, **kw) -> float:
    if df is None:
        return default
    try:
        return fn(df["close"], **kw)
    except Exception:
        return default


def _vol_ratio(df: pd.DataFrame, length: int = 20) -> float:
    """Current bar volume / average of previous `length` bars. Returns 1.0 as neutral."""
    if "volume" not in df.columns:
        return 1.0
    avg = df["volume"].rolling(window=length, min_periods=length).mean().shift(1)
    ratio = df["volume"] / avg
    v = ratio.iloc[-1]
    return float(v) if not (math.isnan(v) or math.isinf(v)) else 1.0


# ─────────────────────── public API ───────────────────────


def extract_features(
    signal: "Signal",
    store: "FeatureStore",
    exchange: str,
    cvd_store: "CVDStore | None" = None,
    oi_store: "OIStore | None" = None,
) -> list[float] | None:
    """
    Extract the 25-element ML feature vector for a signal.

    Returns None if insufficient 15m data is available (< 30 bars).
    1h / 4h / CVD / OI features fall back to neutral defaults when sparse.

    Results are cached by (symbol, exchange, bar_ts_ms_15m) so that
    multiple strategies evaluating the same symbol+bar do not recompute
    the full indicator suite. Cache holds at most 50 entries (oldest evicted).
    """
    from src.data.feature_store import FeatureKey

    key_15m = FeatureKey(exchange=exchange, symbol=signal.symbol, timeframe="15m")
    key_1h = FeatureKey(exchange=exchange, symbol=signal.symbol, timeframe="1h")
    key_4h = FeatureKey(exchange=exchange, symbol=signal.symbol, timeframe="4h")

    # ── cache lookup ──────────────────────────────────────
    # The cache key includes the 15m bar timestamp so we automatically
    # return a fresh vector when a new bar arrives.
    latest_bar = store.latest(key_15m)
    if latest_bar is not None:
        cache_key = (signal.symbol, exchange, latest_bar.ts_ms)
        cached_vec = _feature_cache.get(cache_key)
        if cached_vec is not None:
            return cached_vec
    else:
        cache_key = None  # will return None below after the min_bars check

    df15 = store.as_df(key_15m, min_bars=30)
    if df15 is None:
        return None  # not enough data

    df1h = store.as_df(key_1h, min_bars=30)
    df4h = store.as_df(key_4h, min_bars=30)

    meta = signal.meta

    # ── 15m features ──────────────────────────────────────
    f0 = _safe_close_feature(_rsi, df15, default=50.0) / 100.0
    f1 = _safe_close_feature(_macd_hist, df15)
    f2 = _safe_close_feature(_ema_slope, df15, span=9)
    f3 = _safe_close_feature(_bb_width, df15)
    f4 = _safe_df_feature(_adx, df15) / 100.0
    f5 = _safe_df_feature(_donch_pos, df15, default=0.5)
    f6 = _safe_df_feature(_atr_ratio, df15)

    # ── 1h features ───────────────────────────────────────
    f7 = _safe_close_feature(_rsi, df1h, default=50.0) / 100.0
    f8 = _safe_close_feature(_ema_slope, df1h, span=9)
    f9 = _safe_close_feature(_ema_slope, df1h, span=21)
    f10 = _safe_df_feature(_atr_ratio, df1h)

    # ── 4h features ───────────────────────────────────────
    f11 = _safe_close_feature(_rsi, df4h, default=50.0) / 100.0
    f12 = _safe_close_feature(_ema_slope, df4h, span=9)

    # ── regime one-hot (5 binary features) ───────────────
    # Ensemble meta (net_vote, weighted_long, weighted_short, confidence)
    # intentionally excluded — including them caused the ML filter to
    # degenerate into a noisy copy of the ensemble's own confidence ranking.
    regime_str = str(meta.get("regime", "chop"))
    f13 = 1.0 if regime_str == "trend_high_vol" else 0.0
    f14 = 1.0 if regime_str == "trend_low_vol" else 0.0
    f15 = 1.0 if regime_str == "range_high_vol" else 0.0
    f16 = 1.0 if regime_str == "range_low_vol" else 0.0
    f17 = 1.0 if regime_str == "chop" else 0.0

    # ── time features ─────────────────────────────────────
    ts: datetime = signal.ts
    hour_frac = ts.hour + ts.minute / 60.0
    wday = ts.weekday()
    f18 = math.sin(2 * math.pi * hour_frac / 24.0)
    f19 = math.cos(2 * math.pi * hour_frac / 24.0)
    f20 = math.sin(2 * math.pi * wday / 7.0)
    f21 = math.cos(2 * math.pi * wday / 7.0)

    # ── 4h ATR ratio ──────────────────────────────────────
    f22 = _safe_df_feature(_atr_ratio, df4h)

    # ── perp-native signals ────────────────────────────────
    # CVD ratio: 0.5 = neutral; > 0.55 buy-dominant; < 0.45 sell-dominant.
    # Falls back to 0.5 if CVDStore not wired or insufficient data.
    f23: float = 0.5
    if cvd_store is not None and cvd_store.has_data(exchange, signal.symbol, min_bars=3):
        f23 = cvd_store.cvd_ratio(exchange, signal.symbol, lookback=20)

    # OI ROC: normalised ΔOI; clamped to [-0.1, +0.1] for stability.
    f24: float = 0.0
    if oi_store is not None and oi_store.has_data(exchange, signal.symbol, min_samples=6):
        raw_roc = oi_store.oi_roc(exchange, signal.symbol, periods=5)
        f24 = max(-0.1, min(0.1, raw_roc))

    result = [
        f0, f1, f2, f3, f4, f5, f6,
        f7, f8, f9, f10,
        f11, f12,
        f13, f14, f15, f16, f17,
        f18, f19, f20, f21,
        f22,
        f23, f24,
    ]

    # ── cache store ───────────────────────────────────────
    if cache_key is not None:
        if len(_feature_cache) >= _FEATURE_CACHE_MAX:
            # Evict the oldest inserted key (dict preserves insertion order in 3.7+)
            oldest = next(iter(_feature_cache))
            del _feature_cache[oldest]
        _feature_cache[cache_key] = result

    return result
