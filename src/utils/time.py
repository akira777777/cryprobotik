"""
Time helpers. All internal timestamps are UTC; exchange timestamps are normalized
at the connector boundary so nothing downstream ever sees a tz-naive datetime.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

# Timeframe strings used across strategies and connectors.
# Maps to seconds for bar-alignment and windowing math.
TIMEFRAME_SECONDS: dict[str, int] = {
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def now_utc() -> datetime:
    """Return current UTC time, tz-aware."""
    return datetime.now(UTC)


def now_ms() -> int:
    """Return current UTC time as Unix milliseconds."""
    return int(now_utc().timestamp() * 1000)


def ms_to_datetime(ms: int | float) -> datetime:
    """Convert Unix milliseconds to tz-aware UTC datetime."""
    return datetime.fromtimestamp(float(ms) / 1000.0, tz=UTC)


def datetime_to_ms(dt: datetime) -> int:
    """Convert a datetime (assumed UTC if tz-naive) to Unix milliseconds."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def timeframe_to_seconds(tf: str) -> int:
    """Parse a timeframe string to seconds. Raises ValueError on unknown timeframes."""
    if tf not in TIMEFRAME_SECONDS:
        raise ValueError(f"unsupported timeframe: {tf}")
    return TIMEFRAME_SECONDS[tf]


def align_to_timeframe(ts: datetime, tf: str) -> datetime:
    """Floor a timestamp to the start of its timeframe bucket."""
    secs = timeframe_to_seconds(tf)
    epoch = int(ts.timestamp())
    aligned = epoch - (epoch % secs)
    return datetime.fromtimestamp(aligned, tz=UTC)


def start_of_utc_day(ts: datetime | None = None) -> datetime:
    """Return midnight UTC for the given timestamp (default: now)."""
    ts = ts or now_utc()
    return ts.replace(hour=0, minute=0, second=0, microsecond=0)


def seconds_until_next_utc_hour(hour: int, ts: datetime | None = None) -> float:
    """Seconds from `ts` until the next occurrence of `hour:00:00` UTC."""
    ts = ts or now_utc()
    target = ts.replace(hour=hour, minute=0, second=0, microsecond=0)
    if target <= ts:
        target += timedelta(days=1)
    return (target - ts).total_seconds()
