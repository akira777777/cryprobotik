"""
Prometheus metrics registry.

Exposed via FastAPI at /metrics in monitoring/health.py. All metrics are
process-level — the orchestrator updates them directly on events.
"""

from __future__ import annotations

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    CollectorRegistry,
    ProcessCollector,
    PlatformCollector,
    GCCollector,
)

# Use a private registry so the default one (which third-party libs may pollute)
# doesn't end up on /metrics.
REGISTRY = CollectorRegistry()
ProcessCollector(registry=REGISTRY)
PlatformCollector(registry=REGISTRY)
GCCollector(registry=REGISTRY)

# ─────────────────────── counters ───────────────────────
orders_placed_total = Counter(
    "cryprobotik_orders_placed_total",
    "Total orders sent to exchanges",
    ["exchange", "symbol", "side", "status"],
    registry=REGISTRY,
)
orders_filled_total = Counter(
    "cryprobotik_orders_filled_total",
    "Total fills received",
    ["exchange", "symbol", "side"],
    registry=REGISTRY,
)
orders_rejected_total = Counter(
    "cryprobotik_orders_rejected_total",
    "Total orders rejected by exchange or risk manager",
    ["reason"],
    registry=REGISTRY,
)
signals_emitted_total = Counter(
    "cryprobotik_signals_emitted_total",
    "Total signals produced by strategies",
    ["strategy", "side", "regime"],
    registry=REGISTRY,
)
ws_reconnects_total = Counter(
    "cryprobotik_ws_reconnects_total",
    "Total WebSocket reconnections",
    ["exchange", "channel"],
    registry=REGISTRY,
)

# ─────────────────────── gauges ───────────────────────
equity_gauge = Gauge(
    "cryprobotik_equity_usd",
    "Total account equity across all exchanges",
    ["mode"],
    registry=REGISTRY,
)
open_positions_gauge = Gauge(
    "cryprobotik_open_positions",
    "Number of currently open positions",
    registry=REGISTRY,
)
drawdown_gauge = Gauge(
    "cryprobotik_drawdown_pct",
    "Current drawdown from day start",
    registry=REGISTRY,
)
halt_state_gauge = Gauge(
    "cryprobotik_halt_state",
    "Kill switch state (0=running, 1=warning, 2=halted)",
    registry=REGISTRY,
)
ws_connected_gauge = Gauge(
    "cryprobotik_ws_connected",
    "WebSocket connection state (1=connected, 0=disconnected)",
    ["exchange", "channel"],
    registry=REGISTRY,
)

# ─────────────────────── histograms ───────────────────────
order_latency_seconds = Histogram(
    "cryprobotik_order_latency_seconds",
    "Wall time from place_order call to exchange acknowledgement",
    ["exchange"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)
signal_to_fill_seconds = Histogram(
    "cryprobotik_signal_to_fill_seconds",
    "Wall time from signal emit to first fill event",
    ["strategy"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
    registry=REGISTRY,
)

# ─────────────────────── additional gauges ───────────────────────
last_bar_age_gauge = Gauge(
    "cryprobotik_last_bar_timestamp_seconds",
    "Unix timestamp of the last closed bar received (derive staleness with time() - metric)",
    ["exchange", "symbol", "timeframe"],
    registry=REGISTRY,
)
ml_training_samples_gauge = Gauge(
    "cryprobotik_ml_training_samples",
    "Number of labeled training examples in ML buffer",
    registry=REGISTRY,
)
ml_model_version_gauge = Gauge(
    "cryprobotik_ml_model_version",
    "Current ML model version",
    registry=REGISTRY,
)

# ─────────────────────── additional counters ───────────────────────
telegram_send_failures_total = Counter(
    "cryprobotik_telegram_send_failures_total",
    "Total Telegram notification failures",
    ["reason"],
    registry=REGISTRY,
)

# ─────────────────────── exit manager counters ───────────────────────
exits_breakeven_total = Counter(
    "cryprobotik_exits_breakeven_total",
    "Times SL was moved to entry price (breakeven)",
    ["exchange"],
    registry=REGISTRY,
)
exits_trailed_total = Counter(
    "cryprobotik_exits_trailed_total",
    "Times trailing SL was updated after +2R",
    ["exchange"],
    registry=REGISTRY,
)
exits_partial_tp_total = Counter(
    "cryprobotik_exits_partial_tp_total",
    "Times 50% of a position was closed at +1.5R",
    ["exchange"],
    registry=REGISTRY,
)
exits_time_stop_total = Counter(
    "cryprobotik_exits_time_stop_total",
    "Times a position was closed by the time-based exit rule",
    ["exchange"],
    registry=REGISTRY,
)
