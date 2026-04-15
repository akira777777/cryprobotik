"""
Structured JSON logging with structlog.

All events are emitted as JSON to stdout (picked up by Docker's json-file driver).
Trade and order events carry correlation IDs so a single trade can be followed
from signal → routing → fill → PnL across the log stream.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

_configured: bool = False


def _drop_color_message_key(
    _: logging.Logger, __: str, event_dict: EventDict
) -> EventDict:
    """Uvicorn duplicates the formatted message under `color_message`. Drop it."""
    event_dict.pop("color_message", None)
    return event_dict


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    """
    Install structlog as the root logging backend.

    Args:
        level: one of DEBUG, INFO, WARN, WARNING, ERROR, CRITICAL.
        fmt:   'json' for production, 'console' for local dev.
    """
    global _configured
    if _configured:
        return

    level_num = logging.getLevelName(level.upper())
    if not isinstance(level_num, int):
        level_num = logging.INFO

    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _drop_color_message_key,
    ]

    if fmt == "json":
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    # Configure structlog's own loggers
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level_num),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Route stdlib logging (used by ccxt, asyncio, uvicorn, telegram, etc) through
    # structlog so everything comes out as JSON.
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    # Remove any pre-existing handlers (basicConfig from imports, etc.)
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level_num)

    # Quiet the noisiest third-party loggers by a level.
    for noisy in ("websockets.client", "websockets.protocol", "httpx", "httpcore", "ccxt"):
        logging.getLogger(noisy).setLevel(max(level_num, logging.WARNING))

    _configured = True


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger with bound initial context."""
    log = structlog.get_logger(name)
    if initial_context:
        log = log.bind(**initial_context)
    return log  # type: ignore[return-value]


def bind_trade_context(trade_id: str, symbol: str, strategy: str) -> None:
    """Bind trade-scoped context to the current async task via contextvars."""
    structlog.contextvars.bind_contextvars(
        trade_id=trade_id, symbol=symbol, strategy=strategy
    )


def clear_trade_context() -> None:
    structlog.contextvars.clear_contextvars()
