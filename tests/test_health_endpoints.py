"""
Tests for FastAPI endpoints in src.monitoring.health.build_app().

Uses httpx.AsyncClient with ASGITransport to call the app in-process.
Storage is fully mocked — no real DB connection is made.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.monitoring.health import LiveBroadcaster, build_app


# ─────────────────────── mock helpers ────────────────────────────────────────


def _make_mock_conn(rows: list[Any] | None = None) -> MagicMock:
    """Return a mock asyncpg connection that returns `rows` for fetch/fetchval."""
    conn = AsyncMock()
    conn.fetch = AsyncMock(return_value=rows or [])
    conn.fetchval = AsyncMock(return_value=1)
    conn.execute = AsyncMock(return_value=None)
    return conn


def _make_mock_storage(rows: list[Any] | None = None) -> MagicMock:
    """
    Return a mock Storage whose pool.acquire() is a working async context manager.
    """
    conn = _make_mock_conn(rows)

    pool = MagicMock()

    @asynccontextmanager
    async def _acquire():
        yield conn

    pool.acquire = _acquire

    storage = MagicMock()
    storage.pool = pool
    return storage


def _make_mock_kill_switch(halted: bool = False) -> MagicMock:
    ks = MagicMock()
    ks.is_halted = halted
    return ks


def _make_app(
    rows: list[Any] | None = None,
    ml_filter: Any = None,
    broadcaster: Any = None,
    halted: bool = False,
) -> Any:
    storage = _make_mock_storage(rows)
    ks = _make_mock_kill_switch(halted)
    return build_app(
        connectors={},
        storage=storage,
        kill_switch=ks,
        ml_filter=ml_filter,
        broadcaster=broadcaster,
    )


# ─────────────────────── /health ─────────────────────────────────────────────


async def test_health_returns_ok() -> None:
    """GET /health must return HTTP 200 with body {"status": "ok"}."""
    app = _make_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# ─────────────────────── /app ────────────────────────────────────────────────


async def test_app_returns_html() -> None:
    """GET /app must return 200 HTML containing 'Cryprobotik Live'."""
    app = _make_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/app")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cryprobotik Live" in response.text


# ─────────────────────── /api/positions ──────────────────────────────────────


async def test_api_positions_empty() -> None:
    """GET /api/positions returns [] when the DB has no open positions."""
    app = _make_app(rows=[])
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/positions")

    assert response.status_code == 200
    assert response.json() == []


async def test_api_positions_db_error_returns_empty() -> None:
    """GET /api/positions must return [] (not 500) if the DB raises."""
    storage = _make_mock_storage()

    # Make acquire() raise so the endpoint catches and returns [].
    pool = MagicMock()

    @asynccontextmanager
    async def _failing_acquire():
        raise RuntimeError("db gone")
        yield  # pragma: no cover

    pool.acquire = _failing_acquire
    storage.pool = pool

    ks = _make_mock_kill_switch()
    app = build_app(connectors={}, storage=storage, kill_switch=ks)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/positions")

    assert response.status_code == 200
    assert response.json() == []


# ─────────────────────── /ml/stats ───────────────────────────────────────────


async def test_ml_stats_no_filter() -> None:
    """GET /ml/stats with ml_filter=None returns a dict with an 'error' key."""
    app = _make_app(ml_filter=None)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ml/stats")

    assert response.status_code == 200
    body = response.json()
    assert "error" in body


async def test_ml_stats_with_filter() -> None:
    """GET /ml/stats returns the filter's stats() dict when ml_filter is set."""
    fake_stats = {
        "model_version": 3,
        "n_samples": 120,
        "n_profitable": 70,
        "win_rate": 0.583,
        "cold_start": False,
        "accept_threshold": 0.6,
        "feature_importances": None,
        "pending_positions": 0,
    }
    ml_filter = MagicMock()
    ml_filter.stats = MagicMock(return_value=fake_stats)

    app = _make_app(ml_filter=ml_filter)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ml/stats")

    assert response.status_code == 200
    assert response.json()["model_version"] == 3


# ─────────────────────── /ml/stream ──────────────────────────────────────────


async def test_ml_stream_ping() -> None:
    """GET /ml/stream returns text/event-stream content-type."""
    # Without ml_filter the endpoint yields a single empty SSE event then keeps
    # streaming, so we read just the headers.
    app = _make_app(ml_filter=None)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        async with client.stream("GET", "/ml/stream") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]


# ─────────────────────── /ml/dashboard ───────────────────────────────────────


async def test_ml_dashboard_returns_html() -> None:
    """GET /ml/dashboard returns 200 with HTML dashboard content."""
    app = _make_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/ml/dashboard")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Cryprobotik" in response.text
