"""Tests for LiveBroadcaster in src.monitoring.health."""

from __future__ import annotations

import asyncio

from src.monitoring.health import LiveBroadcaster


async def test_subscribe_returns_queue() -> None:
    """subscribe() must return an asyncio.Queue."""
    broadcaster = LiveBroadcaster()
    q = broadcaster.subscribe()
    assert isinstance(q, asyncio.Queue)


async def test_push_delivers_to_subscribers() -> None:
    """push() puts JSON-encoded event into every subscriber queue."""
    broadcaster = LiveBroadcaster()
    q1 = broadcaster.subscribe()
    q2 = broadcaster.subscribe()

    event = {"type": "signal", "symbol": "BTC/USDT"}
    broadcaster.push(event)

    import json

    payload1 = q1.get_nowait()
    payload2 = q2.get_nowait()

    assert json.loads(payload1) == event
    assert json.loads(payload2) == event


async def test_push_drops_when_queue_full() -> None:
    """When a subscriber queue is full, push() drops silently without raising."""
    broadcaster = LiveBroadcaster()

    # Override the subscriber's queue with a tiny maxsize=1 queue.
    small_q: asyncio.Queue[str] = asyncio.Queue(maxsize=1)
    broadcaster._subscribers.add(small_q)

    # Fill it so the next put_nowait would raise QueueFull.
    small_q.put_nowait("existing_item")

    # This must not raise.
    broadcaster.push({"type": "signal"})

    # The queue still holds only the pre-existing item.
    assert small_q.qsize() == 1
    assert small_q.get_nowait() == "existing_item"


async def test_unsubscribe_removes_subscriber() -> None:
    """After unsubscribe(), push() no longer delivers to that queue."""
    broadcaster = LiveBroadcaster()
    q = broadcaster.subscribe()
    broadcaster.unsubscribe(q)

    broadcaster.push({"type": "signal"})

    assert q.empty()


async def test_push_multiple_subscribers() -> None:
    """push() delivers to N simultaneous subscribers."""
    import json

    broadcaster = LiveBroadcaster()
    n = 5
    queues = [broadcaster.subscribe() for _ in range(n)]

    event = {"type": "position", "symbol": "ETH/USDT", "upnl": 42.0}
    broadcaster.push(event)

    for q in queues:
        assert not q.empty()
        assert json.loads(q.get_nowait()) == event
