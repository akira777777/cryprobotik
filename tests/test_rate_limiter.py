"""Token-bucket rate limiter tests."""

from __future__ import annotations

import asyncio

import pytest

from src.execution.rate_limiter import RateLimiter


def test_rejects_non_positive_refill() -> None:
    with pytest.raises(ValueError, match="refill_per_sec"):
        RateLimiter(refill_per_sec=0.0)
    with pytest.raises(ValueError, match="refill_per_sec"):
        RateLimiter(refill_per_sec=-1.0)


async def test_initial_capacity_grants_burst() -> None:
    """With capacity=5, the first 5 acquires must be effectively instant."""
    limiter = RateLimiter(refill_per_sec=1.0, capacity=5)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    for _ in range(5):
        await limiter.acquire()
    elapsed = loop.time() - t0
    # Should be basically instant — allow generous slack for CI jitter
    assert elapsed < 0.15


async def test_blocks_after_burst() -> None:
    """
    Acquiring one more token than the bucket holds must block until the
    bucket refills.
    """
    limiter = RateLimiter(refill_per_sec=10.0, capacity=2)
    await limiter.acquire()
    await limiter.acquire()

    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await limiter.acquire()  # bucket empty → must wait ~0.1s
    elapsed = loop.time() - t0
    # At 10/sec, one token takes 0.1s. Allow jitter in both directions.
    assert elapsed >= 0.05
    assert elapsed < 0.5


async def test_oversize_request_raises() -> None:
    limiter = RateLimiter(refill_per_sec=10.0, capacity=5)
    with pytest.raises(ValueError, match="capacity"):
        await limiter.acquire(tokens=10)


async def test_context_manager_consumes_token() -> None:
    limiter = RateLimiter(refill_per_sec=1.0, capacity=1)
    async with limiter:
        pass
    # Bucket is now empty — the next acquire must wait ~1s.
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await limiter.acquire()
    assert loop.time() - t0 >= 0.5


async def test_refill_continues_over_time() -> None:
    """After a wait the bucket refills and a subsequent acquire is instant."""
    limiter = RateLimiter(refill_per_sec=20.0, capacity=1)
    await limiter.acquire()
    # Wait longer than the refill interval (1/20s = 50ms) so the bucket refills.
    await asyncio.sleep(0.2)
    loop = asyncio.get_running_loop()
    t0 = loop.time()
    await limiter.acquire()
    assert loop.time() - t0 < 0.05
