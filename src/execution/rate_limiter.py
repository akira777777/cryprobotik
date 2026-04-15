"""
Token-bucket rate limiter.

ccxt's REST client has its own sleep-based rate limit handler but it's global
to the client. We layer a finer-grained limiter in front of it so that bursty
public calls (e.g. universe refresh) don't eat into the private/trade budget.

Usage:
    limiter = RateLimiter(refill_per_sec=10, capacity=20)
    async with limiter:
        await rest.create_order(...)
"""

from __future__ import annotations

import asyncio
from typing import Self


class RateLimiter:
    """Classic token bucket. Thread-safe via asyncio.Lock."""

    def __init__(self, refill_per_sec: float, capacity: float | None = None) -> None:
        if refill_per_sec <= 0:
            raise ValueError("refill_per_sec must be > 0")
        self._refill_per_sec = refill_per_sec
        # Cap default capacity at 1 second of tokens to avoid burst triggering
        # exchange sliding-window rate limits after quiet periods.
        self._capacity = capacity if capacity is not None else refill_per_sec
        self._tokens = self._capacity
        self._last_refill: float = 0.0  # set on first acquire() inside event loop
        self._lock = asyncio.Lock()

    async def acquire(self, tokens: float = 1.0) -> None:
        """Block until `tokens` are available, then consume them."""
        if tokens > self._capacity:
            raise ValueError(
                f"requested tokens {tokens} > bucket capacity {self._capacity}"
            )
        # Release lock before sleeping so other waiters can enter and consume
        # tokens that have refilled while we sleep.
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= tokens:
                    self._tokens -= tokens
                    return
                needed = tokens - self._tokens
                wait = needed / self._refill_per_sec
            # Sleep OUTSIDE the lock — other callers can now acquire tokens.
            await asyncio.sleep(wait)

    def _refill(self) -> None:
        now = asyncio.get_running_loop().time()
        if self._last_refill == 0.0:
            self._last_refill = now
            return
        elapsed = now - self._last_refill
        if elapsed > 0:
            self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_sec)
            self._last_refill = now

    async def __aenter__(self) -> Self:
        await self.acquire()
        return self

    async def __aexit__(self, *_: object) -> None:
        return None
