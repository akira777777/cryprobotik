"""
Generic WebSocket connection manager.

Handles:
- connection lifecycle with auto-reconnect
- jittered exponential backoff on failures
- heartbeat / ping scheduling
- subscription state tracking so reconnect can re-subscribe

Exchange-specific code (OKX, Bybit) wraps this manager with a message builder
(`build_subscribe_msg`) and a message parser (`handle_message`) — the manager
owns the socket, the wrappers own the protocol semantics.
"""

from __future__ import annotations

import asyncio
import json
import random
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import ConnectionClosed

from src.utils.logging import get_logger
from src.utils.time import now_utc

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Subscription:
    """A logical subscription. Opaque key; its meaning is exchange-specific."""

    key: str  # e.g. "kline:BTC/USDT:USDT:15m"
    payload: dict[str, Any]  # the subscribe message body to resend on reconnect


MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]
SubscribeMsgBuilder = Callable[[dict[str, Any]], dict[str, Any]]


@dataclass
class WSManagerConfig:
    url: str
    name: str  # free-text tag for logging/metrics
    ping_interval_sec: float = 20.0
    ping_timeout_sec: float = 10.0
    max_backoff_sec: float = 30.0
    initial_backoff_sec: float = 1.0
    # Called after the socket opens, before any subscribe messages. Use for login.
    on_connected: Callable[["WSManager"], Awaitable[None]] | None = None


class WSManager:
    """
    One persistent WebSocket connection with automatic reconnect.

    Usage:
        mgr = WSManager(cfg, handle_message=handle)
        await mgr.start()            # spawns the run loop
        await mgr.subscribe(sub)     # queued; will be sent on (re)connect
        ...
        await mgr.stop()
    """

    def __init__(self, config: WSManagerConfig, handle_message: MessageHandler) -> None:
        self._config = config
        self._handle = handle_message
        self._ws: ClientConnection | None = None
        self._subscriptions: dict[str, Subscription] = {}
        self._run_task: asyncio.Task[None] | None = None
        self._ping_task: asyncio.Task[None] | None = None
        self._connected = asyncio.Event()
        self._stopping = False
        self._reconnects = 0
        self._last_pong_ts: float = 0.0

    # ──────────── public API ────────────

    async def start(self) -> None:
        """Kick off the run loop as a background task."""
        if self._run_task and not self._run_task.done():
            return
        self._stopping = False
        self._run_task = asyncio.create_task(self._run_forever(), name=f"ws-{self._config.name}")

    async def stop(self) -> None:
        self._stopping = True
        if self._ping_task and not self._ping_task.done():
            self._ping_task.cancel()
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                # Expected during teardown: the socket may already be closed or
                # the network may be gone. Logging here would always fire on shutdown.
                pass
        if self._run_task and not self._run_task.done():
            self._run_task.cancel()
            try:
                await self._run_task
            except (asyncio.CancelledError, Exception):
                # CancelledError is the normal result of cancel(); any other exception
                # means _run_forever already exited with an error (already logged there).
                pass
        log.info("ws.stopped", name=self._config.name)

    async def send(self, msg: dict[str, Any]) -> None:
        """Send a message; raises if not connected."""
        if self._ws is None:
            raise RuntimeError(f"WS[{self._config.name}] not connected")
        await self._ws.send(json.dumps(msg))

    async def subscribe(self, sub: Subscription) -> None:
        """Register a subscription and send it if currently connected."""
        self._subscriptions[sub.key] = sub
        if self._connected.is_set() and self._ws is not None:
            try:
                await self.send(sub.payload)
            except Exception as e:
                log.warning("ws.subscribe_failed", name=self._config.name, key=sub.key, error=str(e))

    async def unsubscribe(self, key: str, unsub_payload: dict[str, Any]) -> None:
        """Forget a subscription and send an unsubscribe if possible."""
        self._subscriptions.pop(key, None)
        if self._connected.is_set() and self._ws is not None:
            try:
                await self.send(unsub_payload)
            except Exception:
                # Best-effort: the subscription is already removed from local state.
                # If the WS send fails the server will drop it on the next reconnect.
                pass

    async def wait_connected(self, timeout: float | None = None) -> bool:
        try:
            await asyncio.wait_for(self._connected.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    @property
    def reconnect_count(self) -> int:
        return self._reconnects

    # ──────────── run loop ────────────

    async def _run_forever(self) -> None:
        backoff = self._config.initial_backoff_sec
        while not self._stopping:
            try:
                await self._connect_and_read()
                backoff = self._config.initial_backoff_sec  # reset on clean exit
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("ws.connection_error", name=self._config.name, error=str(e), backoff=backoff)
            finally:
                self._connected.clear()
                if self._ping_task and not self._ping_task.done():
                    self._ping_task.cancel()
                from src.monitoring import prom_metrics as m

                m.ws_connected_gauge.labels(exchange=self._config.name, channel="public").set(0)
                m.ws_reconnects_total.labels(exchange=self._config.name, channel="public").inc()

            if self._stopping:
                break

            # Jittered exponential backoff before reconnecting.
            jitter = random.uniform(0, 0.3 * backoff)
            await asyncio.sleep(backoff + jitter)
            backoff = min(backoff * 2, self._config.max_backoff_sec)
            self._reconnects += 1

    async def _connect_and_read(self) -> None:
        log.info("ws.connecting", name=self._config.name, url=self._config.url)
        async with websockets.connect(
            self._config.url,
            ping_interval=None,  # we drive our own ping loop
            open_timeout=15,
            close_timeout=5,
            max_size=2**24,
        ) as ws:
            self._ws = ws
            self._connected.set()
            self._last_pong_ts = now_utc().timestamp()
            log.info("ws.connected", name=self._config.name)
            from src.monitoring import prom_metrics as m

            m.ws_connected_gauge.labels(exchange=self._config.name, channel="public").set(1)

            # Hook for exchange-specific login
            if self._config.on_connected is not None:
                try:
                    await self._config.on_connected(self)
                except Exception as e:
                    log.error("ws.on_connected_failed", name=self._config.name, error=str(e))
                    raise

            # Resubscribe all known subscriptions.
            for sub in self._subscriptions.values():
                try:
                    await self.send(sub.payload)
                except Exception as e:
                    log.warning("ws.resubscribe_failed", name=self._config.name, key=sub.key, error=str(e))

            # Kick off the ping task
            self._ping_task = asyncio.create_task(self._ping_loop(), name=f"ws-ping-{self._config.name}")

            # Read loop
            try:
                async for raw in ws:
                    if isinstance(raw, bytes):
                        raw = raw.decode("utf-8")
                    # Plain string "pong" is used by OKX — handle out of band.
                    if raw == "pong":
                        self._last_pong_ts = now_utc().timestamp()
                        continue
                    try:
                        msg = json.loads(raw)
                    except json.JSONDecodeError:
                        log.warning("ws.invalid_json", name=self._config.name, raw=raw[:200])
                        continue
                    try:
                        await self._handle(msg)
                    except Exception as e:
                        # Do NOT spread msg — may contain private channel data.
                        log.error("ws.handler_failed", name=self._config.name, error=str(e), exc_info=True)
            except ConnectionClosed as e:
                log.info("ws.closed", name=self._config.name, code=e.code, reason=e.reason)

    async def _ping_loop(self) -> None:
        """
        Send a text "ping" every interval. OKX requires literal "ping"/"pong"
        strings at the app layer; Bybit accepts {"op":"ping"}. Exchange-specific
        overrides are applied via `send_ping_payload` if set.

        Any exception (including ConnectionClosed) forcefully closes the socket so
        that _run_forever's reconnect logic is triggered — a silent return here
        would leave the socket "connected" with no heartbeats.
        """
        while True:
            try:
                await asyncio.sleep(self._config.ping_interval_sec)
                if self._ws is None:
                    return
                # Default ping: send both a plain "ping" text and a JSON op=ping.
                # Exchange code overrides _send_ping if needed.
                await self._send_ping()
                # Reset last_pong_ts right after a successful ping so the stale
                # check doesn't false-fire immediately after reconnect.
                self._last_pong_ts = now_utc().timestamp()
                # Detect stale connection: no pong for 2x ping interval → kill it.
                stale = now_utc().timestamp() - self._last_pong_ts > 2 * self._config.ping_interval_sec
                if stale:
                    log.warning("ws.stale_detected", name=self._config.name)
                    try:
                        await self._ws.close(code=1011, reason="stale")
                    except Exception:
                        # Close may fail if the socket is already gone — that's exactly
                        # why it was detected as stale. _run_forever handles reconnect.
                        pass
                    return  # triggers reconnect via _run_forever
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.warning(
                    "ws.ping_failed_forcing_reconnect",
                    name=self._config.name,
                    error=str(e),
                )
                # Force reconnect — do NOT silently return without closing.
                try:
                    if self._ws is not None:
                        await self._ws.close(code=1011, reason="ping_failed")
                except Exception:
                    # Socket may already be gone if ping failed due to connection drop.
                    # _run_forever handles reconnect regardless.
                    pass
                return  # triggers reconnect via _run_forever

    async def _send_ping(self) -> None:
        """Default ping — subclasses / configs can replace via monkey-patching or by
        sending their own op=ping messages through send()."""
        if self._ws is None:
            return
        await self._ws.send("ping")

    def mark_pong(self) -> None:
        """Called by message handlers when an exchange's pong message arrives."""
        self._last_pong_ts = now_utc().timestamp()
