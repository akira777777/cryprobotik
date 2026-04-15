"""
Daily drawdown kill switch.

The kill switch is the last line of defense. It:
1. Records the starting-of-UTC-day equity.
2. On every equity update, computes current drawdown from that anchor.
3. If drawdown > max_daily_drawdown_pct → emits HALT.
4. The HALT signal persists to bot_state so a crash+restart does NOT clear it.
5. On HALT, the orchestrator cancels all open orders, flattens all positions,
   refuses to open new ones, and sends a CRITICAL Telegram alert.
6. Clearing the halt requires running `scripts/reset_halt.py` manually.

Warning thresholds (e.g. 15% DD) emit a WARN Telegram alert without halting.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Callable, Awaitable

from src.utils.logging import get_logger
from src.utils.time import now_utc, start_of_utc_day

if TYPE_CHECKING:
    from src.data.storage import Storage
    from src.settings import RiskConfig

log = get_logger(__name__)

BOT_STATE_KEY = "kill_switch.state"


class HaltState(StrEnum):
    RUNNING = "running"
    WARNING = "warning"
    HALTED = "halted"


@dataclass
class KillSwitchState:
    state: HaltState = HaltState.RUNNING
    day_ts_ms: int = 0
    day_start_equity: float = 0.0
    lowest_equity_today: float = 0.0
    peak_drawdown_today: float = 0.0
    intraday_peak_equity: float = 0.0  # highest equity seen today (for peak-DD calc)
    last_halt_reason: str | None = None
    last_halt_ts_ms: int = 0


HaltCallback = Callable[[str, float], Awaitable[None]]
"""Signature: async def on_halt(reason: str, drawdown_pct: float) -> None"""

WarningCallback = Callable[[float], Awaitable[None]]
"""Signature: async def on_warning(drawdown_pct: float) -> None"""


class KillSwitch:
    def __init__(
        self,
        config: "RiskConfig",
        storage: "Storage",
        on_halt: HaltCallback | None = None,
        on_warning: WarningCallback | None = None,
    ) -> None:
        self._config = config
        self._storage = storage
        self._state = KillSwitchState()
        self._on_halt = on_halt
        self._on_warning = on_warning
        self._warned_today = False
        self._lock = asyncio.Lock()

    # ─────────────────────── lifecycle ───────────────────────

    async def load(self) -> None:
        """Called on startup. Restores persisted state from DB if present."""
        raw = await self._storage.get_state(BOT_STATE_KEY)
        if raw:
            self._state = KillSwitchState(
                state=HaltState(raw.get("state", "running")),
                day_ts_ms=int(raw.get("day_ts_ms", 0)),
                day_start_equity=float(raw.get("day_start_equity", 0.0)),
                lowest_equity_today=float(raw.get("lowest_equity_today", 0.0)),
                peak_drawdown_today=float(raw.get("peak_drawdown_today", 0.0)),
                intraday_peak_equity=float(raw.get("intraday_peak_equity", 0.0)),
                last_halt_reason=raw.get("last_halt_reason"),
                last_halt_ts_ms=int(raw.get("last_halt_ts_ms", 0)),
            )
            log.info("kill_switch.loaded", **self._state_dict())
        else:
            log.info("kill_switch.fresh_state")

    async def _persist(self) -> None:
        await self._storage.set_state(BOT_STATE_KEY, self._state_dict())

    def _state_dict(self) -> dict[str, object]:
        return {
            "state": self._state.state.value,
            "day_ts_ms": self._state.day_ts_ms,
            "day_start_equity": self._state.day_start_equity,
            "lowest_equity_today": self._state.lowest_equity_today,
            "peak_drawdown_today": self._state.peak_drawdown_today,
            "intraday_peak_equity": self._state.intraday_peak_equity,
            "last_halt_reason": self._state.last_halt_reason,
            "last_halt_ts_ms": self._state.last_halt_ts_ms,
        }

    # ─────────────────────── inspection ───────────────────────

    @property
    def is_halted(self) -> bool:
        return self._state.state == HaltState.HALTED

    @property
    def is_warning(self) -> bool:
        return self._state.state == HaltState.WARNING

    @property
    def state(self) -> HaltState:
        return self._state.state

    def current_drawdown(self, equity: float) -> float:
        """
        Return the worst drawdown between two anchors:
        1. Drawdown from the UTC-day start equity.
        2. Drawdown from the intraday equity peak (covers intraday run-up + reversal).
        """
        if self._state.day_start_equity <= 0:
            return 0.0
        dd_from_start = max(
            0.0,
            (self._state.day_start_equity - equity) / self._state.day_start_equity,
        )
        if self._state.intraday_peak_equity > 0:
            dd_from_peak = max(
                0.0,
                (self._state.intraday_peak_equity - equity) / self._state.intraday_peak_equity,
            )
        else:
            dd_from_peak = 0.0
        return max(dd_from_start, dd_from_peak)

    # ─────────────────────── main check ───────────────────────

    async def on_equity_update(self, equity: float, ts: datetime | None = None) -> None:
        """
        Call this on every equity change. Idempotent and safe to call at any
        frequency. Halting is triggered only ONCE per day.
        """
        warning_cb: WarningCallback | None = None
        halt_cb: HaltCallback | None = None
        halt_reason: str = ""
        dd: float = 0.0

        async with self._lock:
            ts = ts or now_utc()

            # Roll over at UTC midnight.
            day_start = start_of_utc_day(ts)
            day_ms = int(day_start.timestamp() * 1000)
            if self._state.day_ts_ms != day_ms:
                await self._roll_day(day_ms, equity)

            # Track daily lowest + intraday peak equity
            if equity < self._state.lowest_equity_today or self._state.lowest_equity_today == 0:
                self._state.lowest_equity_today = equity
            if equity > self._state.intraday_peak_equity:
                self._state.intraday_peak_equity = equity
            dd = self.current_drawdown(equity)
            if dd > self._state.peak_drawdown_today:
                self._state.peak_drawdown_today = dd

            if self._state.state == HaltState.HALTED:
                # Already halted — nothing else to do.
                return

            # Determine what callbacks to fire (do NOT await inside the lock —
            # Telegram / other async I/O can block indefinitely).

            # Warning level
            if (
                dd >= self._config.warning_drawdown_pct
                and not self._warned_today
                and self._state.state == HaltState.RUNNING
            ):
                self._state.state = HaltState.WARNING
                self._warned_today = True
                log.warning("kill_switch.warning", drawdown_pct=dd, equity=equity)
                warning_cb = self._on_warning

            # Halt level
            if dd >= self._config.max_daily_drawdown_pct:
                self._state.state = HaltState.HALTED
                halt_reason = f"daily_drawdown={dd:.4f}"
                self._state.last_halt_reason = halt_reason
                self._state.last_halt_ts_ms = int(ts.timestamp() * 1000)
                log.error(
                    "kill_switch.HALT",
                    drawdown_pct=dd,
                    equity=equity,
                    day_start=self._state.day_start_equity,
                )
                halt_cb = self._on_halt

            await self._persist()

        # ── callbacks OUTSIDE the lock to avoid Telegram blocking other equity updates ──
        if halt_cb is not None:
            try:
                await halt_cb(halt_reason, dd)
            except Exception as e:
                log.error("kill_switch.halt_callback_failed", error=str(e))
            return
        if warning_cb is not None:
            try:
                await warning_cb(dd)
            except Exception as e:
                log.error("kill_switch.warning_callback_failed", error=str(e))

    async def _roll_day(self, day_ms: int, equity: float) -> None:
        log.info(
            "kill_switch.day_roll",
            prev_day_start_equity=self._state.day_start_equity,
            prev_peak_dd=self._state.peak_drawdown_today,
            new_day_start_equity=equity,
        )
        self._state.day_ts_ms = day_ms
        self._state.day_start_equity = equity
        self._state.lowest_equity_today = equity
        self._state.intraday_peak_equity = equity
        self._state.peak_drawdown_today = 0.0
        self._warned_today = False
        # Do NOT auto-clear a HALT state across day boundaries — halting is
        # sticky by design and requires manual reset.
        await self._persist()

    # ─────────────────────── manual control ───────────────────────

    async def force_halt(self, reason: str) -> None:
        """Programmatic halt — e.g. from a Telegram /halt command."""
        async with self._lock:
            self._state.state = HaltState.HALTED
            self._state.last_halt_reason = reason
            self._state.last_halt_ts_ms = int(now_utc().timestamp() * 1000)
            log.error("kill_switch.force_halt", reason=reason)
            await self._persist()
            if self._on_halt is not None:
                try:
                    await self._on_halt(reason, self._state.peak_drawdown_today)
                except Exception as e:
                    log.error("kill_switch.halt_callback_failed", error=str(e))

    async def reset(self) -> None:
        """
        Manually clear a HALT state. Intended for scripts/reset_halt.py — do NOT
        wire this to any automated path.
        """
        async with self._lock:
            log.warning("kill_switch.reset", previous=self._state.state.value)
            self._state.state = HaltState.RUNNING
            self._state.last_halt_reason = None
            self._state.last_halt_ts_ms = 0
            self._warned_today = False
            await self._persist()
