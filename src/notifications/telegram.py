"""
Telegram notifications.

Thin wrapper around python-telegram-bot v21. Provides:
    - notify(level, text)   — fire-and-forget messaging
    - /status, /positions, /pnl, /halt, /resume command handlers
    - /minimap — live auto-refreshing dashboard (edited every 60s)
    - daily report scheduling at configured UTC hour

All messages are HTML-escaped and rate-limited by the library's built-in
rate limiter plugin (we enable it via the [rate-limiter] extra in requirements).
"""

from __future__ import annotations

import asyncio
import html
from collections import deque
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from src.utils.logging import get_logger
from src.utils.time import now_utc, seconds_until_next_utc_hour

if TYPE_CHECKING:
    from src.ml.model import MLSignalFilter
    from src.portfolio.analytics import Analytics
    from src.portfolio.tracker import PortfolioTracker
    from src.risk.kill_switch import KillSwitch
    from src.settings import NotificationsConfig, Settings

log = get_logger(__name__)

# How often (seconds) the minimap message is refreshed in Telegram.
MINIMAP_REFRESH_SEC: int = 60
# Maximum signal lines shown in the minimap feed.
MINIMAP_FEED_SIZE: int = 7


class _DummyBot:
    """Used when telegram is disabled or token is blank."""
    async def send_message(self, *_: object, **__: object) -> None:
        return None

    async def edit_message_text(self, *_: object, **__: object) -> None:
        return None

    async def pin_chat_message(self, *_: object, **__: object) -> None:
        return None


class TelegramNotifier:
    def __init__(
        self,
        settings: "Settings",
        notif_config: "NotificationsConfig",
        tracker: "PortfolioTracker | None" = None,
        analytics: "Analytics | None" = None,
        kill_switch: "KillSwitch | None" = None,
        ml_filter: "MLSignalFilter | None" = None,
        get_regimes: "Callable[[], dict[str, str]] | None" = None,
        miniapp_url: str = "",
    ) -> None:
        self._settings = settings
        self._notif_config = notif_config
        self._tracker = tracker
        self._analytics = analytics
        self._kill_switch = kill_switch
        self._ml_filter = ml_filter
        self._get_regimes = get_regimes  # () → {symbol: regime_str}
        self._miniapp_url = miniapp_url  # URL for the Telegram Mini App

        self._bot: object = _DummyBot()
        self._app: object | None = None  # Application from python-telegram-bot
        self._enabled = False
        self._chat_ids: list[int] = []
        self._levels: set[str] = set()
        self._commands_enabled = False
        self._daily_report_task: asyncio.Task[None] | None = None

        # Minimap state — one pinned message per chat
        self._minimap_active: bool = False
        self._minimap_message_ids: dict[int, int] = {}   # chat_id → message_id
        self._minimap_task: asyncio.Task[None] | None = None
        # Rolling feed: last MINIMAP_FEED_SIZE pre-formatted HTML lines
        self._signal_feed: deque[str] = deque(maxlen=MINIMAP_FEED_SIZE)
        # Running today stats (reset at daily report time)
        self._today_trades: int = 0
        self._today_wins: int = 0
        self._today_losses: int = 0
        self._today_pnl: float = 0.0

    async def start(self) -> None:
        """Initialize the Telegram bot (lazy import to keep the module light)."""
        if not self._notif_config.telegram.enabled:
            log.info("telegram.disabled")
            return
        token = self._settings.telegram_bot_token.get_secret_value()
        if not token:
            log.warning("telegram.no_token_configured")
            return
        self._chat_ids = self._settings.telegram_chat_ids
        if not self._chat_ids:
            log.warning("telegram.no_chat_ids_configured")
            return

        # Lazy import so the whole module can be imported without the library.
        from telegram.ext import Application, CommandHandler  # type: ignore[import-not-found]

        self._app = Application.builder().token(token).build()
        self._bot = self._app.bot

        self._levels = set(lvl.upper() for lvl in self._notif_config.telegram.levels)
        self._commands_enabled = self._notif_config.telegram.enable_commands

        if self._commands_enabled:
            self._app.add_handler(CommandHandler("status", self._cmd_status))
            self._app.add_handler(CommandHandler("positions", self._cmd_positions))
            self._app.add_handler(CommandHandler("pnl", self._cmd_pnl))
            self._app.add_handler(CommandHandler("halt", self._cmd_halt))
            self._app.add_handler(CommandHandler("resume", self._cmd_resume))
            self._app.add_handler(CommandHandler("minimap", self._cmd_minimap))
            self._app.add_handler(CommandHandler("webapp", self._cmd_webapp))

        await self._app.initialize()
        await self._app.start()
        await self._app.updater.start_polling(drop_pending_updates=True)

        self._enabled = True
        self._daily_report_task = asyncio.create_task(
            self._run_daily_report_scheduler(), name="telegram-daily-report"
        )
        log.info("telegram.started", chat_ids=self._chat_ids)
        await self.notify("INFO", "🚀 Cryprobotik started")

    async def stop(self) -> None:
        if self._minimap_task:
            self._minimap_task.cancel()
            try:
                await self._minimap_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._daily_report_task:
            self._daily_report_task.cancel()
            try:
                await self._daily_report_task
            except (asyncio.CancelledError, Exception):
                pass
        if self._app is not None:
            try:
                await self._app.updater.stop()  # type: ignore[attr-defined]
                await self._app.stop()          # type: ignore[attr-defined]
                await self._app.shutdown()      # type: ignore[attr-defined]
            except Exception as e:
                log.warning("telegram.stop_failed", error=str(e))
        self._enabled = False

    # ─────────────────────── public setters for late injection ───────────────

    def set_ml_filter(self, ml_filter: "MLSignalFilter") -> None:
        self._ml_filter = ml_filter

    def set_regime_provider(self, fn: "Callable[[], dict[str, str]]") -> None:
        """Inject a callable that returns {symbol: regime_name} for the minimap."""
        self._get_regimes = fn

    # ─────────────────────── notify ───────────────────────

    async def notify(self, level: str, text: str) -> None:
        level = level.upper()
        if not self._enabled or level not in self._levels:
            return
        emoji = {"INFO": "ℹ️", "WARN": "⚠️", "CRITICAL": "🚨"}.get(level, "")
        body = f"{emoji} <b>{html.escape(level)}</b>\n{html.escape(text)}"
        for chat_id in self._chat_ids:
            try:
                await self._bot.send_message(  # type: ignore[attr-defined]
                    chat_id=chat_id, text=body, parse_mode="HTML"
                )
            except Exception as e:
                log.warning("telegram.send_failed", chat_id=chat_id, error=str(e))

    async def notify_trade_open(
        self, strategy: str, symbol: str, side: str, qty: float,
        entry: float, sl: float, tp: float,
    ) -> None:
        rr = abs(tp - entry) / abs(entry - sl) if abs(entry - sl) > 0 else 0.0
        text = (
            f"📈 TRADE OPENED\n"
            f"Strategy: {strategy}\n"
            f"Symbol: {symbol}\n"
            f"Side: {side}\n"
            f"Qty: {qty:g}\n"
            f"Entry: {entry:g}\n"
            f"SL: {sl:g}   TP: {tp:g}   RR: {rr:.1f}x"
        )
        await self.notify("INFO", text)

    async def notify_trade_close(
        self, symbol: str, pnl: float, reason: str,
    ) -> None:
        emoji = "✅" if pnl >= 0 else "❌"
        text = f"{emoji} TRADE CLOSED {symbol}  PnL: {pnl:+.4f}  ({reason})"
        await self.notify("INFO", text)
        # Update today stats for minimap
        self._today_trades += 1
        self._today_pnl += pnl
        if pnl >= 0:
            self._today_wins += 1
        else:
            self._today_losses += 1

    async def notify_halt(self, reason: str, drawdown: float) -> None:
        text = (
            f"KILL SWITCH TRIPPED\n"
            f"Reason: {reason}\n"
            f"Drawdown: {drawdown:.2%}\n"
            f"All positions flattened. Manual /resume or reset_halt.py required."
        )
        await self.notify("CRITICAL", text)

    async def notify_warning(self, drawdown: float) -> None:
        await self.notify("WARN", f"Drawdown at {drawdown:.2%} — approaching halt threshold.")

    # ─────────────────────── minimap event feed ───────────────────────

    def push_signal_event(
        self,
        *,
        symbol: str,
        side: str,
        strategy: str,
        confidence: float,
        regime: str,
        outcome: str,          # "exec" | "ml_rej" | "risk_rej" | "limit_rej"
    ) -> None:
        """
        Record a signal event in the rolling minimap feed.
        Called synchronously from the orchestrator — no await needed.
        """
        now = datetime.now(timezone.utc).strftime("%H:%M")
        side_arrow = "▲" if side.lower() == "buy" else "▼"
        outcome_tag = {
            "exec":      "✅",
            "ml_rej":    "❌ML",
            "risk_rej":  "❌risk",
            "limit_rej": "❌lim",
        }.get(outcome, outcome)
        regime_short = regime.replace("_high_vol", "↑").replace("_low_vol", "↓").replace("trend", "TR").replace("range", "RNG").replace("chop", "CHOP")
        line = (
            f"<code>{now}</code> {html.escape(strategy[:8])} "
            f"{html.escape(symbol.split('/')[0])} {side_arrow} "
            f"<code>{confidence:.2f}</code> [{html.escape(regime_short)}] {outcome_tag}"
        )
        self._signal_feed.appendleft(line)

    # ─────────────────────── minimap dashboard ───────────────────────

    async def _cmd_minimap(self, update: Any, context: Any) -> None:  # type: ignore[no-untyped-def]
        if not self._authorized(update):
            return
        chat_id: int = update.effective_chat.id

        if self._minimap_active and chat_id in self._minimap_message_ids:
            # Toggle off
            self._minimap_active = False
            if self._minimap_task:
                self._minimap_task.cancel()
                self._minimap_task = None
            await update.message.reply_text("📊 Minimap stopped.")
            return

        # Send the first minimap message
        text = await self._build_minimap_text()
        try:
            msg = await self._bot.send_message(  # type: ignore[attr-defined]
                chat_id=chat_id, text=text, parse_mode="HTML",
                disable_web_page_preview=True,
            )
            self._minimap_message_ids[chat_id] = msg.message_id
            # Pin it (silently)
            try:
                await self._bot.pin_chat_message(  # type: ignore[attr-defined]
                    chat_id=chat_id,
                    message_id=msg.message_id,
                    disable_notification=True,
                )
            except Exception:
                pass  # pinning may fail in channels/groups without admin rights
        except Exception as e:
            log.warning("telegram.minimap_send_failed", error=str(e))
            await update.message.reply_text(f"❌ Minimap failed to send: {e}")
            return

        self._minimap_active = True
        if self._minimap_task is None or self._minimap_task.done():
            self._minimap_task = asyncio.create_task(
                self._minimap_refresh_loop(), name="telegram-minimap"
            )
        log.info("telegram.minimap_started", chat_id=chat_id)

    async def _minimap_refresh_loop(self) -> None:
        """Background task: re-edit the minimap message every MINIMAP_REFRESH_SEC."""
        while self._minimap_active:
            try:
                await asyncio.sleep(MINIMAP_REFRESH_SEC)
                if not self._minimap_active:
                    return
                text = await self._build_minimap_text()
                for chat_id, msg_id in list(self._minimap_message_ids.items()):
                    try:
                        await self._bot.edit_message_text(  # type: ignore[attr-defined]
                            chat_id=chat_id,
                            message_id=msg_id,
                            text=text,
                            parse_mode="HTML",
                            disable_web_page_preview=True,
                        )
                    except Exception as e:
                        # "Message is not modified" is harmless — log only real errors
                        err_str = str(e).lower()
                        if "not modified" not in err_str:
                            log.warning("telegram.minimap_edit_failed",
                                        chat_id=chat_id, error=str(e))
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("telegram.minimap_loop_error", error=str(e))
                await asyncio.sleep(10)

    async def _build_minimap_text(self) -> str:
        """Build the full minimap HTML string."""
        mode = self._settings.config.mode.value
        now_str = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        # ── Equity & drawdown ─────────────────────────────────────────────────
        equity_line = "💰 <b>Equity</b>: n/a"
        dd_line = "📉 DD: n/a"
        if self._tracker is not None:
            eq = self._tracker.total_equity()
            ks = self._kill_switch
            dd_pct = 0.0
            halt_pct = self._settings.config.risk.max_daily_drawdown_pct
            if ks is not None:
                dd_pct = getattr(ks, "peak_drawdown_today", 0.0)
                try:
                    dd_pct = ks._state.peak_drawdown_today  # type: ignore[attr-defined]
                except AttributeError:
                    pass
            day_start = self._settings.config.paper.starting_balance_usd if mode == "paper" else eq
            try:
                day_start = (self._kill_switch._state.day_start_equity  # type: ignore[attr-defined]
                             if self._kill_switch else eq)
            except AttributeError:
                pass
            change_pct = (eq - day_start) / day_start if day_start else 0.0
            sign = "+" if change_pct >= 0 else ""
            equity_line = f"💰 <b>Equity</b>: ${eq:,.2f}  <code>{sign}{change_pct:.2%}</code>"
            dd_bar = _mini_bar(dd_pct, halt_pct, width=8)
            dd_line = f"📉 DD: <code>{dd_pct:.2%}</code> {dd_bar} halt@{halt_pct:.0%}"

        # ── ML filter ─────────────────────────────────────────────────────────
        ml_line = "🤖 <b>ML</b>: cold (need 50 trades)"
        if self._ml_filter is not None:
            stats = self._ml_filter.stats()
            n = stats["n_samples"]
            wr = stats["win_rate"]
            v = stats["model_version"]
            cold = stats["cold_start"]
            thr = stats["accept_threshold"]
            pending = stats["pending_positions"]
            if cold:
                ml_line = f"🤖 <b>ML</b>: cold  {n}/50 samples  score≥{thr}"
            else:
                wr_str = f"{wr:.1%}" if wr is not None else "n/a"
                ml_line = (
                    f"🤖 <b>ML</b>: warm v{v}  score≥{thr}\n"
                    f"   {n} samples  win rate <b>{wr_str}</b>  pending:{pending}"
                )

        # ── Kill switch state ─────────────────────────────────────────────────
        ks_state = ""
        if self._kill_switch is not None:
            try:
                state_val = self._kill_switch.state.value  # type: ignore[attr-defined]
                ks_state = f"  KS:<b>{state_val}</b>" if state_val != "running" else ""
            except AttributeError:
                pass

        # ── Regime snapshot ───────────────────────────────────────────────────
        regime_section = ""
        if self._get_regimes is not None:
            try:
                regimes: dict[str, str] = self._get_regimes()
                if regimes:
                    # Group symbols by regime
                    by_regime: dict[str, list[str]] = {}
                    for sym, reg in regimes.items():
                        ticker = sym.split("/")[0]
                        by_regime.setdefault(reg, []).append(ticker)
                    lines = []
                    for reg, syms in sorted(by_regime.items()):
                        emoji = _regime_emoji(reg)
                        lines.append(f"   {emoji} {reg.replace('_', ' ')}: {', '.join(syms)}")
                    regime_section = "📡 <b>Regime</b>:\n" + "\n".join(lines)
            except Exception:
                pass

        # ── Open positions ────────────────────────────────────────────────────
        positions_section = ""
        if self._tracker is not None:
            max_pos = self._settings.config.risk.max_open_positions
            positions = self._tracker.open_positions()
            if positions:
                pos_lines = []
                for p in positions[:6]:  # cap display at 6
                    arrow = "▲" if p.side.value.lower() in ("long", "buy") else "▼"
                    upnl = p.unrealized_pnl
                    sign = "+" if upnl >= 0 else ""
                    pos_lines.append(
                        f"   {arrow} {html.escape(p.symbol.split('/')[0])}  "
                        f"qty={p.quantity:g}  "
                        f"uPnL:<code>{sign}{upnl:+.2f}</code>"
                    )
                positions_section = (
                    f"📈 <b>Positions</b> {len(positions)}/{max_pos}:\n"
                    + "\n".join(pos_lines)
                )
            else:
                positions_section = f"📈 <b>Positions</b>: none  (max {max_pos})"

        # ── Today's stats ─────────────────────────────────────────────────────
        if self._today_trades > 0:
            wr_today = self._today_wins / self._today_trades
            pf_today = (
                self._today_wins / self._today_losses
                if self._today_losses > 0 else float("inf")
            )
            pf_str = f"{pf_today:.2f}" if pf_today != float("inf") else "∞"
            sign = "+" if self._today_pnl >= 0 else ""
            today_section = (
                f"📊 <b>Today</b>: {self._today_trades} trades  "
                f"{self._today_wins}W {self._today_losses}L  "
                f"<code>{wr_today:.0%}</code>\n"
                f"   Net PnL: <b>{sign}{self._today_pnl:+.4f}</b>  PF: {pf_str}"
            )
        else:
            today_section = "📊 <b>Today</b>: no trades yet"

        # ── Signal feed ───────────────────────────────────────────────────────
        if self._signal_feed:
            feed_section = "📝 <b>Signal feed</b>:\n" + "\n".join(self._signal_feed)
        else:
            feed_section = "📝 <b>Signal feed</b>: waiting for first bar close…"

        # ── Assemble ──────────────────────────────────────────────────────────
        sep = "━━━━━━━━━━━━━━━━━━━━"
        parts = [
            f"📊 <b>CRYPROBOTIK LIVE</b> — {mode}{ks_state}",
            sep,
            equity_line,
            dd_line,
            "",
            ml_line,
        ]
        if regime_section:
            parts += ["", regime_section]
        parts += [
            "",
            positions_section,
            "",
            today_section,
            "",
            feed_section,
            "",
            f"🔄 <i>{now_str}</i>",
        ]
        return "\n".join(parts)

    # ─────────────────────── command handlers ───────────────────────

    def _authorized(self, update: Any) -> bool:
        chat = getattr(getattr(update, "effective_chat", None), "id", None)
        return chat in self._chat_ids

    async def _cmd_webapp(self, update: Any, context: Any) -> None:  # type: ignore[no-untyped-def]
        """Send a button that opens the live Mini App dashboard."""
        if not self._authorized(update):
            return
        url = self._miniapp_url or "http://localhost:8080/app"
        try:
            from telegram import InlineKeyboardButton, InlineKeyboardMarkup, WebAppInfo  # type: ignore[import-not-found]
            # Telegram Mini App requires HTTPS. Fall back to plain link if URL is HTTP.
            if url.startswith("https://"):
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("📊 Open Live Dashboard", web_app=WebAppInfo(url=url))
                ]])
            else:
                kb = InlineKeyboardMarkup([[
                    InlineKeyboardButton("📊 Open Live Dashboard", url=url)
                ]])
            await update.message.reply_text(
                "📊 <b>Cryprobotik Live Dashboard</b>\nReal-time signals, positions, ML & regime monitor.",
                parse_mode="HTML",
                reply_markup=kb,
            )
        except Exception as e:
            # Fallback: just send the URL as text
            await update.message.reply_text(
                f"📊 Live dashboard: {url}\n\n"
                "For Telegram Mini App support, set MINIAPP_URL=https://your-domain:8080/app"
            )

    async def _cmd_status(self, update: Any, context: Any) -> None:  # type: ignore[no-untyped-def]
        if not self._authorized(update):
            return
        lines = [f"<b>Status</b>  mode={self._settings.config.mode.value}"]
        if self._tracker is not None:
            lines.append(f"Equity: {self._tracker.total_equity():.2f}")
            lines.append(f"Open positions: {len(self._tracker.open_positions())}")
        if self._kill_switch is not None:
            lines.append(f"Kill switch: {self._kill_switch.state.value}")  # type: ignore[attr-defined]
        if self._ml_filter is not None:
            s = self._ml_filter.stats()
            ml_state = "cold" if s["cold_start"] else f"warm v{s['model_version']}"
            lines.append(f"ML: {ml_state}  {s['n_samples']} samples")
        lines.append("Use /minimap for live dashboard | /webapp for Mini App")
        await update.message.reply_html("\n".join(lines))

    async def _cmd_positions(self, update: Any, context: Any) -> None:  # type: ignore[no-untyped-def]
        if not self._authorized(update):
            return
        if self._tracker is None:
            return
        positions = self._tracker.open_positions()
        if not positions:
            await update.message.reply_text("No open positions.")
            return
        lines = []
        for p in positions:
            lines.append(
                f"{p.exchange} {p.symbol} {p.side.value} qty={p.quantity:g} "
                f"entry={p.entry_price:g} mark={p.mark_price:g} uPnL={p.unrealized_pnl:+.4f}"
            )
        await update.message.reply_text("\n".join(lines))

    async def _cmd_pnl(self, update: Any, context: Any) -> None:  # type: ignore[no-untyped-def]
        if not self._authorized(update):
            return
        if self._analytics is None:
            return
        rep = await self._analytics.report()
        text = (
            f"<b>PnL (last 24h)</b>\n"
            f"Trades: {rep.trades}  Wins: {rep.winners}  Losses: {rep.losers}\n"
            f"Win rate: {rep.win_rate:.1%}  PF: {rep.profit_factor:.2f}\n"
            f"Net PnL: {rep.net_pnl:+.4f}\n"
            f"Sharpe: {rep.sharpe:.2f}  Sortino: {rep.sortino:.2f}\n"
            f"Max DD: {rep.max_drawdown_pct:.2%}"
        )
        await update.message.reply_html(text)

    async def _cmd_halt(self, update: Any, context: Any) -> None:  # type: ignore[no-untyped-def]
        if not self._authorized(update):
            return
        if self._kill_switch is not None:
            await self._kill_switch.force_halt("operator_telegram_halt")  # type: ignore[attr-defined]
            await update.message.reply_text("Halted. Bot will stop placing new trades.")

    async def _cmd_resume(self, update: Any, context: Any) -> None:  # type: ignore[no-untyped-def]
        if not self._authorized(update):
            return
        if self._kill_switch is not None:
            await self._kill_switch.reset()  # type: ignore[attr-defined]
            await update.message.reply_text("Halt cleared. Trading resumed.")

    # ─────────────────────── daily report ───────────────────────

    async def _run_daily_report_scheduler(self) -> None:
        target_hour = self._notif_config.telegram.daily_report_hour_utc
        while True:
            try:
                delay = seconds_until_next_utc_hour(target_hour)
                await asyncio.sleep(delay)
                # Reset today stats
                self._today_trades = 0
                self._today_wins = 0
                self._today_losses = 0
                self._today_pnl = 0.0
                if self._analytics is not None:
                    rep = await self._analytics.report()
                    text = (
                        f"📊 Daily report\n"
                        f"Trades: {rep.trades}  Wins: {rep.winners}\n"
                        f"Win rate: {rep.win_rate:.1%}\n"
                        f"Net PnL: {rep.net_pnl:+.4f}\n"
                        f"Sharpe: {rep.sharpe:.2f}\n"
                        f"Max DD: {rep.max_drawdown_pct:.2%}"
                    )
                    await self.notify("INFO", text)
            except asyncio.CancelledError:
                return
            except Exception as e:
                log.error("telegram.daily_report_failed", error=str(e))
                await asyncio.sleep(300)


# ─────────────────────── helpers ───────────────────────


def _mini_bar(value: float, maximum: float, width: int = 8) -> str:
    """ASCII progress bar: value / maximum → e.g. [███░░░░░]"""
    if maximum <= 0:
        return ""
    filled = int(min(1.0, value / maximum) * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def _regime_emoji(regime: str) -> str:
    return {
        "trend_high_vol": "🚀",
        "trend_low_vol":  "📈",
        "range_high_vol": "↔️",
        "range_low_vol":  "😴",
        "chop":           "🌀",
    }.get(regime, "❓")
