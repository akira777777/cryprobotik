"""
Cross-exchange funding rate arbitrage.

Both OKX and Bybit pay funding on USDT perps every 8 hours (in most cases).
When the same symbol has a funding rate that differs significantly between
venues, we can open a delta-neutral pair:
    - LONG on the venue where funding is lower (or negative  → we receive)
    - SHORT on the venue where funding is higher (we receive)

The PnL comes (almost) entirely from the funding payments; the price risk is
neutralized by the opposing legs.

This strategy does NOT use OHLCV — it operates on the latest funding rates
received from both exchanges via WebSocket funding streams + periodic REST
polls. It's fed externally via `update_rates()`.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from src.strategies.base import PairSignal, Strategy
from src.utils.logging import get_logger
from src.utils.time import now_utc

if TYPE_CHECKING:
    from src.data.feature_store import FeatureStore
    from src.strategies.base import Signal

log = get_logger(__name__)


@dataclass(slots=True)
class _RateSnapshot:
    rate: float
    ts: datetime
    next_funding_ts: datetime | None


class FundingArbStrategy(Strategy):
    """
    Cross-exchange funding arbitrage strategy.

    Note: evaluate() is unused for this strategy (it doesn't react to kline
    bars). The orchestrator calls `scan_arb_opportunities()` directly after
    each funding update. We still inherit from Strategy so it fits the
    ensemble's config/enable switches.
    """
    name = "funding_arb"

    def __init__(
        self,
        *,
        min_rate_delta: float,
        min_notional_usd: float,
        close_before_funding_sec: float,
        base_confidence: float,
    ) -> None:
        super().__init__(base_confidence=base_confidence)
        self._min_rate_delta = min_rate_delta
        self._min_notional = min_notional_usd
        self._close_before_funding = close_before_funding_sec
        # latest_rates[symbol][exchange] = _RateSnapshot
        self._latest_rates: dict[str, dict[str, _RateSnapshot]] = {}

    def evaluate(
        self, symbol: str, store: "FeatureStore", exchange: str, ts: datetime
    ) -> list["Signal"]:
        # Funding arb doesn't produce single-leg Signals via the bar loop.
        # It uses scan_arb_opportunities() instead.
        return []

    # ─────────────────────── external rate feed ───────────────────────

    def update_rate(
        self,
        exchange: str,
        symbol: str,
        rate: float,
        next_funding_ts: datetime | None,
        ts: datetime | None = None,
    ) -> None:
        """Called by the orchestrator whenever a funding rate event arrives."""
        bucket = self._latest_rates.setdefault(symbol, {})
        bucket[exchange] = _RateSnapshot(
            rate=rate, ts=ts or now_utc(), next_funding_ts=next_funding_ts
        )

    # ─────────────────────── scanner ───────────────────────

    def scan_arb_opportunities(self) -> list[PairSignal]:
        """
        Walk the latest-rate table and emit a PairSignal for every symbol
        where the delta between venues exceeds `min_rate_delta`.

        This is called on a timer (e.g. once per minute) by the orchestrator
        and after every funding-rate update.
        """
        out: list[PairSignal] = []
        # Snapshot the dict before iterating — update_rate() can mutate it
        # concurrently from the WebSocket push handler.
        snapshot = dict(self._latest_rates)
        for symbol, per_ex in snapshot.items():
            if len(per_ex) < 2:
                continue
            # Take the two known venues. For v1 we only support OKX vs Bybit.
            if "okx" not in per_ex or "bybit" not in per_ex:
                continue
            okx_snap = per_ex["okx"]
            bybit_snap = per_ex["bybit"]

            # Stale check — skip if either side's rate is older than 15 minutes.
            now = now_utc()
            if (now - okx_snap.ts).total_seconds() > 900:
                continue
            if (now - bybit_snap.ts).total_seconds() > 900:
                continue

            delta = okx_snap.rate - bybit_snap.rate
            if abs(delta) < self._min_rate_delta:
                continue

            # delta > 0 → OKX funding is higher → short OKX, long Bybit.
            # delta < 0 → opposite.
            if delta > 0:
                long_ex, short_ex = "bybit", "okx"
            else:
                long_ex, short_ex = "okx", "bybit"

            # Don't open if we're within close-before-funding window of the
            # NEXT funding settlement on the short leg — we'd settle late.
            short_next_ts = (okx_snap.next_funding_ts if short_ex == "okx"
                             else bybit_snap.next_funding_ts)
            if short_next_ts is not None:
                seconds_to_funding = (short_next_ts - now).total_seconds()
                if seconds_to_funding < 0:
                    # next_funding_ts is stale (past epoch not yet updated).
                    # We don't know when the next settlement is — skip.
                    log.debug(
                        "funding_arb.stale_next_funding_ts",
                        symbol=symbol, exchange=short_ex,
                        seconds_overdue=abs(seconds_to_funding),
                    )
                    continue
                if seconds_to_funding < self._close_before_funding:
                    continue

            # Confidence scales with delta magnitude (capped at 2x threshold).
            scale = min(1.0, abs(delta) / (2 * self._min_rate_delta))
            conf = self._clip(self._base_confidence * (0.6 + 0.4 * scale))

            out.append(PairSignal(
                strategy=self.name,
                symbol=symbol,
                ts=now,
                long_exchange=long_ex,
                short_exchange=short_ex,
                confidence=conf,
                meta={
                    "okx_rate": round(okx_snap.rate, 6),
                    "bybit_rate": round(bybit_snap.rate, 6),
                    "delta": round(delta, 6),
                    "next_funding_ts": short_next_ts.isoformat() if short_next_ts else None,
                },
            ))
        return out
