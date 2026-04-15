"""
Performance analytics.

Reads fills + equity history from TimescaleDB and produces:
    - total realized PnL
    - win rate, profit factor
    - Sharpe ratio (daily, 0% rf)
    - Sortino ratio
    - max drawdown
    - trade count + average RR

Used to format the daily Telegram report and to expose Prometheus gauges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import numpy as np

from src.utils.logging import get_logger
from src.utils.time import now_utc

if TYPE_CHECKING:
    from src.data.storage import Storage

log = get_logger(__name__)


@dataclass(slots=True)
class PerformanceReport:
    period_start: datetime
    period_end: datetime
    trades: int = 0
    winners: int = 0
    losers: int = 0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    max_drawdown_pct: float = 0.0
    start_equity: float = 0.0
    end_equity: float = 0.0
    pnl_by_strategy: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "trades": self.trades,
            "winners": self.winners,
            "losers": self.losers,
            "gross_profit": round(self.gross_profit, 4),
            "gross_loss": round(self.gross_loss, 4),
            "net_pnl": round(self.net_pnl, 4),
            "win_rate": round(self.win_rate, 4),
            "profit_factor": round(self.profit_factor, 4),
            "sharpe": round(self.sharpe, 4),
            "sortino": round(self.sortino, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "start_equity": round(self.start_equity, 2),
            "end_equity": round(self.end_equity, 2),
            "pnl_by_strategy": {k: round(v, 4) for k, v in self.pnl_by_strategy.items()},
        }


class Analytics:
    def __init__(self, storage: "Storage", mode: str) -> None:
        self._storage = storage
        self._mode = mode

    async def report(
        self, period_start: datetime | None = None, period_end: datetime | None = None
    ) -> PerformanceReport:
        """
        Compute a report for the given period (default: last 24 hours).
        """
        period_end = period_end or now_utc()
        period_start = period_start or (period_end - timedelta(days=1))

        async with self._storage.pool.acquire() as conn:
            fills = await conn.fetch(
                """
                SELECT f.ts, f.symbol, f.side, f.quantity, f.price, f.fee,
                       f.realized_pnl, o.strategy
                FROM fills f
                LEFT JOIN orders o ON o.id = f.order_id
                WHERE f.ts >= $1 AND f.ts < $2
                ORDER BY f.ts ASC
                """,
                period_start, period_end,
            )
            equity_rows = await conn.fetch(
                """
                SELECT ts, equity
                FROM equity
                WHERE ts >= $1 AND ts < $2 AND mode = $3
                ORDER BY ts ASC
                """,
                period_start, period_end, self._mode,
            )

        rep = PerformanceReport(period_start=period_start, period_end=period_end)

        # Fills
        pnl_by_strategy: dict[str, float] = {}
        for row in fills:
            pnl = float(row["realized_pnl"] or 0.0) - float(row["fee"] or 0.0)
            strat = row["strategy"] or "unknown"
            if pnl == 0:
                continue
            rep.net_pnl += pnl
            if pnl > 0:
                rep.winners += 1
                rep.gross_profit += pnl
            else:
                rep.losers += 1
                rep.gross_loss += -pnl
            rep.trades += 1
            pnl_by_strategy[strat] = pnl_by_strategy.get(strat, 0.0) + pnl

        rep.pnl_by_strategy = pnl_by_strategy
        rep.win_rate = (rep.winners / rep.trades) if rep.trades else 0.0
        rep.profit_factor = (rep.gross_profit / rep.gross_loss) if rep.gross_loss else float("inf") if rep.gross_profit > 0 else 0.0

        # Equity curve: compute daily returns → sharpe / sortino / maxDD
        if equity_rows:
            eq_values = np.array([float(r["equity"]) for r in equity_rows])
            rep.start_equity = float(eq_values[0])
            rep.end_equity = float(eq_values[-1])

            # Max drawdown
            peak = np.maximum.accumulate(eq_values)
            dd = (peak - eq_values) / np.where(peak == 0, 1, peak)
            rep.max_drawdown_pct = float(np.max(dd)) if len(dd) else 0.0

            # Returns: diff of equity / prior value
            if len(eq_values) > 1:
                returns = np.diff(eq_values) / np.where(eq_values[:-1] == 0, 1, eq_values[:-1])
                if len(returns):
                    mean_r = float(np.mean(returns))
                    std_r = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
                    rep.sharpe = (mean_r / std_r * np.sqrt(365)) if std_r > 0 else 0.0
                    downside = returns[returns < 0]
                    down_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
                    rep.sortino = (mean_r / down_std * np.sqrt(365)) if down_std > 0 else 0.0

        return rep
