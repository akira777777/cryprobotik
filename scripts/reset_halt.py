"""
Manual kill-switch reset script.

Usage:
    python -m scripts.reset_halt [--config config/config.yaml]

Run this AFTER you have:
    1. Investigated what caused the halt (check events table + Telegram).
    2. Confirmed the cause is addressed (e.g. bug fixed, market sanity-checked).
    3. Verified that no positions are stuck open that shouldn't be.

The halt reset does NOT re-open positions — it only allows the bot to start
taking new trades on its next loop iteration.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.data.storage import Storage
from src.risk.kill_switch import KillSwitch
from src.settings import load_settings
from src.utils.logging import configure_logging, get_logger


async def _run(config_path: str) -> int:
    settings = load_settings(config_path)
    configure_logging(
        level=settings.config.logging.level, fmt=settings.config.logging.format
    )
    log = get_logger("reset_halt")

    storage = Storage(
        dsn=settings.database_url.get_secret_value(),
        pool_min=1, pool_max=2,
    )
    await storage.connect()
    try:
        ks = KillSwitch(config=settings.config.risk, storage=storage)
        await ks.load()
        if not ks.is_halted:
            log.info("reset_halt.not_halted", current_state=ks.state.value)
            print(f"Kill switch is not halted (state={ks.state.value}). Nothing to reset.")
            return 0
        await ks.reset()
        log.info("reset_halt.done")
        print("✅ Halt cleared. Bot will resume trading on next loop iteration.")
        return 0
    finally:
        await storage.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Reset the kill-switch halt flag")
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    return asyncio.run(_run(args.config))


if __name__ == "__main__":
    sys.exit(main())
