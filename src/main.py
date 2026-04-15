"""
Main entrypoint.

Usage:
    python -m src.main [--config config/config.yaml] [--mode testnet|paper|live] [--confirm-live]

Live mode requires BOTH:
    - `mode: live` in config.yaml (or --mode live CLI override)
    - `--confirm-live` flag

This double-confirmation exists so a stray typo or env-var can't accidentally
send orders with real money.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from src.orchestrator import Orchestrator
from src.settings import RuntimeMode, load_settings
from src.utils.logging import configure_logging, get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="cryprobotik",
        description="Autonomous multi-exchange crypto trading bot",
    )
    parser.add_argument(
        "--config", default="config/config.yaml",
        help="Path to config YAML (default: config/config.yaml)",
    )
    parser.add_argument(
        "--mode", choices=[m.value for m in RuntimeMode], default=None,
        help="Override runtime mode from config.yaml",
    )
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="REQUIRED to enable live trading. Without this flag, live mode refuses to start.",
    )
    return parser.parse_args()


async def _run(orchestrator: Orchestrator) -> None:
    await orchestrator.setup()
    await orchestrator.run()


def main() -> int:
    args = parse_args()

    settings = load_settings(args.config)
    if args.mode is not None:
        settings.config.mode = RuntimeMode(args.mode)

    configure_logging(
        level=settings.config.logging.level,
        fmt=settings.config.logging.format,
    )
    log = get_logger("main")

    # Live mode guard
    if settings.config.mode == RuntimeMode.LIVE and not args.confirm_live:
        log.error(
            "main.live_mode_refused",
            message=(
                "Refusing to start in LIVE mode without --confirm-live. "
                "This is a deliberate safety check. Re-run with --confirm-live "
                "ONLY after completing the full validation protocol in DEPLOYMENT.md."
            ),
        )
        print(
            "\n❌ REFUSED: live mode requires --confirm-live flag.\n"
            "   See DEPLOYMENT.md for the validation protocol that must be\n"
            "   completed before enabling live trading.\n",
            file=sys.stderr,
        )
        return 2

    log.info("main.starting", mode=settings.config.mode.value, config_path=args.config)

    orchestrator = Orchestrator(settings)
    try:
        asyncio.run(_run(orchestrator))
    except KeyboardInterrupt:
        log.info("main.keyboard_interrupt")
    except Exception as e:
        log.error("main.fatal_error", error=str(e), exc_info=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
