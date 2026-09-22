from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .adapters import create_adapter
from .config import default_config_path, load_config, write_default_config
from .doctor import run_checks
from .engine import SecondPlayerEngine
from .errors import SecondPlayerError
from .policy import LayaVisionPolicy


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="secondplayer", description="Local visual AI players for stock emulators")
    p.add_argument("--config", type=Path, default=None, help="SecondPlayer TOML config")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a starter configuration")
    init.add_argument("--force", action="store_true")

    play = sub.add_parser("play", help="launch the configured emulator adapter with SecondPlayer AI seats")
    play.add_argument("rom", type=Path)

    sub.add_parser("doctor", help="check core runtime and selected-adapter prerequisites")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        if args.command == "init":
            path = write_default_config(args.config or default_config_path(), overwrite=args.force)
            print(path)
            return 0

        cfg = load_config(args.config)
        if args.command == "doctor":
            checks = run_checks(cfg)
            for c in checks:
                print(f"{'OK' if c.ok else 'FAIL':4} {c.name:14} {c.detail}")
            return 0 if all(c.ok for c in checks) else 2

        if args.command == "play":
            adapter = create_adapter(cfg.adapter.name, cfg.adapter.options, cfg.session, args.rom)
            policy = LayaVisionPolicy(cfg.runtime)
            engine = SecondPlayerEngine(adapter, policy, cfg.session.ai_players, cfg.runtime.reaction_ms)
            stats = engine.run()
            print(
                f"SecondPlayer stopped: decisions={stats.decisions}, failures={stats.failures}, "
                f"last_inference_ms={stats.last_inference_ms:.0f}"
            )
            return 0
    except KeyboardInterrupt:
        return 130
    except SecondPlayerError as exc:
        print(f"secondplayer: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
