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
from .policy import create_policy


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="secondplayer", description="Local visual AI players for stock emulators"
    )
    p.add_argument("--config", type=Path, default=None, help="SecondPlayer TOML config")
    p.add_argument("-v", "--verbose", action="count", default=0)
    sub = p.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="write a starter configuration")
    init.add_argument("--force", action="store_true")

    play = sub.add_parser("play", help="launch the configured emulator adapter with SecondPlayer AI seats")
    play.add_argument("rom", type=Path)

    bench = sub.add_parser("bench", help="run a pinned benchmark from pixels and write a JSON report")
    bench.add_argument("game", choices=["tetris"], help="benchmark to run")
    bench.add_argument("rom", type=Path)
    bench.add_argument("--timeout", type=float, default=1800.0, help="run ceiling in seconds")
    bench.add_argument("--out-dir", type=Path, default=None, help="report directory")
    bench.add_argument(
        "--reaction-ms",
        type=float,
        default=None,
        help="decision cadence floor in ms (default: config reaction_ms; 0 = as fast as inference allows)",
    )
    bench.add_argument(
        "--from-state",
        type=Path,
        default=None,
        help="debug: load this savestate after boot instead of cold-booting (default: cold boot)",
    )
    bench.add_argument(
        "--snap-every",
        type=int,
        default=0,
        help="save every Nth poll frame to <out-dir>/snaps (default: 0 = off)",
    )
    sample_group = bench.add_mutually_exclusive_group()
    sample_group.add_argument("--sample", dest="sample", action="store_true", default=None)
    sample_group.add_argument("--no-sample", dest="sample", action="store_false")
    bench.set_defaults(sample=None)

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
            policy = create_policy(cfg.runtime)
            engine = SecondPlayerEngine(
                adapter, policy, cfg.session.ai_players, cfg.runtime.reaction_ms, cfg.runtime.poll_ms
            )
            stats = engine.run()
            print(
                f"SecondPlayer stopped: decisions={stats.decisions}, failures={stats.failures}, "
                f"last_inference_ms={stats.last_inference_ms:.0f}"
            )
            return 0

        if args.command == "bench":
            from .benchmarks.tetris.run import run_benchmark, write_report

            if cfg.adapter.name != "mesence":
                raise SecondPlayerError(
                    f"tetris benchmark requires the mesence adapter, got {cfg.adapter.name!r}"
                )
            if cfg.session.ai_players != [1]:
                raise SecondPlayerError("tetris benchmark requires exactly AI player 1 (P2 disabled)")
            options = dict(cfg.adapter.options)
            if options.get("headless", False):
                print("tetris benchmark is realtime-only: forcing headless=false")
                options["headless"] = False
            adapter = create_adapter(cfg.adapter.name, options, cfg.session, args.rom)
            if args.sample is not None:
                cfg.runtime.sample = args.sample
            policy = create_policy(cfg.runtime)
            reaction_ms = args.reaction_ms if args.reaction_ms is not None else cfg.runtime.reaction_ms
            if reaction_ms < 0:
                raise SecondPlayerError("--reaction-ms must be >= 0")
            if args.snap_every < 0:
                raise SecondPlayerError("--snap-every must be >= 0")
            out_dir = args.out_dir or (Path(__file__).with_name("benchmarks") / "tetris" / "results")
            report = run_benchmark(
                adapter,
                policy,
                args.rom,
                timeout_s=args.timeout,
                min_interval_s=reaction_ms / 1000.0,
                poll_interval_s=cfg.runtime.poll_ms / 1000.0,
                from_state=args.from_state,
                snap_every=args.snap_every,
                snap_dir=out_dir / "snaps" if args.snap_every > 0 else None,
                extra={
                    "model": cfg.runtime.model,
                    "screen_grid": cfg.runtime.screen_grid,
                    "openjev_weights": cfg.runtime.openjev_weights,
                    "quant_backend": cfg.runtime.quant_backend,
                    "device": cfg.runtime.device,
                    "reaction_ms": reaction_ms,
                    "poll_ms": cfg.runtime.poll_ms,
                    "permutations": cfg.runtime.permutations,
                    "window_frames": cfg.runtime.window_frames,
                    "sample": cfg.runtime.sample,
                    "learn": cfg.runtime.learn,
                    "memory_turns": cfg.runtime.memory_turns,
                },
            )
            path = write_report(report, args.out_dir)
            final = report["final"]
            gameplay = "yes" if report["reached_gameplay"] else "no"
            print(
                f"tetris-v1: reason={report['reason']} score={final['score']} lines={report['lines_cleared']} "
                f"mode={report['mode']} level={final['level']} gameplay={gameplay} "
                f"decisions={report['decisions']} polls={report['polls']} cadence={report['cadence_hz']:.1f}Hz "
                f"infer_p50={report['infer_ms']['p50']:.0f}ms ({path})"
            )
            return 0 if report["reason"] in ("game_over", "timeout", "interrupted") else 2
    except KeyboardInterrupt:
        return 130
    except SecondPlayerError as exc:
        print(f"secondplayer: {exc}", file=sys.stderr)
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
