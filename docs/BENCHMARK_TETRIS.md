# Tetris Benchmark v1

Plays Tetris from pixels only, from a cold boot — the AI navigates boot,
menus, and gameplay itself, exactly like product play mode — until GAME
OVER or a 30-minute ceiling. Produces a JSON report with a directly
comparable scalar: final score.

## Pins

- ROM: Tetris & Dr. Mario (USA), SMC 1049088 bytes (512-byte header),
  sha256 `62cef84f03ea6f3d0a05cc399731da193e01f0aee54e01b13ad08fb98b4a427c`.
  The runner refuses any other ROM.
- Cold boot is the start. `bench --from-state` loads a savestate instead
  (debug only); `secondplayer/benchmarks/tetris/bench0.state` (A-TYPE
  level 0, empty board, score 0, first piece falling) is kept for that.
  Regenerate with `make_start_state.py` (dev tool).
- Config: `config/bench-tetris.toml` (auto device, 100ms poll, inference
  only on scene change with max 10 decisions/s, 6-frame sliding window,
  learn scaffold on, 9-frame input pulses, realtime graphical MesenCE
  under Xvfb). Cadence overridable per run with `bench --reaction-ms`,
  recorded in the report.

## Rules

- The policy sees only native-resolution RGB frames; no RAM, no state peeking.
- Realtime emulation (`bench` forces `headless=false`).
- The loop polls visuals at the poll rate and runs inference only when the
  scene changed (1s heartbeat on frozen screens so stuck recovery can act).
- The model state carries a sliding window of recent frames + their control
  choices, trimmed per decision to fit the checkpoint's context window.
- GAME OVER ends a run only after gameplay was reached (first valid score
  decode), confirmed across reads at least a second apart, so a
  sub-second line-clear flash cannot end it early.
- HUD digits decode by exact white-mask template match (all 10 glyphs,
  verified across menu/gameplay/game-over screens). Any non-matching cell
  reads as unknown and is retried next frame — never guessed.

## Metrics (JSON report)

`reason` (game_over/timeout/interrupted/error), `final` score/top/lines/
level/stats, `duration_s`, `decisions`, `polls`, `reached_gameplay`,
`gameplay_at_s`, `mode` (a-type/b-type/unknown from the lines counter),
`lines_start`, `lines_cleared`, `failures`, `decode_gaps`, `infer_ms`
(n/mean/p50/p95/max), `confidence` (movement/buttons summaries),
`learning` (turns/facts/tried/forced/effects/per-turn log),
`trace` of wall time + score on decisions and score changes.

## Run

```sh
secondplayer --config config/bench-tetris.toml bench tetris <rom> \
    --timeout 1800 --out-dir results/
```

## Baseline (2026-09-23, 192.168.0.200, CPU)

First Laya run: `game_over`, score **5**, lines 0, 78 decisions in 92s,
infer p50 983ms / p95 1218ms, 0 failures, 0 decode gaps. The agent survived
~12 pieces on soft-drop points without clearing a line — the number to beat.

Historical checkpoint (do not relabel): that run used laya-vision code
`05ba04399f0e34ff31f7c8adf95d127c7fd009fc` with the mutable
`thaitea/laya-vision` alias, which on 2026-09-23 resolved to
`d1fbdc0612fbe3b3d8ec6f54d328b195d35bb338` (cauldron-score-2ep-bidir-full,
237M, no game training). The Hub history shows no push to that repo between
2026-09-22 15:19 UTC and 2026-09-24 05:35 UTC, so the score-5 run predates
the current checkpoint. The committed JSON report for that run is lost
(only the numbers above survive); current benchmarks pin
`thaitea/laya-vision-201m@0b6228f7a0762566de1c4539e9aa4eb1c1aef5f4`
and record resolved revisions in `model_provenance`.
