# Probe archive 2026-09-24: can the 4B drive?

Scratch evidence for `docs/DRIVE_QUESTION.md`. NOT maintained code: these
ran once each on the bench box (Intel Arc, `~/projects/secondplayer`
checkout, XPU venv) and are kept for audit, not reuse. They import the
repo (`sys.path`) for the real prompt path and load Qwen3.5-4B qint4.

Map (script -> result in DRIVE_QUESTION.md section 4):

- asccal.py, asccal2.py — well/palette calibration (no quiz).
- ascii2.py — ASCII vs text vs vision quiz, clean repainted synths.
- ascii3gen.py, ascii3gen2.py — free generation on grids (3/5).
- coordprobe.py — coordinate lists, thinking-enabled (2/6, misconfigured).
- thinkoff.py — TEXT/COORD/ASCII/IMG with thinking disabled (4/6, 5/6, 5/6, 4/6).
- motionprobe.py — two-frame coords + changed cells (4/4).
- actionquiz.py — movement decisions, standard + rotated (0/8).
- factquiz.py — side facts + urgent actions (4/10).
- rulequiz.py — halves digest + if/else rule (2/5).
- ladderquiz.py — binary A/B ladder (3/4).
- thinkact.py — think-then-act (0/4, ~200s/case).
- directquiz.py — direct question (0/4) + few-shot (1/2).
- firstperson.py — player framing (0/4).
- pocketquiz.py — valid pockets +- dynamics line (1/3, 1/3).
- directopt.py — direct options (1/3).
- bench-val1.json, bench-val2.json — validation bench reports (score 0).
- thinkact.out, thinkoff.out — raw outputs incl. think traces.
