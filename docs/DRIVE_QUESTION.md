# Can this 4B brain drive? An open question with full evidence

Status: unresolved dispute, 2026-09-24. The operator doubts the verdict;
this file hands a reviewer AI everything needed to judge it or break it.

## Prompt to give the reviewer

> Inspect this repository. Read `docs/DRIVE_QUESTION.md` in full, plus the
> probe scripts and reports under `probes/2026-09-24-drive/`. The question:
> **given the measured results, is the verdict correct that Qwen3.5-4B
> cannot select Tetris actions in this single-forward letter readout — or
> is there a flaw in the quiz design, ground truth, statistics, or
> reasoning that, fixed, unlocks driving?** If you believe it can drive,
> propose ONE concrete prompt (exact system line, question line, option
> list with descriptions, state-text additions) that stays inside the
> thesis constraints in section 2. It will be tested on held-out pocket
> scenarios generated after your proposal: 3/4 passes to win.

## 1. The dispute

I (the coding agent) claim, after 12 prompt framings and ~60 GPU forwards:
**Qwen3.5-4B (qint4) demonstrably understands the board as text
(column-set extraction at 0.99) yet cannot select actions from it in a
single-forward letter readout.** Every action probe collapses to neutral
abstention, including urgency controls, explicit rules, binary choices,
few-shot, first-person framing, and 150-token think-then-act (0/4,
~200s/case, hallucinating traces).

The operator does not believe a 4B model with demonstrated comprehension
is terminally incapable of deciding. The operator is right that several
of my intermediate theories were wrong (details in section 5), which is
why this file exists instead of a verdict.

## 2. Thesis constraints (non-negotiable for any proposal)

- Pixels-only, local-only, in-process. No external server, no side process.
- General core, no game-specific policy. No Tetris tactics, steering rules,
  adapters, or per-game code in core. Precedent for what IS allowed:
  `MENU_HINT` ("if a menu is shown, press confirm") — generic controller
  convention, no game named.
- No extra information beyond what a player sees on screen. Per-game
  manuals, tactics injections, and harness-side play-calling are out.
  Fixed display geometry (grid origin/pitch for a renderer) is allowed:
  it is renderer data, like screen resolution, and the operator approved it.
- One universal 4B qint4 quant (quanto, no int8). Model swap is a
  product decision for the operator, not something a prompt proposal needs.
- Zero interaction with `~/projects/openjev` (the oracle; not in this repo).

A proposal violates the thesis if the harness decides instead of the
brain (servo control with a decorative model), if Tetris strategy lives
anywhere but the model's own weights, or if any input is not derivable
from framebuffer pixels by fixed generic algorithms.

## 3. The system under test (all in this repo)

- `secondplayer/policy/grid.py` — deterministic framebuffer-to-text
  reader. Fixed cell grid, center-sampled cells (bilinear single pixel),
  background = dominant color, foreground = distance >= 60. Emits
  filled-cell `(row,col)` coords plus the changed-cell set vs the previous
  frame. No game words anywhere. Golden test on the reference frame:
  `tests/test_policy_grid.py`.
- `secondplayer/policy/openjev.py` — builds the state text
  (`Game:` / `Objective:` / role / `Screen:` grid block / previous
  controller / menu hint / episodic memory) and asks two letter questions
  per decision: movement (10 options: neutral, up, down, left, right,
  4 diagonals, keep) and buttons (14 SNES options).
- `secondplayer/policy/ojcore/` — clean-room in-process scorer: one
  forward, softmax over verified single-token letter slots (A=32, B=33,
  ...), chat template with `enable_thinking=False` (verified: template
  ends with an empty closed `<think></think>` block, so the scored
  position is the answer position).
- Weights: Qwen3.5-4B quanto qint4 (`/data/tmp/ojcore/qwen3.5-4b-qint4e`
  on the bench box), text-only load, 2.79GB peak, ~1.3-2.3s/score warm
  on Intel Arc (XPU reference kernels; the 200ms budget is unachievable
  on this box for any forward — latencies are reported, not promised).
- Bench: `secondplayer/benchmarks/tetris/` — realtime Mesen play from
  pixels, HUD decoded from pixels for scoring only (the policy never
  sees decoder output). Pins: Tetris & Dr. Mario (USA) ROM sha256
  `62cef84f...`, 30-minute ceiling.

Reference frame: `secondplayer/benchmarks/tetris/bench0-reference.png`
(256x224). Validated well geometry: origin (93,44), pitch 8x8.2,
10 cols x 20 rows. Its falling blob reads exactly
`(6,4) (6,5) (6,6) (7,4) (7,5)`.

## 4. Measured results (all on Intel Arc XPU unless noted)

Hardware note: all GPU probes ran on 192.168.0.200 (Arc A770, torch 2.8
XPU, reference linear-attention kernels). The reviewer likely lacks this
box; every number needed is inline below, scripts are in `probes/` for
audit, and any proposed prompt will be run on held-out scenarios here.

### 4a. Comprehension: the brain reads (probes: thinkoff, motion, ascii3gen2)

Deployment-faithful readout (thinking disabled, letter slots), 6 cases
over the reference frame plus two synthetic column-repaints:

- COORD (filled-cell list): **5/6**, confident (synth stacks at 0.99).
  Only miss: single-frame falling-vs-stacked ambiguity.
- ASCII-WELL (10x20 grid text, verified faithful): **5/6** (0.89-0.93).
- TEXT-ONLY control: 4/6 (prior luck: piece is middle in all frames).
- IMG (native full-frame pixels): 4/6 BUT emits near-identical
  distributions across different images — zero image contribution.
  The vision tower is dead weight; this is settled, not disputed.
- Two-frame COORD + changed-cell list: **4/4** (moving object located
  at 0.82, static stacks identified with the mover excluded, true-empty
  called empty at 0.81).
- Free generation on grids (thinking disabled, 80 tokens): 3/5 —
  correct columns/empty/zone, but self-contradictory on repeats.

Strongest single datum for comprehension: synthA's coord list
(`(17,2)...(19,5)` plus falling cells) -> "columns 2-5" at **0.99**,
while synthB's list -> "columns 0-1 and 8-9" at **0.99**. This needs
pair parsing, second-element projection, and range formation — more
than token overlap (distractor digits appear inside row numbers, yet
the model is decisive and correct per image).

### 4b. Action: the brain does not drive (12 framings, all fail)

All through the real `_state_text` + real options + thinking-disabled
readout. Scenarios R1/L1/R2/L2: side tower + falling square, steer to
open space (later judged teleport-physics; see 4c). Pocket scenarios
S1/S2/S3: square one tap from a perfect 2-wide well fill (valid,
reachable, provable ground truth).

1. STD movement question, 10 options: **0/4**. Neutral 0.75-0.81;
   left/right mass 0.01-0.04 always.
2. ROT (neutral moved last): **0/4**. Neutral still wins 0.48-0.60.
   Not position bias.
3. Urgency (piece one row above tower): neutral 0.71-0.75. Not patience.
4. Explicit if/else rule on a halves-fill digest: **2/5**; steer branch
   never taken; one wrong-branch, one tie.
5. Binary A/B ladder: stated answers pass at 0.95 either letter
   (mechanics fine); one-hop map fails into A-bias. 3/4 with the pass
   being luck.
6. Few-shot (2 abstract examples): 1/2, recency copy (both B).
7. Think-then-act (150 tokens generated, letters after): **0/4 at
   ~170-211s/case**. Traces hallucinate dimensions ("10x10 grid",
   "0-15"); post-think distributions flat. Thinking confirmed ON
   (template tail shows open `<think>`).
8. Direct question ("which way should the square move"), same options:
   **0/4** but lateral mass rises to 0.06-0.17, neutral drops to ~0.53.
9. First-person ("you are the player", "your move"): **0/4**, neutral
   0.81-0.89 — worse than baseline.
10. Pockets without dynamics line: **1/3** (S1 neutral>right,
    S2 neutral, S3 neutral which is acceptable).
11. Pockets with "each decision shifts one cell" line: **1/3, numbers
    identical to (10)** (0.44->0.44). The line is ignored.
12. Pockets with fully direct options ("Move the square one cell
    left", etc.): **1/3**. S2 and S3 emit IDENTICAL distributions
    (0.38/0.20/0.16/0.26) for different states — the state is
    discounted, only a fixed prior answers.

Cleanest single datum for incapability: S3 (square perfectly aligned
over an open well) assigns DOWN — a zero-risk pure-gain drop — **0.02**.

### 4c. Bench: loop works, driver abstains (reports in `probes/`)

Two validation runs (gameplay savestate start, 300s ceiling), full
emu+policy+HUD loop, 0 failures, HUD decode perfect:

- Run 1 (no objective line): 19 decisions, tried={neutral:20},
  score 0, lines 0, top-out ~87s, infer p50 4.4s (0.2Hz).
- Run 2 (objective rendered): 19 decisions, genuine neutral/none
  20/20 (move conf 0.957 decaying to ~0.55), score 0, top-out ~88s.
- Perception on live frames verified perfect (piece tracked across
  rows, motion diffs exact, stack growth visible to row 0).

## 5. Theories proposed and their fates

| # | Theory | Fate |
|---|--------|------|
| 1 | Hop depth: 1-hop reads work, 2+ hops fail | Supported, then challenged: range formation (multi-hop) works at 0.99 while S3-down (zero-hop pure gain) gets 0.02 |
| 2 | Positives are surface overlap + priors | Challenged: distractor digits inside row numbers don't confuse it; ASCII discriminates identical-ruler grids |
| 3 | Position bias (neutral is option A) | Killed: neutral wins from last position; both letters reachable at 0.95 |
| 4 | Patience (piece high, decide later) | Killed: urgent low-piece cases still neutral |
| 5 | Advisor/RLHF hedging ("should" -> play safe) | Killed: first-person "your move" scores worse (0.81-0.89) |
| 6 | Controller-semantics gap ("hold" unclear) | Partial: direct language triples lateral mass but flips nothing; direct options still 1/3 with S2 identical to S3 |
| 7 | Missing dynamics knowledge | Killed: dynamics line changes numbers 0.00 |
| 8 | My ground truth assumed teleport physics | CONFIRMED FLAW: R-scenario "correct" moves were unreachable multi-tap fantasies (one analysis: the "correct" RIGHT creates a 5-deep hole; neutral lands flat). Rebuilt as pockets — model still fails on valid truth |
| 9 | Thinking unlocks driving | Killed: 0/4, hallucinating traces, flat posteriors |

Surviving observation, mechanism unknown: comprehension routes state
into letters decisively and correctly; action questions route almost
entirely around the state into a fixed prior (neutral > right > down >
left in 4-option direct; neutral ~0.8 in 10-option). The state is read
but not acted on.

## 6. Open question for the reviewer

1. Is there a flaw in the quiz design, ground truth (especially the
   pockets S1/S2/S3), statistics (n=3-8 per quiz — but note the
   cross-framing consistency over ~60 forwards), or reasoning above
   that, fixed, changes the verdict? Name it precisely.
2. If you believe the brain can drive, propose ONE concrete prompt
   inside the section-2 constraints: exact system line, question line,
   option list with descriptions, state-text additions. No harness
   servo, no Tetris tactics, no extra inputs.
3. If you believe the verdict, say which of these you would do next:
   (a) quiz a different <=3GB brain the same way (which one, and what
   evidence suggests it routes state to actions?); (b) abandon the
   OpenJev-Tetris path for the Laya vision backend (scores 5 today);
   (c) something else thesis-compatible.

## 7. Held-out test protocol (for a proposed prompt)

1. Reviewer sends the exact prompt words. No scenarios are shared first.
2. I generate four FRESH pocket scenarios (new well columns/widths,
   two must-go-left, two must-go-right, one-tap reachable, provable
   ground truth by the same hole analysis as S1/S2) and run the prompt
   verbatim through the real `_state_text` + thinking-disabled readout.
3. 3/4 passes: the operator was right; the framing ships. Otherwise
   the verdict stands, tested on the reviewer's own terms.

## 8. Repo pointers

- `secondplayer/policy/grid.py` — the reader (shipped, tested).
- `secondplayer/policy/openjev.py` — prompt builder (`_state_text`),
  questions, options, memory wiring.
- `secondplayer/policy/ojcore/` — scorer, template flags, loader.
- `tests/test_policy_grid.py` — golden grid tests incl. bench0 cells.
- `probes/2026-09-24-drive/` — all probe scripts from this campaign
  (scratch evidence, not maintained code), both bench JSON reports,
  think-act and think-off raw outputs.
- `docs/OJCORE_PROBE.md` — full probe history incl. Step-8/8b.
- Prior vision work (native 4B 1/6, Florence unloadable) predates this
  campaign; its scripts live only on the bench box, results are in
  `docs/OJCORE_PROBE.md` and section 4a above.

## 9. Held-out outcome (2026-09-24, same day)

The reviewer accepted the section-7 protocol with one requirement (two
observations per state, no `render(None, ...)`) and proposed a prompt:
controller-role system line, five temporal occupancy fields from mask
set-ops (FILLED NOW / PREVIOUSLY / NEWLY FILLED / NEWLY EMPTIED /
UNCHANGED), objective-driven question, directions defined in row/column
geometry. Verbatim in `probes/2026-09-24-drive/heldout.py`. Four fresh
pockets (new wells 6-7, 2-3, 5-6, 3-4; two must-go-right, two
must-go-left; previous frame = square one row up, mover signature clean).

Result: **0/4**. All four answered down-right (0.28-0.33); neutral fell
to 0.19-0.25, up 0.17-0.22, correct laterals 0.02-0.07. Mirrored states
again emit near-identical distributions (max delta 0.04). The
`render(None)` flaw was real but not load-bearing: fixed, the verdict
got stronger. Per the protocol's own terms this is materially stronger
evidence against this 4B quant in single-forward letter readout.

Standing concessions to the review (all verified against my scripts):
the old action quizzes withheld temporal info they graded on; the live
"moving object" sentence overclaims (symmetric difference + stale prev);
think-act scored mid-thought; S3-down-0.02 rhetoric was spin over a
pass; "~60 forwards" overstated independent evidence; "open space" was
vague. None of them, fixed, changed the outcome.

## 10. Clean round on the deployed quant (2026-09-24)

The reviewer objected the held-out ran on runtime-quantized base weights
(BF16 embeddings) rather than the shipped `qwen3.5-4b-qint4e` (int8
embeddings + sidecar), that `dims_line`'s overclaim sentence was still
prepended, and that down-right (DOWN+RIGHT) should count toward RIGHT
cases. All three addressed: `probes/2026-09-24-drive/cleanround.py`
loads qint4e through production `ojcore.load_model` + `ojcore.score`
(XPU, dropped head, sidecar active — only the system line, which is the
prompt under test, is swapped), uses a convention-only legend, computes
temporal fields and the FrameDiff observation from the masks, and runs
four NEW wells (7-8, 1-2, 8-9, 0-1) with fresh prev/current pairs.

Result: **0/4 exact, 2/4 effective**. All four answered down-right
(0.24-0.26); the two "effective" passes are the bias pattern (same
answer where LEFT is needed), not discrimination — mirror states again
near-identical, correct laterals at 0.02-0.05. Gate was 3/4. The verdict
now holds across three load paths (runtime qint4, deployed qint4e via
production scorer, live bench play) with numerically matching
abstention/invariance signatures.

## 11. PlayJev-0.8B factorized: transfer fails (2026-09-24)

Per the reviewer's redirect, tested PlayJev-0.8B (Qwen3.5-0.8B finetune,
10 browser games incl. Tetris, 2.2M frames, moves shuffled in training;
weights + code from public upstream, no local-oracle contact) on the
held-out SNES pockets with factorized decisions inside its trained
2-7-action range: lateral {left, right, none} + drop {drop, none},
upstream instructions verbatim, 1-frame and 2-frame modes, training
resolution (long_side=448), real hook descriptions.

Result: **3/10 in 1-frame and 3/10 in 2-frame mode**. With five
3-way and five 2-way questions, chance expectation is 4.17/10 and
scoring <=3 happens ~34% of the time, so the bare 3/10 aggregate is
weak evidence on its own and must not be read as "below chance".
The decisive evidence is the state invariance: all five lateral
answers go left at 0.86-0.89 regardless of well side, all five drop
answers go drop at 0.57-0.64, and mirrored states emit
near-identical distributions. The 26-option uniformity the reviewer
diagnosed was never the blocker: with 2-3 options the model still
cannot distinguish mirrored SNES states. Zero-shot end-to-end SNES
transfer fails (browser Tetris rendering -> SNES rendering), so
controller factorization cannot help. This experiment cannot
separate "vision representation fails" from "vision representation
changes but the trained decision layers ignore it on this OOD
rendering"; the record claims only the end-to-end failure.
`allowed_mass=0.000` throughout is genuine behavior (CPU float32
reference bit-matches XPU); relative letter order still decides, and
it decides by fixed prior. Scripts: `probes/2026-09-24-drive/pj*.py`.

## 12. Laya-201M pinned baseline: timeout, never reached gameplay (2026-09-25)

Per the reviewer's work order: laya-vision code pinned to
`404fc4a63208d14b544d5c852f81e88972d267bb` (with `load_vlm(revision=)`),
model pinned to `thaitea/laya-vision-201m` at
`0b6228f7a0762566de1c4539e9aa4eb1c1aef5f4`, revision wired through
`RuntimeConfig.model_revision`, and every report now carries
`model_provenance` (repo, requested/resolved revision, code SHA, params).
No gameplay prompts, learning behavior, pulse timing, or bench semantics
changed. Historical score-5 run attributed (Hub history + local cache) to
the old `thaitea/laya-vision@d1fbdc06` 237M/no-game checkpoint under code
`05ba043`; its JSON report is lost, only the numbers survive.

Mirror diagnostic (`probes/2026-09-25-laya/layamirror.py`, production
`LayaVisionPolicy.decide`, learn=False, H5-H8/A1 frames, XPU): load 5.5s
cached, resolved==requested, params **201161347**. Distributions DO change
with pixels (H5 keep .161/down-left .152; H6 neutral .169; A1 down-left
.214) - no PlayJev-style invariance, deterministic across runs. But the
required lateral mass stays .05-.08 on all four H states, confidences
.025-.042, argmax never the required direction. Steady inference
~140-155ms, first 2.1s.

Full bench (`config/bench-tetris.toml` unchanged except the model pin,
.200 Arc XPU, 30-min ceiling): **reason=timeout, 1801.8s, gameplay=no,
score/lines None, 3544 decisions, 3885 polls, cadence 1.97Hz, infer p50
452ms / p95 464ms, failures 0, decode_gaps 3885.** Forced actions 8/3545
turns, so early play was model-selected, not harness-driven. The model
held `UP+LEFT+X+Y` (2201x) and `DOWN+LEFT+X+Y` (1212x); START was tried
once (sweep, effect menu) and never model-selected; effects read ambient
3522x. Movement conf p50 .087, buttons p50 .038. The harness worked (0
failures, inputs flowed, live video); the 201M brain never started a game.
Caveat: score-5 was likewise a single run of a menu-lottery benchmark, so
single-run variance cuts both ways; this run establishes the pinned
baseline, not a model ranking.

## 13. Laya-201M motor test: survives paused, plays nothing (2026-09-25)

The reviewer's narrow follow-up: same pinned 201M/code, `--from-state
bench0.state` (level 0, empty board, first piece falling), 5-min ceiling,
nothing else changed. Result: `timeout`, 301.5s, gameplay=yes from 0.51s,
**score 0, lines 0, level 0**, 579 decisions, 1.92Hz, infer p50 363ms /
p95 566ms, 0 failures, 7 decode gaps. Forced 14/580 turns.

Only 8 pieces were dealt in 300s (stats sum): the agent pressed START
twice (pausing/unpausing around its own sweep) and spent the run spamming
A (269x) and UP+LEFT+A (294x) - rotation without steering or dropping
(DOWN 2x, no model-selected drops). Effects read ambient 451x; lateral
motion was never intentionally produced. Survival was trivial (8 pieces
cannot top out an empty board), progress zero.

Verdict per the reviewer's rule: Laya cannot function as a motor/reflex
layer for Tetris even when placed in gameplay. Standalone System-1
benchmarking stops here: Qwen/OpenJev (no action routing), PlayJev
(no SNES transfer), Laya-201M (no policy, no motor function) are all
eliminated as solo brains. Next direction under discussion: hierarchical
cognition (Qwen-class intention generator) over a fast execution policy.

## 14. Qwen3.5-4B native vision: reads text, blind to fine space (2026-09-25)

The reviewer's authorized test: qualify stock Qwen vision with the official
BF16 model (snapshot `851bf6e8`) + official Qwen3VLProcessor, offline, XPU,
plain generation (not logits), thinking disabled - no action selection, no
quantization. 140 questions (134 main + 6 controls) over real emulator
frames (Tetris/Dr. Mario gameplay, menus, title), every label human-verified
against zoomed crops, 23 mirrored pairs, blank/shuffled controls.
Artifacts: `probes/2026-09-25-vision/` (manifest, runner, captures, results).

Result: **79/134 = 59%, gate 90%+ FAILED.** Controls 1/6 (collapse as
required). The split is the finding: text/menus/selection are strong
(tsub/dsub 12/12, gsel 7/8, sel 6/6, hiscore 4/4, virus-OCR 4/4, speed 4/4,
drm-more 4/4), while fine spatial grounding is ABSENT: thirds 8/24 all
answered CENTER regardless of truth, directions DOWN-leaning prior 6/18,
pill colors one fixed option 1/6, 17/23 mirror pairs answered identically
across flips. Stats-OCR errors were row swaps (digits read, rows
misattributed). 6/6 geo-column answers are VOID (prose cut at 24 tokens,
never reached a letter - method flaw, excluded from interpretation).
Resolution is not the cause: the processor patches full-res (288 patches),
and 2x-upscaled thirds still failed. Latency p50 1.5s / p95 5.1s also
misses 200ms. One gap: no true higher-res material exists (single SNES
cart); 2x scale probe substituted.

Verdict per the progressive plan: BF16 fails the gate, so abandon Qwen
vision FOR SPATIAL/CONTROLLER GROUNDING. It reads dialogue/menus/HUDs and
could still serve a System-2 reader, but it cannot localize game objects -
no "east door", no "second enemy", no piece steering. This refines (not
reverses) the old probe: Qwen vision contributes text/salient-marker
signal but no fine-spatial signal. Infra note: the XPU UR runtime leaks
pinned host memory per generate call, so the quiz ran as 18 fresh-process
chunks of 8; the first single-process attempt yielded 125/140 infra errors
and was discarded, and one teardown segfaulted.
