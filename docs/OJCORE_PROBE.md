# ojcore Step-2 Probe Record (2026-09-23)

Method: `/tmp/oj_probe.py` — one device × backend combo per run, JSON record,
real `ojcore` prompt + letter readout, 2 scored reps. Never crashes the matrix:
failures are recorded with error + trace tail.

## Matrix (measured)

| Device | Backend | Result |
|---|---|---|
| .200 CPU (6t) | fp (fp32) | OK — load 5.6s, 8.5s/score, argmax up@0.43 |
| .200 CPU | quanto qint4 (weights-only, fp activations, no int8) | OK — load+quant 42.8s, 12.6s/score, argmax up@0.53 |
| .200 CPU | torchao int4 | ABSENT — torchao 0.18 removed: its eager import breaks transformers on torch 2.8 (needs torch ≥2.11 APIs) |
| .200 XPU (Arc 16GB) | fp | BLOCKED — VRAM held by operator processes (fleet router :8080, minicpm :59559); OOM at 7.8GB alloc |
| local CUDA (2×32GB + 12GB) | fp / quanto / torchao / AWQ | BLOCKED — VRAM held by operator jobs (~30GB ea PG500, A2000 full) |
| local CPU | fp / quanto | BLOCKED — RAM 57/62GB held by operator jobs |
| AWQ 4-bit-g128 (5.7GB, config-verified) | CUDA run | PENDING — same local VRAM block; no CPU/XPU kernel path exists (architectural) |
| MPS / ROCm | — | NO HARDWARE |

## Environment findings (measured)

- transformers ≥5.15 needs `ScalingType`/`scaled_grouped_mm` (torch ≥2.11 API).
  Local venv now torch 2.11.0+cu128 + torchvision 0.26 (isolated; user-site untouched).
- .200 XPU torch is pinned at 2.8.0+xpu: 2.10/2.11 import-fail (box SYCL runtime
  too old, undefined `sycl::queue` symbol); 2.14 segfaulted historically. No torch
  upgrade possible without a system oneAPI upgrade (out of scope, shared box).
- torchao must stay OUT of the .200 venv (breaks transformers import on torch 2.8).
- Qwen3.5 is a linear-attention hybrid: `causal_conv1d` and `flash-linear-attention`
  are absent everywhere → reference kernels (correct, slower). Speed work later.
- Local HF cache lacked `refs/main` (offline ID resolution failed); fixed by writing
  the known snapshot hash. .200 cache resolves IDs fine via `HF_HOME=/data/tmp/hf-cache`.
- Letter slots A–J verified single round-trip tokens (ids 32–41) on the Qwen3.5
  tokenizer; readout contract holds in-process.

## Fixes applied from probe evidence

- `score()` uses `torch.no_grad`, not `inference_mode` (quanto bumps version
  counters that inference tensors forbid).
- quanto serialization API confirmed: `safe_save(state_dict, path)` (0.2.0).

## Step-3 agreement + weight file (measured on .200 CPU)

- 20 authored rows (10 states × movement/buttons): fp vs qint4 argmax **14/20**.
- All 6 flips go abstain→action (fp keep/none → qint4 up/left/right/A/START);
  2 close-call, 4 confident. No reverse flips, no action↔action flips.
- File: `/data/tmp/ojcore/qwen3.5-4b-qint4/model.safetensors`, 4.6GB
  (linears qint4 ≈2.1GB + fp32 embeddings/norms ≈2.7GB; config + tokenizer alongside).
- Load-back verified: from_config → quantize → safe_load → dtype cast →
  freeze → score matches in-memory qint4 exactly (movement-0 up@0.6058).
- File-load path is what the step-4 `quanto_qint4` loader implements.
- Agreement bar is the operator's call (spec section 5.2).

## Step-6 XPU acceptance (measured on .200 Arc, alongside operator servers)

- qint4 file loads to XPU (4.6GB fits in remaining VRAM): load 57s (cold CPU-side
  unpack), score 3.05s cold / **1.35s warm** — ~9x the CPU 12.6s.
- Cross-device determinism: XPU up@0.5308 == CPU qint4 up@0.5308 exactly.
- Policy e2e (real Tetris reference frame, stub caption, learn on): 3 turns,
  LEFT/RIGHT/UP, memory/sweep/attribute path active, ~3-4.6s/turn for 2 scores.
- Caveats: one `solve_triangular` op falls back XPU→CPU per forward (linear-attn
  reference path; correct, costs time); bf16 fp baseline still OOMs while the
  Arc is shared (needs ~8GB free; only the 4.6GB quant fits today).

## Step-7 sub-3GB readout (measured on .200 Arc + local CUDA)

Peak was dominated by the 248k-row lm_head: quanto unpacks the whole head on
any slice (~2GB transient), so the scorer now runs the backbone only and dots
the final hidden state against slot rows from a `head_rows.pt` sidecar written
at quant time (`scripts/build_qint4.py`; 16 rows, ~100KB). The loader drops the
dead int4 head when the sidecar is present. Subset-vs-full probs match to 2e-6
(Qwen3-0.6B, single + chunked prefill).

- 2B (`qwen3.5-2b-qint4b`, bf16): **1.55GB peak every rep**, 1.85s cold /
  0.61s warm per score on Arc.
- 4B (`qwen3.5-4b-qint4e`, fp32+QEmbed): **2.79GB peak every rep**, 3.6s cold /
  1.3s warm per score on Arc. Shipped brain (build script reproduces it).
- Gate A (file-qint4 CPU vs XPU, 20 rows): **20/20** both sizes
  (max prob diff 0.03 on 2B, 1e-5 on 4B).
- Gate B (fp vs file-qint4, 20 rows): **14/20 on 4B** (all 6 flips
  abstain→action, as before), 10/20 on 2B (same flip direction; smaller
  model quantizes noisier).
- Captioner SmolVLM-256M on Arc: 0.99GB peak, ~1.7s warm per caption;
  reads HUD text + piece mentions. Florence-2 fails on this transformers
  (`forced_bos_token_id`); structured 3-sentence prompts degenerate into
  repetition loops, so the shipped prompt stays in short simple sentences
  with repetition_penalty=1.2.
- Full decide() (caption + 2 scores, 4B): ~4-6s warm, 3.63GB combined peak.
  Every warm jev request over 200ms logs the suboptimal-performance warning.
- Tetris quiz (text states, 4B): title→START@0.80, fresh board→neutral/none,
  piece-over-well→down@0.77 with the shipped game-conditional hint, no lateral
  steering (single-token readout does not do spatial tactics; a lateral rule
  hint mirrors left/right and stays out).

## Pending (needs operator)

- Full cold-boot Tetris bench needs the benchmark ROM on .200 (not on either
  host or the mounts; searched 2026-09-23). Run when placed:
  `secondplayer --config config/bench-tetris-openjev.toml bench tetris <rom>`.
- CUDA verdict (local, 2026-09-23): 2B qint4 fits + computes on PG500-216
  (Volta CC 7.0) under a torch-2.5.1/cu124 probe env: 1.29GB warm peak,
  1.43s warm (cold rep 638s one-time). Project torch 2.11/cu128 has no
  Volta kernels, so `auto` there skips the PG500s (compute probe) and takes
  the A2000 (0.8GB free, too tight for the 2B) — documented, not silent.
  torchao/AWQ CUDA cells still need freed VRAM.
- bench-tetris-openjev.toml + scripts/build_qint4.py synced to .200 2026-09-23.
- Spec section 5: agreement bar (14/20 4B, abstain→action flips), CUDA path,
  permanent weight home.

## Step-8 screen reader: caption out, cell grid in (measured on .200 Arc, 2026-09-24)

The caption path was never validated on spatial detail, cost ~1.7s + 0.99GB
per decision, and native 4B vision scored 1/6 on a spatial quiz (same as its
text-only control) at 6.34GB peak. Follow-up probes then found the quiz
readout itself was misconfigured: the probes scored pre-thinking logits
while the shipped scorer disables thinking. Re-ran deployment-faithful
(thinking off, letter slots, qint4 4B, same 6 cases over the reference frame
plus two repainted-column synths):

- TEXT-ONLY: 4/6 (prior luck: the piece is middle in all 3 frames).
- COORD (filled-cell list + column ruler): 5/6, confident (0.87-0.99).
- ASCII-WELL (10x20 grid text): 5/6, confident (0.89-0.93).
- IMG (native vision, full-frame pixels): 4/6 but emits near-identical
  distributions across different images -- zero image contribution, prior only.
- The one shared COORD/ASCII miss is sensor ambiguity (single frame cannot
  separate the falling piece from stacked blocks), not model failure.

Two-frame motion resolves it: current-frame coords plus the changed-cell set
scores 4/4 (moving object located, static stacks identified with the mover
excluded, true-empty called empty). The shipped reader
(`secondplayer/policy/grid.py`) is this exact renderer: dominant-color
background, center-sampled cells, `ox,oy,cell_w,cell_h,cols,rows` display
geometry from `screen_grid` (blank = whole-frame adaptive). Well ROI
`93,44,8,8.2,10,20` reproduces the probe cells bit-exact on bench0
(`tests/test_policy_grid.py` golden). Captioner, caption prompt, and
`caption_model` config removed from the OpenJev backend.

## Step-8b verdict: the 4B reads but cannot drive (measured on .200 Arc, 2026-09-24)

The Step-8 quizzes tested comprehension, never action selection. Action
probes through the real `_state_text` + options + thinking-disabled readout:

- Action quiz (tower + falling square, indisputable steer direction): **0/8**
  across standard and rotated option orders. Neutral wins everywhere
  (0.75-0.81 standard, 0.48-0.60 rotated-last); left/right mass ~0.01-0.04
  always. Not position bias: genuine abstention, state-invariant.
- Urgency control (piece one row above the tower): still neutral, 0.71-0.75.
- Explicit-rule quiz (halves-fill digest + stated if/else rule): 2/5; the
  steer branch is never taken.
- Binary A/B ladder: answer-stated cases pass at 0.95 either letter
  (mechanics fine); one-hop map fails into A-bias.
- Think-then-act (thinking on, 150-token trace, letters scored after):
  **0/4 at ~200s/case**; traces hallucinate grid dimensions, post-think
  distributions go flat.

Mechanism: the single-forward readout matches options by surface overlap
plus priors. It does not compare, evaluate conditionals, isolate subsets,
or map relations -- the Step-8 positives reduce to overlap (stack columns
share digits with the state) plus prior luck (middle). Consequence: no
prompt, sensor, manual, or cadence change makes Qwen3.5-4B select Tetris
actions. Validation benches confirm: 2 runs x 19 decisions, genuine
neutral/none every turn (0.5-0.95 conf), score 0, lines 0, top-out ~88s,
0.2Hz at 4.4s/decision, 0 failures, HUD decode perfect. Loop works;
driver abstains.
